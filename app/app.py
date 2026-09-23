"""Single-page Streamlit chat app over a LangChain agent on Databricks.

The agent is built with ``langchain.agents.create_agent`` and talks to a
Databricks-hosted model through the Unity Gateway — no agent server, no
FastAPI, no ResponsesAgent wrapper. Streamlit calls the compiled graph directly.

Adapted from the ``agent-langgraph-advanced`` app template, reduced to just the
agent: the template's agent_server (FastAPI + MLflow ResponsesAgent) is left out,
while its Lakebase memory layer is kept — see ``memory``.
"""

import logging
import os
from datetime import datetime
from typing import Any, NamedTuple

import streamlit as st
from databricks.sdk import WorkspaceClient
from databricks_langchain import ChatDatabricks
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.tools import tool

import memory as mem
import tracing as trc
from async_bridge import iter_coroutine
from mcp_client import genie_mcp_url, load_genie_tools

# Streamlit leaves the root logger at WARNING, which drops this module's INFO
# lines — including which credentials the app ended up authenticating with.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Databricks serving endpoint backing the agent, reached through the Unity Gateway.
LLM_ENDPOINT_NAME = os.environ.get("LLM_ENDPOINT_NAME", "databricks-gpt-6-astra")

SYSTEM_PROMPT = """You are a helpful assistant. Use the available tools to answer questions.

For questions about data, use the Genie tools. `genie_ask` starts a question and
may return before the answer is ready — when it reports an in-flight status, call
`genie_poll_response` until the status is terminal, then use
`genie_get_query_result` if you need the actual rows behind an answer. Pass the
`conversation_id` from a previous Genie call when a question follows up on it, so
Genie keeps its own context; omit it entirely when starting a new question.

You have long-term memory across conversations, but only what you write with
`save_user_memory` survives — acknowledging a fact in your reply does NOT store
it, and the next conversation starts blank. So whenever the user states a durable
fact or preference about themselves — their name, role or team, how they like
answers formatted, the tables or metrics they care about, anything they'd expect
you to know next time — you MUST call `save_user_memory` in that same turn,
before you reply. Use a short descriptive key and a JSON-object value, e.g. key
`preferred_units` with value `{"preferred_units": "metric"}`. Do this
proactively, without being asked to remember, and don't announce it unless it's
relevant. Skip one-off questions and anything they'd expect you to forget.

Call `get_user_memory` at the start of a turn when your answer depends on
something the user told you earlier. Use `delete_user_memory` when they correct
or retract something you saved."""

st.set_page_config(page_title="Agent Chat", page_icon="🤖", layout="wide")


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
@tool
def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().isoformat()


LOCAL_TOOLS = [get_current_time]


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
# Env vars the Databricks Apps runtime sets for the app's service principal.
# They must be absent when building an on-behalf-of-user client, or the SDK
# raises "more than one authorization method configured".
_SP_OAUTH_KEYS = (
    "DATABRICKS_CLIENT_ID",
    "DATABRICKS_CLIENT_SECRET",
    "DATABRICKS_TOKEN",
)


def _obo_token() -> str | None:
    """Return the on-behalf-of-user token forwarded by Databricks Apps, if any."""
    try:
        headers = st.context.headers
    except Exception:
        return None
    if not headers:
        return None
    return headers.get("X-Forwarded-Access-Token")


def _workspace_client(token: str | None, host: str) -> WorkspaceClient:
    """Authenticate as the logged-in user in Databricks Apps, else via the CLI profile."""
    if token and host:
        removed = {k: os.environ.pop(k) for k in _SP_OAUTH_KEYS if k in os.environ}
        try:
            logger.info("Authenticating as the signed-in user (forwarded token).")
            return WorkspaceClient(host=host, token=token)
        finally:
            os.environ.update(removed)
    # Locally this is the CLI profile. Deployed it is the app's service principal,
    # which is worth knowing: the SP can query the serving endpoint but has no
    # Genie grants, so Genie failures look identical to a missing scope.
    logger.info(
        "No forwarded user token (token=%s, host=%s); using default credentials.",
        bool(token),
        bool(host),
    )
    return WorkspaceClient()


def _user_id(workspace_client: WorkspaceClient) -> str | None:
    """Who the memories belong to — their email, or ``None`` if it can't be known.

    The email is the stable key: it survives token rotation, and locally it
    matches the profile's own user so a developer sees the memories they wrote.

    The SDK call is primary because it's what the memories already on disk were
    keyed by. When it fails — a blipping on-behalf-of-user token 401s
    ``current_user.me()`` — fall back to the identity header Databricks Apps
    forwards: it comes from the verified session and makes no API call, so it
    can't 401 the way the SDK call just did. Returning ``None`` rather than a
    literal ``"unknown"`` is deliberate: the caller then runs *without* memory for
    the session instead of pooling every unidentified visitor under one shared
    key, which used to both hide the real user's memories and mix strangers
    together.
    """
    try:
        if name := workspace_client.current_user.me().user_name:
            return name
    except Exception:  # noqa: BLE001 — fall back to the forwarded header
        logger.warning(
            "current_user.me() failed; trying the forwarded identity header.",
            exc_info=True,
        )

    try:
        headers = st.context.headers or {}
    except Exception:  # noqa: BLE001 — no request context (e.g. local)
        headers = {}
    for header in ("X-Forwarded-Email", "X-Forwarded-Preferred-Username"):
        if email := headers.get(header):
            return email
    return None


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
class Built(NamedTuple):
    """A compiled agent plus what the sidebar needs to describe it."""

    graph: Any
    genie_tools: list[str]
    mcp_url: str
    genie_error: str
    memory: mem.Memory
    tracing: trc.Tracing
    user_id: str | None


@st.cache_resource(show_spinner=False)
def build_agent(token: str | None, host: str, endpoint: str) -> Built:
    """Compile the LangChain agent, cached per user so each rerun reuses it.

    Returns the agent alongside the Genie MCP tool names, so the sidebar can say
    whether Genie actually loaded instead of claiming tools that aren't there.

    ``use_responses_api`` is required, not optional: ``databricks-gpt-6-astra``
    is a reasoning model, and the Databricks /v1/chat/completions route rejects
    function tools alongside reasoning_effort ("To use function tools, use
    /v1/responses"). Setting reasoning_effort="none" is not a way out either —
    this model doesn't accept that value. ``use_ai_gateway`` routes the request
    through Unity Gateway V2.
    """
    workspace_client = _workspace_client(token, host)
    model = ChatDatabricks(
        model=endpoint,
        use_ai_gateway=True,
        use_responses_api=True,
        workspace_client=workspace_client,
    )
    # Fetched under the signed-in user's credentials, so Genie answers from the
    # data that user can see.
    genie_tools, genie_error = load_genie_tools(workspace_client)

    # Lakebase, by contrast, is reached with the app's own credentials: the
    # memories are the app's storage, partitioned by user.
    memory = mem.open_memory(mem.init_lakebase_config())
    user_id = _user_id(workspace_client)
    if memory.enabled and not user_id:
        # No stable identity to key memories on — run without memory rather than
        # pool it under a shared bucket. A mislabeled memory is worse than none.
        memory = mem.Memory(None, None, "Couldn't identify the signed-in user.")
    memory_tools = (
        mem.memory_tools(memory.store, user_id) if memory.enabled else []
    )

    # Enable MLflow autolog before the agent runs, so every turn is traced. This
    # is process-global and best-effort; a tracing problem never blocks chat.
    tracing = trc.init_tracing()

    agent = create_agent(
        model=model,
        tools=[*LOCAL_TOOLS, *genie_tools, *memory_tools],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=memory.checkpointer,
        store=memory.store,
    )
    return Built(
        graph=agent,
        genie_tools=[t.name for t in genie_tools],
        mcp_url=genie_mcp_url(workspace_client),
        genie_error=genie_error,
        memory=memory,
        tracing=tracing,
        user_id=user_id,
    )


def get_agent() -> Built:
    host = (
        os.environ.get("DATABRICKS_HOST")
        or os.environ.get("DATABRICKS_WORKSPACE_URL")
        or ""
    ).strip()
    return build_agent(_obo_token(), host, LLM_ENDPOINT_NAME)


# ---------------------------------------------------------------------------
# Message helpers
# ---------------------------------------------------------------------------
def _text_blocks(content) -> list[dict]:
    """Return the ``text`` blocks of a message's content.

    The Responses API returns content as a list of typed blocks — ``text``
    alongside ``reasoning`` and ``function_call`` — so plain concatenation would
    dump encrypted reasoning payloads into the chat. Chat-completions responses
    arrive as a plain string instead.
    """
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if not isinstance(content, list):
        return []
    return [
        block
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    ]


def text_of(content) -> str:
    """Full text of a *completed* message — used to replay the transcript.

    When token streaming is on, the accumulated message keeps the incremental
    deltas *and* a closing block repeating the whole answer (marked by an
    ``annotations`` key), so concatenating everything renders it twice. The
    closing blocks are authoritative when present; without streaming there is
    only one of them, and chat-completions content has none at all.
    """
    blocks = _text_blocks(content)
    complete = [block for block in blocks if "annotations" in block]
    return "".join(block.get("text", "") for block in (complete or blocks))


def delta_of(content) -> str:
    """Text of a streamed *chunk*, dropping the Responses API's closing repeat.

    A streaming response emits incremental deltas and then one final block
    carrying the whole answer again, marked by an ``annotations`` key. Without
    dropping it the answer renders twice. Completed messages hold that block
    only once, which is why ``text_of`` keeps it and this does not.
    """
    return "".join(
        block.get("text", "")
        for block in _text_blocks(content)
        if "annotations" not in block
    )


def render_tool_activity(container, messages: list) -> None:
    """Render each tool call and its result into ``container``."""
    calls = [
        call
        for msg in messages
        if isinstance(msg, AIMessage)
        for call in (msg.tool_calls or [])
    ]
    if not calls:
        return

    results = {
        msg.tool_call_id: msg.content
        for msg in messages
        if isinstance(msg, ToolMessage)
    }
    with container:
        label = f"🔧 {len(calls)} tool call{'s' if len(calls) > 1 else ''}"
        with st.expander(label):
            for call in calls:
                st.markdown(f"**{call['name']}**")
                if call.get("args"):
                    st.json(call["args"], expanded=False)
                result = results.get(call.get("id"))
                if result is not None:
                    st.caption("Result")
                    st.code(str(result), language="text")


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.title("🤖 Agent Chat")

with st.sidebar:
    st.subheader("Settings")
    st.text_input("Model endpoint", value=LLM_ENDPOINT_NAME, disabled=True)
    st.caption("Served through the Unity Gateway (`use_ai_gateway`).")

    # Builds the agent on first load, which is also the first Genie MCP call and
    # the first Lakebase connection — so a broken endpoint, a missing `genie`
    # scope or an unreachable database shows up here rather than silently in the
    # middle of someone's question.
    with st.spinner("Connecting to Genie and Lakebase…"):
        try:
            built = get_agent()
            agent_error = None
        except Exception as exc:  # noqa: BLE001 — surfaced in the sidebar
            logger.exception("Agent construction failed")
            built, agent_error = None, exc

    if agent_error or built is None:
        st.error(f"Agent unavailable: {agent_error}")
    else:
        st.caption(f"Local tools: {', '.join(t.name for t in LOCAL_TOOLS)}")
        if built.genie_tools:
            st.caption(f"Genie One MCP: {', '.join(built.genie_tools)}")
            st.caption(f"`{built.mcp_url}`")
        else:
            st.warning(
                "Genie MCP tools unavailable — the agent is running without them. "
                "If this says the token is missing the `genie` scope, sign out of "
                "the app and back in to consent to it."
            )
            st.caption(f"Genie said: {built.genie_error}")

        if built.memory.enabled:
            st.caption(f"Long-term memory: Lakebase, as `{built.user_id}`")
        else:
            st.warning(
                "Long-term memory unavailable — this conversation won't be "
                "remembered after you close the tab."
            )
            if built.memory.error:
                st.caption(f"Lakebase said: {built.memory.error}")

        if built.tracing.enabled:
            st.caption(f"MLflow tracing: `{built.tracing.experiment}`")
        elif built.tracing.error:
            st.caption(f"MLflow tracing off — {built.tracing.error}")

    if st.button("New conversation", width="stretch"):
        # With a checkpointer the transcript lives in Postgres, so starting over
        # means pointing at a new thread rather than clearing a local list. The
        # pointer is stored too, so a reload doesn't resurrect the old thread.
        if built is not None and built.memory.enabled:
            st.session_state["thread_id"] = mem.start_new_thread(
                built.memory, built.user_id
            )
        st.session_state["history"] = []
        st.rerun()

if built is None:
    st.stop()

# Where the transcript comes from. With memory the checkpointer is the source of
# truth, keyed by a thread id that outlives the browser tab; without it the
# transcript is whatever this session happens to be holding.
if built.memory.enabled:
    if "thread_id" not in st.session_state:
        st.session_state["thread_id"] = mem.current_thread_id(
            built.memory, built.user_id
        )
    run_config = {"configurable": {"thread_id": st.session_state["thread_id"]}}
    history = mem.load_history(built.graph, run_config)
else:
    run_config = {}
    history = st.session_state.setdefault("history", [])

# Read on the script thread and closed over below: the traced span runs on the
# shared event loop, which has no Streamlit ScriptRunContext, so it can't touch
# st.session_state itself. The thread id doubles as the trace session id, so a
# whole conversation's turns group together in the MLflow UI.
session_id = st.session_state.get("thread_id")

# Replay the transcript. It holds real LangChain messages, so a rerun re-renders
# without calling the model again.
#
# One user question can produce several messages (a tool-calling AIMessage, the
# ToolMessages, then the answering AIMessage). They are replayed as one
# assistant bubble so the transcript matches what was rendered live.
index = 0
while index < len(history):
    message = history[index]
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(text_of(message.content))
        index += 1
        continue

    run_end = index
    while run_end < len(history) and not isinstance(history[run_end], HumanMessage):
        run_end += 1
    run = history[index:run_end]

    body = "".join(
        text_of(msg.content) for msg in run if isinstance(msg, AIMessage)
    )
    if body or any(isinstance(m, AIMessage) and m.tool_calls for m in run):
        with st.chat_message("assistant"):
            render_tool_activity(st.container(), run)
            if body:
                st.markdown(body)
    index = run_end

if question := st.chat_input("Ask the agent…"):
    asked = HumanMessage(question)
    if not built.memory.enabled:
        history.append(asked)
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        # Reserved above the answer so tool activity renders before the text,
        # even though it is only known once the stream has run.
        tool_slot = st.container()
        new_messages: list = []

        def token_stream():
            """Yield answer text as it arrives, collecting full messages as a side effect.

            ``updates`` carries each node's completed messages (what history
            needs), ``messages`` carries token deltas (what the user sees), so
            one pass over the stream serves both.

            The turn runs through ``astream``, not ``stream``: the Genie MCP tools
            are coroutine-only, and a sync run raises NotImplementedError as soon
            as one is called. ``iter_coroutine`` bridges it back to the plain
            iterator ``st.write_stream`` wants.
            """
            agent = built.graph
            # With a checkpointer the graph already has the conversation, so the
            # turn sends only the new question; resending the transcript would
            # append it to the checkpoint a second time. Without one, the whole
            # list has to go every turn because nothing else holds context.
            #
            # Either way this is read here, on the script thread: the lambda below
            # runs on the shared event loop, which has no Streamlit
            # ScriptRunContext, and touching st.session_state there raises
            # "st.session_state has no key".
            payload_messages = [asked] if built.memory.enabled else list(history)

            async def astream():
                """Drive the agent, wrapped in one MLflow trace per turn.

                autolog already traces the run; the enclosing ``chat_turn`` span
                exists to hang the whole turn under a single named root, stamp it
                with the session and user (so a conversation groups in the UI),
                and — crucially — carry the final answer as the trace's output.
                MLflow shows a trace's response from its root span, so without
                set_outputs here the trace reads as having produced nothing even
                though the answer is nested in the model span. The answer is
                rebuilt from the ``updates`` messages via ``text_of`` rather than
                the token deltas, so it isn't the Responses API's doubled text.

                Both this span and the autolog spans it parents run here on the
                shared loop, so trace context propagates cleanly — opening the
                span on the Streamlit thread would not reach them.
                """
                graph_input = {"messages": payload_messages}
                kwargs = dict(config=run_config, stream_mode=["updates", "messages"])
                if not built.tracing.enabled:
                    async for item in agent.astream(graph_input, **kwargs):
                        yield item
                    return

                import mlflow

                with mlflow.start_span(name="chat_turn") as span:
                    span.set_inputs({"question": question})
                    metadata = trc.turn_metadata(session_id, built.user_id)
                    if metadata:
                        mlflow.update_current_trace(metadata=metadata)
                    answered: list = []
                    async for item in agent.astream(graph_input, **kwargs):
                        mode, payload = item
                        if mode == "updates":
                            for node_update in (payload or {}).values():
                                answered.extend((node_update or {}).get("messages", []))
                        yield item
                    span.set_outputs(
                        {
                            "response": "".join(
                                text_of(m.content)
                                for m in answered
                                if isinstance(m, AIMessage)
                            )
                        }
                    )

            events = iter_coroutine(astream)
            for mode, payload in events:
                if mode == "messages":
                    chunk, _meta = payload
                    if isinstance(chunk, AIMessageChunk):
                        if delta := delta_of(chunk.content):
                            yield delta
                elif mode == "updates":
                    for node_update in (payload or {}).values():
                        new_messages.extend((node_update or {}).get("messages", []))

        try:
            with st.spinner("Thinking…"):
                st.write_stream(token_stream())
        except Exception as exc:
            logger.exception("Agent invocation failed")
            st.error(f"Agent request failed: {exc}")
            # Drop the unanswered question so the next turn isn't sent a
            # trailing user message with no reply. The checkpointed case needs
            # nothing: a turn that raised was never committed to the checkpoint.
            if not built.memory.enabled:
                history.pop()
        else:
            render_tool_activity(tool_slot, new_messages)
            if not built.memory.enabled:
                history.extend(new_messages)
