# agent-genie-one-long-term-mem

Long-term memory for a Databricks agent.

A single-page Streamlit chat app over a [LangChain](https://docs.langchain.com/oss/python/langchain/agents)
agent built with `langchain.agents.create_agent`, talking to a Databricks-hosted
model through the [Unity Gateway](https://docs.databricks.com/aws/en/ai-gateway/).
Ask a question in the chat box; the answer streams in token by token, with any
tool calls shown inline.

Data questions go to **Genie One** over MCP, called on behalf of the signed-in
user — so Genie answers from the data that person can see, and finds its own
tables rather than being pinned to one Genie space.

The agent construction is adapted from the
[`agent-langgraph-advanced`](https://github.com/databricks/app-templates/tree/main/agent-langgraph-advanced)
app template, reduced to just the agent: no agent server (FastAPI), no MLflow
`ResponsesAgent` wrapper, no Lakebase memory, no MCP. Streamlit calls the
compiled LangGraph graph directly.

## How the chat works

`st.session_state["history"]` holds real LangChain message objects
(`HumanMessage`, `AIMessage`, `ToolMessage`), and the whole list is passed to the
agent on every turn — that list *is* the conversation memory, so a follow-up like
"now do that again in UTC" resolves against the previous turn. A rerun replays
those same objects without calling the model again.

The turn streams with `stream_mode=["updates", "messages"]`, which serves both
needs in one pass: `messages` carries token deltas for `st.write_stream`, and
`updates` carries each node's completed messages, which is what gets appended to
history.

**"New conversation"** in the sidebar clears the history, starting fresh.

### Two things worth knowing

**The Responses API is required, not a preference.** `databricks-gpt-6-astra` is
a reasoning model, and the `/v1/chat/completions` route rejects function tools
alongside `reasoning_effort`: *"To use function tools, use /v1/responses."*
Setting `reasoning_effort="none"` isn't a way out either — this model doesn't
accept that value. Hence `use_responses_api=True` next to `use_ai_gateway=True`.

**Responses API content is a list of typed blocks**, not a string — `text`
blocks alongside `reasoning` (carrying `encrypted_content`) and `function_call`.
Concatenating everything would dump encrypted reasoning payloads into the chat,
so `_text_blocks()` filters to `text`. A streamed response also emits incremental
deltas *and* a closing block repeating the whole answer, marked by an
`annotations` key, which is why there are two extractors: `delta_of()` drops that
closing block (it would render the answer twice live) and `text_of()` prefers it
(the accumulated message keeps both).

## Layout

```
app/
  app.py                  # the app — auth, agent, streaming, rendering
  mcp_client.py           # Genie One MCP tools, OBO auth, async→sync bridges
  app.yaml                # Databricks Apps runtime config
  requirements.txt
  .streamlit/config.toml  # theme
databricks.yml            # asset bundle
resources/
  agent_chat_app.app.yml  # the app resource + user_api_scopes
```

## Run locally

```bash
uv venv --python 3.12
uv pip install -r app/requirements.txt

cp .env.example .env      # then set DATABRICKS_CONFIG_PROFILE
```

Locally there's no forwarded user token, so the app falls back to the SDK's
default credential resolution — i.e. your CLI profile:

```bash
cd app
DATABRICKS_CONFIG_PROFILE=<your-profile> \
  ../.venv/bin/python -m streamlit run app.py
```

The model endpoint defaults to `databricks-gpt-6-astra`; override it with
`LLM_ENDPOINT_NAME`. Check what your workspace serves with:

```bash
databricks serving-endpoints list --profile <your-profile>
```

## Tools

**Genie One, over MCP.** `app/mcp_client.py` connects to
`https://<workspace-hostname>/api/2.0/mcp/genie` with streamable HTTP and loads
whatever tools it advertises — currently `genie_ask`, `genie_poll_response`,
`genie_get_query_result` and `genie_cancel_response`. Note the bare path: adding
a space id (`/api/2.0/mcp/genie/<space_id>`) gets you space-scoped query tools
instead, while the bare path is Genie One and searches across the user's data.

Auth is the pattern from
[databricks-agent-on-apps-mcp-blog](https://github.com/bcheng004/databricks-agent-on-apps-mcp-blog)'s
`agent_server`: a `WorkspaceClientAuth(httpx.Auth)` that copies
`WorkspaceClient.config.authenticate()` onto every outgoing request. Because it
re-reads the client on each request rather than capturing a header once, the same
class covers both cases — the forwarded user token when deployed, the CLI profile
locally.

If the MCP server can't be reached the agent is built without those tools and the
sidebar says so — along with what the server actually replied — rather than the
whole app failing.

Getting that reply into the log took a custom `httpx_client_factory`. MCP calls
`raise_for_status()` on a *streaming* response, so the body is never read: the
`HTTPStatusError` carries only a status line, and reading `.text` later raises
`ResponseNotRead` because the stream is closed by then. The factory attaches a
response event hook that reads error bodies while the stream is still live. It
also sets `terminate_on_close: False`, since this endpoint answers the session-
ending `DELETE` with *"Session termination is not supported"*.

One more wrinkle on the way out: the transport runs inside an anyio task group, so
the failure surfaces as an `ExceptionGroup`. `exc.response` doesn't exist on the
outer exception, which is why `_server_detail` walks `exceptions`, `__cause__` and
`__context__` to find the response.

`get_current_time` remains as a trivial local tool. Add your own to `LOCAL_TOOLS`
in `app/app.py`.

### Three things that will bite you

**MCP tools are coroutine-only.** They expose `coroutine` and no `func`, so a
synchronous `agent.stream()` raises `NotImplementedError` the moment Genie is
called. The turn has to run through `agent.astream()`, and Streamlit is
synchronous — hence `iter_coroutine` in `mcp_client.py`, which runs the async
stream on a worker thread and hands items back through a queue so tokens still
appear as they arrive. The lambda it wraps must not touch `st.session_state`:
that worker thread has no Streamlit `ScriptRunContext`, so history is read on the
script thread first.

**Tool results need flattening.** `langchain_mcp_adapters` returns content as
`[{"type": "text", ...}]`, but a Responses API `function_call_output` only accepts
`output_text`/`refusal`/… — a `text` block fails the *next* model call with
`Invalid value: 'text'` pointing at `input[N].output[0].type`, which reads like a
model problem rather than a tool one. Both paths need it: the success path via the
coroutine's return value, and the error path via `handle_tool_error`, since
failures are raised rather than returned.

**Blank optional args are rejected.** The model likes to send
`conversation_id: ""`, and Genie validates strictly — *"conversation_id must be
alphanumeric (with - and _) and 1-128 chars; got ''"*. `_wrap_mcp_tool` drops
blank values for non-required fields, which is more reliable than asking the
prompt to remember.

## Deploy as a Databricks App

No workspace host or profile is pinned in `databricks.yml` on purpose — pass the
profile at deploy time so the target workspace is always an explicit choice:

```bash
databricks bundle deploy -t dev --profile <your-profile>
databricks bundle run agent_chat -t dev --profile <your-profile>
```

The `run` command prints the app URL.

The app declares the `model-serving`, `ai-gateway` and `genie` user API scopes and
authenticates as the logged-in user (on-behalf-of), so both the model call and the
Genie query run under that user's grants rather than the app's service principal.

**The Genie MCP scope is `genie`, not `mcp.genie`.** Both are accepted by
`databricks bundle validate`, and `mcp.genie` reads like the obvious choice for an
MCP endpoint, but `/api/2.0/mcp/genie` refuses a token carrying only that one:

```
403 {"error_code":403,"message":"Provided OAuth token does not have required scopes: genie"}
```

Without it the MCP handshake is refused and the app falls back to running with no
Genie tools.

If you change the scopes, existing users must re-consent before their forwarded
token carries them — sign out of Databricks and back in, then reload the app.
Restarting or redeploying the app is *not* enough: the consent grant is stored
per-user, server-side, so it survives `apps stop`/`apps start`, a new deployment
and a browser reload. Until re-consent the sidebar shows the server's own
explanation, so it's clear which of the two problems you have.

## Next step: the memory layer

Nothing persists across sessions yet — history lives in `st.session_state`, which
dies with the browser tab. `create_agent` takes `checkpointer` (thread-scoped
conversation state) and `store` (cross-thread long-term memory), so that's the
hook: pass a durable backend in `build_agent()` instead of keeping the transcript
in session state.
