"""Conversation checkpoints and long-term memory, backed by Lakebase.

Adapted from ``utils_memory.py`` in the
`agent-langgraph-advanced <https://github.com/databricks/app-templates/tree/main/agent-langgraph-advanced>`_
app template: the same ``AsyncCheckpointSaver`` + ``AsyncDatabricksStore`` pair
and the same three memory tools. Two roles, deliberately separate:

* the **checkpointer** holds one conversation's messages, keyed by ``thread_id``,
  so a transcript survives a browser reload
* the **store** holds durable facts about a user, keyed by namespace and searched
  semantically through an embedding endpoint, so the agent can recall something
  from a conversation weeks ago

The template opened both in FastAPI's ``lifespan`` and reached them through
``RunnableConfig`` on each request. Streamlit has no lifespan, so they are opened
once on the shared event loop (see ``async_bridge``) and kept for the life of the
process. That also means the store and the signed-in user are both known when the
agent is built, so the memory tools simply close over them instead of digging
them out of the config — the same behaviour with less plumbing.

Lakebase is reached with the app's own credentials, not the user's: the memories
are this app's storage, partitioned by user, and the app's service principal is
what owns the schema. Genie and the model still run on behalf of the user.
"""

import json
import logging
import os
import uuid
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any, NamedTuple

from databricks_langchain import AsyncCheckpointSaver, AsyncDatabricksStore
from langchain_core.tools import tool

from async_bridge import run_coroutine

logger = logging.getLogger(__name__)

# Where a user's remembered facts live. The reference template's namespace, kept
# as-is: dots are replaced because namespace segments are path-like.
MEMORY_NAMESPACE = "user_memories"
# Which conversation a user is currently in. Kept in the store rather than in
# st.session_state so that reloading the tab returns to the same transcript —
# session state dies with the browser tab, which is the thing we're fixing.
THREAD_NAMESPACE = "agent_chat_threads"


@dataclass(frozen=True)
class LakebaseConfig:
    """Which Lakebase to use. Endpoint and project/branch are mutually exclusive."""

    endpoint: str | None
    project: str | None
    branch: str | None
    schema: str | None
    embedding_endpoint: str
    embedding_dims: int

    @property
    def description(self) -> str:
        return self.endpoint or f"{self.project}/{self.branch}"


def init_lakebase_config() -> LakebaseConfig | None:
    """Read the Lakebase target from the environment, or ``None`` if unset.

    Returning ``None`` rather than raising lets the app start without a memory
    layer — the sidebar says so and the chat still works, it just forgets.
    """
    endpoint = os.environ.get("LAKEBASE_AUTOSCALING_ENDPOINT") or None
    project = os.environ.get("LAKEBASE_AUTOSCALING_PROJECT") or None
    branch = os.environ.get("LAKEBASE_AUTOSCALING_BRANCH") or None

    if not endpoint and not (project and branch):
        return None
    if endpoint:  # the library rejects both being set
        project = branch = None

    return LakebaseConfig(
        endpoint=endpoint,
        project=project,
        branch=branch,
        schema=os.environ.get("LAKEBASE_AGENT_MEMORY_SCHEMA") or None,
        embedding_endpoint=os.environ.get(
            "DATABRICKS_EMBEDDING_ENDPOINT", "databricks-gte-large-en"
        ),
        embedding_dims=int(os.environ.get("DATABRICKS_EMBEDDING_DIMS", "1024")),
    )


class Memory(NamedTuple):
    """The opened memory layer, or the reason there isn't one."""

    checkpointer: Any | None
    store: Any | None
    error: str

    @property
    def enabled(self) -> bool:
        return self.checkpointer is not None and self.store is not None


DISABLED = Memory(checkpointer=None, store=None, error="")

# Holds the open connection pools for the life of the process. Without a
# reference they would be garbage collected and the pools closed underneath us.
_stack: AsyncExitStack | None = None


def open_memory(config: LakebaseConfig | None) -> Memory:
    """Open the checkpointer and store, running their migrations once.

    Failure is returned rather than raised: a Lakebase that is unreachable, or a
    schema the app's service principal cannot create, shouldn't take the whole
    chat down with it.
    """
    global _stack
    if config is None:
        return Memory(None, None, "No Lakebase configured.")

    async def _open() -> tuple[AsyncExitStack, Any, Any]:
        stack = AsyncExitStack()
        try:
            shared = {
                "autoscaling_endpoint": config.endpoint,
                "project": config.project,
                "branch": config.branch,
                "schema": config.schema,
            }
            checkpointer = await stack.enter_async_context(
                AsyncCheckpointSaver(**shared)
            )
            store = await stack.enter_async_context(
                AsyncDatabricksStore(
                    **shared,
                    embedding_endpoint=config.embedding_endpoint,
                    embedding_dims=config.embedding_dims,
                )
            )
            # Creates the checkpoint and store tables if they aren't there yet.
            await checkpointer.setup()
            await store.setup()
            return stack, checkpointer, store
        except BaseException:
            await stack.aclose()
            raise

    try:
        stack, checkpointer, store = run_coroutine(_open)
    except Exception as exc:  # noqa: BLE001 — surfaced in the sidebar
        logger.warning(
            "Lakebase memory unavailable (%s); continuing without it.",
            config.description,
            exc_info=True,
        )
        return Memory(None, None, f"{type(exc).__name__}: {exc}")

    _stack = stack
    logger.info("Lakebase memory ready on %s", config.description)
    return Memory(checkpointer=checkpointer, store=store, error="")


# ---------------------------------------------------------------------------
# Which conversation the user is in
# ---------------------------------------------------------------------------
def _thread_namespace(user_id: str) -> tuple[str, str]:
    return (THREAD_NAMESPACE, user_id.replace(".", "-"))


def current_thread_id(memory: Memory, user_id: str) -> str:
    """The user's open conversation, starting one if they have none.

    Looked up in the store so it outlives the browser tab. If the store can't be
    read the conversation still works, it just won't be found again later.
    """
    if not memory.enabled:
        return str(uuid.uuid4())
    try:
        item = run_coroutine(
            lambda: memory.store.aget(_thread_namespace(user_id), "current")
        )
    except Exception:  # noqa: BLE001 — a fresh thread is a fine fallback
        logger.warning("Could not read the current thread id.", exc_info=True)
        return str(uuid.uuid4())

    if item and isinstance(item.value, dict) and item.value.get("thread_id"):
        return str(item.value["thread_id"])
    return start_new_thread(memory, user_id)


def start_new_thread(memory: Memory, user_id: str) -> str:
    """Point the user at a brand-new conversation and remember that."""
    thread_id = str(uuid.uuid4())
    if memory.enabled:
        try:
            run_coroutine(
                lambda: memory.store.aput(
                    _thread_namespace(user_id), "current", {"thread_id": thread_id}
                )
            )
        except Exception:  # noqa: BLE001 — the new thread still works this session
            logger.warning("Could not record the new thread id.", exc_info=True)
    return thread_id


def load_history(agent: Any, config: dict) -> list:
    """The messages already in this conversation, from the checkpoint.

    Returns ``[]`` for a thread that has never been written to, which is also
    what a failed read degrades to — an empty transcript rather than an error
    page.
    """
    try:
        state = run_coroutine(lambda: agent.aget_state(config))
    except Exception:  # noqa: BLE001 — render an empty transcript instead
        logger.warning("Could not load checkpointed history.", exc_info=True)
        return []
    return list((getattr(state, "values", None) or {}).get("messages") or [])


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
def memory_tools(store: Any, user_id: str) -> list:
    """The three long-term memory tools, scoped to one user.

    Both the store and the user are fixed for the life of the built agent, so
    these close over them; the reference template passed them through
    ``RunnableConfig`` because its store was per-request.
    """
    namespace = (MEMORY_NAMESPACE, user_id.replace(".", "-"))

    @tool
    async def get_user_memory(query: str) -> str:
        """Search for relevant information about the user from long-term memory."""
        results = await store.asearch(namespace, query=query, limit=5)
        if not results:
            return "No memories found for this user."
        found = "\n".join(f"- [{item.key}]: {json.dumps(item.value)}" for item in results)
        return f"Found {len(results)} relevant memories:\n{found}"

    @tool
    async def save_user_memory(memory_key: str, memory_data_json: str) -> str:
        """Save information about the user to long-term memory.

        memory_data_json must be a JSON object, e.g. {"preference": "metric units"}.
        """
        try:
            memory_data = json.loads(memory_data_json)
        except json.JSONDecodeError as exc:
            return f"Failed to save memory: invalid JSON - {exc}"
        if not isinstance(memory_data, dict):
            return (
                "Failed: memory_data must be a JSON object, not "
                f"{type(memory_data).__name__}"
            )
        await store.aput(namespace, memory_key, memory_data)
        return f"Successfully saved memory '{memory_key}'."

    @tool
    async def delete_user_memory(memory_key: str) -> str:
        """Delete a specific memory from the user's long-term memory."""
        await store.adelete(namespace, memory_key)
        return f"Successfully deleted memory '{memory_key}'."

    return [get_user_memory, save_user_memory, delete_user_memory]
