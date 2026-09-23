"""Genie One MCP tools, called on behalf of the signed-in user.

The OBO auth logic is adapted from the ``agent_server`` package of
`databricks-agent-on-apps-mcp-blog <https://github.com/bcheng004/databricks-agent-on-apps-mcp-blog>`_:
a thin ``httpx.Auth`` wrapper around ``WorkspaceClient.config.authenticate()``,
so every MCP request carries freshly-resolved credentials from whatever client
the app built — the forwarded user token when deployed, the CLI profile locally.

MCP tools are coroutine-only and Streamlit is synchronous; the two meet on the
shared event loop in ``async_bridge``.
"""

import json
import logging
from typing import Any

import httpx
from databricks.sdk import WorkspaceClient
from langchain_mcp_adapters.client import MultiServerMCPClient
from mcp.shared._httpx_utils import create_mcp_http_client

import tracing as trc
from async_bridge import run_coroutine

logger = logging.getLogger(__name__)

# The Genie One MCP server. Unlike /api/2.0/mcp/genie/<space_id>, which exposes
# per-space query tools, the bare path exposes Genie One: genie_ask searches the
# user's enterprise data without being pinned to one space.
GENIE_MCP_PATH = "/api/2.0/mcp/genie"

SERVER_NAME = "genie"


class WorkspaceClientAuth(httpx.Auth):
    """httpx auth that injects fresh credentials from a WorkspaceClient on every request."""

    requires_request_body = False
    requires_response_body = False

    def __init__(self, workspace_client: WorkspaceClient):
        self.workspace_client = workspace_client

    def auth_flow(self, request):
        for key, value in self.workspace_client.config.authenticate().items():
            request.headers[key] = value
        yield request


def genie_mcp_url(workspace_client: WorkspaceClient) -> str:
    """The Genie One MCP endpoint on the workspace this client points at."""
    return f"{(workspace_client.config.host or '').rstrip('/')}{GENIE_MCP_PATH}"


async def _read_error_body(response: httpx.Response) -> None:
    """Read the body of a failed response before anything closes the stream.

    MCP calls ``raise_for_status()`` on a *streaming* response, so the body is
    never read: the resulting ``HTTPStatusError`` carries only a status line, and
    touching ``.text`` afterwards raises ``ResponseNotRead``. Reading it here,
    while the stream is live, both logs the server's explanation and caches the
    content so ``_server_detail`` can recover it from the exception later.
    """
    if response.status_code < 400:
        return
    try:
        await response.aread()
    except Exception:  # noqa: BLE001 — diagnostics must never mask the real error
        return
    logger.warning(
        "MCP %s %s -> %d: %s",
        response.request.method,
        response.request.url,
        response.status_code,
        response.text[:2000],
    )


def _http_client(
    headers: dict[str, str] | None = None,
    timeout: httpx.Timeout | None = None,
    auth: httpx.Auth | None = None,
) -> httpx.AsyncClient:
    """The MCP transport's client, with an error-body hook attached.

    Built through ``create_mcp_http_client`` to keep the transport's own timeout
    defaults (300s read, for long-lived streams); the hook is attached afterwards
    because that factory takes no ``event_hooks``.
    """
    client = create_mcp_http_client(headers=headers, timeout=timeout, auth=auth)
    client.event_hooks = {"response": [_read_error_body]}
    return client


def _mcp_client(workspace_client: WorkspaceClient) -> MultiServerMCPClient:
    return MultiServerMCPClient(
        connections={
            SERVER_NAME: {
                "transport": "streamable_http",
                "url": genie_mcp_url(workspace_client),
                "auth": WorkspaceClientAuth(workspace_client),
                "httpx_client_factory": _http_client,
                # The default close sends DELETE to end the MCP session, which
                # this endpoint answers with 405 "Session termination is not
                # supported" — a wasted request on every call.
                "terminate_on_close": False,
            }
        }
    )


def _flatten_content(content: Any) -> Any:
    """Collapse MCP content blocks into a plain string.

    ``langchain_mcp_adapters`` hands back tool output as typed blocks, e.g.
    ``[{"type": "text", "text": "{...}", "id": "lc_..."}]``. The Responses API
    only accepts ``output_text``/``refusal``/… inside a ``function_call_output``,
    so a ``text`` block fails the *next* model call with "Invalid value: 'text'.
    Supported values are: 'input_text', 'output_text', …" — pointing at the tool
    output rather than at the tool, which makes it easy to misread. Returning a
    plain string sidesteps the typed-block schema altogether.
    """
    if isinstance(content, str) or not isinstance(content, list):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict):
            parts.append(
                block.get("text", "")
                if block.get("type") == "text"
                else json.dumps(block, default=str)
            )
        else:
            parts.append(str(block))
    return "".join(parts)


def _wrap_mcp_tool(tool: Any) -> Any:
    """Adapt an MCP tool to this model and to the Genie server's strict validation.

    Two transforms, both learned the hard way:

    * **Blank optional args are dropped.** The model fills the optional
      ``conversation_id`` with ``""``, which Genie rejects outright
      ("conversation_id must be alphanumeric (with - and _) and 1-128 chars; got
      ''"). Dropping blanks turns that into an ordinary new-conversation call.
      Required fields are left alone so a genuinely missing argument still errors
      where you'd expect.
    * **Result content is flattened to a string on both paths.** The success path
      goes through the coroutine's return value; the error path does not — the
      coroutine raises and ``langchain_mcp_adapters`` renders the failure through
      its own ``handle_tool_error``, which also produces ``text``-typed blocks.
      Overriding that handler is the only way to catch the second path.
    """
    schema = tool.args_schema if isinstance(tool.args_schema, dict) else {}
    required = set(schema.get("required") or ())

    name = tool.name
    original = tool.coroutine
    if original is not None:

        async def wrapped(**kwargs: Any) -> Any:
            cleaned = {
                key: value
                for key, value in kwargs.items()
                if key in required or value not in ("", None)
            }
            # The span carries what autolog's own tool span drops — the cleaned
            # args actually sent to Genie and, for content_and_artifact tools,
            # the artifact (SQL, query_id, rows). A raised call still propagates
            # through here so the span records the error and the tool's own
            # handle_tool_error runs.
            with trc.tool_span(name, cleaned) as record:
                result = await original(**cleaned)
                # response_format="content_and_artifact" -> (content, artifact)
                if isinstance(result, tuple) and len(result) == 2:
                    content, artifact = result
                    content = _flatten_content(content)
                    record(content, artifact)
                    return content, artifact
                content = _flatten_content(result)
                record(content)
                return content

        tool.coroutine = wrapped

    tool.handle_tool_error = lambda exc: str(exc)
    return tool


def _server_detail(exc: BaseException, _depth: int = 0) -> str:
    """Find an HTTP response body inside a possibly-nested exception.

    The MCP client runs its transport inside an anyio task group, so a 403
    surfaces wrapped in an ``ExceptionGroup``: ``exc.response`` is absent on the
    outer exception and only the status line reaches the log. The body is the
    part that names what was actually refused, so it's worth digging for.
    """
    if _depth > 5:
        return ""
    if text := getattr(getattr(exc, "response", None), "text", ""):
        return text
    for nested in (*getattr(exc, "exceptions", ()), exc.__cause__, exc.__context__):
        if isinstance(nested, BaseException) and (
            found := _server_detail(nested, _depth + 1)
        ):
            return found
    return ""


def load_genie_tools(workspace_client: WorkspaceClient) -> tuple[list, str]:
    """Fetch the Genie One MCP tools, or return ``([], reason)`` if unreachable.

    Degrading instead of raising keeps the rest of the chat usable when the MCP
    endpoint is down or the signed-in user's token lacks the ``genie`` scope. The
    reason comes back with it so the UI can show what the server actually said
    instead of a generic "unavailable".
    """
    try:
        tools = run_coroutine(lambda: _mcp_client(workspace_client).get_tools(
            server_name=SERVER_NAME
        ))
    except Exception as exc:
        # The status line alone is not actionable — a 403 from the MCP gateway
        # carries a body naming what was refused, so surface it.
        detail = _server_detail(exc)
        logger.warning(
            "Failed to fetch Genie MCP tools; continuing without them.%s",
            f" Server said: {detail}" if detail else "",
            exc_info=True,
        )
        return [], detail or f"{type(exc).__name__}: {exc}"
    logger.info("Loaded %d Genie MCP tool(s): %s", len(tools), [t.name for t in tools])
    return [_wrap_mcp_tool(t) for t in tools], ""
