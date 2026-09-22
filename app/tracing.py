"""MLflow tracing for the agent, sent to a Databricks experiment.

``mlflow.langchain.autolog()`` turns every LangGraph turn into a trace tree: the
model call, each Genie MCP tool and each memory tool become spans carrying their
inputs, outputs, latency and token counts. A turn that behaved oddly — a memory
that wasn't saved, a Genie call that timed out — is then inspectable after the
fact in the experiment's *Traces* tab, which is the whole point of adding this.

Setup is best-effort, like the rest of the app: if the experiment can't be
reached the chat still runs, just untraced, and the sidebar says so. Everything
is keyed off ``MLFLOW_EXPERIMENT_NAME`` — leave it unset to turn tracing off.

The tracking URI is ``databricks``, so in the Apps runtime traces are written
under the app's own credentials (ambient service-principal auth), the same
identity that owns the Lakebase memory schema.
"""

import logging
import os
from contextlib import contextmanager
from typing import Callable, Iterator, NamedTuple

logger = logging.getLogger(__name__)


class Tracing(NamedTuple):
    """The result of wiring up tracing, or the reason there isn't any."""

    experiment: str  # "" when disabled or when setup failed
    error: str

    @property
    def enabled(self) -> bool:
        return bool(self.experiment)


# Setup is process-global and done once; the result is cached so that calling
# init_tracing() on every Streamlit rerun is free after the first.
_result: Tracing | None = None


def init_tracing() -> Tracing:
    """Point MLflow at the configured experiment and enable LangChain autolog.

    Idempotent: the actual setup runs once per process. Returns a disabled
    ``Tracing`` (no experiment) when ``MLFLOW_EXPERIMENT_NAME`` is unset or when
    setup raised — never propagates, so a tracing problem can't take chat down.
    """
    global _result
    if _result is not None:
        return _result

    name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "").strip()
    if not name:
        _result = Tracing(experiment="", error="")
        return _result

    try:
        import mlflow

        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "databricks"))
        # Creates the experiment on first run if it isn't there yet.
        mlflow.set_experiment(name)
        # Traces every LangGraph / LangChain invocation, tool calls included.
        mlflow.langchain.autolog()
        _result = Tracing(experiment=name, error="")
        logger.info("MLflow tracing enabled on experiment %s", name)
    except Exception as exc:  # noqa: BLE001 — surfaced in the sidebar
        logger.warning(
            "MLflow tracing unavailable; continuing without it.", exc_info=True
        )
        _result = Tracing(experiment="", error=f"{type(exc).__name__}: {exc}")
    return _result


def turn_metadata(session_id: str | None, user_id: str | None) -> dict:
    """Reserved trace metadata that groups traces by conversation and user.

    ``mlflow.trace.session`` and ``mlflow.trace.user`` are the keys the MLflow UI
    reads to group a conversation's turns and to attribute them to a user, so a
    whole chat can be followed end to end rather than as scattered traces.
    """
    metadata = {}
    if session_id:
        metadata["mlflow.trace.session"] = str(session_id)
    if user_id:
        metadata["mlflow.trace.user"] = str(user_id)
    return metadata


def _noop_record(content: object = None, artifact: object = None) -> None:
    return None


@contextmanager
def tool_span(name: str, inputs: dict) -> Iterator[Callable[..., None]]:
    """Best-effort MLflow TOOL span around an MCP tool call.

    autolog already traces the call's name, args and *text* result, but it drops
    the Genie ``artifact`` — the generated SQL, the ``query_id`` and the result
    rows. This span keeps that: it yields a ``record(content, artifact)`` callback
    the caller uses to attach both, so the interesting half of a Genie answer
    lands in the trace instead of only the prose summary.

    It is a no-op when tracing is off or MLflow is unavailable, and it never
    suppresses the tool's own exception — that propagates (recorded on the span,
    which is why the error path gets a span at all) so langchain's
    ``handle_tool_error`` still turns it into a ToolMessage.
    """
    result = _result
    if result is None or not result.enabled:
        yield _noop_record
        return

    try:
        import mlflow
    except Exception:  # noqa: BLE001 — tracing is optional, never fatal
        yield _noop_record
        return

    with mlflow.start_span(name=name, span_type="TOOL") as span:
        try:
            span.set_inputs(inputs)
        except Exception:  # noqa: BLE001 — recording must not break the call
            pass

        def record(content: object = None, artifact: object = None) -> None:
            outputs = {"content": content}
            if artifact is not None:
                outputs["artifact"] = artifact
            try:
                span.set_outputs(outputs)
            except Exception:  # noqa: BLE001
                pass

        yield record
