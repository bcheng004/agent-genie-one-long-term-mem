"""One background event loop, shared by everything async in the app.

Streamlit is synchronous; three things here are not. The Genie MCP tools are
coroutine-only, the agent turn therefore has to run through ``astream``, and the
Lakebase checkpointer and store hold asyncio connection pools.

That last one is why this module exists. A connection pool belongs to the event
loop that opened it, so the obvious bridge — spawn a thread, ``asyncio.run()``,
throw the loop away — cannot work once there are pools: they would outlive the
loop they were created on and every later call would fail. Instead a single loop
runs for the life of the process and all coroutines are handed to it.
"""

import asyncio
import queue
import threading
from typing import Any, Awaitable, Callable, Iterator

_loop: asyncio.AbstractEventLoop | None = None
_lock = threading.Lock()


def loop() -> asyncio.AbstractEventLoop:
    """The process-wide event loop, started on first use.

    Streamlit re-executes the page script on every interaction, but imported
    modules are cached in ``sys.modules``, so this runs once per process rather
    than once per rerun.
    """
    global _loop
    with _lock:
        if _loop is None:
            _loop = asyncio.new_event_loop()
            threading.Thread(
                target=_loop.run_forever, name="agent-asyncio", daemon=True
            ).start()
        return _loop


def run_coroutine(make_coro: Callable[[], Awaitable[Any]]) -> Any:
    """Run a coroutine on the shared loop and block until it returns.

    Exceptions surface on the calling thread, as they would from a normal call.
    """
    return asyncio.run_coroutine_threadsafe(make_coro(), loop()).result()


_DONE = object()


def iter_coroutine(make_async_iter: Callable[[], Any]) -> Iterator[Any]:
    """Consume an async iterator from synchronous code, yielding as items arrive.

    ``st.write_stream`` wants a plain iterator, and collecting the whole turn
    before returning would throw away live token streaming — so the loop pushes
    each item onto a queue and this drains it.
    """
    channel: queue.Queue = queue.Queue()

    async def pump() -> None:
        try:
            async for item in make_async_iter():
                channel.put(item)
        except BaseException as exc:  # noqa: BLE001 — re-raised on the calling thread
            channel.put(exc)
        finally:
            channel.put(_DONE)

    future = asyncio.run_coroutine_threadsafe(pump(), loop())
    try:
        while True:
            item = channel.get()
            if item is _DONE:
                break
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        # A consumer that stops early — an exception mid-render, or a rerun
        # tearing down st.write_stream — would otherwise leave the pump running
        # against a stream nobody reads.
        future.cancel()
