"""
Trace context helpers.

These are used to extract the current trace_id / span_id at the point where
the async eval job is scheduled (still inside the request span).  The IDs are
passed into the EvalJob so that eval results emitted minutes later can be
correlated back to the exact trace that produced the response.

    async with agent_run_span(...) as span:
        ...  # agent runs, response returned to client

        # Before we fire the background eval task, capture the trace context:
        trace_id = get_current_trace_id()   # e.g. "4bf92f3577b34da6..."
        span_id  = get_current_span_id()    # e.g. "00f067aa0ba902b7"

        asyncio.create_task(
            eval_scheduler.dispatch(EvalJob(trace_id=trace_id, ...))
        )
"""

from opentelemetry import trace
from opentelemetry.trace import INVALID_SPAN_CONTEXT


def get_current_trace_id() -> str | None:
    """
    Return the active trace_id as a 32-char lowercase hex string.
    Returns None if there is no active span (e.g. called outside a request).
    """
    ctx = trace.get_current_span().get_span_context()
    if ctx is INVALID_SPAN_CONTEXT or not ctx.is_valid:
        return None
    return format(ctx.trace_id, "032x")


def get_current_span_id() -> str | None:
    """
    Return the active span_id as a 16-char lowercase hex string.
    Returns None if there is no active span.
    """
    ctx = trace.get_current_span().get_span_context()
    if ctx is INVALID_SPAN_CONTEXT or not ctx.is_valid:
        return None
    return format(ctx.span_id, "016x")
