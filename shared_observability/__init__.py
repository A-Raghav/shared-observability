"""
shared_observability
====================
Shared OTel instrumentation SDK for the agent observability PoC.

Quick-start
-----------
In your app's startup handler, call setup_otel() once:

    from shared_observability import setup_otel

    @app.on_event("startup")
    async def startup():
        setup_otel(
            service_name="langgraph-app",
            service_version="1.0.0",
            framework="langgraph",
        )

Then use the instrumentation context managers in your agent code:

    from shared_observability.otel_core.instrumentors import (
        agent_run_span, llm_call_span, tool_call_span, record_reasoning_turns
    )
"""

from .otel_core.provider import setup_otel
from .otel_core.propagation import get_current_trace_id, get_current_span_id

__all__ = [
    "setup_otel",
    "get_current_trace_id",
    "get_current_span_id",
]
