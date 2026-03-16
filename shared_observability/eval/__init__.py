"""
Async eval pipeline.

Public interface
----------------
    from shared_observability.eval import EvalJob, dispatch

    # Inside run_agent(), after the agent loop, still inside agent_run_span:
    job = EvalJob(
        session_id=session_id,
        framework=FRAMEWORK,
        model=model,
        prompt=message,
        response=response_text,
        tools_used=list(tools_used),
        reasoning_turns=turn,
        trace_id=get_current_trace_id(),
        span_id=get_current_span_id(),
    )
    asyncio.create_task(dispatch(job))
"""

from .models import EvalJob
from .scheduler import dispatch
from .catalog import CATALOG, BY_NAME, BY_RUNNER, BY_CATEGORY

__all__ = [
    "EvalJob",
    "dispatch",
    "CATALOG",
    "BY_NAME",
    "BY_RUNNER",
    "BY_CATEGORY",
]
