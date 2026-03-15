"""
Framework-agnostic instrumentation context managers.

Both LangGraph and ADK apps use these identical wrappers — the OTel signals
they emit are indistinguishable at the Collector level, which is what makes
cross-framework comparison possible in Grafana.

Span hierarchy produced per /chat request:

    POST /chat            ← FastAPI auto-instrumentation
      └─ agent_run        ← agent_run_span()
           ├─ llm_call    ← llm_call_span()   [turn 1]
           │    └─ tool_call:web_search  ← tool_call_span()
           ├─ llm_call    ← llm_call_span()   [turn 2]
           │    └─ tool_call:math_eval   ← tool_call_span()
           └─ llm_call    ← llm_call_span()   [turn N — final response]

Usage pattern (same for both frameworks):

    async with agent_run_span(session_id=..., model=..., framework=...) as span:
        turn = 0
        async for event in <framework_event_stream>:
            if is_llm_call(event):
                turn += 1
                async with llm_call_span(model=..., framework=..., turn=turn) as ls:
                    response = await process_llm_event(event)
                    ls.set_attribute("llm.input_tokens", response.input_tokens)
                    ls.set_attribute("llm.output_tokens", response.output_tokens)

            elif is_tool_call(event):
                async with tool_call_span(tool_name=..., framework=..., turn=turn):
                    await process_tool_event(event)

        record_reasoning_turns(turn, framework=..., model=...)

Metric instruments defined here:
    agent.request.duration   Histogram  ms    — end-to-end /chat latency
    agent.llm.call.duration  Histogram  ms    — per LLM inference latency
    agent.llm.tokens.input   Counter    tokens — cumulative input tokens
    agent.llm.tokens.output  Counter    tokens — cumulative output tokens
    agent.tool.call.duration Histogram  ms    — per tool execution latency
    agent.tool.call.count    Counter          — tool invocation count
    agent.reasoning.turns    Histogram        — LLM turns per agent run
    agent.error.count        Counter          — errors by type
"""

import logging
import time
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import AsyncGenerator

from opentelemetry import metrics, trace
from opentelemetry.trace import Span, Status, StatusCode

_log = logging.getLogger(__name__)

# Instrument name used as the meter/tracer scope — appears in Prometheus labels
# as `otel_scope_name` and in Tempo's service graph.
_SCOPE = "agent.observability"
_SCOPE_VERSION = "1.0.0"


# ── Lazy instrument accessors ─────────────────────────────────────────────────
# Each function is called at span-close time (inside `finally`), never at import
# time. `lru_cache` ensures each instrument is created exactly once after the
# MeterProvider has been configured by setup_otel(), and the OTel SDK deduplicates
# by name within the same meter, so this is safe across concurrent calls.

@lru_cache(maxsize=None)
def _request_duration():
    return metrics.get_meter(_SCOPE, _SCOPE_VERSION).create_histogram(
        name="agent.request.duration",
        unit="ms",
        description="End-to-end /chat request duration",
    )

@lru_cache(maxsize=None)
def _llm_call_duration():
    return metrics.get_meter(_SCOPE, _SCOPE_VERSION).create_histogram(
        name="agent.llm.call.duration",
        unit="ms",
        description="Duration of a single LLM inference call",
    )

@lru_cache(maxsize=None)
def _llm_tokens_input():
    return metrics.get_meter(_SCOPE, _SCOPE_VERSION).create_counter(
        name="agent.llm.tokens.input",
        unit="tokens",
        description="Cumulative input tokens consumed",
    )

@lru_cache(maxsize=None)
def _llm_tokens_output():
    return metrics.get_meter(_SCOPE, _SCOPE_VERSION).create_counter(
        name="agent.llm.tokens.output",
        unit="tokens",
        description="Cumulative output tokens generated",
    )

@lru_cache(maxsize=None)
def _tool_call_duration():
    return metrics.get_meter(_SCOPE, _SCOPE_VERSION).create_histogram(
        name="agent.tool.call.duration",
        unit="ms",
        description="Duration of a single tool execution",
    )

@lru_cache(maxsize=None)
def _tool_call_count():
    return metrics.get_meter(_SCOPE, _SCOPE_VERSION).create_counter(
        name="agent.tool.call.count",
        description="Number of tool invocations",
    )

@lru_cache(maxsize=None)
def _reasoning_turns():
    return metrics.get_meter(_SCOPE, _SCOPE_VERSION).create_histogram(
        name="agent.reasoning.turns",
        description="Number of LLM reasoning turns per agent run",
    )

@lru_cache(maxsize=None)
def _error_count():
    return metrics.get_meter(_SCOPE, _SCOPE_VERSION).create_counter(
        name="agent.error.count",
        description="Number of errors by type",
    )


# ── Tracer ────────────────────────────────────────────────────────────────────

def _tracer() -> trace.Tracer:
    """Resolved against the global TracerProvider at call time (after setup_otel)."""
    return trace.get_tracer(_SCOPE, _SCOPE_VERSION)


# ── Context managers ──────────────────────────────────────────────────────────

@asynccontextmanager
async def agent_run_span(
    session_id: str,
    model: str,
    framework: str,
    tools_available: list[str] | None = None,
) -> AsyncGenerator[Span, None]:
    """
    Root span for one complete agent run (one /chat request).

    Created as a child of the FastAPI request span, so the full hierarchy is:
        POST /chat → agent_run → llm_call(s) → tool_call(s)

    Yields the Span so callers can attach additional attributes mid-run
    (e.g. final response length, total turns).

    Records:
        agent.request.duration  — on exit
        agent.error.count       — on unhandled exception
    """
    start = time.monotonic()
    with _tracer().start_as_current_span("agent_run") as span:
        span.set_attribute("agent.session_id", session_id)
        span.set_attribute("agent.model", model)
        span.set_attribute("agent.framework", framework)
        if tools_available:
            span.set_attribute("agent.tools_available", tools_available)

        try:
            yield span
            span.set_status(Status(StatusCode.OK))

        except Exception as exc:
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            span.record_exception(exc)
            _error_count().add(
                1,
                {
                    "agent.framework": framework,
                    "agent.model": model,
                    "error.type": type(exc).__name__,
                },
            )
            raise

        finally:
            duration_ms = (time.monotonic() - start) * 1000
            _request_duration().record(
                duration_ms,
                {"agent.framework": framework, "agent.model": model},
            )


@asynccontextmanager
async def llm_call_span(
    model: str,
    framework: str,
    turn: int = 0,
) -> AsyncGenerator[Span, None]:
    """
    Span for a single LLM inference call within an agent run.

    Caller sets token counts on the yielded span:
        span.set_attribute("llm.input_tokens", N)
        span.set_attribute("llm.output_tokens", N)

    The context manager reads those attributes in its finally block and records
    them as OTel counter metrics, so the caller only needs one handle (the span)
    rather than having to call a separate metrics function.

    Records:
        agent.llm.call.duration  — on exit
        agent.llm.tokens.input   — read from span.attributes["llm.input_tokens"]
        agent.llm.tokens.output  — read from span.attributes["llm.output_tokens"]
    """
    start = time.monotonic()
    with _tracer().start_as_current_span("llm_call") as span:
        span.set_attribute("llm.model", model)
        span.set_attribute("llm.turn", turn)
        span.set_attribute("agent.framework", framework)

        try:
            yield span
            span.set_status(Status(StatusCode.OK))

        except Exception as exc:
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            span.record_exception(exc)
            raise

        finally:
            duration_ms = (time.monotonic() - start) * 1000
            attrs = {"agent.framework": framework, "llm.model": model}

            _llm_call_duration().record(duration_ms, attrs)

            # Read token counts set by the caller (span is still live here,
            # inside the `with start_as_current_span` block).
            raw_attrs = span.attributes or {}
            input_tokens = int(raw_attrs.get("llm.input_tokens", 0))
            output_tokens = int(raw_attrs.get("llm.output_tokens", 0))

            if input_tokens > 0:
                _llm_tokens_input().add(input_tokens, attrs)
            if output_tokens > 0:
                _llm_tokens_output().add(output_tokens, attrs)


@asynccontextmanager
async def tool_call_span(
    tool_name: str,
    framework: str,
    turn: int = 0,
) -> AsyncGenerator[Span, None]:
    """
    Span for a single tool execution, nested inside an llm_call_span.

    Records:
        agent.tool.call.duration — on exit
        agent.tool.call.count    — on exit (success flag attached as label)
    """
    start = time.monotonic()
    # Span name includes tool name for instant visibility in Tempo waterfall
    with _tracer().start_as_current_span(f"tool_call:{tool_name}") as span:
        span.set_attribute("tool.name", tool_name)
        span.set_attribute("tool.turn", turn)
        span.set_attribute("agent.framework", framework)
        success = True

        try:
            yield span
            span.set_status(Status(StatusCode.OK))

        except Exception as exc:
            success = False
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            span.record_exception(exc)
            raise

        finally:
            duration_ms = (time.monotonic() - start) * 1000
            attrs = {
                "agent.framework": framework,
                "tool.name": tool_name,
                "tool.success": str(success).lower(),  # "true" / "false"
            }
            _tool_call_duration().record(duration_ms, attrs)
            _tool_call_count().add(1, attrs)


# ── Standalone metric helpers ─────────────────────────────────────────────────

def record_reasoning_turns(turn_count: int, framework: str, model: str) -> None:
    """
    Record the total number of LLM reasoning turns for a completed agent run.

    Call this once, after the agent run loop finishes, inside the agent_run_span
    context but after all llm_call_spans have closed:

        async with agent_run_span(...) as span:
            turn = 0
            async for event in stream:
                ...
                turn += 1
            record_reasoning_turns(turn, framework=framework, model=model)
            span.set_attribute("agent.reasoning_turns", turn)
    """
    _reasoning_turns().record(
        turn_count,
        {"agent.framework": framework, "llm.model": model},
    )
