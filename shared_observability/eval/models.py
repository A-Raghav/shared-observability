"""
Eval pipeline data models.

EvalJob is the single unit of work dispatched from the agent run handler
into the async eval scheduler.  It carries everything the runners need:
the original prompt, the agent's response, execution metadata, and the
OTel trace/span IDs so eval metric results can be correlated back to the
exact trace in Tempo.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvalJob:
    """
    All context needed to evaluate one agent response.

    Created inside agent_run_span (while the OTel context is still active)
    so that trace_id / span_id are valid and can be attached to eval metrics.

    Fields
    ------
    job_id          : Unique ID for this eval job (auto-generated UUID4).
    session_id      : Agent session that produced the response.
    trace_id        : OTel trace ID — 32-char hex — for correlation in Grafana.
    span_id         : OTel span ID — 16-char hex — for finer correlation.
    framework       : "langgraph" | "adk" — which framework ran the agent.
    model           : Model name used, e.g. "gemini-2.0-flash".
    prompt          : The user message sent to the agent.
    response        : The agent's text response.
    tools_used      : List of tool names that were called during the run.
    reasoning_turns : Number of LLM reasoning turns taken.
    metadata        : Optional catch-all dict for future fields.
    """

    session_id: str
    framework: str
    model: str
    prompt: str
    response: str
    tools_used: list[str] = field(default_factory=list)
    reasoning_turns: int = 0
    trace_id: str | None = None
    span_id: str | None = None
    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: dict[str, Any] = field(default_factory=dict)
