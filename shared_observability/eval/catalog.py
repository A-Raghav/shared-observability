"""
Eval metric catalog.

Defines every eval metric this pipeline can produce.  Each entry specifies:

    name        — OTel metric name (also used as the Prometheus metric name
                  after the exporter applies its naming rules)
    category    — "llm_quality" | "agent_behavior" | "business"
    runner      — "heuristic" | "llm_flash" | "llm_pro"
    description — Human-readable meaning; shown in Grafana panel descriptions.
    unit        — OTel unit string ("1" = dimensionless ratio, "" = count)
    value_range — Expected (min, max) for validation & dashboard axis config.

Runner guide
------------
  heuristic   Zero LLM cost.  Deterministic rules on the EvalJob fields.
              Example: response length, tool_call_count, turns_taken.

  llm_flash   Gemini 2.5-flash.  Lighter, cheaper LLM judge calls.
              Example: task_completion_flag, response_relevance.

  llm_pro     Gemini 2.5-pro.  Highest quality, reserved for critical evals.
              Example: hallucination_risk, completeness_score.

All metrics are emitted as OTel Gauge observations so Prometheus can scrape
the latest value per (framework, model, session_id, trace_id) label-set.

Adding a new metric
-------------------
1. Add an entry to CATALOG below.
2. Implement the corresponding runner method in heuristic.py or llm_judge.py.
3. The scheduler picks it up automatically — no other changes needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

RunnerKind = Literal["heuristic", "llm_flash", "llm_pro"]
Category = Literal["llm_quality", "agent_behavior", "business"]


@dataclass(frozen=True)
class EvalMetricDef:
    name: str
    category: Category
    runner: RunnerKind
    description: str
    unit: str = "1"
    value_range: tuple[float, float] = (0.0, 1.0)


# ─── Metric catalog ───────────────────────────────────────────────────────────

CATALOG: list[EvalMetricDef] = [

    # ── LLM Quality ──────────────────────────────────────────────────────────

    EvalMetricDef(
        name="eval.llm.response_relevance",
        category="llm_quality",
        runner="llm_flash",
        description=(
            "0-1 score: how well the response addresses the user's prompt. "
            "Judged by Gemini 2.5-flash using a structured rubric."
        ),
    ),
    EvalMetricDef(
        name="eval.llm.completeness",
        category="llm_quality",
        runner="llm_flash",
        description=(
            "0-1 score: whether the response fully answers all parts of "
            "the prompt without omissions.  Judged by Gemini 2.5-flash."
        ),
    ),
    EvalMetricDef(
        name="eval.llm.hallucination_risk",
        category="llm_quality",
        runner="llm_pro",
        description=(
            "0-1 score: likelihood that the response contains fabricated or "
            "unverifiable claims.  High scores are bad. Judged by Gemini 2.5-pro."
        ),
    ),
    EvalMetricDef(
        name="eval.llm.conciseness",
        category="llm_quality",
        runner="llm_flash",
        description=(
            "0-1 score: whether the response is appropriately brief — "
            "penalises unnecessary verbosity. Judged by Gemini 2.5-flash."
        ),
    ),

    # ── Agent Behaviour ───────────────────────────────────────────────────────

    EvalMetricDef(
        name="eval.agent.tool_call_count",
        category="agent_behavior",
        runner="heuristic",
        description="Raw count of tool invocations made during the agent run.",
        unit="",
        value_range=(0.0, 20.0),
    ),
    EvalMetricDef(
        name="eval.agent.reasoning_turns",
        category="agent_behavior",
        runner="heuristic",
        description="Number of LLM reasoning turns (each turn = one LLM call).",
        unit="",
        value_range=(0.0, 20.0),
    ),
    EvalMetricDef(
        name="eval.agent.tool_appropriateness",
        category="agent_behavior",
        runner="llm_flash",
        description=(
            "0-1 score: whether the agent called the right tools for the "
            "given task — penalises unnecessary or missing tool use."
        ),
    ),
    EvalMetricDef(
        name="eval.agent.response_length_chars",
        category="agent_behavior",
        runner="heuristic",
        description="Character count of the agent's final response.",
        unit="chars",
        value_range=(0.0, 10_000.0),
    ),

    # ── Business ──────────────────────────────────────────────────────────────

    EvalMetricDef(
        name="eval.business.task_completion",
        category="business",
        runner="llm_pro",
        description=(
            "0 or 1 flag: did the agent successfully complete the user's "
            "requested task?  Judged by Gemini 2.5-pro with chain-of-thought."
        ),
        value_range=(0.0, 1.0),
    ),
    EvalMetricDef(
        name="eval.business.session_goal_achieved",
        category="business",
        runner="llm_pro",
        description=(
            "0-1 score: overall goal achievement across the session, "
            "accounting for multi-turn context. Judged by Gemini 2.5-pro."
        ),
    ),
]

# ── Convenience look-ups ──────────────────────────────────────────────────────

BY_NAME: dict[str, EvalMetricDef] = {m.name: m for m in CATALOG}
BY_RUNNER: dict[RunnerKind, list[EvalMetricDef]] = {
    "heuristic": [m for m in CATALOG if m.runner == "heuristic"],
    "llm_flash": [m for m in CATALOG if m.runner == "llm_flash"],
    "llm_pro":   [m for m in CATALOG if m.runner == "llm_pro"],
}
BY_CATEGORY: dict[Category, list[EvalMetricDef]] = {
    "llm_quality":    [m for m in CATALOG if m.category == "llm_quality"],
    "agent_behavior": [m for m in CATALOG if m.category == "agent_behavior"],
    "business":       [m for m in CATALOG if m.category == "business"],
}
