"""
Heuristic runner — zero LLM cost, instant, deterministic.

Computes eval metrics that can be derived purely from the EvalJob fields
without calling any external model.  These run synchronously (wrapped in
an async interface to match the BaseRunner contract).

Metrics produced
----------------
  eval.agent.tool_call_count       — len(job.tools_used)
  eval.agent.reasoning_turns       — job.reasoning_turns
  eval.agent.response_length_chars — len(job.response)
"""

from __future__ import annotations

import logging

from ..catalog import BY_RUNNER
from ..models import EvalJob
from .base import BaseRunner, EvalResult

_log = logging.getLogger(__name__)

# Names of metrics this runner is responsible for (from catalog)
_OWNED = {m.name for m in BY_RUNNER["heuristic"]}


class HeuristicRunner(BaseRunner):
    """
    Stateless, synchronous eval runner for rule-based metrics.

    All computation is O(1) — no I/O, no model calls.
    """

    async def run(self, job: EvalJob) -> list[EvalResult]:
        results: list[EvalResult] = []

        # ── eval.agent.tool_call_count ────────────────────────────────────────
        if "eval.agent.tool_call_count" in _OWNED:
            results.append(
                EvalResult(
                    metric_name="eval.agent.tool_call_count",
                    value=float(len(job.tools_used)),
                )
            )

        # ── eval.agent.reasoning_turns ────────────────────────────────────────
        if "eval.agent.reasoning_turns" in _OWNED:
            results.append(
                EvalResult(
                    metric_name="eval.agent.reasoning_turns",
                    value=float(job.reasoning_turns),
                )
            )

        # ── eval.agent.response_length_chars ─────────────────────────────────
        if "eval.agent.response_length_chars" in _OWNED:
            results.append(
                EvalResult(
                    metric_name="eval.agent.response_length_chars",
                    value=float(len(job.response)),
                )
            )

        _log.debug(
            "heuristic_eval_complete job=%s results=%d",
            job.job_id, len(results),
        )
        return results
