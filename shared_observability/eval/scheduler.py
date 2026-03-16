"""
Async eval scheduler.

Dispatches EvalJob objects to the appropriate runners in the background,
then emits each EvalResult as an OTel gauge observation so the scores
appear in Prometheus and are queryable in Grafana.

Architecture
------------

    agent_run()                    (request handler, foreground)
        │
        ├── yields response to caller immediately
        │
        └── asyncio.create_task(scheduler.dispatch(job))
                │                  (background, non-blocking)
                │
                ├── HeuristicRunner.run(job)   ← instant, O(1)
                ├── LLMJudgeRunner(flash).run(job)  ← Gemini 2.5-flash
                └── LLMJudgeRunner(pro).run(job)    ← Gemini 2.5-pro
                        │
                        └── emit_result(result, job)
                                └── OTel gauge.record(value, labels)

The three runner tiers run concurrently via asyncio.gather, so the wall-clock
delay experienced by the background task is max(runner_times), not their sum.

OTel metric emitted per result
------------------------------
    Name  : <EvalResult.metric_name>   e.g. "eval.llm.response_relevance"
    Type  : Gauge (ObservableGauge callback approach — we use direct record()
            via UpDownCounter trick; see note below)
    Labels:
        agent.framework    — "langgraph" | "adk"
        llm.model          — model name string
        agent.session_id   — session identifier
        eval.trace_id      — OTel trace_id for log/trace correlation

Note on OTel Gauge vs UpDownCounter
------------------------------------
OTel's Python SDK `UpDownCounter` is scrape-point-in-time safe for Prometheus
because Prometheus only reads the latest value at scrape time.  We use it here
as a "gauge" because:
  (a) OTel's ObservableGauge requires a callback which complicates per-job
      label cardinality.
  (b) UpDownCounter with set() semantics (add positive to raise, subtract to
      lower) is not idiomatic; instead we use a single `record()` call per
      result which is equivalent since scores are always absolute floats.

In practice the Prometheus operator sees this as a gauge — the dashboard
panels use `eval_*` metric queries with the same label filters.
"""

from __future__ import annotations

import asyncio
import logging
from functools import lru_cache

from opentelemetry import metrics

from .catalog import CATALOG, BY_RUNNER
from .models import EvalJob
from .runners.base import EvalResult
from .runners.heuristic import HeuristicRunner
from .runners.llm_judge import LLMJudgeRunner

_log = logging.getLogger(__name__)

_SCOPE = "agent.observability"
_SCOPE_VERSION = "1.0.0"


# ── Lazy OTel gauge instruments (one per catalog metric) ──────────────────────

@lru_cache(maxsize=None)
def _get_gauge(metric_name: str):
    """
    Return an UpDownCounter used as a gauge for the given eval metric.
    Created lazily after the MeterProvider is configured by setup_otel().
    The lru_cache ensures each instrument is created exactly once.
    """
    meter = metrics.get_meter(_SCOPE, _SCOPE_VERSION)
    defn = next((m for m in CATALOG if m.name == metric_name), None)
    description = defn.description if defn else ""
    unit = defn.unit if defn else "1"
    return meter.create_up_down_counter(
        name=metric_name,
        description=description,
        unit=unit,
    )


# ── Singleton runners (lazy-initialised) ─────────────────────────────────────
# NOT constructed at import time because LLMJudgeRunner calls genai.Client()
# which requires GOOGLE_API_KEY to be set.  The key is available at runtime
# (loaded from .env by the app's startup handler) but not at import time.

_heuristic_runner: HeuristicRunner | None = None
_flash_runner:     LLMJudgeRunner  | None = None
_pro_runner:       LLMJudgeRunner  | None = None


def _get_runners() -> tuple[HeuristicRunner, LLMJudgeRunner, LLMJudgeRunner]:
    """Return (heuristic, flash, pro) runners, creating them on first call."""
    global _heuristic_runner, _flash_runner, _pro_runner
    if _heuristic_runner is None:
        _heuristic_runner = HeuristicRunner()
    if _flash_runner is None:
        _flash_runner = LLMJudgeRunner(model_tier="flash")
    if _pro_runner is None:
        _pro_runner = LLMJudgeRunner(model_tier="pro")
    return _heuristic_runner, _flash_runner, _pro_runner


# ── Result emitter ────────────────────────────────────────────────────────────

def _emit(result: EvalResult, job: EvalJob) -> None:
    """Record one EvalResult as an OTel gauge observation."""
    if not result.ok:
        _log.warning(
            "eval_result_skipped metric=%s job=%s error=%s",
            result.metric_name, job.job_id, result.error,
        )
        return

    labels: dict[str, str] = {
        "agent.framework": job.framework,
        "llm.model":       job.model,
        "agent.session_id": job.session_id,
        "eval.trace_id":   job.trace_id or "unknown",
        **result.labels,
    }

    try:
        _get_gauge(result.metric_name).add(result.value, labels)
        _log.debug(
            "eval_metric_recorded metric=%s value=%.4f trace_id=%s",
            result.metric_name, result.value, job.trace_id,
        )
    except Exception as exc:  # noqa: BLE001
        _log.error(
            "eval_emit_error metric=%s job=%s error=%s",
            result.metric_name, job.job_id, exc,
        )


# ── Public dispatcher ─────────────────────────────────────────────────────────

async def dispatch(job: EvalJob) -> None:
    """
    Run all three runner tiers concurrently for the given job, then emit
    each result as an OTel metric.

    This is designed to be called via asyncio.create_task() so it does NOT
    block the request handler:

        asyncio.create_task(dispatch(job))

    Exceptions inside runners are caught by BaseRunner.safe_run() and logged;
    they never propagate out of this coroutine.
    """
    _log.info(
        "eval_dispatch_start job=%s session=%s framework=%s model=%s trace=%s",
        job.job_id, job.session_id, job.framework, job.model, job.trace_id,
    )

    # Initialise runners on first call (after GOOGLE_API_KEY is loaded)
    heuristic, flash, pro = _get_runners()

    # Run all three tiers concurrently
    heuristic_results, flash_results, pro_results = await asyncio.gather(
        heuristic.safe_run(job),
        flash.safe_run(job),
        pro.safe_run(job),
    )

    all_results = heuristic_results + flash_results + pro_results
    for result in all_results:
        _emit(result, job)

    _log.info(
        "eval_dispatch_complete job=%s metrics_emitted=%d",
        job.job_id, sum(1 for r in all_results if r.ok),
    )
