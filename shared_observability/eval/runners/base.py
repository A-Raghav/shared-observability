"""
Abstract base runner interface.

Every runner (heuristic, llm_flash, llm_pro) implements `run()` which
accepts an EvalJob and returns a list of EvalResult objects.

EvalResult carries:
    metric_name  — catalog metric name (e.g. "eval.llm.response_relevance")
    value        — float score/count to be recorded as an OTel gauge
    labels       — extra Prometheus labels beyond the standard set
    ok           — False if the runner itself errored (result unreliable)
    error        — error message string when ok=False

The scheduler calls runner.run() and then emits each EvalResult as an
OTel gauge observation with standard labels:
    agent.framework, llm.model, agent.session_id, eval.trace_id
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ..models import EvalJob

_log = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """One scored eval metric from a runner."""

    metric_name: str
    value: float
    labels: dict[str, str] = field(default_factory=dict)
    ok: bool = True
    error: str = ""


class BaseRunner(ABC):
    """
    Abstract runner.  Subclasses implement `run()`.

    The `safe_run()` wrapper catches all exceptions and converts them into
    EvalResult(ok=False) entries so a failing runner never crashes the
    scheduler loop.
    """

    @abstractmethod
    async def run(self, job: EvalJob) -> list[EvalResult]:
        """Return a list of EvalResult objects for the given job."""

    async def safe_run(self, job: EvalJob) -> list[EvalResult]:
        """
        Call `run()` and catch any exception, returning an empty list
        plus a logged error rather than propagating.
        """
        try:
            return await self.run(job)
        except Exception as exc:  # noqa: BLE001
            _log.exception(
                "eval_runner_error runner=%s job=%s error=%s",
                type(self).__name__, job.job_id, exc,
            )
            return []
