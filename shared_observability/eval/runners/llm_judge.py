"""
LLM Judge runner — uses Gemini models to score agent responses.

Two tiers (configured via `model_tier` at construction time):

    "flash"  →  gemini-2.5-flash   Lighter, cheaper; used for relevance /
                                   completeness / conciseness / tool_appropriateness.

    "pro"    →  gemini-2.5-pro     Highest quality; used for hallucination_risk /
                                   task_completion / session_goal_achieved.

Each judge call uses Gemini's structured output (response_schema=...) so
scores are always valid floats in [0, 1] without any fragile string parsing.

Prompt design principles
------------------------
  • System instruction defines the role of an impartial evaluator.
  • User turn provides the rubric, prompt, response, and tool context.
  • Asks for a JSON object with a single `score` key (float 0-1).
  • Chain-of-thought is elicited via a `reasoning` key (discarded after
    scoring) — this improves calibration without increasing response size.
  • Temperature = 0 for reproducibility.

Environment variable
--------------------
  GOOGLE_API_KEY  must be set (same key as the agent apps).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types as genai_types

from ..catalog import BY_RUNNER
from ..models import EvalJob
from .base import BaseRunner, EvalResult

_log = logging.getLogger(__name__)

# ── Model aliases ─────────────────────────────────────────────────────────────

_MODELS: dict[str, str] = {
    "flash": "gemini-2.5-flash",
    "pro":   "gemini-2.5-pro",
}

# ── JSON schema for all judge responses ──────────────────────────────────────
# Gemini structured output guarantees the response matches this schema exactly.

_SCORE_SCHEMA = genai_types.Schema(
    type=genai_types.Type.OBJECT,
    properties={
        "reasoning": genai_types.Schema(type=genai_types.Type.STRING),
        "score":     genai_types.Schema(
            type=genai_types.Type.NUMBER,
            description="Float in [0.0, 1.0]",
        ),
    },
    required=["reasoning", "score"],
)

# ── System instruction (same for all judges) ──────────────────────────────────

_SYSTEM = (
    "You are an impartial AI quality evaluator. "
    "You assess AI agent responses based on the provided rubric. "
    "Always respond with valid JSON matching the requested schema. "
    "Scores must be floats in the range [0.0, 1.0] where 1.0 is best."
)


# ── Per-metric rubrics ────────────────────────────────────────────────────────

def _prompt_relevance(job: EvalJob) -> str:
    return (
        f"## Task\nScore the AI response's RELEVANCE to the user prompt.\n\n"
        f"**User prompt:** {job.prompt}\n\n"
        f"**AI response:** {job.response}\n\n"
        "## Rubric\n"
        "1.0 = Response directly and fully addresses the prompt.\n"
        "0.7 = Mostly relevant with minor tangents.\n"
        "0.4 = Partially relevant; some important aspects missed.\n"
        "0.1 = Barely relevant; mostly off-topic.\n"
        "0.0 = Completely irrelevant.\n\n"
        "Provide your reasoning first, then a score."
    )


def _prompt_completeness(job: EvalJob) -> str:
    return (
        f"## Task\nScore the COMPLETENESS of the AI response.\n\n"
        f"**User prompt:** {job.prompt}\n\n"
        f"**AI response:** {job.response}\n\n"
        "## Rubric\n"
        "1.0 = All parts of the prompt are fully addressed.\n"
        "0.7 = Most parts addressed; one minor omission.\n"
        "0.4 = Several significant omissions.\n"
        "0.0 = Response barely addresses the prompt.\n\n"
        "Provide your reasoning first, then a score."
    )


def _prompt_hallucination(job: EvalJob) -> str:
    return (
        f"## Task\nScore the HALLUCINATION RISK of the AI response.\n\n"
        f"**User prompt:** {job.prompt}\n\n"
        f"**AI response:** {job.response}\n\n"
        f"**Tools used:** {', '.join(job.tools_used) if job.tools_used else 'none'}\n\n"
        "## Rubric\n"
        "Score = probability that the response contains fabricated, unverifiable, "
        "or factually incorrect claims.\n"
        "1.0 = High risk; clear hallucinations present.\n"
        "0.5 = Some unverifiable claims that may be incorrect.\n"
        "0.1 = Very low risk; response appears factually grounded.\n"
        "0.0 = No hallucination risk detected.\n\n"
        "Provide your reasoning first, then a score."
    )


def _prompt_conciseness(job: EvalJob) -> str:
    return (
        f"## Task\nScore the CONCISENESS of the AI response.\n\n"
        f"**User prompt:** {job.prompt}\n\n"
        f"**AI response:** {job.response}\n\n"
        "## Rubric\n"
        "1.0 = Response is appropriately brief; no unnecessary padding.\n"
        "0.7 = Slightly verbose but acceptable.\n"
        "0.4 = Noticeably verbose; could be 30%+ shorter without losing value.\n"
        "0.0 = Extremely verbose; mostly filler.\n\n"
        "Provide your reasoning first, then a score."
    )


def _prompt_tool_appropriateness(job: EvalJob) -> str:
    return (
        f"## Task\nScore the TOOL APPROPRIATENESS of the agent's tool usage.\n\n"
        f"**User prompt:** {job.prompt}\n\n"
        f"**Tools available:** web_search, math\n"
        f"**Tools actually used:** {', '.join(job.tools_used) if job.tools_used else 'none'}\n\n"
        f"**AI response:** {job.response}\n\n"
        "## Rubric\n"
        "1.0 = Agent called exactly the right tools for this task.\n"
        "0.7 = Mostly appropriate; one minor unnecessary or missed tool call.\n"
        "0.4 = Some inappropriate tool use or significant missed tool calls.\n"
        "0.0 = Completely wrong tool selection or unnecessary tool spam.\n\n"
        "Provide your reasoning first, then a score."
    )


def _prompt_task_completion(job: EvalJob) -> str:
    return (
        f"## Task\nDid the AI agent successfully COMPLETE the user's task?\n\n"
        f"**User prompt:** {job.prompt}\n\n"
        f"**AI response:** {job.response}\n\n"
        f"**Tools used:** {', '.join(job.tools_used) if job.tools_used else 'none'}\n"
        f"**Reasoning turns:** {job.reasoning_turns}\n\n"
        "## Rubric\n"
        "Return 1.0 if the task was fully completed, 0.0 if it was not.\n"
        "For partial completion, use intermediate values.\n"
        "Consider: Did the agent provide a concrete, actionable answer? "
        "Did it use the right tools when needed?\n\n"
        "Provide your reasoning first, then a score."
    )


def _prompt_session_goal(job: EvalJob) -> str:
    return (
        f"## Task\nScore whether the overall SESSION GOAL was achieved.\n\n"
        f"**User prompt:** {job.prompt}\n\n"
        f"**AI response:** {job.response}\n\n"
        f"**Session ID:** {job.session_id}\n"
        f"**Tools used:** {', '.join(job.tools_used) if job.tools_used else 'none'}\n"
        f"**Reasoning turns:** {job.reasoning_turns}\n\n"
        "## Rubric\n"
        "1.0 = Session goal fully achieved; user would be satisfied.\n"
        "0.7 = Goal mostly achieved; minor gaps.\n"
        "0.4 = Partial progress; user would need to follow up significantly.\n"
        "0.0 = Goal not achieved.\n\n"
        "Provide your reasoning first, then a score."
    )


# ── Routing table: metric_name → prompt_builder function ─────────────────────

_PROMPTS: dict[str, Any] = {
    "eval.llm.response_relevance":        _prompt_relevance,
    "eval.llm.completeness":              _prompt_completeness,
    "eval.llm.hallucination_risk":        _prompt_hallucination,
    "eval.llm.conciseness":               _prompt_conciseness,
    "eval.agent.tool_appropriateness":    _prompt_tool_appropriateness,
    "eval.business.task_completion":      _prompt_task_completion,
    "eval.business.session_goal_achieved": _prompt_session_goal,
}


# ── Runner ────────────────────────────────────────────────────────────────────

class LLMJudgeRunner(BaseRunner):
    """
    Async LLM judge runner.

    Parameters
    ----------
    model_tier : "flash" | "pro"
        Selects which Gemini model to use.  The tier is set at construction
        time; one runner instance is created per tier in the scheduler.
    """

    def __init__(self, model_tier: str = "flash") -> None:
        self._model_tier = model_tier
        self._model_name = _MODELS[model_tier]
        self._metrics = [
            m.name for m in BY_RUNNER.get(f"llm_{model_tier}", [])
        ]
        self._client = self._build_client()

    @staticmethod
    def _build_client() -> genai.Client:
        """
        Build a genai.Client, finding GOOGLE_API_KEY by:
          1. os.environ (set by the app's config module at startup)
          2. Walk up from CWD looking for a .env file (belt-and-suspenders
             for cases where the app is launched from an unexpected directory)
          3. Raise a clear error if the key is still not found.
        """
        api_key = os.environ.get("GOOGLE_API_KEY", "").strip()

        if not api_key:
            # Try to load from a .env file found anywhere up the directory tree
            search_dir = Path.cwd()
            for _ in range(6):  # walk up at most 6 levels
                dotenv_path = search_dir / ".env"
                if dotenv_path.is_file():
                    _log.debug("llm_judge loading .env from %s", dotenv_path)
                    # Parse the file manually — avoids a hard dotenv dependency
                    with dotenv_path.open() as fh:
                        for line in fh:
                            line = line.strip()
                            if line.startswith("GOOGLE_API_KEY="):
                                api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                                os.environ["GOOGLE_API_KEY"] = api_key
                                break
                    if api_key:
                        break
                parent = search_dir.parent
                if parent == search_dir:
                    break
                search_dir = parent

        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY not found in os.environ or any .env file. "
                "Set the key before starting the app."
            )

        return genai.Client(api_key=api_key)

    async def run(self, job: EvalJob) -> list[EvalResult]:
        results: list[EvalResult] = []

        for metric_name in self._metrics:
            prompt_fn = _PROMPTS.get(metric_name)
            if prompt_fn is None:
                _log.warning(
                    "llm_judge_no_prompt metric=%s — skipping", metric_name
                )
                continue

            result = await self._judge(job, metric_name, prompt_fn(job))
            results.append(result)

        return results

    async def _judge(
        self, job: EvalJob, metric_name: str, user_prompt: str
    ) -> EvalResult:
        """Call Gemini and return an EvalResult for one metric."""
        try:
            response = await self._client.aio.models.generate_content(
                model=self._model_name,
                contents=user_prompt,
                config=genai_types.GenerateContentConfig(
                    system_instruction=_SYSTEM,
                    response_mime_type="application/json",
                    response_schema=_SCORE_SCHEMA,
                    temperature=0.0,
                    max_output_tokens=512,
                ),
            )

            # Structured output guarantees valid JSON with `score` key
            data: dict = response.parsed or {}
            score = float(data.get("score", 0.0))
            # Clamp to [0, 1] as a safety net
            score = max(0.0, min(1.0, score))

            _log.debug(
                "llm_judge_complete metric=%s model=%s job=%s score=%.3f "
                "reasoning=%s",
                metric_name, self._model_name, job.job_id, score,
                str(data.get("reasoning", ""))[:120],
            )
            return EvalResult(metric_name=metric_name, value=score)

        except Exception as exc:  # noqa: BLE001
            _log.error(
                "llm_judge_error metric=%s model=%s job=%s error=%s",
                metric_name, self._model_name, job.job_id, exc,
            )
            return EvalResult(
                metric_name=metric_name,
                value=0.0,
                ok=False,
                error=str(exc),
            )
