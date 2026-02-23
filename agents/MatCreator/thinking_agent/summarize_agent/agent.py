"""Summarize-agent submodule for thinking phase post-execution synthesis.

This agent summarizes execution outcomes using:
- confirmed goal (session state)
- approved execution plan (session state)
- execution history / results (session history passed by caller)
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Literal

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from pydantic import BaseModel, Field

from ...constants import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL

_model_name = os.environ.get("LLM_MODEL", LLM_MODEL)
_model_api_key = os.environ.get("LLM_API_KEY", LLM_API_KEY)
_model_base_url = os.environ.get("LLM_BASE_URL", LLM_BASE_URL)


class ExecutionSummaryInput(BaseModel):
    """Input for execution-result summarization."""

    goal: str = Field(
        ...,
        description="Confirmed user goal from session state.",
        max_length=500,
    )
    plan: Dict[str, Any] = Field(
        ...,
        description="Approved execution plan object from session state.",
    )
    execution_history: str = Field(
        ...,
        description=(
            "Relevant execution transcript/history from session messages and tool results, "
            "including errors and produced outputs/artifacts."
        ),
        max_length=12000,
    )


class ExecutionSummary(BaseModel):
    """Structured summary of execution outcomes for decision making."""

    goal: str = Field(
        ...,
        description="Goal that was evaluated.",
        max_length=500,
    )
    completion_status: Literal["completed", "in_progress", "blocked", "not_started"] = Field(
        ...,
        description="Overall execution status against the approved plan and goal.",
    )
    completed_steps: List[int] = Field(
        default_factory=list,
        description="Step numbers completed successfully.",
    )
    pending_steps: List[int] = Field(
        default_factory=list,
        description="Step numbers not yet completed.",
    )
    failed_steps: List[int] = Field(
        default_factory=list,
        description="Step numbers that failed or were blocked.",
    )
    key_results: str = Field(
        ...,
        description="Concise summary of what was produced or learned during execution.",
        max_length=1200,
    )
    artifacts: List[str] = Field(
        default_factory=list,
        description="Important generated files/paths/IDs observed in execution outputs.",
    )
    blockers: str = Field(
        default="",
        description="Main blockers or errors preventing completion. Empty if none.",
        max_length=800,
    )
    recommended_next_action: Literal[
        "continue_execution",
        "request_user_input",
        "replan",
        "mark_complete",
    ] = Field(
        ...,
        description="Recommended next action for the orchestrator.",
    )
    concise_summary: str = Field(
        ...,
        description="User-facing one-paragraph execution summary.",
        max_length=800,
    )


_SUMMARIZE_INSTRUCTION = """
You are a summarize-agent used by the thinking agent after or during execution.

Input (ExecutionSummaryInput):
- goal: {goal}
- plan: {plan}
- execution_history: execution messages/tool outputs/errors/artifacts

Task:
1) Compare execution history against plan steps and goal.
2) Determine completion status and infer completed/pending/failed step numbers.
3) Summarize key outcomes and extract concrete artifacts (absolute paths/IDs when present).
4) Provide blockers and recommend one next action.

Rules:
- Be conservative: if evidence is incomplete, use in_progress.
- Do not invent artifacts or metrics; only include what appears in history.
- If all required steps are complete and goal achieved -> mark_complete.
- If blocked by errors or missing prerequisites -> replan or request_user_input as appropriate.
- Return only JSON conforming to ExecutionSummary.
"""


summarize_agent = LlmAgent(
    name="summarize_agent",
    model=LiteLlm(
        model=_model_name,
        base_url=_model_base_url,
        api_key=_model_api_key,
    ),
    description=(
        "Summarizes execution progress/results from goal, approved plan, and execution history, "
        "returning a structured ExecutionSummary."
    ),
    instruction=_SUMMARIZE_INSTRUCTION,
    input_schema=ExecutionSummaryInput,
    output_schema=ExecutionSummary,
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
)
