"""Plan-builder sub-agent for the ThinkingAgent.

This agent is the structural equivalent of `database_agent/sql_agent`:
it accepts a structured planning request via `PlanBuilderInput` and returns
a fully-formed `ExecutionPlan` JSON without any custom orchestration logic.
"""

from __future__ import annotations

import os
from typing import List, Literal

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from pydantic import BaseModel, Field

from ...constants import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL

_model_name = os.environ.get("LLM_MODEL", LLM_MODEL)
_model_api_key = os.environ.get("LLM_API_KEY", LLM_API_KEY)
_model_base_url = os.environ.get("LLM_BASE_URL", LLM_BASE_URL)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class PlanStep(BaseModel):
    """Single step in the execution plan."""

    step_number: int = Field(..., description="Sequential step number (1, 2, 3, ...)")
    agent: str = Field(
        ...,
        description=(
            "Agent or tool that will execute this step."
        ),
    )
    action: str = Field(
        ...,
        description="Clear, concise description of what this step does (1-2 sentences)",
        max_length=500,
    )
    inputs_required: str = Field(
        ...,
        description="Expected inputs (models, parameters, etc.)",
        max_length=300,
    )
    expected_output: str = Field(
        ...,
        description="What result this step produces",
        max_length=300,
    )


class ExecutionPlan(BaseModel):
    """Structured execution plan for user approval."""
    steps: List[PlanStep] = Field(
        ...,
        description="Ordered list of execution steps",
        min_items=1,
        max_items=10,
    )
    fallback_strategy: str = Field(
        ...,
        description=(
            "If any step fails or is not feasible, describe an alternative approach "
            "or contingency plan (1-2 sentences)"
        ),
        max_length=500,
    )
    additional_notes: str = Field(
        ...,
        description="Any extra information or considerations for the user",
        max_length=500,
    )


class PlanBuilderInput(BaseModel):
    """Structured request passed from the ThinkingAgent to the plan-builder."""

    goal: str = Field(
        ...,
        description="Confirmed user goal in one sentence.",
    )


# ---------------------------------------------------------------------------
# Instruction
# ---------------------------------------------------------------------------

_PLAN_BUILDER_INSTRUCTION = """
You are a plan-builder sub-agent. You produce a
detailed, actionable execution plan.

Input:
- goal: {goal}
- skills: {skills}
- agent_descriptions: {agents}

Output requirements:
- Use ONLY agent names that appear in agent_descriptions.
- Keep each step specific: clear action, named inputs, concrete expected output.
- Prefer reuse/fine-tuning over expensive retraining when feasible.
- Respect workflow_guidance ordering/constraints and align with matched_skills.
- Return only valid JSON conforming to ExecutionPlan. No markdown fences or extra commentary.
"""


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

plan_builder_agent = LlmAgent(
    name="plan_builder_agent",
    model=LiteLlm(
        model=_model_name,
        base_url=_model_base_url,
        api_key=_model_api_key,
    ),
    description=(
        "Accepts a structured planning request (goal, workflow type, guidance, available agents) "
        "and produces a detailed ExecutionPlan JSON. Used as a tool by the ThinkingAgent."
    ),
    instruction=_PLAN_BUILDER_INSTRUCTION,
    input_schema=PlanBuilderInput,
    output_schema=ExecutionPlan,
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
)
