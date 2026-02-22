from __future__ import annotations

import os
from typing import Literal, List
import json
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.models import llm_response
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.tool_context import ToolContext
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools.base_tool import BaseTool
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

from ...constants import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL

from .skill import search_skills, _load_skill_registry


_model_name = os.environ.get("LLM_MODEL", LLM_MODEL)
_model_api_key = os.environ.get("LLM_API_KEY", LLM_API_KEY)
_model_base_url = os.environ.get("LLM_BASE_URL", LLM_BASE_URL)

class SkillSearchInput(BaseModel):
    """Structured request passed from the ThinkingAgent to the plan-builder."""

    goal: str = Field(
        ...,
        description="Confirmed user goal in one sentence.",
    )

class SkillSearchOutput(BaseModel):
    """Name of relevant skills."""
    selected_skills: List[str] = Field(
        default_factory=list,
        description="Relevant skills for the task on-hand.",
    )

def search_markdown_skills(query: str, workflow_type: str, top_k: int = 2) -> dict:
    """Deterministic markdown skill retrieval tool."""
    hits = search_skills(query=query, top_k=top_k)
    return {
        "matched_skills": [
            {
                "name": item.name,
                "source_path": item.source_path,
                "tags": item.tags,
                "score": item.score,
                "guidance": item.guidance
            }
            for item in hits
        ]
    }


_SKILL_SEARCH_INSTRUCTION="""
You are a search agent that search for skills relevant to the goal.

Goal: {goal}
Skills: {skills}
"""

def before_agent_callback(callback_context: CallbackContext):
    """Load available skills from registry and inject them into the agent instruction via state."""
    registry = _load_skill_registry()
    lines = []
    for skill in registry.values():
        tags_str = ", ".join(skill.tags) if skill.tags else "none"
        lines.append(f"- {skill.name}: {skill.description} (tags: {tags_str})")
    skills_text = "\n".join(lines) if lines else "No skills available."
    callback_context.state["skills"] = skills_text
    return None


skill_search_tool_agent = LlmAgent(
    name="skill_search_tool_agent",
    model=LiteLlm(
        model=_model_name,
        base_url=_model_base_url,
        api_key=_model_api_key,
    ),
    description=(
        "Retrieves markdown-based workflow skills and returns planning-ready guidance, "
        "matched skill metadata, and allowed agent descriptions."
    ),
    instruction=_SKILL_SEARCH_INSTRUCTION,
    input_schema=SkillSearchInput,
    output_schema=SkillSearchOutput,
    output_key="skills",
    #before_agent_callback=before_agent_callback,
    #after_agent_callback=after_agent_callback,
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
)