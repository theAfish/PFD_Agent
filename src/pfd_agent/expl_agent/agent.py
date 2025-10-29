"""
Exploration Agent
Handles structure optimization, molecular dynamics, and dataset curation.
"""

from google.adk.agents import LlmAgent
from .prompt import (
    EXPLAGENT_NAME,
    EXPLAGENTInstruction,
    EXPLAGENTDescription
)
from ..utils.llm_config import LlmConfig


from ..tools.expl import (
    list_calculators,
    optimize_structure,
    run_molecular_dynamics,
    get_base_model_path,
    filter_by_entropy,
)

from ..tools.artifacts.artifact_file_bridge import (
    list_files_and_artifacts_tool,
    artifact_write_tool,
    file_read_artifact_tool,
    get_artifact_path_tool,
)
# Import calculator module to ensure all calculators are registered
from ..tools.expl import calculator


def init_expl_agent(config):
    """
    Initialize exploration agent with the given configuration.

    Args:
        config: LLM configuration object

    Returns:
        LlmAgent: Configured exploration agent instance
    """
    expl_agent = LlmAgent(
        name=EXPLAGENT_NAME,
        model=config.deepseek_chat,
        instruction=EXPLAGENTInstruction,
        description=EXPLAGENTDescription,
        tools=[
            # Exploration tools
            list_calculators,
            optimize_structure,
            # run_molecular_dynamics,
            get_base_model_path,
            filter_by_entropy,
            # Artifact tools
            list_files_and_artifacts_tool,
            file_read_artifact_tool,
            get_artifact_path_tool,
            artifact_write_tool,
        ],
    )
    return expl_agent


# Example usage
root_agent = init_expl_agent(LlmConfig)
