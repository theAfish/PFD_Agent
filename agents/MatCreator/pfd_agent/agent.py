from google.adk.agents import  LlmAgent,Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.mcp_tool import MCPToolset
from google.adk.tools.mcp_tool.mcp_session_manager import SseServerParams
from google.adk.tools import agent_tool
#from .abacus_agent.agent import abacus_agent
from ..abacus_agent.agent import abacus_agent
from ..dpa_agent.agent import dpa_agent
#from .dpa_agent.agent import dpa_agent
#from .structure_agent.agent import structure_agent
import os, json

# Set the secret key in ~/.abacusagent/env.json or as an environment variable, or modify the code to set it directly.
env_file = os.path.expanduser("~/.pfd_agent/env.json")
if os.path.isfile(env_file):
    env = json.load(open(env_file, "r"))
else:
    env = {}
model_name = env.get("LLM_MODEL", os.environ.get("LLM_MODEL", ""))
model_api_key = env.get("LLM_API_KEY", os.environ.get("LLM_API_KEY", ""))
model_base_url = env.get("LLM_BASE_URL", os.environ.get("LLM_BASE_URL", ""))
bohrium_username = env.get("BOHRIUM_USERNAME", os.environ.get("BOHRIUM_USERNAME", ""))
bohrium_password = env.get("BOHRIUM_PASSWORD", os.environ.get("BOHRIUM_PASSWORD", ""))
bohrium_project_id = env.get("BOHRIUM_PROJECT_ID", os.environ.get("BOHRIUM_PROJECT_ID", ""))

description="""
The main coordinator agent for PFD (pretrain-finetuning-distillation) workflow. Handles PFD workflow and delegates DPA/ABACUS tasks to specialized sub-agents.
"""

instruction ="""
Mission
- Orchestrate the standard PFD workflow with minimal, safe steps and clear outputs:
    MD exploration → data curation (entropy selection) → DFT labeling (ABACUS) → model training (DPA).

Before detailed planning (must verify with user)
- MD: ensemble (NVT/NPT/NVE), temperature(s), total simulation time (ps), timestep/expected steps.
- Curation: max_sel (and chunk_size if applicable).
- Training: target epochs (or equivalent); propose a short validation run if long.

You have two specialized sub‑agents: 
1. 'dpa_agent_pfd': Handles molecular dynamic (MD) simulation with DPA model and training of DPA model. Delegate to it for these.
2. 'abacus_agent_pfd': Handles DFT calculations using ABACUS software. Delegate to it for these.

Workflow rules
- Create a workflow log for NEW PFD runs; show the initial plan; refine via update_workflow_log_plan until agreed. Read the log before delegating to sub-agent steps.
- In each step, either delegate to one sub-agent or execute a tool in this agent; do not mix.
- After each step, summarize artifacts with absolute paths and key metrics; propose the next step.

Failure and resume
- If a tool fails or is unavailable, show the exact error and propose a concrete alternative.
- To resume or resubmit, use resubmit_workflow_log then read_workflow_log to determine the next action.

Response format (strict)
- Plan: 1–3 bullets (why these steps).
- Action: the exact tool name you will call or the sub-agent you will delegate to.
- Result: concise outputs with absolute paths and critical metrics (e.g., frames selected, final energy).
- Next: the immediate next step or a final recap with follow‑ups.
"""

executor = {
    "bohr": {
        "type": "dispatcher",
        "machine": {
            "batch_type": "Bohrium",
            "context_type": "Bohrium",
            "remote_profile": {
                "email": bohrium_username,
                "password": bohrium_password,
                "program_id": bohrium_project_id,
                "input_data": {
                    "image_name": "registry.dp.tech/dptech/dp/native/prod-22618/abacus-agent-tools:v0.2",
                    "job_type": "container",
                    "platform": "ali",
                    "scass_type": "c32_m64_cpu",
                },
            },
        }
    },
    "local": {"type": "local",}
}

EXECUTOR_MAP = {
    "generate_bulk_structure": executor["local"],
    "generate_molecule_structure": executor["local"],
    "abacus_prepare": executor["local"],
    "abacus_modify_input": executor["local"],
    "abacus_modify_stru": executor["local"],
    "abacus_collect_data": executor["local"],
    "abacus_prepare_inputs_from_relax_results": executor["local"],
    "generate_bulk_structure_from_wyckoff_position": executor["local"],
}

STORAGE = {
    "type": "https",
    "plugin":{
        "type": "bohrium",
        "username": bohrium_username,
        "password": bohrium_password,
        "project_id": bohrium_project_id,
    }
}

toolset = MCPToolset(
    connection_params=SseServerParams(
        url="http://localhost:50001/sse", # Or any other MCP server URL
        sse_read_timeout=3600,  # Set SSE timeout to 3600 seconds
    ),
    tool_filter=[
        "create_workflow_log",
        "update_workflow_log_plan",
        "read_workflow_log",
        "resubmit_workflow_log",
        "filter_by_entropy",
    ],
)

abacus_agent= abacus_agent.clone(
    update={
        "name": "abacus_agent_pfd",
        },
)

dpa_agent= dpa_agent.clone(
    update={"name": "dpa_agent_pfd"},
)

pfd_agent = LlmAgent(
    name='pfd_agent',
    model=LiteLlm(
        model=model_name,
        base_url=model_base_url,
        api_key=model_api_key
    ),
    description=description,
    instruction=instruction,
    tools=[
        toolset,
        #abacus_tools,
        #dpa_tools,
        #structure_tools
        ],
    sub_agents=[
        abacus_agent,
        dpa_agent,
        #structure_agent
    ]
)