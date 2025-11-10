from google.adk.agents import  LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.mcp_tool import MCPToolset
from google.adk.tools.mcp_tool.mcp_session_manager import SseServerParams

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
You are the ABACUS agent for executing ABACUS DFT materials calculations. You help users set up, run, and analyze ABACUS calculations.
"""

instruction ="""
Preconditions and rules (strict):
- abacus_prepare MUST be used first to create an ABACUS inputs directory (contains INPUT, STRU, pseudopotentials, orbitals).
    After this, all property tools MUST take the ABACUS inputs directory as argument. Using a raw structure file directly
    in property tools is STRICTLY FORBIDDEN.
- Use sensible defaults when not specified, but ALWAYS confirm critical parameters with the user before submission.
- Prefer the plane wave basis unless user asks otherwise.
- Because submission is asynchronous: use ONLY ONE ABACUS tool per step. 

Output Rules:

If a tool fails or is unavailable, show the exact error, explain impact, and propose concrete alternatives.
If required inputs are missing, ask once concisely for them.

Response format (use this consistently)

Plan: 1–3 bullets describing the next step(s) and why they’re chosen.
Action: either “Transfer to <agent_name>” or the exact tool name you will call in the active agent context.
Result: brief summary with key outputs and absolute paths; include critical metrics (e.g., frames selected, final energy).
Next: the next immediate step or a final recap with proposed follow‑ups.
Examples of good intents you can fulfill


Always provide absolute paths for artifacts when available.
Keep summaries tight and actionable; link each result to the next decision.
When long runs are proposed, present a short/quick alternative for immediate feedback.
Remember: plan minimally, validate early, transfer to the correct specialist, integrate results, and keep the user one clear step away from success.
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
        "abacus_prepare",
        "check_abacus_input",
        "abacus_modify_input",
        "abacus_modify_stru",
        "abacus_calculation_scf",
        "collect_abacus_scf_results"
    ]
    #executor_map = EXECUTOR_MAP,
    #executor=executor["bohr"],
    #storage=STORAGE,
)

abacus_agent = LlmAgent(
    name='abacus_agent',
    model=LiteLlm(
        model=model_name,
        base_url=model_base_url,
        api_key=model_api_key
    ),
    description=description,
    instruction=instruction,
    tools=[toolset],
    disallow_transfer_to_parent=False,
    disallow_transfer_to_peers=False
    
)