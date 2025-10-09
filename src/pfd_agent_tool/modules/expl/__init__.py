"""Exploration MCP server bootstrap.

Import the exploration tool modules (ase.py and atoms.py) so their
@mcp.tool-decorated functions register with the shared MCP instance,
then provide a simple entrypoint to run the server.
"""
from __future__ import annotations

import os
from pfd_agent_tool.init_mcp import mcp

# Import tool modules so that their @mcp.tool decorators execute and register tools
# Note: name carefully to avoid clashing with the external 'ase' package symbol
from pfd_agent_tool.modules.expl import ase as expl_tools_ase  # noqa: F401
from pfd_agent_tool.modules.expl import atoms as expl_tools_atoms  # noqa: F401