"""MatCreator CLI entry point.

This module bootstraps the project root onto sys.path so that the ``script``
and ``agents`` packages are importable, then delegates to the Click CLI group
defined in ``script.start_agent``.
"""

import sys
from pathlib import Path

# The installed package lives in src/matcreator; the project root is two levels up.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from script.start_agent import main  # noqa: E402

__all__ = ["main"]
