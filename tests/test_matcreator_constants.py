from __future__ import annotations

import importlib
import sys
from pathlib import Path


def test_default_kdg_db_path_points_to_repo_local_adk(monkeypatch) -> None:
    monkeypatch.delenv("KDG_DB_PATH", raising=False)
    sys.modules.pop("src.matcreator.constants", None)

    constants = importlib.import_module("src.matcreator.constants")

    project_root = Path(__file__).resolve().parents[1]
    expected = project_root / "agents" / "MatCreator" / ".adk" / "know_do_graph.db"

    assert constants.DEFAULT_KDG_DB_PATH == expected
    assert constants.KNOW_DO_GRAPH_DB == expected


def test_home_adk_db_is_treated_as_legacy_source(monkeypatch) -> None:
    monkeypatch.delenv("KDG_DB_PATH", raising=False)
    sys.modules.pop("src.matcreator.constants", None)

    constants = importlib.import_module("src.matcreator.constants")

    assert constants.LEGACY_UNIFIED_GRAPH_DB == (
        Path.home() / ".matcreator" / ".adk" / "know_do_graph.db"
    )
