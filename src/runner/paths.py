"""Runner environment & path helpers.

The runner needs stable ways to locate project-level resources such as:

- `configs/runner.yaml`
- per-experiment YAML configs (`configs/experiment_*.yaml`)
- the web dashboard HTML
- results and persistence files

Historically this was implemented with `Path(__file__).parent.parent.parent` in
multiple modules. That breaks when files move.

This module centralizes root discovery so the runner can be reorganized without
changing behavior.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def project_root() -> Path:
    """Return the repository root directory.

    Strategy: walk upward from this file until a directory containing either
    `pyproject.toml` or `configs/` is found.

    This is intentionally conservative: it prefers *finding* the root rather
    than assuming a fixed parent depth.
    """

    start = Path(__file__).resolve()

    for candidate in (start.parent, *start.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
        if (candidate / "configs").is_dir():
            return candidate

    # Fallback: for the current repo layout `src/runner/paths.py` → root.
    try:
        return start.parents[2]
    except Exception:
        return start.parent


def runner_config_path() -> Path:
    """Return the path to `configs/runner.yaml` (may not exist)."""

    return project_root() / "configs" / "runner.yaml"


def experiment_config_path(experiment_id: int) -> Path:
    """Return the path to a specific experiment YAML config."""

    return project_root() / "configs" / f"experiment_{experiment_id}.yaml"


def dashboard_path() -> Path:
    """Return the path to the dashboard HTML file."""

    return Path(__file__).resolve().parent / "dashboard.html"
