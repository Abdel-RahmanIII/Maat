"""Turn-level checkpoint persistence for resumable experiments.

Extends the existing game-level checkpoint (``output_dir/.checkpoint``)
with per-game state files that capture mid-game progress:

    output_dir/.game_state/{game_id}.json

On resume, incomplete games are detected and continued from the last
completed turn.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── Game-state I/O ───────────────────────────────────────────────────────


def _state_dir(output_dir: Path) -> Path:
    """Return the ``.game_state`` directory under *output_dir*."""
    return output_dir / ".game_state"


def save_game_state(
    output_dir: Path,
    *,
    game_id: str,
    condition: str,
    experiment: int,
    starting_fen: str,
    board_fen: str,
    move_stack_uci: list[str],
    half_moves_played: int,
    turn_records: list[dict[str, Any]],
    game_status: str,
    input_mode: str = "fen",
    generation_strategy: str = "generator_only",
) -> Path:
    """Atomically write a game-state checkpoint.

    Returns the path to the written file.
    """

    d = _state_dir(output_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{game_id}.json"

    payload = {
        "game_id": game_id,
        "condition": condition,
        "experiment": experiment,
        "starting_fen": starting_fen,
        "board_fen": board_fen,
        "move_stack_uci": move_stack_uci,
        "half_moves_played": half_moves_played,
        "turn_records": turn_records,
        "game_status": game_status,
        "input_mode": input_mode,
        "generation_strategy": generation_strategy,
    }

    # Atomic write: write to tmp then rename
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)

    return path


def load_game_state(output_dir: Path, game_id: str) -> dict[str, Any] | None:
    """Load a saved game state, or return ``None`` if not found."""

    path = _state_dir(output_dir) / f"{game_id}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Corrupt game-state file %s", path, exc_info=True)
        return None


def delete_game_state(output_dir: Path, game_id: str) -> None:
    """Remove a game-state file (called after game completes)."""

    path = _state_dir(output_dir) / f"{game_id}.json"
    if path.exists():
        path.unlink()


def list_incomplete_games(output_dir: Path) -> list[str]:
    """Return game IDs that have state files (i.e. in-progress games)."""

    d = _state_dir(output_dir)
    if not d.exists():
        return []
    return [p.stem for p in d.glob("*.json")]


# ── Experiment-level progress persistence ────────────────────────────────


def save_run_progress(
    output_dir: Path,
    *,
    experiment: int,
    conditions: list[str],
    condition_progress: dict[str, dict[str, Any]],
    status: str,
    started_at: str | None = None,
    paused_at: str | None = None,
) -> Path:
    """Save overall experiment-run progress for dashboard display on resume.

    Written to ``output_dir/.run_progress.json``.
    """

    path = output_dir / ".run_progress.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "experiment": experiment,
        "conditions": conditions,
        "condition_progress": condition_progress,
        "status": status,
        "started_at": started_at,
        "paused_at": paused_at,
    }

    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)

    return path


def load_run_progress(output_dir: Path) -> dict[str, Any] | None:
    """Load experiment-run progress, or ``None`` if not found."""

    path = output_dir / ".run_progress.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Corrupt run-progress file %s", path, exc_info=True)
        return None
