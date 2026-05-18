"""Disk-based progress tracking for the runner dashboard.

All progress is derived from files on disk so it is never stale and
survives process restarts. No in-memory counters are needed.

Functions
---------
- ``get_progress_from_disk`` — read results JSONL + checkpoint directories
  + failed-items files to compute completed / in-progress / failed /
  remaining counts.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.engine.result_store import load_completed_game_ids, load_failed_item_ids

logger = logging.getLogger(__name__)


def get_progress_from_disk(
    *,
    experiment: int,
    condition: str,
    generation_strategy: str,
    output_dir: Path,
    total_items: int,
) -> dict[str, Any]:
    """Compute current progress by reading disk artifacts.

    Parameters
    ----------
    experiment:
        Experiment ID (1, 2, or 3).
    condition:
        Condition letter (e.g. ``"A"``).
    generation_strategy:
        Generation strategy name.
    output_dir:
        Experiment output directory containing results and checkpoints.
    total_items:
        Total number of puzzles or games expected.

    Returns
    -------
    dict
        ``{"total", "completed", "in_progress", "failed", "remaining"}``
    """

    results_path = output_dir / f"{generation_strategy}_{condition}" / "results.jsonl"
    if experiment == 1:
        checkpoint_path = output_dir / ".checkpoint"
    else:
        checkpoint_path = None
    completed_ids = load_completed_game_ids(results_path, checkpoint_path)

    # Filter completed IDs to only those matching this experiment/condition
    prefix = f"exp{experiment}_"
    suffix = f"_{condition}"
    relevant_completed = {
        gid for gid in completed_ids
        if gid.startswith(prefix) and (
            # Puzzles: exp1_{puzzle_id}_{condition}
            (experiment == 1 and gid.endswith(suffix))
            # Games: exp{N}_{condition}_game{NNN}
            or (experiment != 1 and f"_{condition}_" in gid)
        )
    }

    # Failed items: from per-condition/strategy failed JSONL
    failed_path = output_dir / f"{generation_strategy}_{condition}" / "fail.jsonl"
    failed_ids = load_failed_item_ids(failed_path)

    # In-progress items: mid-game checkpoints (games only)
    in_progress = 0
    if experiment != 1:
        midgame_dir = output_dir / f"{generation_strategy}_{condition}" / "checkpoints"
        checkpoint_glob = "*.jsonl"
        if midgame_dir.exists():
            in_progress = len(list(midgame_dir.glob(checkpoint_glob)))

    completed = len(relevant_completed)
    failed = len(failed_ids)
    remaining = max(0, total_items - completed - failed)

    return {
        "total": total_items,
        "completed": completed,
        "in_progress": in_progress,
        "failed": failed,
        "remaining": remaining,
    }
