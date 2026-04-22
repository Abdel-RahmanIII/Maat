"""Result persistence and checkpoint utilities.

Provides JSONL-based storage for :class:`~src.metrics.definitions.GameRecord`
objects, a simple checkpoint mechanism for resumable experiment runs, and a
summary CSV writer.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

from src.metrics.definitions import GameRecord

logger = logging.getLogger(__name__)


# ── JSONL game-record persistence ────────────────────────────────────────


def append_game_record(record: GameRecord, filepath: Path) -> None:
    """Append one ``GameRecord`` as a JSON line to *filepath*.

    Creates the file and parent directories if they don't exist.
    """

    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("a", encoding="utf-8") as fh:
        fh.write(record.model_dump_json() + "\n")


def load_game_records(filepath: Path) -> list[GameRecord]:
    """Load all ``GameRecord`` objects from a JSONL file.

    Returns an empty list if the file does not exist.
    """

    if not filepath.exists():
        return []

    records: list[GameRecord] = []
    with filepath.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(GameRecord.model_validate_json(stripped))
            except Exception:
                logger.warning(
                    "Skipping malformed JSON at %s:%d",
                    filepath,
                    line_no,
                )
    return records


# ── Checkpoint management ────────────────────────────────────────────────


def load_checkpoint(filepath: Path) -> set[str]:
    """Load the set of completed game IDs from a checkpoint file.

    Each line in the checkpoint file is a single ``game_id``.
    Returns an empty set if the file does not exist.
    """

    if not filepath.exists():
        return set()

    completed: set[str] = set()
    with filepath.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                completed.add(stripped)
    return completed


def append_checkpoint(game_id: str, filepath: Path) -> None:
    """Mark *game_id* as completed in the checkpoint file."""

    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("a", encoding="utf-8") as fh:
        fh.write(game_id + "\n")


# ── Summary CSV writer ───────────────────────────────────────────────────


def write_summary_csv(records: list[GameRecord], filepath: Path) -> None:
    """Write a per-game summary CSV.

    Columns: ``game_id``, ``condition``, ``experiment``, ``input_mode``,
    ``total_turns``, ``total_llm_calls``, ``total_tokens``,
    ``final_status``, ``n_forfeits``, ``n_first_try_invalid``,
    ``fir``, ``ftir``.
    """

    filepath.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "game_id",
        "condition",
        "experiment",
        "input_mode",
        "total_turns",
        "total_llm_calls",
        "total_tokens",
        "final_status",
        "n_forfeits",
        "n_first_try_invalid",
        "fir",
        "ftir",
    ]

    with filepath.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()

        for rec in records:
            n_invalid = sum(1 for t in rec.turns if not t.is_valid)
            n_first_invalid = sum(1 for t in rec.turns if not t.first_try_valid)
            n_turns = len(rec.turns) or 1  # avoid division by zero

            writer.writerow({
                "game_id": rec.game_id,
                "condition": rec.condition,
                "experiment": rec.experiment,
                "input_mode": rec.input_mode,
                "total_turns": rec.total_turns,
                "total_llm_calls": rec.total_llm_calls,
                "total_tokens": rec.total_tokens,
                "final_status": rec.final_status,
                "n_forfeits": n_invalid,
                "n_first_try_invalid": n_first_invalid,
                "fir": round(n_invalid / n_turns, 4),
                "ftir": round(n_first_invalid / n_turns, 4),
            })

    logger.info("Wrote summary CSV with %d records to %s", len(records), filepath)
