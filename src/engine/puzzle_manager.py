"""PuzzleManager — Experiment 1 orchestrator.

Evaluates each puzzle position once per condition (A–F), collecting
metrics via :class:`~src.metrics.collector.MetricsCollector` and
persisting results as JSONL.  Supports checkpoint-based resumption
and configurable rate limiting with exponential backoff.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from src.config import ModelConfig, config_for_condition
from src.engine.condition_dispatch import dispatch_turn_with_backoff
from src.engine.result_store import (
    append_checkpoint,
    append_game_record,
    load_checkpoint,
    load_game_records,
    write_summary_csv,
)
from src.metrics.collector import MetricsCollector
from src.metrics.definitions import GameRecord

logger = logging.getLogger(__name__)


# ── Puzzle loading ───────────────────────────────────────────────────────


def load_puzzle_inputs(jsonl_path: Path | str) -> list[dict[str, Any]]:
    """Load experiment input dicts from a JSONL file.

    Each line must be a JSON object with at least ``puzzle_id`` and
    ``fen`` keys (as produced by
    :func:`~src.data.puzzle_sampler.write_experiment_inputs_jsonl`).
    """

    path = Path(jsonl_path)
    inputs: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                inputs.append(json.loads(stripped))
    return inputs


# ── PuzzleManager ────────────────────────────────────────────────────────


class PuzzleManager:
    """Orchestrates Experiment 1 — isolated-position evaluation.

    For every ``(puzzle, condition)`` pair the manager:

    1. Creates a :class:`MetricsCollector`.
    2. Dispatches one turn to the appropriate condition runner.
    3. Records the result as a :class:`GameRecord` (single-turn "game").
    4. Persists the record to JSONL and updates the checkpoint file.

    Usage::

        mgr = PuzzleManager(
            puzzles=load_puzzle_inputs("experiment_inputs.jsonl"),
            conditions=["A", "B", "C", "D", "E", "F"],
            output_dir=Path("results/exp1"),
        )
        records = mgr.run_all()
    """

    def __init__(
        self,
        puzzles: list[dict[str, Any]],
        conditions: list[str],
        output_dir: Path | str,
        *,
        model_config: ModelConfig | None = None,
        generation_strategy: str = "generator_only",
        delay_seconds: float = 0.0,
        max_api_retries: int = 5,
        backoff_base: float = 2.0,
        backoff_max: float = 60.0,
    ) -> None:
        self.puzzles = puzzles
        self.conditions = [c.upper() for c in conditions]
        self.output_dir = Path(output_dir)
        self.model_config = model_config
        self.generation_strategy = generation_strategy
        self.delay_seconds = delay_seconds
        self.max_api_retries = max_api_retries
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max

        # Derived paths
        self._checkpoint_path = self.output_dir / ".checkpoint"

    # ── Public API ───────────────────────────────────────────────────

    def run_all(self) -> list[GameRecord]:
        """Evaluate every puzzle under every condition.

        Returns all :class:`GameRecord` instances produced.
        """

        self.output_dir.mkdir(parents=True, exist_ok=True)
        completed = load_checkpoint(self._checkpoint_path)
        all_records: list[GameRecord] = []

        total_pairs = len(self.puzzles) * len(self.conditions)
        done_count = 0

        for condition in self.conditions:
            cond_records = self._run_condition(condition, completed)
            all_records.extend(cond_records)
            done_count += len(cond_records)

        # Write aggregate summary
        if all_records:
            write_summary_csv(
                all_records,
                self.output_dir / "exp1_summary.csv",
            )

        logger.info(
            "[Exp1] Finished: %d records across %d conditions",
            len(all_records),
            len(self.conditions),
        )
        return all_records

    def run_condition(self, condition: str) -> list[GameRecord]:
        """Evaluate all puzzles under a single condition."""

        self.output_dir.mkdir(parents=True, exist_ok=True)
        completed = load_checkpoint(self._checkpoint_path)
        return self._run_condition(condition.upper(), completed)

    def run_single(
        self,
        puzzle: dict[str, Any],
        condition: str,
    ) -> GameRecord:
        """Evaluate a single puzzle under a single condition."""

        return self._evaluate_puzzle(puzzle, condition.upper())

    # ── Internal helpers ─────────────────────────────────────────────

    def _run_condition(
        self,
        condition: str,
        completed: set[str],
    ) -> list[GameRecord]:
        """Run all puzzles for *condition*, skipping checkpointed ones."""

        results_path = self.output_dir / f"exp1_{condition}_results.jsonl"
        records: list[GameRecord] = []

        for idx, puzzle in enumerate(self.puzzles):
            puzzle_id = puzzle.get("puzzle_id", f"pos_{idx}")
            game_id = f"exp1_{puzzle_id}_{condition}"

            if game_id in completed:
                logger.debug("[Exp1] Skipping %s (checkpointed)", game_id)
                continue

            try:
                record = self._evaluate_puzzle(puzzle, condition, game_id=game_id)
            except Exception:
                logger.exception(
                    "[Exp1] Fatal error on %s — skipping", game_id,
                )
                continue

            # Persist
            append_game_record(record, results_path)
            append_checkpoint(game_id, self._checkpoint_path)
            completed.add(game_id)
            records.append(record)

            # Progress log
            status = "✓" if record.final_status != "forfeit" else "✗"
            logger.info(
                "[Exp1] %s | Puzzle %d/%d | %s | %s | %.1fs",
                condition,
                idx + 1,
                len(self.puzzles),
                puzzle_id,
                status,
                sum(t.wall_clock_ms for t in record.turns) / 1000,
            )

            # Rate-limit delay
            if self.delay_seconds > 0:
                time.sleep(self.delay_seconds)

        return records

    def _evaluate_puzzle(
        self,
        puzzle: dict[str, Any],
        condition: str,
        game_id: str | None = None,
    ) -> GameRecord:
        """Run one turn for a puzzle-condition pair."""

        puzzle_id = puzzle.get("puzzle_id", "unknown")
        fen = puzzle["fen"]
        gid = game_id or f"exp1_{puzzle_id}_{condition}"

        cond_cfg = config_for_condition(condition)

        collector = MetricsCollector(
            game_id=gid,
            condition=condition,
            experiment=1,
            input_mode="fen",
            starting_fen=fen,
        )

        # ── Execute the turn ──
        collector.start_turn()

        state = dispatch_turn_with_backoff(
            condition,
            fen=fen,
            move_history=[],
            move_number=1,
            game_id=gid,
            input_mode="fen",
            generation_strategy=self.generation_strategy,
            model_config=self.model_config,
            max_react_steps=cond_cfg.max_react_steps,
            max_api_retries=self.max_api_retries,
            base_delay=self.backoff_base,
            max_delay=self.backoff_max,
        )

        collector.end_turn(state)

        # ── Finalize ──
        final_status = state.get("game_status", "ongoing")
        if final_status == "ongoing":
            final_status = "completed"

        return collector.finalize_game(
            final_status=final_status,
            starting_fen=fen,
        )
