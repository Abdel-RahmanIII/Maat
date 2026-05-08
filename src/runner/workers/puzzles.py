"""Puzzle worker (Experiment 1).

A puzzle worker evaluates exactly one position under one condition.

It:
- checks stop/pause signals,
- runs the condition dispatcher once,
- records metrics,
- appends a JSONL game record,
- updates the checkpoint file,
- emits progress events via a callback.

This module is part of the *data plane*.
"""

from __future__ import annotations

import logging
import threading
import traceback
from pathlib import Path
from typing import Any, Callable

from src.config import ModelConfig, config_for_condition
from src.context import ConversationContext
from src.engine.condition_dispatch import dispatch_turn_with_backoff
from src.engine.result_store import append_checkpoint, append_game_record
from src.metrics.collector import MetricsCollector
from src.metrics.definitions import GameRecord

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[dict[str, Any]], None]


def run_puzzle_worker(
    puzzle: dict[str, Any],
    condition: str,
    game_id: str,
    output_dir: Any,
    *,
    model_config: ModelConfig | None = None,
    generation_strategy: str = "generator_only",
    max_api_retries: int = 5,
    backoff_base: float = 2.0,
    backoff_max: float = 60.0,
    pause_event: threading.Event | None = None,
    stop_event: threading.Event | None = None,
    on_progress: ProgressCallback | None = None,
) -> GameRecord | None:
    """Evaluate a single puzzle under a condition.

    Returns the `GameRecord` on success, or None if stopped/failed.
    """

    # ── Check stop/pause ──
    if stop_event and stop_event.is_set():
        return None
    if pause_event:
        pause_event.wait()  # blocks if paused

    # Stop may be requested *while paused*; `Orchestrator.stop()` unpauses
    # workers to let them exit. Re-check here to avoid starting a new puzzle.
    if stop_event and stop_event.is_set():
        _emit(on_progress, {
            "type": "worker_status",
            "game_id": game_id,
            "condition": condition,
            "experiment": 1,
            "status": "stopped",
            "detail": "Stopped before puzzle start",
        })
        return None

    fen = puzzle["fen"]
    cond_cfg = config_for_condition(condition)

    collector = MetricsCollector(
        game_id=game_id,
        condition=condition,
        experiment=1,
        input_mode="fen",
        starting_fen=fen,
    )

    try:
        _emit(on_progress, {
            "type": "worker_status",
            "game_id": game_id,
            "condition": condition,
            "experiment": 1,
            "status": "running",
            "detail": f"Puzzle {puzzle.get('puzzle_id', '?')}",
        })

        collector.start_turn()

        # One conversation context per puzzle
        context = ConversationContext()

        state = dispatch_turn_with_backoff(
            condition,
            fen=fen,
            move_history=[],
            move_number=1,
            game_id=game_id,
            input_mode="fen",
            generation_strategy=generation_strategy,
            model_config=model_config,
            max_react_steps=cond_cfg.max_react_steps,
            max_api_retries=max_api_retries,
            base_delay=backoff_base,
            max_delay=backoff_max,
            context=context,
        )

        collector.end_turn(state)

        final_status = state.get("game_status", "ongoing")
        if final_status == "ongoing":
            final_status = "completed"

        record = collector.finalize_game(final_status=final_status, starting_fen=fen)

        output_path = Path(output_dir)
        results_path = output_path / f"exp1_{condition}_results.jsonl"

        append_game_record(record, results_path)
        append_checkpoint(game_id, output_path / ".checkpoint")

        _emit(on_progress, {
            "type": "puzzle_complete",
            "game_id": game_id,
            "condition": condition,
            "experiment": 1,
            "status": final_status,
            "is_valid": state.get("is_valid", False),
            "proposed_move": state.get("proposed_move", ""),
            "raw_llm_response": state.get("raw_llm_response", ""),
        })

        return record

    except Exception as exc:
        logger.exception("Puzzle worker error on %s", game_id)
        _emit(on_progress, {
            "type": "worker_error",
            "game_id": game_id,
            "condition": condition,
            "experiment": 1,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })
        return None


def _emit(callback: ProgressCallback | None, data: dict[str, Any]) -> None:
    """Fire a progress callback, swallowing exceptions."""

    if callback:
        try:
            callback(data)
        except Exception:
            pass
