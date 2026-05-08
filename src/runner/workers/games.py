"""Game worker (Experiments 2 & 3).

A game worker plays exactly one full game:

- White is the LLM condition dispatcher.
- Black is Stockfish.

The worker supports turn-level checkpointing for resumability:

- If `.game_state/{game_id}.json` exists and status is `ongoing`, resume from it.
- After every White (LLM) turn, persist the mid-game state.
- On completion, write the final JSONL record and delete the state file.

This module is part of the runner's *data plane*.
"""

from __future__ import annotations

import logging
import threading
import traceback
from pathlib import Path
from typing import Any, Callable

import chess

from src.config import ModelConfig, config_for_condition
from src.context import ConversationContext
from src.engine.condition_dispatch import dispatch_turn_with_backoff
from src.engine.result_store import append_checkpoint, append_game_record
from src.engine.stockfish_wrapper import StockfishWrapper
from src.metrics.collector import MetricsCollector
from src.metrics.definitions import GameRecord, TurnRecord
from src.runner.persistence.checkpoint import delete_game_state, load_game_state, save_game_state
from src.state import InputMode

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[dict[str, Any]], None]


def run_game_worker(
    game_id: str,
    starting_fen: str,
    condition: str,
    experiment: int,
    output_dir: Any,
    *,
    input_mode: InputMode = "fen",
    model_config: ModelConfig | None = None,
    generation_strategy: str = "generator_only",
    max_half_moves: int = 150,
    stockfish_elo: int = 1000,
    stockfish_path: str | None = None,
    max_api_retries: int = 5,
    backoff_base: float = 2.0,
    backoff_max: float = 60.0,
    pause_event: threading.Event | None = None,
    stop_event: threading.Event | None = None,
    on_progress: ProgressCallback | None = None,
) -> GameRecord | None:
    """Play one full game with turn-level checkpointing.

    Returns the `GameRecord` on success, or None if stopped/failed.
    """

    output_path = Path(output_dir)
    cond_cfg = config_for_condition(condition)

    # ── Resume from checkpoint if available ──
    saved = load_game_state(output_path, game_id)

    if saved and saved.get("game_status") == "ongoing":
        board = chess.Board(saved["starting_fen"])
        for uci in saved["move_stack_uci"]:
            board.push(chess.Move.from_uci(uci))
        half_moves_played = saved["half_moves_played"]
        existing_turn_records = [TurnRecord.model_validate(t) for t in saved["turn_records"]]
        logger.info("Resuming game %s from half-move %d", game_id, half_moves_played)
    else:
        board = chess.Board(starting_fen)
        half_moves_played = 0
        existing_turn_records = []

    collector = MetricsCollector(
        game_id=game_id,
        condition=condition,
        experiment=experiment,
        input_mode=input_mode,
        starting_fen=starting_fen,
    )
    # Inject existing turn records
    collector._turn_records = list(existing_turn_records)

    game_status = "ongoing"

    # One conversation context per game
    context = ConversationContext()

    # Per-game Stockfish instance (not thread-safe)
    sf = StockfishWrapper(engine_path=stockfish_path, elo=stockfish_elo)

    try:
        sf.start()

        _emit(on_progress, {
            "type": "worker_status",
            "game_id": game_id,
            "condition": condition,
            "experiment": experiment,
            "status": "running",
            "detail": f"Turn {half_moves_played}",
        })

        while game_status == "ongoing":
            # ── Check stop/pause ──
            if stop_event and stop_event.is_set():
                _save_mid_game(
                    output_path, game_id, condition, experiment,
                    starting_fen, board, half_moves_played,
                    collector.turn_records, game_status, input_mode,
                    generation_strategy,
                )
                _emit(on_progress, {
                    "type": "worker_status",
                    "game_id": game_id,
                    "condition": condition,
                    "experiment": experiment,
                    "status": "stopped",
                    "detail": f"Saved at half-move {half_moves_played}",
                })
                return None

            if pause_event and not pause_event.is_set():
                _save_mid_game(
                    output_path, game_id, condition, experiment,
                    starting_fen, board, half_moves_played,
                    collector.turn_records, game_status, input_mode,
                    generation_strategy,
                )
                _emit(on_progress, {
                    "type": "worker_status",
                    "game_id": game_id,
                    "condition": condition,
                    "experiment": experiment,
                    "status": "paused",
                    "detail": f"Paused at half-move {half_moves_played}",
                })
                pause_event.wait()

            # ── Half-move cap ──
            if half_moves_played >= max_half_moves:
                game_status = "max_moves"
                break

            side_to_move = board.turn

            if side_to_move == chess.WHITE:
                # LLM turn
                game_status = _llm_turn(
                    board=board,
                    condition=condition,
                    cond_cfg=cond_cfg,
                    collector=collector,
                    game_id=game_id,
                    input_mode=input_mode,
                    generation_strategy=generation_strategy,
                    model_config=model_config,
                    max_api_retries=max_api_retries,
                    backoff_base=backoff_base,
                    backoff_max=backoff_max,
                    experiment=experiment,
                    context=context,
                )
            else:
                # Stockfish turn
                sf_move_uci = sf.choose_move(board.fen())
                move = chess.Move.from_uci(sf_move_uci)
                board.push(move)

            half_moves_played += 1

            # Natural termination
            if game_status == "ongoing":
                game_status = _check_termination(board)

            # Turn-level checkpoint (save every LLM turn)
            if side_to_move == chess.WHITE:
                _save_mid_game(
                    output_path, game_id, condition, experiment,
                    starting_fen, board, half_moves_played,
                    collector.turn_records, game_status, input_mode,
                    generation_strategy,
                )

            # Capture last turn info for the event
            last_move = ""
            last_valid = False
            last_response = ""
            if side_to_move == chess.WHITE and collector.turn_records:
                last_tr = collector.turn_records[-1]
                last_move = last_tr.proposed_move
                last_valid = last_tr.is_valid
                last_response = last_tr.raw_llm_response

            _emit(on_progress, {
                "type": "game_turn",
                "game_id": game_id,
                "condition": condition,
                "experiment": experiment,
                "half_moves": half_moves_played,
                "game_status": game_status,
                "proposed_move": last_move,
                "is_valid": last_valid,
                "raw_llm_response": last_response,
            })

        # ── Finalize ──
        record = collector.finalize_game(final_status=game_status, starting_fen=starting_fen)

        results_path = output_path / f"exp{experiment}_{condition}_results.jsonl"
        append_game_record(record, results_path)
        append_checkpoint(game_id, output_path / ".checkpoint")
        delete_game_state(output_path, game_id)

        _emit(on_progress, {
            "type": "game_complete",
            "game_id": game_id,
            "condition": condition,
            "experiment": experiment,
            "status": game_status,
            "total_turns": record.total_turns,
        })

        return record

    except Exception as exc:
        logger.exception("Game worker error on %s", game_id)
        _save_mid_game(
            output_path, game_id, condition, experiment,
            starting_fen, board, half_moves_played,
            collector.turn_records, "ongoing", input_mode,
            generation_strategy,
        )
        _emit(on_progress, {
            "type": "worker_error",
            "game_id": game_id,
            "condition": condition,
            "experiment": experiment,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })
        return None
    finally:
        sf.close()


# ── Internal helpers ─────────────────────────────────────────────────────


def _llm_turn(
    *,
    board: chess.Board,
    condition: str,
    cond_cfg: Any,
    collector: MetricsCollector,
    game_id: str,
    input_mode: InputMode,
    generation_strategy: str,
    model_config: ModelConfig | None,
    max_api_retries: int,
    backoff_base: float,
    backoff_max: float,
    experiment: int,
    context: ConversationContext | None = None,
) -> str:
    """Execute one LLM move. Returns the updated game status."""

    fen = board.fen()
    move_history = [m.uci() for m in board.move_stack]
    move_number = board.fullmove_number

    collector.start_turn()

    state = dispatch_turn_with_backoff(
        condition,
        fen=fen,
        move_history=move_history,
        move_number=move_number,
        game_id=game_id,
        input_mode=input_mode,
        generation_strategy=generation_strategy,
        model_config=model_config,
        max_react_steps=cond_cfg.max_react_steps,
        max_api_retries=max_api_retries,
        base_delay=backoff_base,
        max_delay=backoff_max,
        context=context,
    )

    collector.end_turn(state)

    if state.get("game_status") == "forfeit" or not state.get("is_valid"):
        return "forfeit"

    proposed = state.get("proposed_move", "")
    try:
        move = chess.Move.from_uci(proposed)
        if move in board.legal_moves:
            board.push(move)
        else:
            return "forfeit"
    except ValueError:
        return "forfeit"

    return "ongoing"


def _check_termination(board: chess.Board) -> str:
    """Check if the game has ended naturally."""

    if board.is_checkmate():
        return "checkmate"
    if board.is_stalemate():
        return "stalemate"
    if board.is_insufficient_material():
        return "draw"
    if board.can_claim_fifty_moves():
        return "draw"
    if board.can_claim_threefold_repetition():
        return "draw"
    return "ongoing"


def _save_mid_game(
    output_dir: Any,
    game_id: str,
    condition: str,
    experiment: int,
    starting_fen: str,
    board: chess.Board,
    half_moves_played: int,
    turn_records: list[TurnRecord],
    game_status: str,
    input_mode: str,
    generation_strategy: str,
) -> None:
    """Save current game state for resume."""

    save_game_state(
        Path(output_dir),
        game_id=game_id,
        condition=condition,
        experiment=experiment,
        starting_fen=starting_fen,
        board_fen=board.fen(),
        move_stack_uci=[m.uci() for m in board.move_stack],
        half_moves_played=half_moves_played,
        turn_records=[t.model_dump() for t in turn_records],
        game_status=game_status,
        input_mode=input_mode,
        generation_strategy=generation_strategy,
    )


def _emit(callback: ProgressCallback | None, data: dict[str, Any]) -> None:
    """Fire a progress callback, swallowing exceptions."""

    if callback:
        try:
            callback(data)
        except Exception:
            pass
