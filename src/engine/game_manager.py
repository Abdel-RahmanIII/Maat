"""GameManager — Experiment 2 & 3 orchestrator.

Runs full chess games between the LLM (White) and Stockfish (Black),
collecting per-turn metrics and producing :class:`GameRecord` objects.

* **Experiment 2** (``input_mode="fen"``): Full FEN + ASCII board sent
  every turn.
* **Experiment 3** (``input_mode="history"``): Move history only (FEN
  withheld from the LLM prompt, still used internally for validation).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import chess

from src.config import ModelConfig, config_for_condition
from src.engine.condition_dispatch import dispatch_turn_with_backoff
from src.engine.result_store import (
    append_checkpoint,
    append_game_record,
    load_checkpoint,
    write_summary_csv,
)
from src.engine.stockfish_wrapper import StockfishWrapper
from src.metrics.collector import MetricsCollector
from src.metrics.definitions import GameRecord
from src.state import InputMode

logger = logging.getLogger(__name__)


# ── Starting-position loading ────────────────────────────────────────────


def load_starting_positions(filepath: Path | str) -> list[str]:
    """Load starting FEN strings from a text file (one per line).

    Blank lines and lines starting with ``#`` are ignored.
    """

    path = Path(filepath)
    fens: list[str] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                fens.append(stripped)
    return fens


# ── GameManager ──────────────────────────────────────────────────────────


class GameManager:
    """Orchestrates full-game experiments (Exp 2 and 3).

    For each ``(starting_position, condition)`` pair the manager plays a
    complete game — LLM as White, Stockfish as Black — up to
    ``max_half_moves`` half-moves.

    Usage::

        mgr = GameManager(
            starting_positions=load_starting_positions("starting_positions.txt"),
            conditions=["A", "B", "D"],
            experiment=2,
            output_dir=Path("results/exp2"),
        )
        records = mgr.run_all()
    """

    def __init__(
        self,
        starting_positions: list[str],
        conditions: list[str],
        experiment: int,
        output_dir: Path | str,
        *,
        stockfish_elo: int = 1000,
        stockfish_path: str | None = None,
        max_half_moves: int = 150,
        model_config: ModelConfig | None = None,
        generation_strategy: str = "generator_only",
        delay_seconds: float = 0.0,
        max_api_retries: int = 5,
        backoff_base: float = 2.0,
        backoff_max: float = 60.0,
    ) -> None:
        if experiment not in (2, 3):
            raise ValueError(f"experiment must be 2 or 3, got {experiment}")

        self.starting_positions = starting_positions
        self.conditions = [c.upper() for c in conditions]
        self.experiment = experiment
        self.output_dir = Path(output_dir)
        self.stockfish_elo = stockfish_elo
        self.stockfish_path = stockfish_path
        self.max_half_moves = max_half_moves
        self.model_config = model_config
        self.generation_strategy = generation_strategy
        self.delay_seconds = delay_seconds
        self.max_api_retries = max_api_retries
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max

        self.input_mode: InputMode = "fen" if experiment == 2 else "history"

        # Derived paths
        self._checkpoint_path = self.output_dir / ".checkpoint"

    # ── Public API ───────────────────────────────────────────────────

    def run_all(self) -> list[GameRecord]:
        """Run all games across all conditions.

        Stockfish is started once and shared across all games for
        efficiency.

        Returns all :class:`GameRecord` instances produced.
        """

        self.output_dir.mkdir(parents=True, exist_ok=True)
        completed = load_checkpoint(self._checkpoint_path)
        all_records: list[GameRecord] = []

        sf = StockfishWrapper(
            engine_path=self.stockfish_path,
            elo=self.stockfish_elo,
        )

        with sf:
            for condition in self.conditions:
                cond_records = self._run_condition(condition, sf, completed)
                all_records.extend(cond_records)

        # Write aggregate summary
        if all_records:
            write_summary_csv(
                all_records,
                self.output_dir / f"exp{self.experiment}_summary.csv",
            )

        logger.info(
            "[Exp%d] Finished: %d games across %d conditions",
            self.experiment,
            len(all_records),
            len(self.conditions),
        )
        return all_records

    def run_single_game(
        self,
        starting_fen: str,
        condition: str,
        game_index: int = 0,
        stockfish: StockfishWrapper | None = None,
    ) -> GameRecord:
        """Play one game.  Optionally accepts a pre-started Stockfish."""

        owns_sf = stockfish is None
        sf = stockfish or StockfishWrapper(
            engine_path=self.stockfish_path,
            elo=self.stockfish_elo,
        )

        try:
            if owns_sf:
                sf.start()
            return self._play_game(
                starting_fen=starting_fen,
                condition=condition.upper(),
                game_index=game_index,
                stockfish=sf,
            )
        finally:
            if owns_sf:
                sf.close()

    # ── Internal helpers ─────────────────────────────────────────────

    def _run_condition(
        self,
        condition: str,
        stockfish: StockfishWrapper,
        completed: set[str],
    ) -> list[GameRecord]:
        """Play all games for *condition*, skipping checkpointed ones."""

        results_path = (
            self.output_dir / f"exp{self.experiment}_{condition}_results.jsonl"
        )
        records: list[GameRecord] = []

        for idx, fen in enumerate(self.starting_positions):
            game_id = f"exp{self.experiment}_{condition}_game{idx:03d}"

            if game_id in completed:
                logger.debug("[Exp%d] Skipping %s (checkpointed)",
                             self.experiment, game_id)
                continue

            try:
                record = self._play_game(
                    starting_fen=fen,
                    condition=condition,
                    game_index=idx,
                    stockfish=stockfish,
                    game_id=game_id,
                )
            except Exception:
                logger.exception(
                    "[Exp%d] Fatal error on %s — skipping",
                    self.experiment,
                    game_id,
                )
                continue

            # Persist
            append_game_record(record, results_path)
            append_checkpoint(game_id, self._checkpoint_path)
            completed.add(game_id)
            records.append(record)

            # Progress log
            logger.info(
                "[Exp%d] %s | Game %d/%d | %d turns | %s",
                self.experiment,
                condition,
                idx + 1,
                len(self.starting_positions),
                record.total_turns,
                record.final_status,
            )

            # Rate-limit delay
            if self.delay_seconds > 0:
                time.sleep(self.delay_seconds)

        return records

    def _play_game(
        self,
        *,
        starting_fen: str,
        condition: str,
        game_index: int,
        stockfish: StockfishWrapper,
        game_id: str | None = None,
    ) -> GameRecord:
        """Play one full game: LLM (White) vs Stockfish (Black)."""

        gid = game_id or f"exp{self.experiment}_{condition}_game{game_index:03d}"
        cond_cfg = config_for_condition(condition)

        board = chess.Board(starting_fen)

        collector = MetricsCollector(
            game_id=gid,
            condition=condition,
            experiment=self.experiment,
            input_mode=self.input_mode,
            starting_fen=starting_fen,
        )

        game_status = "ongoing"
        half_moves_played = 0

        while game_status == "ongoing":
            # ── Check half-move cap ──
            if half_moves_played >= self.max_half_moves:
                game_status = "max_moves"
                break

            side_to_move = board.turn  # chess.WHITE or chess.BLACK

            if side_to_move == chess.WHITE:
                # ── LLM's turn ──
                game_status = self._llm_turn(
                    board=board,
                    condition=condition,
                    cond_cfg=cond_cfg,
                    collector=collector,
                    game_id=gid,
                )
            else:
                # ── Stockfish's turn ──
                self._stockfish_turn(board, stockfish)

            half_moves_played += 1

            # ── Check board termination ──
            if game_status == "ongoing":
                game_status = self._check_termination(board)

        return collector.finalize_game(
            final_status=game_status,
            starting_fen=starting_fen,
        )

    def _llm_turn(
        self,
        *,
        board: chess.Board,
        condition: str,
        cond_cfg: Any,
        collector: MetricsCollector,
        game_id: str,
    ) -> str:
        """Execute one LLM turn.  Returns the updated game status."""

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
            input_mode=self.input_mode,
            generation_strategy=self.generation_strategy,
            model_config=self.model_config,
            max_react_steps=cond_cfg.max_react_steps,
            max_api_retries=self.max_api_retries,
            base_delay=self.backoff_base,
            max_delay=self.backoff_max,
        )

        collector.end_turn(state)

        # Process result
        if state.get("game_status") == "forfeit" or not state.get("is_valid"):
            return "forfeit"

        # Apply the valid move to the board
        proposed = state.get("proposed_move", "")
        try:
            move = chess.Move.from_uci(proposed)
            if move in board.legal_moves:
                board.push(move)
            else:
                # Should not happen if validation passed, but be safe
                logger.error(
                    "[Exp%d] Move %s validated but not in legal_moves for %s",
                    self.experiment,
                    proposed,
                    game_id,
                )
                return "forfeit"
        except ValueError:
            logger.error(
                "[Exp%d] Failed to parse validated move %r for %s",
                self.experiment,
                proposed,
                game_id,
            )
            return "forfeit"

        return "ongoing"

    def _stockfish_turn(
        self,
        board: chess.Board,
        stockfish: StockfishWrapper,
    ) -> None:
        """Execute one Stockfish turn and push the move to the board."""

        sf_move_uci = stockfish.choose_move(board.fen())
        try:
            move = chess.Move.from_uci(sf_move_uci)
            board.push(move)
        except (ValueError, chess.IllegalMoveError) as exc:
            logger.error("Stockfish returned invalid move %r: %s", sf_move_uci, exc)
            # Push anyway — Stockfish shouldn't return illegal moves
            raise RuntimeError(
                f"Stockfish produced illegal move: {sf_move_uci}"
            ) from exc

    @staticmethod
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
