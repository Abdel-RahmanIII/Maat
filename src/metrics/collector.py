"""Real-time per-turn metrics collector.

The :class:`MetricsCollector` is instantiated per-game (or per-position
batch in Experiment 1) and wraps each ``graph.invoke()`` call with
:meth:`start_turn` / :meth:`end_turn` hooks.  It handles wall-clock
timing, game-phase inference, and produces structured
:class:`~src.metrics.definitions.TurnRecord` and
:class:`~src.metrics.definitions.GameRecord` objects.
"""

from __future__ import annotations

import time
from typing import Any

import chess

from src.metrics.definitions import GameRecord, TurnRecord
from src.state import TurnState


# ── Game phase inference ─────────────────────────────────────────────────


def _count_non_pawn_material(board: chess.Board) -> int:
    """Return total non-pawn material points on the board.

    Uses standard piece values:
    Knight/Bishop = 3, Rook = 5, Queen = 9.
    Kings and pawns are excluded.
    """

    values = {
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
    }

    total = 0
    for piece_type, value in values.items():
        total += len(board.pieces(piece_type, chess.WHITE)) * value
        total += len(board.pieces(piece_type, chess.BLACK)) * value
    return total


def infer_game_phase(fen: str, move_number: int) -> str:
    """Classify the current position into a game phase.

    Rules (from Section 4.1 of the research plan):

    * **Opening**: ``move_number`` ≤ 12
    * **Endgame**: ``move_number`` ≥ 40 **or** total non-pawn material ≤ 10
    * **Middlegame**: everything else

    Parameters
    ----------
    fen:
        Board position in FEN notation.
    move_number:
        Full-move number (1-indexed).

    Returns
    -------
    str
        One of ``"opening"``, ``"middlegame"``, ``"endgame"``.
    """

    if move_number <= 12:
        return "opening"

    try:
        board = chess.Board(fen)
    except ValueError:
        # Fall back to move-number heuristic if FEN is invalid
        return "endgame" if move_number >= 40 else "middlegame"

    material = _count_non_pawn_material(board)

    if move_number >= 40 or material <= 10:
        return "endgame"

    return "middlegame"


# ── Turn record builder ──────────────────────────────────────────────────


def _build_turn_record(
    state: TurnState,
    wall_clock_ms: float,
    game_phase: str,
) -> TurnRecord:
    """Create a :class:`TurnRecord` from a completed ``TurnState``.

    Reads the last entry in ``state["turn_results"]`` if available,
    otherwise falls back to reading fields directly from the state.
    """

    # Prefer the snapshot dict if available (it's the canonical record
    # written by the accept/forfeit nodes).
    turn_results = state.get("turn_results", [])
    if turn_results:
        raw: dict[str, Any] = turn_results[-1]
    else:
        # Fallback: build from state fields directly
        raw = {
            "move_number": state.get("move_number", 0),
            "proposed_move": state.get("proposed_move", ""),
            "is_valid": state.get("is_valid", False),
            "first_try_valid": state.get("first_try_valid", False),
            "total_attempts": state.get("total_attempts", 0),
            "error_types": list(state.get("error_types", [])),
            "retry_count": state.get("retry_count", 0),
            "llm_calls_this_turn": state.get("llm_calls_this_turn", 0),
            "tokens_this_turn": state.get("tokens_this_turn", 0),
            "prompt_token_count": state.get("prompt_token_count", 0),
            "tool_calls": list(state.get("tool_calls", [])),
            "critic_verdict": state.get("critic_verdict"),
            "ground_truth_verdict": state.get("ground_truth_verdict"),
            "generation_strategy": state.get("generation_strategy", "generator_only"),
            "strategic_plan": state.get("strategic_plan", ""),
            "routed_phase": state.get("routed_phase", ""),
            "feedback_history": list(state.get("feedback_history", [])),
            "board_fen": state.get("board_fen", ""),
        }

    # Override timing / phase with collector-computed values
    raw["wall_clock_ms"] = wall_clock_ms
    raw["game_phase"] = game_phase

    # Ensure board_fen is present
    if "board_fen" not in raw or not raw["board_fen"]:
        raw["board_fen"] = state.get("board_fen", "")

    return TurnRecord.model_validate(raw)


# ── Metrics collector ────────────────────────────────────────────────────


class MetricsCollector:
    """Per-game real-time metrics collector.

    Wraps each turn's ``graph.invoke()`` call with :meth:`start_turn` /
    :meth:`end_turn` hooks.  Accumulates :class:`TurnRecord` objects and
    produces a final :class:`GameRecord` via :meth:`finalize_game`.

    Usage::

        collector = MetricsCollector(
            game_id="exp1_pos_042",
            condition="D",
            experiment=1,
        )

        collector.start_turn()
        final_state = run_condition_d(fen=puzzle_fen, ...)
        record = collector.end_turn(final_state)

        game_record = collector.finalize_game(final_status="completed")
    """

    def __init__(
        self,
        game_id: str,
        condition: str,
        experiment: int,
        input_mode: str = "fen",
        starting_fen: str = "",
    ) -> None:
        self.game_id = game_id
        self.condition = condition
        self.experiment = experiment
        self.input_mode = input_mode
        self.starting_fen = starting_fen

        self._turn_records: list[TurnRecord] = []
        self._turn_start_ns: int | None = None

    # ── Turn lifecycle ───────────────────────────────────────────────

    def start_turn(self) -> None:
        """Record the turn start timestamp.  Call before ``graph.invoke()``."""
        self._turn_start_ns = time.perf_counter_ns()

    def end_turn(self, state: TurnState) -> TurnRecord:
        """Finalize the current turn.

        Computes ``wall_clock_ms``, infers ``game_phase``, and creates a
        :class:`TurnRecord` from the final graph state.

        Parameters
        ----------
        state:
            The ``TurnState`` returned by ``graph.invoke()``.

        Returns
        -------
        TurnRecord
            The structured turn record that was appended internally.
        """

        # Compute wall-clock time
        if self._turn_start_ns is not None:
            elapsed_ns = time.perf_counter_ns() - self._turn_start_ns
            wall_clock_ms = elapsed_ns / 1_000_000
        else:
            wall_clock_ms = 0.0

        # Infer game phase
        fen = state.get("board_fen", "")
        move_number = state.get("move_number", 1)
        game_phase = infer_game_phase(fen, move_number)

        # Build the turn record
        record = _build_turn_record(state, wall_clock_ms, game_phase)
        self._turn_records.append(record)

        # Reset for next turn
        self._turn_start_ns = None

        return record

    # ── Game finalization ────────────────────────────────────────────

    def finalize_game(
        self,
        final_status: str,
        starting_fen: str | None = None,
    ) -> GameRecord:
        """Create a :class:`GameRecord` from all accumulated turn records.

        Parameters
        ----------
        final_status:
            Terminal game status (``"checkmate"``, ``"forfeit"``, etc.).
        starting_fen:
            Override the starting FEN (useful if not set at init).

        Returns
        -------
        GameRecord
        """

        fen = starting_fen if starting_fen is not None else self.starting_fen

        return GameRecord(
            game_id=self.game_id,
            condition=self.condition,
            experiment=self.experiment,
            input_mode=self.input_mode,
            turns=list(self._turn_records),
            final_status=final_status,
            total_turns=len(self._turn_records),
            total_llm_calls=sum(t.llm_calls_this_turn for t in self._turn_records),
            total_tokens=sum(t.tokens_this_turn for t in self._turn_records),
            starting_fen=fen,
        )

    # ── Accessors ────────────────────────────────────────────────────

    @property
    def turn_records(self) -> list[TurnRecord]:
        """Return a copy of all accumulated turn records."""
        return list(self._turn_records)

    @property
    def current_turn_count(self) -> int:
        """Number of turns recorded so far."""
        return len(self._turn_records)
