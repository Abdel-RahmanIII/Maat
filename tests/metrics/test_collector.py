"""Tests for src.metrics.collector — MetricsCollector + game phase inference."""

from __future__ import annotations

import time

import pytest

from src.metrics.collector import MetricsCollector, infer_game_phase
from src.metrics.definitions import GameRecord, TurnRecord
from src.state import GameStatus, TurnState, create_initial_turn_state


# ── infer_game_phase ─────────────────────────────────────────────────────

STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
# Full material = 2*9 + 4*5 + 4*3 + 4*3 = 62 non-pawn points
ENDGAME_FEN = "8/8/4k3/8/8/4K3/8/8 w - - 0 40"
# Only kings — 0 non-pawn material


class TestInferGamePhase:
    def test_opening(self):
        assert infer_game_phase(STARTING_FEN, 1) == "opening"
        assert infer_game_phase(STARTING_FEN, 15) == "opening"

    def test_middlegame(self):
        assert infer_game_phase(STARTING_FEN, 16) == "middlegame"
        assert infer_game_phase(STARTING_FEN, 35) == "middlegame"

    def test_endgame_by_move_number(self):
        assert infer_game_phase(STARTING_FEN, 36) == "endgame"

    def test_endgame_by_material(self):
        # Move 20 but only kings on board — material ≤ 13
        assert infer_game_phase(ENDGAME_FEN, 20) == "endgame"

    def test_invalid_fen_fallback(self):
        assert infer_game_phase("NOT_A_FEN", 10) == "opening"
        assert infer_game_phase("NOT_A_FEN", 20) == "middlegame"
        assert infer_game_phase("NOT_A_FEN", 40) == "endgame"

    def test_material_threshold_boundary(self):
        # KR vs K = 5 non-pawn material ≤ 13 → endgame
        fen_kr = "8/8/4k3/8/8/4K3/8/R7 w - - 0 20"
        assert infer_game_phase(fen_kr, 20) == "endgame"

        # QR vs QR = 2*9 + 2*5 = 28 > 13 → middlegame
        fen_heavy = "r3k3/8/8/8/8/8/8/R3K2Q w Kq - 0 20"
        assert infer_game_phase(fen_heavy, 20) == "middlegame"


# ── MetricsCollector ─────────────────────────────────────────────────────


def _make_state(
    *,
    fen: str = STARTING_FEN,
    move_number: int = 1,
    is_valid: bool = True,
    first_try_valid: bool = True,
    total_attempts: int = 1,
    llm_calls: int = 1,
    tokens: int = 200,
    error_types: list[str] | None = None,
    game_status: GameStatus = "ongoing",
    condition: str = "D",
) -> TurnState:
    """Create a mock TurnState suitable for collector testing."""
    state = create_initial_turn_state(
        board_fen=fen,
        game_id="test_game",
        condition=condition,
        move_number=move_number,
    )
    state["is_valid"] = is_valid
    state["first_try_valid"] = first_try_valid
    state["total_attempts"] = total_attempts
    state["llm_calls_this_turn"] = llm_calls
    state["tokens_this_turn"] = tokens
    state["proposed_move"] = "e2e4"
    state["error_types"] = error_types or []
    state["game_status"] = game_status

    # Simulate accept node appending a turn result
    from src.graph.base_graph import snapshot_turn_result
    state["turn_results"] = [snapshot_turn_result(state)]

    return state


class TestMetricsCollector:
    def test_start_end_turn_produces_record(self):
        collector = MetricsCollector(
            game_id="test_001",
            condition="D",
            experiment=1,
        )

        state = _make_state()
        collector.start_turn()
        record = collector.end_turn(state)

        assert isinstance(record, TurnRecord)
        assert record.move_number == 1
        assert record.is_valid is True
        assert record.game_phase == "opening"

    def test_wall_clock_ms_positive(self):
        collector = MetricsCollector(
            game_id="test_002",
            condition="B",
            experiment=1,
        )

        state = _make_state()
        collector.start_turn()
        # Tiny sleep to ensure measurable time
        time.sleep(0.005)
        record = collector.end_turn(state)

        assert record.wall_clock_ms > 0.0

    def test_wall_clock_ms_zero_without_start(self):
        collector = MetricsCollector(
            game_id="test_003",
            condition="A",
            experiment=1,
        )

        state = _make_state()
        # No start_turn() call
        record = collector.end_turn(state)
        assert record.wall_clock_ms == 0.0

    def test_multiple_turns(self):
        collector = MetricsCollector(
            game_id="test_004",
            condition="D",
            experiment=2,
        )

        for i in range(5):
            state = _make_state(move_number=i + 1)
            collector.start_turn()
            collector.end_turn(state)

        assert collector.current_turn_count == 5
        assert len(collector.turn_records) == 5

    def test_finalize_game(self):
        collector = MetricsCollector(
            game_id="game_42",
            condition="D",
            experiment=2,
            starting_fen=STARTING_FEN,
        )

        for i in range(3):
            state = _make_state(
                move_number=i + 1,
                llm_calls=2,
                tokens=300,
            )
            collector.start_turn()
            collector.end_turn(state)

        game_record = collector.finalize_game(final_status="checkmate")

        assert isinstance(game_record, GameRecord)
        assert game_record.game_id == "game_42"
        assert game_record.condition == "D"
        assert game_record.experiment == 2
        assert game_record.total_turns == 3
        assert game_record.total_llm_calls == 6  # 2 * 3
        assert game_record.total_tokens == 900   # 300 * 3
        assert game_record.final_status == "checkmate"
        assert game_record.starting_fen == STARTING_FEN

    def test_finalize_game_override_fen(self):
        collector = MetricsCollector(
            game_id="test",
            condition="A",
            experiment=1,
        )
        state = _make_state()
        collector.start_turn()
        collector.end_turn(state)

        game = collector.finalize_game(
            final_status="forfeit",
            starting_fen="custom_fen",
        )
        assert game.starting_fen == "custom_fen"

    def test_game_phase_inference_in_record(self):
        collector = MetricsCollector(
            game_id="phase_test",
            condition="D",
            experiment=2,
        )

        # Opening
        state = _make_state(move_number=5)
        collector.start_turn()
        r1 = collector.end_turn(state)
        assert r1.game_phase == "opening"

        # Middlegame
        state = _make_state(move_number=25)
        collector.start_turn()
        r2 = collector.end_turn(state)
        assert r2.game_phase == "middlegame"

        # Endgame by material
        state = _make_state(fen=ENDGAME_FEN, move_number=20)
        collector.start_turn()
        r3 = collector.end_turn(state)
        assert r3.game_phase == "endgame"

    def test_error_types_preserved(self):
        collector = MetricsCollector(
            game_id="err",
            condition="D",
            experiment=1,
        )

        state = _make_state(
            is_valid=False,
            first_try_valid=False,
            error_types=["INVALID_PIECE", "ILLEGAL_DESTINATION"],
        )
        collector.start_turn()
        record = collector.end_turn(state)

        assert "INVALID_PIECE" in record.error_types
        assert "ILLEGAL_DESTINATION" in record.error_types
