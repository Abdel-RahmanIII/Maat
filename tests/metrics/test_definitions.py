"""Tests for src.metrics.definitions — Pydantic model validation / serialization."""

from __future__ import annotations

import json

import pytest

from src.metrics.definitions import (
    ConditionMetrics,
    CriticAccuracy,
    DescriptiveStats,
    FSTEntry,
    GameRecord,
    LegalityDegradationBin,
    PhaseStratifiedFIR,
    QuartileErrorDist,
    TurnRecord,
)


# ── TurnRecord ───────────────────────────────────────────────────────────


class TestTurnRecord:
    def test_minimal_creation(self):
        """TurnRecord can be created with just move_number."""
        tr = TurnRecord(move_number=1)
        assert tr.move_number == 1
        assert tr.is_valid is False
        assert tr.error_types == []
        assert tr.wall_clock_ms == 0.0

    def test_full_creation(self):
        tr = TurnRecord(
            move_number=5,
            proposed_move="e2e4",
            is_valid=True,
            first_try_valid=True,
            total_attempts=1,
            error_types=[],
            retry_count=0,
            llm_calls_this_turn=2,
            tokens_this_turn=500,
            prompt_token_count=300,
            wall_clock_ms=123.456,
            game_phase="opening",
            board_fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        )
        assert tr.proposed_move == "e2e4"
        assert tr.llm_calls_this_turn == 2
        assert tr.game_phase == "opening"

    def test_json_round_trip(self):
        tr = TurnRecord(
            move_number=3,
            proposed_move="d7d5",
            is_valid=True,
            error_types=["INVALID_PIECE", "ILLEGAL_DESTINATION"],
            tool_calls=[{"name": "validate_move", "result": True}],
        )
        json_str = tr.model_dump_json()
        restored = TurnRecord.model_validate_json(json_str)
        assert restored == tr

    def test_from_dict(self):
        """model_validate can ingest snapshot_turn_result-style dicts."""
        raw = {
            "move_number": 7,
            "proposed_move": "g1f3",
            "is_valid": True,
            "first_try_valid": True,
            "total_attempts": 1,
            "error_types": [],
            "retry_count": 0,
            "llm_calls_this_turn": 1,
            "tokens_this_turn": 200,
            "prompt_token_count": 150,
            "tool_calls": [],
            "critic_verdict": None,
            "ground_truth_verdict": None,
            "feedback_history": [],
            "wall_clock_ms": 50.0,
            "game_phase": "opening",
            "board_fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
        }
        tr = TurnRecord.model_validate(raw)
        assert tr.move_number == 7
        assert tr.board_fen.startswith("rnbqkbnr")

    def test_optional_critic_fields(self):
        tr = TurnRecord(
            move_number=1,
            critic_verdict=True,
            ground_truth_verdict=False,
        )
        assert tr.critic_verdict is True
        assert tr.ground_truth_verdict is False


# ── GameRecord ───────────────────────────────────────────────────────────


class TestGameRecord:
    def test_creation_with_turns(self):
        turns = [
            TurnRecord(move_number=1, is_valid=True),
            TurnRecord(move_number=2, is_valid=False),
        ]
        gr = GameRecord(
            game_id="test_001",
            condition="D",
            experiment=1,
            turns=turns,
            final_status="forfeit",
            total_turns=2,
            total_llm_calls=4,
            total_tokens=1000,
        )
        assert gr.game_id == "test_001"
        assert len(gr.turns) == 2
        assert gr.final_status == "forfeit"

    def test_json_round_trip(self):
        gr = GameRecord(
            game_id="g_42",
            condition="C",
            experiment=2,
            turns=[TurnRecord(move_number=1)],
            final_status="checkmate",
            total_turns=1,
        )
        json_str = gr.model_dump_json()
        restored = GameRecord.model_validate_json(json_str)
        assert restored.game_id == "g_42"
        assert len(restored.turns) == 1

    def test_empty_turns(self):
        gr = GameRecord(
            game_id="empty",
            condition="A",
            experiment=1,
        )
        assert gr.turns == []
        assert gr.total_turns == 0


# ── Aggregate containers ─────────────────────────────────────────────────


class TestAggregateModels:
    def test_phase_stratified_fir(self):
        p = PhaseStratifiedFIR(opening=0.1, middlegame=0.2, endgame=None)
        assert p.opening == 0.1
        assert p.endgame is None

    def test_critic_accuracy(self):
        ca = CriticAccuracy(
            true_positives=10,
            false_negatives=5,
            tpr=10 / 15,
            fnr=5 / 15,
        )
        assert ca.tpr == pytest.approx(0.6667, abs=1e-3)
        assert ca.fnr == pytest.approx(0.3333, abs=1e-3)

    def test_descriptive_stats(self):
        ds = DescriptiveStats(mean=2.5, median=2.0, min=1.0, max=4.0, std=1.0, n=4)
        assert ds.mean == 2.5

    def test_fst_entry(self):
        e = FSTEntry(game_id="g1", half_moves=42, censored=False)
        assert not e.censored

    def test_quartile_error_dist(self):
        q = QuartileErrorDist(
            quartile="Q1",
            counts={"INVALID_PIECE": 3, "PARSE_ERROR": 1},
            total_errors=4,
        )
        assert q.total_errors == 4

    def test_legality_degradation_bin(self):
        b = LegalityDegradationBin(bin_start=1, bin_end=10, ftir=0.3, n_turns=20)
        assert b.ftir == 0.3

    def test_condition_metrics_defaults(self):
        cm = ConditionMetrics(condition="B", experiment=1)
        assert cm.rsr is None
        assert cm.cafir is None
        assert cm.vta is None
