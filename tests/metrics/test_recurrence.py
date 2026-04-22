"""Tests for src.metrics.recurrence — multi-turn consistency metrics."""

from __future__ import annotations

import pytest

from src.metrics.definitions import TurnRecord
from src.metrics.recurrence import (
    compute_ecc,
    compute_error_type_over_quartiles,
    compute_input_length_vs_error,
    compute_legality_degradation,
    compute_pcrr,
    compute_population_ftir_by_turn,
    compute_serr,
    compute_ttr,
)


# ── Helper ───────────────────────────────────────────────────────────────


def _turn(
    *,
    move_num: int = 1,
    first_valid: bool = True,
    valid: bool = True,
    errors: list[str] | None = None,
    prompt_tokens: int = 100,
) -> TurnRecord:
    return TurnRecord(
        move_number=move_num,
        proposed_move="e2e4",
        is_valid=valid,
        first_try_valid=first_valid,
        total_attempts=1 if first_valid else 2,
        error_types=errors or [],
        retry_count=0 if first_valid else 1,
        llm_calls_this_turn=1,
        tokens_this_turn=200,
        prompt_token_count=prompt_tokens,
    )


# ── SERR ─────────────────────────────────────────────────────────────────


class TestSERR:
    def test_no_errors(self):
        turns = [_turn(errors=[]) for _ in range(5)]
        assert compute_serr(turns) == 0.0

    def test_no_recurrence(self):
        """Each error type appears exactly once."""
        turns = [
            _turn(errors=["INVALID_PIECE"]),
            _turn(errors=["ILLEGAL_DESTINATION"]),
            _turn(errors=["PARSE_ERROR"]),
        ]
        assert compute_serr(turns) == pytest.approx(0.0)

    def test_all_recurring(self):
        """Every error type appears more than once."""
        turns = [
            _turn(errors=["INVALID_PIECE"]),
            _turn(errors=["INVALID_PIECE"]),
            _turn(errors=["PARSE_ERROR"]),
            _turn(errors=["PARSE_ERROR"]),
        ]
        # 2 distinct types, both recurring → 1.0
        assert compute_serr(turns) == pytest.approx(1.0)

    def test_partial_recurrence(self):
        turns = [
            _turn(errors=["INVALID_PIECE"]),
            _turn(errors=["INVALID_PIECE"]),
            _turn(errors=["PARSE_ERROR"]),
        ]
        # 2 distinct types, 1 recurring → 0.5
        assert compute_serr(turns) == pytest.approx(0.5)


# ── PCRR ─────────────────────────────────────────────────────────────────


class TestPCRR:
    def test_no_corrections(self):
        turns = [
            _turn(first_valid=True, valid=True),
            _turn(first_valid=False, valid=False, errors=["INVALID_PIECE"]),
        ]
        assert compute_pcrr(turns) is None

    def test_no_recurrence_after_correction(self):
        turns = [
            _turn(first_valid=False, valid=True, errors=["INVALID_PIECE"], move_num=1),
            _turn(first_valid=True, valid=True, errors=[], move_num=2),
            _turn(first_valid=True, valid=True, errors=[], move_num=3),
        ]
        assert compute_pcrr(turns) == pytest.approx(0.0)

    def test_recurrence_after_correction(self):
        turns = [
            _turn(first_valid=False, valid=True, errors=["INVALID_PIECE"], move_num=1),
            _turn(first_valid=True, valid=True, errors=[], move_num=2),
            _turn(first_valid=False, valid=False, errors=["INVALID_PIECE"], move_num=3),
        ]
        # 1 correction, 1 recurrence → 1.0
        assert compute_pcrr(turns) == pytest.approx(1.0)

    def test_multiple_corrections(self):
        turns = [
            # Correction 1: INVALID_PIECE
            _turn(first_valid=False, valid=True, errors=["INVALID_PIECE"], move_num=1),
            _turn(first_valid=True, valid=True, move_num=2),
            # Correction 2: PARSE_ERROR
            _turn(first_valid=False, valid=True, errors=["PARSE_ERROR"], move_num=3),
            _turn(first_valid=True, valid=True, move_num=4),
            # Recurrence of INVALID_PIECE (but not PARSE_ERROR)
            _turn(first_valid=False, valid=False, errors=["INVALID_PIECE"], move_num=5),
        ]
        # 2 corrections, 1 recurrence → 0.5
        assert compute_pcrr(turns) == pytest.approx(0.5)


# ── TTR ──────────────────────────────────────────────────────────────────


class TestTTR:
    def test_no_corrections(self):
        turns = [_turn(first_valid=True, valid=True) for _ in range(5)]
        assert compute_ttr(turns) == []

    def test_immediate_error_after_correction(self):
        turns = [
            _turn(first_valid=False, valid=True, move_num=1),   # Correction
            _turn(first_valid=False, valid=False, move_num=2),  # Error immediately
        ]
        assert compute_ttr(turns) == [0]

    def test_clean_streak(self):
        turns = [
            _turn(first_valid=False, valid=True, move_num=1),   # Correction
            _turn(first_valid=True, valid=True, move_num=2),    # Clean
            _turn(first_valid=True, valid=True, move_num=3),    # Clean
            _turn(first_valid=True, valid=True, move_num=4),    # Clean
            _turn(first_valid=False, valid=False, move_num=5),  # Error
        ]
        assert compute_ttr(turns) == [3]

    def test_multiple_corrections(self):
        turns = [
            _turn(first_valid=False, valid=True, move_num=1),
            _turn(first_valid=True, valid=True, move_num=2),
            _turn(first_valid=False, valid=True, move_num=3),
            _turn(first_valid=True, valid=True, move_num=4),
            _turn(first_valid=True, valid=True, move_num=5),
        ]
        assert compute_ttr(turns) == [1, 2]


# ── ECC ──────────────────────────────────────────────────────────────────


class TestECC:
    def test_no_errors(self):
        turns = [_turn(first_valid=True) for _ in range(5)]
        pop = {i: 0.0 for i in range(5)}
        assert compute_ecc(turns, pop) is None  # Expected = 0

    def test_independent_errors(self):
        """Errors at uniform rate → ECC should be finite."""
        # 10 games, each 4 turns, with errors at turns 0 and 2 only
        # This way population FTIR(0)=1, FTIR(1)=0, FTIR(2)=1, FTIR(3)=0
        # Expected pairs all involve a zero → expected = 0 → None
        # Instead, use a pattern where consecutive errors can occur:
        # All games have error at every turn → ECC = 1.0 (observed = expected)
        all_games = []
        for _ in range(20):
            game = [_turn(first_valid=False) for _ in range(5)]
            all_games.append(game)

        pop = compute_population_ftir_by_turn(all_games)
        # Pop FTIR = 1.0 at every turn

        test_game = all_games[0]
        ecc = compute_ecc(test_game, pop)
        assert ecc is not None
        # 4 consecutive pairs, expected = 4 * 1.0 * 1.0 = 4.0 → ECC = 1.0
        assert ecc == pytest.approx(1.0)

    def test_clustered_errors(self):
        """Consecutive errors → ECC > 1 (if baseline expects fewer)."""
        # Game with errors at turns 0,1,2 then clean 3,4
        game = [
            _turn(first_valid=False, move_num=1),
            _turn(first_valid=False, move_num=2),
            _turn(first_valid=False, move_num=3),
            _turn(first_valid=True, move_num=4),
            _turn(first_valid=True, move_num=5),
        ]
        # Population: 50% error rate at every turn
        pop = {i: 0.5 for i in range(5)}
        ecc = compute_ecc(game, pop)
        assert ecc is not None
        # 2 consecutive pairs, expected = 4 * 0.25 = 1.0 → ECC = 2.0
        assert ecc == pytest.approx(2.0)

    def test_single_turn(self):
        assert compute_ecc([_turn()], {0: 0.5}) is None


class TestPopulationFTIR:
    def test_basic(self):
        games = [
            [_turn(first_valid=True), _turn(first_valid=False)],
            [_turn(first_valid=False), _turn(first_valid=True)],
        ]
        pop = compute_population_ftir_by_turn(games)
        assert pop[0] == pytest.approx(0.5)
        assert pop[1] == pytest.approx(0.5)

    def test_uneven_lengths(self):
        games = [
            [_turn(first_valid=True), _turn(first_valid=True), _turn(first_valid=False)],
            [_turn(first_valid=False)],
        ]
        pop = compute_population_ftir_by_turn(games)
        assert pop[0] == pytest.approx(0.5)
        assert pop[1] == pytest.approx(0.0)  # Only 1 game has turn 1
        assert pop[2] == pytest.approx(1.0)  # Only 1 game has turn 2


# ── Legality Degradation ─────────────────────────────────────────────────


class TestLegalityDegradation:
    def test_basic_bins(self):
        turns = [
            _turn(first_valid=True, move_num=1),
            _turn(first_valid=True, move_num=5),
            _turn(first_valid=False, move_num=8),
            _turn(first_valid=False, move_num=12),
            _turn(first_valid=False, move_num=15),
        ]
        bins = compute_legality_degradation(turns, bin_size=10)
        assert len(bins) == 2
        # Bin 1-10: 1 invalid out of 3
        assert bins[0].bin_start == 1
        assert bins[0].bin_end == 10
        assert bins[0].ftir == pytest.approx(1 / 3)
        assert bins[0].n_turns == 3
        # Bin 11-20: 2 invalid out of 2
        assert bins[1].ftir == pytest.approx(1.0)
        assert bins[1].n_turns == 2

    def test_empty(self):
        assert compute_legality_degradation([]) == []


# ── Input Length vs Error ────────────────────────────────────────────────


class TestInputLengthVsError:
    def test_output_format(self):
        turns = [
            _turn(first_valid=True, prompt_tokens=100, move_num=1),
            _turn(first_valid=False, prompt_tokens=500, move_num=2),
        ]
        result = compute_input_length_vs_error(turns)
        assert len(result) == 2
        assert result[0]["prompt_token_count"] == 100
        assert result[0]["is_error"] is False
        assert result[1]["prompt_token_count"] == 500
        assert result[1]["is_error"] is True


# ── Error-Type over Quartiles ────────────────────────────────────────────


class TestErrorTypeOverQuartiles:
    def test_four_quartiles(self):
        turns = [
            _turn(errors=["INVALID_PIECE"], move_num=1),        # Q1
            _turn(errors=["PARSE_ERROR"], move_num=5),           # Q2
            _turn(errors=["ILLEGAL_DESTINATION"], move_num=8),   # Q3
            _turn(errors=["INVALID_PIECE", "PARSE_ERROR"], move_num=10),  # Q4
        ]
        result = compute_error_type_over_quartiles(turns, total_game_moves=10)
        assert len(result) == 4
        # Q1: move 1 → errors: INVALID_PIECE
        q1 = result[0]
        assert q1.quartile == "Q1"
        assert q1.counts.get("INVALID_PIECE", 0) >= 1

    def test_empty(self):
        assert compute_error_type_over_quartiles([], 10) == []

    def test_zero_moves(self):
        turns = [_turn(errors=["INVALID_PIECE"])]
        assert compute_error_type_over_quartiles(turns, 0) == []
