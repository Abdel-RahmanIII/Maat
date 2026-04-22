"""Tests for src.metrics.aggregator — aggregate metric computation."""

from __future__ import annotations

import pytest

from src.metrics.aggregator import (
    compute_all_exp1_metrics,
    compute_all_game_metrics,
    compute_arr,
    compute_avg_reasoning_steps,
    compute_cafir,
    compute_critic_accuracy,
    compute_error_type_rsr,
    compute_fir,
    compute_fir_cross_experiment_delta,
    compute_fst_data,
    compute_ftir,
    compute_game_fir,
    compute_imfr,
    compute_lcpt,
    compute_mfir,
    compute_mrtc,
    compute_parse_failure_counts,
    compute_phase_stratified_fir,
    compute_rsr,
    compute_tcr,
    compute_tool_call_distribution,
    compute_tool_stratified_fir,
    compute_tpt,
    compute_vta,
)
from src.metrics.definitions import GameRecord, TurnRecord


# ── Helpers ──────────────────────────────────────────────────────────────


def _turn(
    *,
    valid: bool = True,
    first_valid: bool = True,
    retries: int = 0,
    llm_calls: int = 1,
    tokens: int = 100,
    prompt_tokens: int = 80,
    errors: list[str] | None = None,
    phase: str = "opening",
    move_num: int = 1,
    tool_calls: list[dict] | None = None,
    critic_verdict: bool | None = None,
    ground_truth_verdict: bool | None = None,
) -> TurnRecord:
    return TurnRecord(
        move_number=move_num,
        proposed_move="e2e4",
        is_valid=valid,
        first_try_valid=first_valid,
        total_attempts=1 + retries,
        error_types=errors or [],
        retry_count=retries,
        llm_calls_this_turn=llm_calls,
        tokens_this_turn=tokens,
        prompt_token_count=prompt_tokens,
        tool_calls=tool_calls or [],
        critic_verdict=critic_verdict,
        ground_truth_verdict=ground_truth_verdict,
        game_phase=phase,
    )


def _game(
    game_id: str,
    condition: str,
    turns: list[TurnRecord],
    status: str = "checkmate",
    experiment: int = 2,
    starting_fen: str = "",
) -> GameRecord:
    return GameRecord(
        game_id=game_id,
        condition=condition,
        experiment=experiment,
        turns=turns,
        final_status=status,
        total_turns=len(turns),
        total_llm_calls=sum(t.llm_calls_this_turn for t in turns),
        total_tokens=sum(t.tokens_this_turn for t in turns),
        starting_fen=starting_fen,
    )


# ── FIR / FTIR ───────────────────────────────────────────────────────────


class TestFIR:
    def test_all_valid(self):
        turns = [_turn(valid=True) for _ in range(10)]
        assert compute_fir(turns) == 0.0

    def test_all_invalid(self):
        turns = [_turn(valid=False) for _ in range(10)]
        assert compute_fir(turns) == 1.0

    def test_mixed(self):
        turns = [_turn(valid=True)] * 7 + [_turn(valid=False)] * 3
        assert compute_fir(turns) == pytest.approx(0.3)

    def test_empty(self):
        assert compute_fir([]) == 0.0


class TestFTIR:
    def test_all_first_valid(self):
        turns = [_turn(first_valid=True) for _ in range(5)]
        assert compute_ftir(turns) == 0.0

    def test_mixed(self):
        turns = [_turn(first_valid=True)] * 6 + [_turn(first_valid=False)] * 4
        assert compute_ftir(turns) == pytest.approx(0.4)

    def test_empty(self):
        assert compute_ftir([]) == 0.0


# ── MFIR / ARR ───────────────────────────────────────────────────────────


class TestMFIR:
    def test_normal(self):
        result = compute_mfir(0.5, 0.3)
        assert result == pytest.approx(0.4)

    def test_baseline_zero(self):
        assert compute_mfir(0.0, 0.3) is None

    def test_no_improvement(self):
        assert compute_mfir(0.5, 0.5) == pytest.approx(0.0)


class TestARR:
    def test_positive(self):
        assert compute_arr(0.5, 0.3) == pytest.approx(0.2)

    def test_negative(self):
        assert compute_arr(0.3, 0.5) == pytest.approx(-0.2)


# ── Phase-stratified FIR ─────────────────────────────────────────────────


class TestPhaseStratifiedFIR:
    def test_all_phases(self):
        turns = [
            _turn(valid=True, phase="opening"),
            _turn(valid=False, phase="opening"),
            _turn(valid=True, phase="middlegame"),
            _turn(valid=True, phase="middlegame"),
            _turn(valid=False, phase="endgame"),
        ]
        result = compute_phase_stratified_fir(turns)
        assert result.opening == pytest.approx(0.5)
        assert result.middlegame == pytest.approx(0.0)
        assert result.endgame == pytest.approx(1.0)

    def test_missing_phase(self):
        turns = [_turn(valid=True, phase="opening")]
        result = compute_phase_stratified_fir(turns)
        assert result.opening == pytest.approx(0.0)
        assert result.middlegame is None
        assert result.endgame is None


# ── Parse failures ───────────────────────────────────────────────────────


class TestParseFailureCounts:
    def test_counts(self):
        turns = [
            _turn(errors=["PARSE_ERROR"]),
            _turn(errors=["PARSE_ERROR", "NO_OUTPUT"]),
            _turn(errors=["INVALID_PIECE"]),
            _turn(errors=[]),
        ]
        assert compute_parse_failure_counts(turns) == 3

    def test_none(self):
        turns = [_turn(errors=["INVALID_PIECE"]), _turn()]
        assert compute_parse_failure_counts(turns) == 0


# ── RSR / MRTC ───────────────────────────────────────────────────────────


class TestRSR:
    def test_all_corrected(self):
        turns = [
            _turn(first_valid=False, valid=True, retries=1),
            _turn(first_valid=False, valid=True, retries=2),
        ]
        assert compute_rsr(turns) == pytest.approx(1.0)

    def test_none_corrected(self):
        turns = [
            _turn(first_valid=False, valid=False),
            _turn(first_valid=False, valid=False),
        ]
        assert compute_rsr(turns) == pytest.approx(0.0)

    def test_mixed(self):
        turns = [
            _turn(first_valid=False, valid=True, retries=1),
            _turn(first_valid=False, valid=False),
            _turn(first_valid=True, valid=True),  # Not initially-invalid
        ]
        assert compute_rsr(turns) == pytest.approx(0.5)

    def test_no_initially_invalid(self):
        turns = [_turn(first_valid=True, valid=True)]
        assert compute_rsr(turns) is None


class TestMRTC:
    def test_mean_retries(self):
        turns = [
            _turn(first_valid=False, valid=True, retries=1),
            _turn(first_valid=False, valid=True, retries=3),
        ]
        assert compute_mrtc(turns) == pytest.approx(2.0)

    def test_no_corrections(self):
        turns = [_turn(first_valid=False, valid=False)]
        assert compute_mrtc(turns) is None


# ── Cost metrics ─────────────────────────────────────────────────────────


class TestCostMetrics:
    def test_lcpt(self):
        turns = [_turn(llm_calls=2), _turn(llm_calls=4), _turn(llm_calls=3)]
        result = compute_lcpt(turns)
        assert result.mean == pytest.approx(3.0)
        assert result.median == pytest.approx(3.0)
        assert result.min == 2.0
        assert result.max == 4.0
        assert result.n == 3

    def test_tpt(self):
        turns = [_turn(tokens=100), _turn(tokens=300)]
        result = compute_tpt(turns)
        assert result.mean == pytest.approx(200.0)

    def test_cafir(self):
        assert compute_cafir(0.3, 2.5) == pytest.approx(0.75)

    def test_cafir_zero(self):
        assert compute_cafir(0.0, 5.0) == pytest.approx(0.0)


# ── Critic accuracy ─────────────────────────────────────────────────────


class TestCriticAccuracy:
    def test_perfect_critic(self):
        turns = [
            # Critic says valid, actually valid → TN
            _turn(critic_verdict=True, ground_truth_verdict=True),
            _turn(critic_verdict=True, ground_truth_verdict=True),
            # Critic says invalid, actually invalid → TP
            _turn(critic_verdict=False, ground_truth_verdict=False),
        ]
        result = compute_critic_accuracy(turns)
        assert result.true_negatives == 2
        assert result.true_positives == 1
        assert result.false_positives == 0
        assert result.false_negatives == 0
        assert result.tpr == pytest.approx(1.0)
        assert result.tnr == pytest.approx(1.0)

    def test_imperfect_critic(self):
        turns = [
            _turn(critic_verdict=True, ground_truth_verdict=True),   # TN
            _turn(critic_verdict=True, ground_truth_verdict=False),  # FN
            _turn(critic_verdict=False, ground_truth_verdict=False), # TP
            _turn(critic_verdict=False, ground_truth_verdict=True),  # FP
        ]
        result = compute_critic_accuracy(turns)
        assert result.true_positives == 1
        assert result.false_negatives == 1
        assert result.true_negatives == 1
        assert result.false_positives == 1
        assert result.tpr == pytest.approx(0.5)
        assert result.fpr == pytest.approx(0.5)
        assert result.tnr == pytest.approx(0.5)
        assert result.fnr == pytest.approx(0.5)

    def test_skips_none_verdicts(self):
        turns = [
            _turn(critic_verdict=None, ground_truth_verdict=None),
            _turn(critic_verdict=True, ground_truth_verdict=True),
        ]
        result = compute_critic_accuracy(turns)
        assert result.true_negatives == 1
        assert result.true_positives == 0


# ── Error-type RSR ───────────────────────────────────────────────────────


class TestErrorTypeRSR:
    def test_basic(self):
        turns = [
            _turn(first_valid=False, valid=True, errors=["INVALID_PIECE"]),
            _turn(first_valid=False, valid=False, errors=["INVALID_PIECE"]),
            _turn(first_valid=False, valid=True, errors=["ILLEGAL_DESTINATION"]),
        ]
        result = compute_error_type_rsr(turns)
        assert result.rsr_by_type["INVALID_PIECE"] == pytest.approx(0.5)
        assert result.rsr_by_type["ILLEGAL_DESTINATION"] == pytest.approx(1.0)
        assert result.counts_by_type["INVALID_PIECE"] == (1, 2)
        assert result.counts_by_type["ILLEGAL_DESTINATION"] == (1, 1)


# ── Condition F metrics ──────────────────────────────────────────────────


class TestConditionFMetrics:
    def test_vta(self):
        turns = [
            _turn(tool_calls=[{"name": "validate_move", "result": True}]),
            _turn(tool_calls=[{"name": "get_board_visual"}]),
            _turn(tool_calls=[]),
        ]
        assert compute_vta(turns) == pytest.approx(1 / 3)

    def test_tcr(self):
        turns = [
            _turn(tool_calls=[{"name": "validate_move"}]),
            _turn(tool_calls=[]),
            _turn(tool_calls=[{"name": "get_board_visual"}]),
        ]
        assert compute_tcr(turns) == pytest.approx(2 / 3)

    def test_tool_call_distribution(self):
        turns = [
            _turn(tool_calls=[
                {"name": "validate_move"},
                {"name": "get_board_visual"},
            ]),
            _turn(tool_calls=[{"name": "validate_move"}]),
        ]
        result = compute_tool_call_distribution(turns)
        assert result.counts["validate_move"] == 2
        assert result.counts["get_board_visual"] == 1
        assert result.total_tool_calls == 3

    def test_tool_stratified_fir(self):
        turns = [
            _turn(valid=True, tool_calls=[{"name": "validate_move"}]),
            _turn(valid=False, tool_calls=[{"name": "validate_move"}]),
            _turn(valid=False, tool_calls=[]),
            _turn(valid=False, tool_calls=[]),
        ]
        result = compute_tool_stratified_fir(turns)
        assert result["with_tools"] == pytest.approx(0.5)
        assert result["without_tools"] == pytest.approx(1.0)

    def test_avg_reasoning_steps(self):
        turns = [
            _turn(tool_calls=[{"name": "a"}, {"name": "b"}]),
            _turn(tool_calls=[{"name": "c"}]),
            _turn(tool_calls=[]),
        ]
        assert compute_avg_reasoning_steps(turns) == pytest.approx(1.0)


# ── Game-level metrics ───────────────────────────────────────────────────


class TestGameLevelMetrics:
    def test_game_fir(self):
        g1 = _game("g1", "D", [_turn(valid=True), _turn(valid=False)])
        g2 = _game("g2", "D", [_turn(valid=True), _turn(valid=True)])
        assert compute_game_fir([g1, g2]) == pytest.approx(0.25)

    def test_imfr(self):
        games = [
            _game("g1", "D", [_turn()], status="forfeit"),
            _game("g2", "D", [_turn()], status="checkmate"),
            _game("g3", "D", [_turn()], status="forfeit"),
            _game("g4", "D", [_turn()], status="draw"),
        ]
        assert compute_imfr(games) == pytest.approx(0.5)

    def test_imfr_none_forfeit(self):
        games = [_game("g1", "A", [_turn()], status="checkmate")]
        assert compute_imfr(games) == pytest.approx(0.0)

    def test_fst_data(self):
        games = [
            _game("g1", "D", [_turn()] * 20, status="forfeit"),
            _game("g2", "D", [_turn()] * 50, status="checkmate"),
        ]
        fst = compute_fst_data(games)
        assert len(fst) == 2
        assert fst[0].half_moves == 20
        assert fst[0].censored is False  # forfeit = uncensored
        assert fst[1].half_moves == 50
        assert fst[1].censored is True   # checkmate = censored


# ── Cross-experiment delta ───────────────────────────────────────────────


class TestFIRCrossExperimentDelta:
    def test_paired(self):
        exp2 = [
            _game("g1", "D", [_turn(valid=True), _turn(valid=False)],
                  experiment=2, starting_fen="fen_A"),
        ]
        exp3 = [
            _game("g2", "D", [_turn(valid=False), _turn(valid=False)],
                  experiment=3, starting_fen="fen_A"),
        ]
        deltas = compute_fir_cross_experiment_delta(exp2, exp3)
        assert len(deltas) == 1
        assert deltas[0].fir_exp2 == pytest.approx(0.5)
        assert deltas[0].fir_exp3 == pytest.approx(1.0)
        assert deltas[0].delta == pytest.approx(0.5)

    def test_unpaired_skipped(self):
        exp2 = [_game("g1", "D", [_turn()], experiment=2, starting_fen="fen_A")]
        exp3 = [_game("g2", "D", [_turn()], experiment=3, starting_fen="fen_B")]
        deltas = compute_fir_cross_experiment_delta(exp2, exp3)
        assert len(deltas) == 0


# ── Dispatcher tests ─────────────────────────────────────────────────────


class TestComputeAllExp1:
    def test_condition_d_has_retry_metrics(self):
        turns = [
            _turn(valid=True, first_valid=False, retries=1,
                  errors=["INVALID_PIECE"], phase="opening"),
            _turn(valid=True, first_valid=True, phase="middlegame"),
        ]
        result = compute_all_exp1_metrics(turns, "D")
        assert result.condition == "D"
        assert result.experiment == 1
        assert result.fir == pytest.approx(0.0)
        assert result.ftir == pytest.approx(0.5)
        assert result.rsr is not None
        assert result.mrtc is not None
        assert result.cafir is not None
        assert result.vta is None  # Not F
        assert result.critic_accuracy is None  # Not C

    def test_condition_a_no_retry_no_cafir(self):
        turns = [_turn(valid=True, phase="opening")]
        result = compute_all_exp1_metrics(turns, "A")
        assert result.rsr is None
        assert result.mrtc is None
        assert result.cafir is None

    def test_condition_f_has_tool_metrics(self):
        turns = [
            _turn(tool_calls=[{"name": "validate_move"}]),
            _turn(tool_calls=[]),
        ]
        result = compute_all_exp1_metrics(turns, "F")
        assert result.vta is not None
        assert result.tcr is not None
        assert result.tool_call_distribution is not None
        assert result.avg_reasoning_steps is not None
        assert result.cafir is not None

    def test_condition_c_has_critic(self):
        turns = [
            _turn(critic_verdict=True, ground_truth_verdict=True, phase="opening"),
            _turn(first_valid=False, valid=True, retries=1,
                  errors=["INVALID_PIECE"], phase="opening",
                  critic_verdict=False, ground_truth_verdict=False),
        ]
        result = compute_all_exp1_metrics(turns, "C")
        assert result.critic_accuracy is not None
        assert result.rsr is not None


class TestComputeAllGameMetrics:
    def test_basic_game_metrics(self):
        games = [
            _game("g1", "D", [
                _turn(valid=True, first_valid=True, move_num=1, phase="opening"),
                _turn(valid=True, first_valid=False, move_num=2, phase="opening",
                      retries=1, errors=["INVALID_PIECE"]),
            ], status="checkmate"),
            _game("g2", "D", [
                _turn(valid=False, first_valid=False, move_num=1, phase="opening",
                      errors=["PARSE_ERROR"]),
            ], status="forfeit"),
        ]
        result = compute_all_game_metrics(games, "D")
        assert result.experiment == 2
        assert result.imfr == pytest.approx(0.5)
        assert result.fst_data is not None
        assert len(result.fst_data) == 2
        assert result.serr is not None
        assert result.legality_degradation is not None
