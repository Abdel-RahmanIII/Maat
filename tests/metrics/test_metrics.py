"""Tests for the metrics layer: definitions, collector, aggregator, recurrence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.metrics.definitions import GameRecord, TurnRecord
from src.metrics.collector import MetricsCollector
from src.metrics import aggregator
from src.metrics import recurrence


# ═══════════════════════════════════════════════════════════════════════
# Helpers — build mock data
# ═══════════════════════════════════════════════════════════════════════


def _turn(
    *,
    game_id: str = "g1",
    condition: str = "B",
    experiment: str = "exp1",
    move_number: int = 1,
    game_phase: str = "opening",
    is_valid: bool = True,
    first_try_valid: bool = True,
    total_attempts: int = 1,
    retry_count: int = 0,
    error_types: list[str] | None = None,
    llm_calls_this_turn: int = 1,
    tokens_this_turn: int = 100,
    prompt_token_count: int = 80,
    wall_clock_ms: float = 500.0,
    tool_calls: list[dict] | None = None,
    critic_verdict: bool | None = None,
    ground_truth_verdict: bool | None = None,
) -> TurnRecord:
    """Build a TurnRecord with sensible defaults."""
    return TurnRecord(
        game_id=game_id,
        condition=condition,
        experiment=experiment,
        move_number=move_number,
        game_phase=game_phase,
        proposed_move="e2e4",
        is_valid=is_valid,
        first_try_valid=first_try_valid,
        total_attempts=total_attempts,
        retry_count=retry_count,
        error_types=error_types or [],
        llm_calls_this_turn=llm_calls_this_turn,
        tokens_this_turn=tokens_this_turn,
        prompt_token_count=prompt_token_count,
        wall_clock_ms=wall_clock_ms,
        tool_calls=tool_calls or [],
        critic_verdict=critic_verdict,
        ground_truth_verdict=ground_truth_verdict,
        feedback_history=[],
    )


def _game(
    turns: list[TurnRecord],
    *,
    game_id: str = "g1",
    condition: str = "B",
    experiment: str = "exp1",
    game_status: str = "ongoing",
) -> GameRecord:
    """Build a GameRecord from a list of TurnRecords."""
    total_forfeits = sum(1 for t in turns if not t.is_valid)
    game_length = sum(1 for t in turns if t.is_valid)
    gcr = game_status in ("checkmate", "stalemate", "draw", "max_moves")
    return GameRecord(
        game_id=game_id,
        condition=condition,
        experiment=experiment,
        turns=turns,
        game_status=game_status,
        total_turns=len(turns),
        total_forfeits=total_forfeits,
        game_length=game_length,
        gcr_contributed=gcr,
    )


# ═══════════════════════════════════════════════════════════════════════
# TurnRecord / GameRecord creation
# ═══════════════════════════════════════════════════════════════════════


class TestDefinitions:
    def test_turn_record_creation(self):
        t = _turn(is_valid=True, error_types=["INVALID_PIECE"])
        assert t.game_id == "g1"
        assert t.is_valid is True
        assert t.error_types == ["INVALID_PIECE"]

    def test_game_record_creation(self):
        turns = [_turn(is_valid=True), _turn(is_valid=False)]
        g = _game(turns, game_status="forfeit")
        assert g.total_turns == 2
        assert g.total_forfeits == 1
        assert g.game_length == 1
        assert g.gcr_contributed is False

    def test_gcr_contributed_checkmate(self):
        g = _game([_turn()], game_status="checkmate")
        assert g.gcr_contributed is True


# ═══════════════════════════════════════════════════════════════════════
# Collector
# ═══════════════════════════════════════════════════════════════════════


class TestCollector:
    def test_record_turn_from_state(self):
        """record_turn correctly maps TurnState dict fields."""
        state = {
            "game_id": "p001",
            "condition": "D",
            "move_number": 3,
            "game_phase": "middlegame",
            "proposed_move": "d2d4",
            "is_valid": True,
            "first_try_valid": False,
            "total_attempts": 2,
            "retry_count": 1,
            "error_types": ["ILLEGAL_DESTINATION"],
            "llm_calls_this_turn": 2,
            "tokens_this_turn": 250,
            "prompt_token_count": 180,
            "wall_clock_ms": 1234.5,
            "tool_calls": [],
            "critic_verdict": None,
            "ground_truth_verdict": None,
            "feedback_history": ["Illegal move e2e5"],
            "generation_strategy": "generator_only",
            "strategic_plan": "",
            "routed_phase": "",
        }

        collector = MetricsCollector(output_dir=Path("test_out"), experiment="exp1")
        rec = collector.record_turn(state, experiment="exp1")

        assert rec.game_id == "p001"
        assert rec.condition == "D"
        assert rec.move_number == 3
        assert rec.is_valid is True
        assert rec.first_try_valid is False
        assert rec.error_types == ["ILLEGAL_DESTINATION"]
        assert rec.wall_clock_ms == 1234.5

    def test_record_game(self):
        turns = [
            _turn(is_valid=True),
            _turn(is_valid=False, error_types=["LEAVES_IN_CHECK"]),
        ]
        collector = MetricsCollector(output_dir=Path("test_out"), experiment="exp1")
        g = collector.record_game("g1", "B", turns, "forfeit")

        assert g.total_turns == 2
        assert g.total_forfeits == 1
        assert g.game_length == 1
        assert g.gcr_contributed is False

    def test_jsonl_round_trip(self, tmp_path):
        """Write records → load → verify equality."""
        collector = MetricsCollector(output_dir=tmp_path, experiment="exp1")

        t1 = _turn(game_id="g1", is_valid=True)
        t2 = _turn(game_id="g1", is_valid=False, error_types=["PARSE_ERROR"])

        collector._turn_buffer = [t1, t2]
        g = collector.record_game("g1", "B", [t1, t2], "forfeit")
        collector.flush()

        loaded_turns = MetricsCollector.load_turns(tmp_path / "turns.jsonl")
        assert len(loaded_turns) == 2
        assert loaded_turns[0].game_id == "g1"
        assert loaded_turns[1].error_types == ["PARSE_ERROR"]

        loaded_games = MetricsCollector.load_games(tmp_path / "games.jsonl")
        assert len(loaded_games) == 1
        assert loaded_games[0].total_forfeits == 1


# ═══════════════════════════════════════════════════════════════════════
# Aggregator — RQ1
# ═══════════════════════════════════════════════════════════════════════


class TestRQ1:
    def test_fir_computation(self):
        """10 turns, 3 forfeits → FIR = 0.3."""
        turns = [_turn(is_valid=True)] * 7 + [
            _turn(is_valid=False) for _ in range(3)
        ]
        games = [_game(turns)]
        assert aggregator.compute_fir(games) == pytest.approx(0.3)

    def test_fir_zero_when_all_valid(self):
        turns = [_turn(is_valid=True)] * 5
        games = [_game(turns)]
        assert aggregator.compute_fir(games) == 0.0

    def test_mfir_computation(self):
        # Baseline FIR=0.4, treatment FIR=0.2 → MFIR=0.5
        assert aggregator.compute_mfir(0.4, 0.2) == pytest.approx(0.5)

    def test_mfir_zero_baseline(self):
        assert aggregator.compute_mfir(0.0, 0.0) == 0.0

    def test_phase_stratified_fir(self):
        turns = [
            _turn(game_phase="opening", is_valid=True),
            _turn(game_phase="opening", is_valid=False),
            _turn(game_phase="middlegame", is_valid=True),
            _turn(game_phase="middlegame", is_valid=True),
            _turn(game_phase="endgame", is_valid=False),
        ]
        games = [_game(turns)]
        result = aggregator.compute_phase_stratified_fir(games)

        assert result["opening"] == pytest.approx(0.5)
        assert result["middlegame"] == pytest.approx(0.0)
        assert result["endgame"] == pytest.approx(1.0)

    def test_gcr(self):
        games = [
            _game([], game_status="checkmate"),
            _game([], game_status="forfeit"),
            _game([], game_status="draw"),
        ]
        assert aggregator.compute_gcr(games) == pytest.approx(2 / 3)

    def test_mbf(self):
        # Game 1: first forfeit at turn index 3 (3 legal moves before)
        turns1 = [_turn(is_valid=True)] * 3 + [_turn(is_valid=False)]
        # Game 2: first forfeit at turn index 1
        turns2 = [_turn(is_valid=True)] + [_turn(is_valid=False)]
        games = [_game(turns1), _game(turns2)]
        # median of [3, 1] = 2.0
        assert aggregator.compute_mbf(games) == pytest.approx(2.0)


# ═══════════════════════════════════════════════════════════════════════
# Aggregator — RQ2
# ═══════════════════════════════════════════════════════════════════════


class TestRQ2:
    def test_ftir_computation(self):
        turns = [
            _turn(first_try_valid=True),
            _turn(first_try_valid=False),
            _turn(first_try_valid=False),
            _turn(first_try_valid=True),
        ]
        games = [_game(turns)]
        assert aggregator.compute_ftir(games) == pytest.approx(0.5)

    def test_legality_degradation(self):
        # 15 turns: moves 1-10 in bin 0, moves 11-15 in bin 10
        turns = []
        for i in range(1, 11):
            turns.append(_turn(move_number=i, first_try_valid=True))
        for i in range(11, 16):
            turns.append(_turn(move_number=i, first_try_valid=False))

        games = [_game(turns)]
        result = aggregator.compute_legality_degradation(games, bin_size=10)

        assert result[0] == pytest.approx(0.0)   # moves 1-10: all valid
        assert result[10] == pytest.approx(1.0)  # moves 11-15: all invalid

    def test_error_type_by_quartile(self):
        # 4 turns: one per quartile
        turns = [
            _turn(move_number=1, error_types=["INVALID_PIECE"]),
            _turn(move_number=2, error_types=["ILLEGAL_DESTINATION"]),
            _turn(move_number=3, error_types=[]),
            _turn(move_number=4, error_types=["LEAVES_IN_CHECK"]),
        ]
        games = [_game(turns)]
        result = aggregator.compute_error_type_by_quartile(games)

        assert result["Q1"]["INVALID_PIECE"] == 1
        assert result["Q2"]["ILLEGAL_DESTINATION"] == 1
        assert result["Q4"]["LEAVES_IN_CHECK"] == 1
        assert "LEAVES_IN_CHECK" not in result["Q3"]


# ═══════════════════════════════════════════════════════════════════════
# Aggregator — RQ3
# ═══════════════════════════════════════════════════════════════════════


class TestRQ3:
    def test_rsr_computation(self):
        """2 initially-invalid turns, 1 corrected → RSR = 0.5."""
        turns = [
            _turn(first_try_valid=False, is_valid=True, retry_count=1),   # corrected
            _turn(first_try_valid=False, is_valid=False, retry_count=3),  # not corrected
            _turn(first_try_valid=True, is_valid=True),                   # valid first try
        ]
        games = [_game(turns, condition="D")]
        assert aggregator.compute_rsr(games) == pytest.approx(0.5)

    def test_mrtc_computation(self):
        turns = [
            _turn(first_try_valid=False, is_valid=True, retry_count=2),
            _turn(first_try_valid=False, is_valid=True, retry_count=1),
            _turn(first_try_valid=False, is_valid=False, retry_count=3),
        ]
        games = [_game(turns, condition="D")]
        # Mean of [2, 1] = 1.5
        assert aggregator.compute_mrtc(games) == pytest.approx(1.5)

    def test_pfr_computation(self):
        turns = [
            _turn(error_types=["PARSE_ERROR"]),
            _turn(error_types=["NO_OUTPUT"]),
            _turn(error_types=["INVALID_PIECE"]),
            _turn(error_types=[]),
        ]
        games = [_game(turns)]
        # 2 parse failures out of 4 = 0.5
        assert aggregator.compute_pfr(games) == pytest.approx(0.5)

    def test_lcpt_computation(self):
        turns = [
            _turn(llm_calls_this_turn=1),
            _turn(llm_calls_this_turn=3),
            _turn(llm_calls_this_turn=2),
        ]
        games = [_game(turns)]
        assert aggregator.compute_lcpt(games) == pytest.approx(2.0)

    def test_cafir_computation(self):
        assert aggregator.compute_cafir(0.3, 2.0) == pytest.approx(0.6)

    def test_critic_confusion_matrix(self):
        turns = [
            # Critic says valid, ground truth valid → TN
            _turn(critic_verdict=True, ground_truth_verdict=True),
            # Critic says valid, ground truth invalid → FN (critical!)
            _turn(critic_verdict=True, ground_truth_verdict=False),
            # Critic says invalid, ground truth invalid → TP
            _turn(critic_verdict=False, ground_truth_verdict=False),
            # Critic says invalid, ground truth valid → FP
            _turn(critic_verdict=False, ground_truth_verdict=True),
        ]
        games = [_game(turns, condition="C")]
        m = aggregator.compute_critic_confusion_matrix(games)

        assert m["tpr"] == pytest.approx(0.5)   # 1 TP / (1 TP + 1 FN)
        assert m["fnr"] == pytest.approx(0.5)   # 1 FN / (1 TP + 1 FN)
        assert m["tnr"] == pytest.approx(0.5)   # 1 TN / (1 TN + 1 FP)
        assert m["fpr"] == pytest.approx(0.5)   # 1 FP / (1 TN + 1 FP)

    def test_error_type_rsr(self):
        turns = [
            _turn(first_try_valid=False, is_valid=True,
                  error_types=["INVALID_PIECE"], retry_count=1),
            _turn(first_try_valid=False, is_valid=False,
                  error_types=["INVALID_PIECE"], retry_count=3),
            _turn(first_try_valid=False, is_valid=True,
                  error_types=["LEAVES_IN_CHECK"], retry_count=1),
        ]
        games = [_game(turns, condition="D")]
        result = aggregator.compute_error_type_rsr(games)

        assert result["INVALID_PIECE"] == pytest.approx(0.5)
        assert result["LEAVES_IN_CHECK"] == pytest.approx(1.0)

    def test_latency_stats(self):
        turns = [_turn(wall_clock_ms=float(i)) for i in range(1, 101)]
        games = [_game(turns)]
        stats = aggregator.compute_latency_stats(games)

        assert stats["median"] == pytest.approx(50.5)
        assert stats["mean"] == pytest.approx(50.5)
        # p95 index = int(100 * 0.95) = 95 → value 96
        assert stats["p95"] == pytest.approx(96.0)


# ═══════════════════════════════════════════════════════════════════════
# Aggregator — Condition F
# ═══════════════════════════════════════════════════════════════════════


class TestConditionF:
    def test_tcr(self):
        turns = [
            _turn(tool_calls=[{"tool": "validate_move"}]),
            _turn(tool_calls=[]),
            _turn(tool_calls=[{"tool": "get_board_visual"}]),
        ]
        games = [_game(turns, condition="F")]
        assert aggregator.compute_tcr(games) == pytest.approx(2 / 3)

    def test_vta(self):
        turns = [
            _turn(tool_calls=[{"tool": "validate_move"}]),
            _turn(tool_calls=[{"tool": "get_board_visual"}]),
            _turn(tool_calls=[]),
        ]
        games = [_game(turns, condition="F")]
        assert aggregator.compute_vta(games) == pytest.approx(1 / 3)

    def test_tool_distribution(self):
        turns = [
            _turn(tool_calls=[
                {"tool": "validate_move"},
                {"tool": "get_board_visual"},
            ]),
            _turn(tool_calls=[{"tool": "validate_move"}]),
        ]
        games = [_game(turns, condition="F")]
        dist = aggregator.compute_tool_distribution(games)

        assert dist["validate_move"] == 2
        assert dist["get_board_visual"] == 1

    def test_tool_stratified_fir(self):
        turns = [
            _turn(is_valid=True, tool_calls=[{"tool": "validate_move"}]),
            _turn(is_valid=False, tool_calls=[{"tool": "validate_move"}]),
            _turn(is_valid=False, tool_calls=[]),
        ]
        games = [_game(turns, condition="F")]
        result = aggregator.compute_tool_stratified_fir(games)

        assert result["with_tools"] == pytest.approx(0.5)
        assert result["without_tools"] == pytest.approx(1.0)

    def test_scr(self):
        turns = [
            # validate_move flagged invalid, but final submission is valid → self-corrected
            _turn(
                is_valid=True,
                tool_calls=[{"tool": "validate_move", "result": {"legal": False}}],
            ),
            # validate_move flagged invalid, final is also invalid → not self-corrected
            _turn(
                is_valid=False,
                tool_calls=[{"tool": "validate_move", "result": {"legal": False}}],
            ),
            # No validation tool called
            _turn(is_valid=True, tool_calls=[]),
        ]
        games = [_game(turns, condition="F")]
        assert aggregator.compute_scr(games) == pytest.approx(0.5)


# ═══════════════════════════════════════════════════════════════════════
# Aggregator — compute_all_metrics filtering
# ═══════════════════════════════════════════════════════════════════════


class TestComputeAllMetrics:
    def test_condition_filtering_baseline(self):
        """Condition B / exp1: includes FIR, no RSR, no critic, no tool metrics."""
        turns = [_turn(is_valid=True)] * 5
        games = [_game(turns, condition="B")]
        result = aggregator.compute_all_metrics(games, condition="B", experiment="exp1")

        assert "fir" in result
        assert "phase_stratified_fir" in result
        assert "cafir" in result
        assert "ftir" in result
        assert "rsr" not in result
        assert "critic_confusion_matrix" not in result
        assert "tcr" not in result

    def test_condition_filtering_retry(self):
        """Condition D: includes RSR, MRTC, MRPT."""
        turns = [_turn(first_try_valid=False, is_valid=True, retry_count=1)]
        games = [_game(turns, condition="D")]
        result = aggregator.compute_all_metrics(games, condition="D", experiment="exp1")

        assert "rsr" in result
        assert "mrtc" in result
        assert "mrpt" in result

    def test_condition_filtering_critic(self):
        """Condition C: includes critic confusion matrix."""
        turns = [_turn(critic_verdict=True, ground_truth_verdict=True)]
        games = [_game(turns, condition="C")]
        result = aggregator.compute_all_metrics(games, condition="C", experiment="exp1")

        assert "critic_confusion_matrix" in result

    def test_condition_filtering_tools(self):
        """Condition F: includes tool metrics."""
        turns = [_turn(tool_calls=[{"tool": "validate_move"}])]
        games = [_game(turns, condition="F")]
        result = aggregator.compute_all_metrics(games, condition="F", experiment="exp1")

        assert "tcr" in result
        assert "vta" in result
        assert "scr" in result

    def test_experiment_filtering_exp2(self):
        """Exp 2: includes GCR, MBF, degradation curve; no FIR/CAFIR."""
        turns = [_turn()] * 3
        games = [_game(turns, experiment="exp2", game_status="checkmate")]
        result = aggregator.compute_all_metrics(games, condition="B", experiment="exp2")

        # Exp 2 game-level metrics present
        assert "gcr" in result
        assert "mbf" in result
        assert "legality_degradation" in result
        # FIR-family NOT present (degenerate in full games)
        assert "fir" not in result
        assert "phase_stratified_fir" not in result
        assert "cafir" not in result
        # Universal metrics still present
        assert "ftir" in result
        assert "lcpt" in result

    def test_experiment_filtering_exp1_no_gcr(self):
        """Exp 1: no GCR/MBF (single-position, not full games)."""
        turns = [_turn()] * 3
        games = [_game(turns, condition="B")]
        result = aggregator.compute_all_metrics(games, condition="B", experiment="exp1")

        assert "gcr" not in result
        assert "mbf" not in result
        assert "fir" in result


# ═══════════════════════════════════════════════════════════════════════
# Recurrence metrics
# ═══════════════════════════════════════════════════════════════════════


class TestRecurrence:
    def test_serr_no_errors(self):
        """No errors → SERR is None."""
        g = _game([_turn(is_valid=True, error_types=[])])
        assert recurrence.compute_serr(g) is None

    def test_serr_all_unique(self):
        """Each error type appears once → SERR = 0."""
        g = _game([
            _turn(error_types=["INVALID_PIECE"]),
            _turn(error_types=["LEAVES_IN_CHECK"]),
        ])
        assert recurrence.compute_serr(g) == pytest.approx(0.0)

    def test_serr_recurrent(self):
        """One type appears in 2 turns, one in 1 turn → SERR = 0.5."""
        g = _game([
            _turn(error_types=["INVALID_PIECE"]),
            _turn(error_types=["INVALID_PIECE", "LEAVES_IN_CHECK"]),
        ])
        # INVALID_PIECE appears in 2 turns (recurrent), LEAVES_IN_CHECK in 1
        # 1 recurrent / 2 total types = 0.5
        assert recurrence.compute_serr(g) == pytest.approx(0.5)

    def test_pcrr_no_corrections(self):
        """No corrections → PCRR is None."""
        g = _game([_turn(first_try_valid=True)])
        assert recurrence.compute_pcrr(g) is None

    def test_pcrr_with_recurrence(self):
        """Error corrected, then same error recurs → PCRR = 1.0."""
        g = _game([
            _turn(first_try_valid=False, is_valid=True,
                  error_types=["INVALID_PIECE"], retry_count=1),
            _turn(first_try_valid=False, is_valid=False,
                  error_types=["INVALID_PIECE"]),
        ])
        assert recurrence.compute_pcrr(g) == pytest.approx(1.0)

    def test_pcrr_without_recurrence(self):
        """Error corrected, different error later → PCRR = 0.0."""
        g = _game([
            _turn(first_try_valid=False, is_valid=True,
                  error_types=["INVALID_PIECE"], retry_count=1),
            _turn(first_try_valid=False, is_valid=False,
                  error_types=["LEAVES_IN_CHECK"]),
        ])
        assert recurrence.compute_pcrr(g) == pytest.approx(0.0)

    def test_ecc_no_errors(self):
        """All valid → ECC is None."""
        g = _game([_turn(first_try_valid=True)] * 5)
        assert recurrence.compute_ecc(g) is None

    def test_ecc_clustered(self):
        """Errors in consecutive turns → ECC > 1."""
        turns = [
            _turn(first_try_valid=True),
            _turn(first_try_valid=False),
            _turn(first_try_valid=False),
            _turn(first_try_valid=True),
            _turn(first_try_valid=True),
        ]
        g = _game(turns)
        ecc = recurrence.compute_ecc(g)
        assert ecc is not None
        assert ecc > 1.0  # errors cluster

    def test_ecc_spread(self):
        """Errors spread out → ECC ≤ 1."""
        turns = [
            _turn(first_try_valid=False),
            _turn(first_try_valid=True),
            _turn(first_try_valid=False),
            _turn(first_try_valid=True),
        ]
        g = _game(turns)
        ecc = recurrence.compute_ecc(g)
        assert ecc is not None
        # FTIR = 2/4 = 0.5, expected = 3 * 0.25 = 0.75
        # observed = 0 consecutive pairs → ECC = 0
        assert ecc == pytest.approx(0.0)

    def test_recurrence_metrics_aggregate(self):
        """compute_recurrence_metrics returns aggregated means."""
        games = [
            _game([
                _turn(first_try_valid=False, error_types=["INVALID_PIECE"]),
                _turn(first_try_valid=False, error_types=["INVALID_PIECE"]),
            ]),
            _game([
                _turn(first_try_valid=True, error_types=[]),
                _turn(first_try_valid=True, error_types=[]),
            ]),
        ]
        result = recurrence.compute_recurrence_metrics(games)

        # First game: SERR=1.0, second game: SERR=None
        assert result["serr_mean"] == pytest.approx(1.0)
        assert len(result["serr_values"]) == 1


# ═══════════════════════════════════════════════════════════════════════
# GCR Cross-Experiment Δ
# ═══════════════════════════════════════════════════════════════════════


class TestGCRCrossExperimentDelta:
    def test_positive_delta_fen_helps(self):
        """GCR_Exp2 > GCR_Exp3 → positive Δ (FEN helps)."""
        # Exp 2: 2/3 games completed
        exp2 = [
            _game([], condition="B", experiment="exp2", game_status="checkmate"),
            _game([], condition="B", experiment="exp2", game_status="checkmate"),
            _game([], condition="B", experiment="exp2", game_status="forfeit"),
        ]
        # Exp 3: 1/3 games completed
        exp3 = [
            _game([], condition="B", experiment="exp3", game_status="checkmate"),
            _game([], condition="B", experiment="exp3", game_status="forfeit"),
            _game([], condition="B", experiment="exp3", game_status="forfeit"),
        ]
        delta = aggregator.compute_gcr_cross_experiment_delta(exp2, exp3)
        # GCR_Exp2 = 2/3, GCR_Exp3 = 1/3 → Δ = 1/3
        assert delta["B"] == pytest.approx(1 / 3)

    def test_zero_delta_same_gcr(self):
        """Same GCR in both experiments → Δ = 0."""
        exp2 = [
            _game([], condition="D", experiment="exp2", game_status="checkmate"),
            _game([], condition="D", experiment="exp2", game_status="forfeit"),
        ]
        exp3 = [
            _game([], condition="D", experiment="exp3", game_status="checkmate"),
            _game([], condition="D", experiment="exp3", game_status="forfeit"),
        ]
        delta = aggregator.compute_gcr_cross_experiment_delta(exp2, exp3)
        assert delta["D"] == pytest.approx(0.0)

    def test_multi_condition(self):
        """Multiple conditions each get their own delta."""
        exp2 = [
            _game([], condition="B", experiment="exp2", game_status="checkmate"),
            _game([], condition="D", experiment="exp2", game_status="forfeit"),
        ]
        exp3 = [
            _game([], condition="B", experiment="exp3", game_status="forfeit"),
            _game([], condition="D", experiment="exp3", game_status="checkmate"),
        ]
        delta = aggregator.compute_gcr_cross_experiment_delta(exp2, exp3)
        assert delta["B"] == pytest.approx(1.0)   # 1.0 - 0.0
        assert delta["D"] == pytest.approx(-1.0)  # 0.0 - 1.0


# ═══════════════════════════════════════════════════════════════════════
# Spearman's ρ (internal helper)
# ═══════════════════════════════════════════════════════════════════════


class TestSpearmanRho:
    def test_perfect_positive(self):
        """Perfect positive correlation → ρ ≈ 1.0."""
        rho = aggregator._spearman_rho(
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [10.0, 20.0, 30.0, 40.0, 50.0],
        )
        assert rho == pytest.approx(1.0)

    def test_perfect_negative(self):
        """Perfect negative correlation → ρ ≈ -1.0."""
        rho = aggregator._spearman_rho(
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [50.0, 40.0, 30.0, 20.0, 10.0],
        )
        assert rho == pytest.approx(-1.0)

    def test_no_correlation(self):
        """Independent data → ρ near 0 (give or take)."""
        rho = aggregator._spearman_rho(
            [1.0, 2.0, 3.0],
            [3.0, 1.0, 2.0],
        )
        # Not necessarily exactly 0, but should be small
        assert abs(rho) < 1.0

    def test_too_few_values(self):
        assert aggregator._spearman_rho([1.0, 2.0], [3.0, 4.0]) == 0.0
