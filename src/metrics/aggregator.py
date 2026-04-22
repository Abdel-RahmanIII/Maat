"""Aggregate metric computation for Experiments 1, 2, and 3.

All public functions are **pure** — they accept lists of
:class:`~src.metrics.definitions.TurnRecord` or
:class:`~src.metrics.definitions.GameRecord` and return computed values.
No side effects, no state mutation.

Statistical tests are intentionally excluded (deferred to ``analysis/``).
"""

from __future__ import annotations

import statistics
from collections import Counter
from typing import Any

from src.error_taxonomy import PARSER_ERROR_TYPES, ErrorType
from src.metrics.definitions import (
    ConditionMetrics,
    CriticAccuracy,
    DescriptiveStats,
    ErrorTypeRSR,
    FIRDeltaEntry,
    FSTEntry,
    GameRecord,
    PhaseStratifiedFIR,
    ToolCallDistribution,
    TurnRecord,
)
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

# ── Helper constants ─────────────────────────────────────────────────────

_RETRY_CONDITIONS = {"C", "D", "E"}
_TOOL_CONDITIONS = {"F"}
_CAFIR_EXCLUDED = {"A", "B"}
_CRITIC_CONDITIONS = {"C"}

_PARSE_ERROR_NAMES = {e.value for e in PARSER_ERROR_TYPES}


# ── Rate metrics ─────────────────────────────────────────────────────────


def compute_fir(turns: list[TurnRecord]) -> float:
    """Final Invalid Rate: fraction of turns ending in an invalid move.

    A turn is invalid when ``is_valid`` is ``False`` after all retries.
    """
    if not turns:
        return 0.0
    invalid = sum(1 for t in turns if not t.is_valid)
    return invalid / len(turns)


def compute_ftir(turns: list[TurnRecord]) -> float:
    """First-Try Invalid Rate: fraction where the first attempt was illegal."""
    if not turns:
        return 0.0
    first_invalid = sum(1 for t in turns if not t.first_try_valid)
    return first_invalid / len(turns)


def compute_mfir(fir_baseline: float, fir_treatment: float) -> float | None:
    """Marginal FIR Reduction.

    Returns ``None`` if ``fir_baseline`` is 0 (division by zero).
    """
    if fir_baseline == 0.0:
        return None
    return (fir_baseline - fir_treatment) / fir_baseline


def compute_arr(fir_baseline: float, fir_treatment: float) -> float:
    """Absolute Risk Reduction."""
    return fir_baseline - fir_treatment


# ── Phase-stratified FIR ─────────────────────────────────────────────────


def compute_phase_stratified_fir(turns: list[TurnRecord]) -> PhaseStratifiedFIR:
    """FIR broken down by opening / middlegame / endgame."""

    buckets: dict[str, list[TurnRecord]] = {
        "opening": [],
        "middlegame": [],
        "endgame": [],
    }

    for t in turns:
        phase = t.game_phase.lower()
        if phase in buckets:
            buckets[phase].append(t)

    return PhaseStratifiedFIR(
        opening=compute_fir(buckets["opening"]) if buckets["opening"] else None,
        middlegame=compute_fir(buckets["middlegame"]) if buckets["middlegame"] else None,
        endgame=compute_fir(buckets["endgame"]) if buckets["endgame"] else None,
    )


# ── Parse failures ───────────────────────────────────────────────────────


def compute_parse_failure_counts(turns: list[TurnRecord]) -> int:
    """Count of PARSE_ERROR + NO_OUTPUT across all turns and attempts."""
    count = 0
    for t in turns:
        for err in t.error_types:
            if err in _PARSE_ERROR_NAMES:
                count += 1
    return count


# ── Retry metrics (C, D, E) ─────────────────────────────────────────────


def compute_rsr(turns: list[TurnRecord]) -> float | None:
    """Retry Success Rate: of initially-invalid moves, fraction corrected.

    Returns ``None`` if there are no initially-invalid turns.
    """
    initially_invalid = [t for t in turns if not t.first_try_valid]
    if not initially_invalid:
        return None
    corrected = sum(1 for t in initially_invalid if t.is_valid)
    return corrected / len(initially_invalid)


def compute_mrtc(turns: list[TurnRecord]) -> float | None:
    """Mean Retries To Correct.

    Average retry count for turns where correction succeeded.
    Returns ``None`` if no correction events occurred.
    """
    corrected = [t for t in turns if not t.first_try_valid and t.is_valid]
    if not corrected:
        return None
    return statistics.mean(t.retry_count for t in corrected)


# ── Cost metrics ─────────────────────────────────────────────────────────


def _descriptive_stats(values: list[float | int]) -> DescriptiveStats:
    """Compute descriptive statistics for a list of numeric values."""
    if not values:
        return DescriptiveStats()
    float_vals = [float(v) for v in values]
    return DescriptiveStats(
        mean=statistics.mean(float_vals),
        median=statistics.median(float_vals),
        min=min(float_vals),
        max=max(float_vals),
        std=statistics.stdev(float_vals) if len(float_vals) >= 2 else 0.0,
        n=len(float_vals),
    )


def compute_lcpt(turns: list[TurnRecord]) -> DescriptiveStats:
    """LLM Calls Per Turn — descriptive statistics."""
    return _descriptive_stats([t.llm_calls_this_turn for t in turns])


def compute_tpt(turns: list[TurnRecord]) -> DescriptiveStats:
    """Tokens Per Turn — descriptive statistics."""
    return _descriptive_stats([t.tokens_this_turn for t in turns])


def compute_cafir(fir: float, mean_lcpt: float) -> float:
    """Cost-Adjusted FIR: ``FIR × mean_LCPT``."""
    return fir * mean_lcpt


# ── Critic accuracy (C only) ────────────────────────────────────────────


def compute_critic_accuracy(turns: list[TurnRecord]) -> CriticAccuracy:
    """Confusion matrix for the LLM Critic vs ground truth.

    Only considers turns where both ``critic_verdict`` and
    ``ground_truth_verdict`` are not None.
    """

    tp = fp = tn = fn = 0

    for t in turns:
        if t.critic_verdict is None or t.ground_truth_verdict is None:
            continue

        critic_says_valid = t.critic_verdict
        actually_valid = t.ground_truth_verdict

        if critic_says_valid and actually_valid:
            tn += 1  # Critic says valid, actually valid (correct pass)
        elif critic_says_valid and not actually_valid:
            fn += 1  # Critic says valid, actually invalid (miss)
        elif not critic_says_valid and not actually_valid:
            tp += 1  # Critic says invalid, actually invalid (correct catch)
        else:  # not critic_says_valid and actually_valid
            fp += 1  # Critic says invalid, actually valid (false alarm)

    total_positive = tp + fn  # Actually invalid
    total_negative = tn + fp  # Actually valid

    return CriticAccuracy(
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
        tpr=tp / total_positive if total_positive > 0 else 0.0,
        fpr=fp / total_negative if total_negative > 0 else 0.0,
        tnr=tn / total_negative if total_negative > 0 else 0.0,
        fnr=fn / total_positive if total_positive > 0 else 0.0,
    )


# ── Error-type RSR (C, D, E) ────────────────────────────────────────────


def compute_error_type_rsr(turns: list[TurnRecord]) -> ErrorTypeRSR:
    """RSR per error type per condition (for heatmap).

    For each error type that appeared as a first-try error, compute:
    corrected / total_initially_invalid_with_that_error.
    """

    # Group turns by their first error type
    error_groups: dict[str, list[TurnRecord]] = {}
    for t in turns:
        if t.first_try_valid or not t.error_types:
            continue
        first_error = t.error_types[0]
        error_groups.setdefault(first_error, []).append(t)

    rsr_by_type: dict[str, float] = {}
    counts_by_type: dict[str, tuple[int, int]] = {}

    for err_type, group in error_groups.items():
        corrected = sum(1 for t in group if t.is_valid)
        total = len(group)
        rsr_by_type[err_type] = corrected / total if total > 0 else 0.0
        counts_by_type[err_type] = (corrected, total)

    return ErrorTypeRSR(
        rsr_by_type=rsr_by_type,
        counts_by_type=counts_by_type,
    )


# ── Condition F agent behaviour ──────────────────────────────────────────


def compute_vta(turns: list[TurnRecord]) -> float:
    """Validation Tool Adoption: fraction of turns using ``validate_move``."""
    if not turns:
        return 0.0
    validate_turns = sum(
        1 for t in turns
        if any(
            tc.get("tool_name") == "validate_move" or tc.get("name") == "validate_move"
            for tc in t.tool_calls
        )
    )
    return validate_turns / len(turns)


def compute_tcr(turns: list[TurnRecord]) -> float:
    """Tool Call Rate: fraction of turns with at least one tool call."""
    if not turns:
        return 0.0
    tool_turns = sum(1 for t in turns if t.tool_calls)
    return tool_turns / len(turns)


def compute_tool_call_distribution(turns: list[TurnRecord]) -> ToolCallDistribution:
    """Frequency breakdown per tool type."""
    counts: Counter[str] = Counter()
    for t in turns:
        for tc in t.tool_calls:
            tool_name = tc.get("tool_name") or tc.get("name") or "unknown"
            counts[tool_name] += 1
    return ToolCallDistribution(
        counts=dict(counts),
        total_tool_calls=sum(counts.values()),
    )


def compute_tool_stratified_fir(turns: list[TurnRecord]) -> dict[str, float]:
    """FIR split by whether tools were used.

    Returns ``{"with_tools": ..., "without_tools": ...}``.
    """
    with_tools = [t for t in turns if t.tool_calls]
    without_tools = [t for t in turns if not t.tool_calls]

    return {
        "with_tools": compute_fir(with_tools),
        "without_tools": compute_fir(without_tools),
    }


def compute_avg_reasoning_steps(turns: list[TurnRecord]) -> float:
    """Mean tool calls (think/act cycles) per turn."""
    if not turns:
        return 0.0
    return statistics.mean(len(t.tool_calls) for t in turns)


# ── Game-level metrics (Exp 2/3) ─────────────────────────────────────────


def compute_game_fir(games: list[GameRecord]) -> float:
    """FIR across all turns in all games."""
    all_turns = [t for g in games for t in g.turns]
    return compute_fir(all_turns)


def compute_imfr(games: list[GameRecord]) -> float:
    """Illegal-Move Forfeit Rate: fraction of games lost to a rule violation."""
    if not games:
        return 0.0
    forfeit_games = sum(1 for g in games if g.final_status == "forfeit")
    return forfeit_games / len(games)


def compute_fst_data(games: list[GameRecord]) -> list[FSTEntry]:
    """Forfeit Survival Time data for Kaplan-Meier analysis.

    For each game, records the number of half-moves before the game
    ended. Games ending in forfeit are uncensored events; games reaching
    natural termination or the move cap are right-censored.
    """
    entries: list[FSTEntry] = []
    for g in games:
        censored = g.final_status != "forfeit"
        entries.append(
            FSTEntry(
                game_id=g.game_id,
                half_moves=g.total_turns,
                censored=censored,
            )
        )
    return entries


def compute_fir_cross_experiment_delta(
    exp2_games: list[GameRecord],
    exp3_games: list[GameRecord],
) -> list[FIRDeltaEntry]:
    """Per-position FIR delta between Exp 2 and Exp 3.

    Games are paired by ``starting_fen``.  For each pair, computes the
    per-game FIR and reports the delta (Exp 3 − Exp 2).
    """

    # Index games by (starting_fen, condition)
    def _game_fir(game: GameRecord) -> float:
        if not game.turns:
            return 0.0
        return compute_fir(game.turns)

    exp2_by_key: dict[tuple[str, str], float] = {}
    for g in exp2_games:
        key = (g.starting_fen, g.condition)
        exp2_by_key[key] = _game_fir(g)

    entries: list[FIRDeltaEntry] = []
    for g in exp3_games:
        key = (g.starting_fen, g.condition)
        if key in exp2_by_key:
            fir2 = exp2_by_key[key]
            fir3 = _game_fir(g)
            entries.append(
                FIRDeltaEntry(
                    starting_fen=g.starting_fen,
                    condition=g.condition,
                    fir_exp2=fir2,
                    fir_exp3=fir3,
                    delta=fir3 - fir2,
                )
            )
    return entries


# ── Top-level dispatchers ────────────────────────────────────────────────


def compute_all_exp1_metrics(
    turn_records: list[TurnRecord],
    condition: str,
) -> ConditionMetrics:
    """Compute all applicable Experiment 1 metrics for one condition.

    Metrics are scope-aware: RSR/MRTC only for C/D/E, critic accuracy
    only for C, tool metrics only for F, CAFIR excludes A/B.
    """

    fir = compute_fir(turn_records)
    ftir = compute_ftir(turn_records)
    lcpt_stats = compute_lcpt(turn_records)
    tpt_stats = compute_tpt(turn_records)

    result = ConditionMetrics(
        condition=condition,
        experiment=1,
        fir=fir,
        ftir=ftir,
        phase_stratified_fir=compute_phase_stratified_fir(turn_records),
        parse_failure_count=compute_parse_failure_counts(turn_records),
        lcpt=lcpt_stats,
        tpt=tpt_stats,
    )

    # Retry metrics (C, D, E)
    if condition in _RETRY_CONDITIONS:
        result.rsr = compute_rsr(turn_records)
        result.mrtc = compute_mrtc(turn_records)
        result.error_type_rsr = compute_error_type_rsr(turn_records)

    # CAFIR (excludes A, B)
    if condition not in _CAFIR_EXCLUDED:
        result.cafir = compute_cafir(fir, lcpt_stats.mean)

    # Critic accuracy (C only)
    if condition in _CRITIC_CONDITIONS:
        result.critic_accuracy = compute_critic_accuracy(turn_records)

    # Condition F agent behaviour
    if condition in _TOOL_CONDITIONS:
        result.vta = compute_vta(turn_records)
        result.tcr = compute_tcr(turn_records)
        result.tool_call_distribution = compute_tool_call_distribution(turn_records)
        result.tool_stratified_fir = compute_tool_stratified_fir(turn_records)
        result.avg_reasoning_steps = compute_avg_reasoning_steps(turn_records)

    return result


def compute_all_game_metrics(
    game_records: list[GameRecord],
    condition: str,
) -> ConditionMetrics:
    """Compute all applicable Experiment 2/3 metrics for one condition.

    Includes both turn-level and game-level metrics, plus multi-turn
    consistency metrics from the recurrence module.
    """

    all_turns = [t for g in game_records for t in g.turns]
    experiment = game_records[0].experiment if game_records else 2

    fir = compute_fir(all_turns)
    ftir = compute_ftir(all_turns)
    lcpt_stats = compute_lcpt(all_turns)
    tpt_stats = compute_tpt(all_turns)

    result = ConditionMetrics(
        condition=condition,
        experiment=experiment,
        fir=fir,
        ftir=ftir,
        parse_failure_count=compute_parse_failure_counts(all_turns),
        lcpt=lcpt_stats,
        tpt=tpt_stats,
        imfr=compute_imfr(game_records),
        fst_data=compute_fst_data(game_records),
    )

    # Retry metrics (if applicable)
    if condition in _RETRY_CONDITIONS:
        result.rsr = compute_rsr(all_turns)
        result.mrtc = compute_mrtc(all_turns)

    # CAFIR (if applicable)
    if condition not in _CAFIR_EXCLUDED:
        result.cafir = compute_cafir(fir, lcpt_stats.mean)

    # Multi-turn consistency metrics (per-game, then averaged)
    all_game_turns = [g.turns for g in game_records]

    # SERR — average across games
    serr_values = [compute_serr(gt) for gt in all_game_turns if gt]
    if serr_values:
        result.serr = statistics.mean(serr_values)

    # PCRR (retry conditions only)
    if condition in _RETRY_CONDITIONS:
        pcrr_values = [compute_pcrr(gt) for gt in all_game_turns if gt]
        pcrr_valid = [v for v in pcrr_values if v is not None]
        if pcrr_valid:
            result.pcrr = statistics.mean(pcrr_valid)

        # TTR
        ttr_all: list[int] = []
        for gt in all_game_turns:
            ttr_all.extend(compute_ttr(gt))
        if ttr_all:
            result.ttr_values = ttr_all

    # ECC
    pop_ftir = compute_population_ftir_by_turn(all_game_turns)
    ecc_values = [
        compute_ecc(gt, pop_ftir) for gt in all_game_turns if gt
    ]
    ecc_valid = [v for v in ecc_values if v is not None]
    if ecc_valid:
        result.ecc = statistics.mean(ecc_valid)

    # Legality degradation (pooled across games)
    result.legality_degradation = compute_legality_degradation(all_turns)

    # Input length vs error
    result.input_length_vs_error = compute_input_length_vs_error(all_turns)

    # Error-type over quartiles
    max_turns_in_game = max((len(gt) for gt in all_game_turns), default=0)
    if max_turns_in_game > 0:
        result.error_type_over_quartiles = compute_error_type_over_quartiles(
            all_turns, max_turns_in_game
        )

    return result
