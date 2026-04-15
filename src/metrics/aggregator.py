"""Condition-level and experiment-level metric aggregation.

All functions are **pure** — they take lists of :class:`GameRecord` objects
and return computed values.  No I/O, no side effects, no state.

Metric formulas follow the definitions in §5 of the research plan.
"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from typing import Any

from src.metrics.definitions import GameRecord, TurnRecord


# ═══════════════════════════════════════════════════════════════════════
# RQ1 — Role Separation → Rule Violations
# ═══════════════════════════════════════════════════════════════════════


def compute_fir(games: list[GameRecord]) -> float:
    """Final Invalid Rate: forfeits / total_positions.

    Fraction of positions where the system produced an illegal final output
    (after all retries).  Meaningful only in **Experiment 1** (isolated
    positions).  In full games (Exp 2/3) FIR is degenerate — use GCR and
    MBF for game-level survivability instead.
    """

    turns = _all_turns(games)
    if not turns:
        return 0.0
    forfeits = sum(1 for t in turns if not t.is_valid)
    return forfeits / len(turns)


def compute_mfir(fir_baseline: float, fir_treatment: float) -> float:
    """Marginal FIR Reduction: ``(fir_base − fir_treat) / fir_base``.

    Returns 0.0 if the baseline FIR is 0 (no room for improvement).
    """

    if fir_baseline == 0.0:
        return 0.0
    return (fir_baseline - fir_treatment) / fir_baseline


def compute_phase_stratified_fir(
    games: list[GameRecord],
) -> dict[str, float]:
    """FIR computed within each game-phase stratum.

    Returns ``{"opening": ..., "middlegame": ..., "endgame": ...}``.
    Phases with no turns are omitted.
    """

    phase_buckets: dict[str, list[TurnRecord]] = defaultdict(list)
    for t in _all_turns(games):
        if t.game_phase:
            phase_buckets[t.game_phase].append(t)

    result: dict[str, float] = {}
    for phase, turns in phase_buckets.items():
        forfeits = sum(1 for t in turns if not t.is_valid)
        result[phase] = forfeits / len(turns) if turns else 0.0
    return result


def compute_gcr(games: list[GameRecord]) -> float:
    """Game Completion Rate: non-forfeit games / total games."""

    if not games:
        return 0.0
    completed = sum(1 for g in games if g.gcr_contributed)
    return completed / len(games)


def compute_mbf(games: list[GameRecord]) -> float:
    """Moves Before Forfeit: median legal moves played before first forfeit.

    Only considers games that contain at least one forfeit turn.
    Returns 0.0 if no games forfeited.
    """

    mbf_values: list[int] = []
    for game in games:
        for i, turn in enumerate(game.turns):
            if not turn.is_valid:
                mbf_values.append(i)  # legal moves before this turn
                break

    if not mbf_values:
        return 0.0
    return float(statistics.median(mbf_values))


# ═══════════════════════════════════════════════════════════════════════
# RQ2 — Multi-Turn Consistency
# ═══════════════════════════════════════════════════════════════════════


def compute_ftir(games: list[GameRecord]) -> float:
    """First-Try Invalid Rate: illegal first attempts / total turns."""

    turns = _all_turns(games)
    if not turns:
        return 0.0
    illegal_first = sum(1 for t in turns if not t.first_try_valid)
    return illegal_first / len(turns)


def compute_legality_degradation(
    games: list[GameRecord],
    bin_size: int = 10,
) -> dict[int, float]:
    """FTIR binned by move number.

    Returns ``{0: ftir_for_moves_1_10, 10: ftir_for_moves_11_20, ...}``.
    """

    bins: dict[int, list[bool]] = defaultdict(list)
    for t in _all_turns(games):
        bucket = ((t.move_number - 1) // bin_size) * bin_size
        bins[bucket].append(t.first_try_valid)

    return {
        bucket: sum(1 for v in vals if not v) / len(vals)
        for bucket, vals in sorted(bins.items())
    }


def compute_gcr_cross_experiment_delta(
    exp2_games: list[GameRecord],
    exp3_games: list[GameRecord],
) -> dict[str, float]:
    """``GCR_Exp2 − GCR_Exp3`` per condition.

    Positive values mean FEN mode (Exp 2) had higher game completion than
    history-only mode (Exp 3), i.e. board state helps survivability.

    Assumes game_ids are comparable across experiments (same starting
    positions).  Groups by condition, computes GCR for each, returns delta.
    """

    def _gcr_by_condition(games: list[GameRecord]) -> dict[str, float]:
        cond_groups: dict[str, list[GameRecord]] = defaultdict(list)
        for g in games:
            cond_groups[g.condition].append(g)
        return {c: compute_gcr(gs) for c, gs in cond_groups.items()}

    gcr2 = _gcr_by_condition(exp2_games)
    gcr3 = _gcr_by_condition(exp3_games)

    conditions = set(gcr2) | set(gcr3)
    return {c: gcr2.get(c, 0.0) - gcr3.get(c, 0.0) for c in conditions}


def compute_input_length_error_correlation(
    games: list[GameRecord],
) -> float:
    """Spearman's ρ between ``prompt_token_count`` and error occurrence.

    Uses a simple rank-based computation to avoid a scipy dependency.
    Returns 0.0 if fewer than 3 turns or no variance.
    """

    turns = _all_turns(games)
    if len(turns) < 3:
        return 0.0

    xs = [float(t.prompt_token_count) for t in turns]
    ys = [0.0 if t.first_try_valid else 1.0 for t in turns]

    return _spearman_rho(xs, ys)


def compute_error_type_by_quartile(
    games: list[GameRecord],
) -> dict[str, dict[str, int]]:
    """Error-type frequency broken down by game-progress quartile (Q1–Q4).

    Returns ``{"Q1": {"INVALID_PIECE": 3, ...}, "Q2": {...}, ...}``.
    """

    result: dict[str, dict[str, int]] = {
        "Q1": {},
        "Q2": {},
        "Q3": {},
        "Q4": {},
    }

    for game in games:
        n = len(game.turns)
        if n == 0:
            continue
        for i, turn in enumerate(game.turns):
            quartile_idx = min(int((i / n) * 4), 3)
            q_label = f"Q{quartile_idx + 1}"
            for err in turn.error_types:
                result[q_label][err] = result[q_label].get(err, 0) + 1

    return result


# ═══════════════════════════════════════════════════════════════════════
# RQ3 — Enforcement Strategy Comparison
# ═══════════════════════════════════════════════════════════════════════


def compute_rsr(games: list[GameRecord]) -> float:
    """Retry Success Rate: corrected / initially_invalid.

    A turn is "initially invalid" if ``first_try_valid`` is False.
    It is "corrected" if ``is_valid`` is True despite the first try being
    invalid (i.e. it was fixed via retries).

    Only meaningful for conditions with retries (C, D, E).
    """

    turns = _all_turns(games)
    initially_invalid = [t for t in turns if not t.first_try_valid]
    if not initially_invalid:
        return 0.0
    corrected = sum(1 for t in initially_invalid if t.is_valid)
    return corrected / len(initially_invalid)


def compute_mrtc(games: list[GameRecord]) -> float:
    """Mean Retries To Correct.

    Average ``retry_count`` for turns where correction succeeded
    (first try invalid but final result valid).
    """

    corrected_retries: list[int] = []
    for t in _all_turns(games):
        if not t.first_try_valid and t.is_valid:
            corrected_retries.append(t.retry_count)

    if not corrected_retries:
        return 0.0
    return statistics.mean(corrected_retries)


def compute_pfr(games: list[GameRecord]) -> float:
    """Parse Failure Rate: (PARSE_ERROR + NO_OUTPUT) / total_turns."""

    turns = _all_turns(games)
    if not turns:
        return 0.0
    parse_failures = sum(
        1
        for t in turns
        if any(e in ("PARSE_ERROR", "NO_OUTPUT") for e in t.error_types)
    )
    return parse_failures / len(turns)


def compute_lcpt(games: list[GameRecord]) -> float:
    """LLM Calls Per Turn: total_llm_calls / total_turns."""

    turns = _all_turns(games)
    if not turns:
        return 0.0
    total_calls = sum(t.llm_calls_this_turn for t in turns)
    return total_calls / len(turns)


def compute_tpt(games: list[GameRecord]) -> float:
    """Tokens Per Turn: total_tokens / total_turns."""

    turns = _all_turns(games)
    if not turns:
        return 0.0
    total_tokens = sum(t.tokens_this_turn for t in turns)
    return total_tokens / len(turns)


def compute_mrpt(games: list[GameRecord]) -> float:
    """Mean Retries Per Turn: total_retries / total_turns."""

    turns = _all_turns(games)
    if not turns:
        return 0.0
    total_retries = sum(t.retry_count for t in turns)
    return total_retries / len(turns)


def compute_latency_stats(games: list[GameRecord]) -> dict[str, float]:
    """Latency Per Turn: median, p95, and mean wall-clock ms."""

    latencies = [t.wall_clock_ms for t in _all_turns(games)]
    if not latencies:
        return {"median": 0.0, "p95": 0.0, "mean": 0.0}

    latencies.sort()
    n = len(latencies)
    p95_idx = min(int(n * 0.95), n - 1)

    return {
        "median": float(statistics.median(latencies)),
        "p95": latencies[p95_idx],
        "mean": statistics.mean(latencies),
    }


def compute_cafir(fir: float, lcpt: float) -> float:
    """Cost-Adjusted FIR: ``FIR × LCPT``.

    Only meaningful for **Experiment 1** where FIR is well-defined.
    """

    return fir * lcpt


def compute_critic_confusion_matrix(
    games: list[GameRecord],
) -> dict[str, float]:
    """Critic TPR, FPR, TNR, FNR for Condition C.

    Uses ``critic_verdict`` (the Critic's judgment) and
    ``ground_truth_verdict`` (python-chess result).

    Returns ``{"tpr": ..., "fpr": ..., "tnr": ..., "fnr": ...}``.
    Missing values default to 0.0.
    """

    tp = fp = tn = fn = 0

    for t in _all_turns(games):
        if t.critic_verdict is None or t.ground_truth_verdict is None:
            continue

        actually_valid = t.ground_truth_verdict
        critic_says_valid = t.critic_verdict

        if actually_valid and critic_says_valid:
            tn += 1  # true negative (correctly said "valid" = no error)
        elif actually_valid and not critic_says_valid:
            fp += 1  # false positive (said "invalid" but was valid)
        elif not actually_valid and not critic_says_valid:
            tp += 1  # true positive (correctly caught invalid)
        else:  # not actually_valid and critic_says_valid
            fn += 1  # false negative (said "valid" but was invalid)

    total_positive = tp + fn  # actually invalid moves
    total_negative = tn + fp  # actually valid moves

    return {
        "tpr": tp / total_positive if total_positive else 0.0,
        "fpr": fp / total_negative if total_negative else 0.0,
        "tnr": tn / total_negative if total_negative else 0.0,
        "fnr": fn / total_positive if total_positive else 0.0,
    }


def compute_error_type_frequency(
    games: list[GameRecord],
) -> dict[str, int]:
    """Error-type frequency distribution across all turns."""

    counter: Counter[str] = Counter()
    for t in _all_turns(games):
        counter.update(t.error_types)
    return dict(counter)


def compute_error_type_rsr(
    games: list[GameRecord],
) -> dict[str, float]:
    """RSR broken down by error type.

    For each error type, computes: of turns whose *first* error was that type
    and ``first_try_valid`` was False, what fraction ended up valid?
    """

    by_type: dict[str, dict[str, int]] = defaultdict(
        lambda: {"invalid": 0, "corrected": 0}
    )

    for t in _all_turns(games):
        if t.first_try_valid or not t.error_types:
            continue
        first_error = t.error_types[0]
        by_type[first_error]["invalid"] += 1
        if t.is_valid:
            by_type[first_error]["corrected"] += 1

    return {
        err: (counts["corrected"] / counts["invalid"] if counts["invalid"] else 0.0)
        for err, counts in by_type.items()
    }


# ═══════════════════════════════════════════════════════════════════════
# Condition F — ReAct + Tools
# ═══════════════════════════════════════════════════════════════════════


def compute_tcr(games: list[GameRecord]) -> float:
    """Tool Call Rate: fraction of turns with at least one tool call."""

    turns = _all_turns(games)
    if not turns:
        return 0.0
    with_tools = sum(1 for t in turns if t.tool_calls)
    return with_tools / len(turns)


def compute_vta(games: list[GameRecord]) -> float:
    """Validation Tool Adoption: fraction of turns where ``validate_move``
    was called before submission."""

    turns = _all_turns(games)
    if not turns:
        return 0.0
    with_validate = sum(
        1
        for t in turns
        if any(
            tc.get("tool") == "validate_move" or tc.get("name") == "validate_move"
            for tc in t.tool_calls
        )
    )
    return with_validate / len(turns)


def compute_tool_distribution(games: list[GameRecord]) -> dict[str, int]:
    """Tool-call frequency by tool name."""

    counter: Counter[str] = Counter()
    for t in _all_turns(games):
        for tc in t.tool_calls:
            name = tc.get("tool") or tc.get("name") or "unknown"
            counter[name] += 1
    return dict(counter)


def compute_tool_stratified_fir(
    games: list[GameRecord],
) -> dict[str, float]:
    """FIR split by tool-use / no-tool-use strata.

    Returns ``{"with_tools": ..., "without_tools": ...}``.
    """

    with_tools: list[TurnRecord] = []
    without_tools: list[TurnRecord] = []

    for t in _all_turns(games):
        if t.tool_calls:
            with_tools.append(t)
        else:
            without_tools.append(t)

    return {
        "with_tools": (
            sum(1 for t in with_tools if not t.is_valid) / len(with_tools)
            if with_tools
            else 0.0
        ),
        "without_tools": (
            sum(1 for t in without_tools if not t.is_valid) / len(without_tools)
            if without_tools
            else 0.0
        ),
    }


def compute_avg_reasoning_steps(games: list[GameRecord]) -> float:
    """Mean reasoning steps per turn (uses ``llm_calls_this_turn`` as proxy)."""

    turns = _all_turns(games)
    if not turns:
        return 0.0
    return statistics.mean(t.llm_calls_this_turn for t in turns)


def compute_scr(games: list[GameRecord]) -> float:
    """Self-Correction Rate: of turns where ``validate_move`` flagged invalid
    during reasoning, fraction that resulted in a valid final submission.

    Scans tool_calls for ``validate_move`` results with ``legal=False``.
    """

    flagged_invalid = 0
    self_corrected = 0

    for t in _all_turns(games):
        had_invalid_validation = any(
            (tc.get("tool") == "validate_move" or tc.get("name") == "validate_move")
            and tc.get("result", {}).get("legal") is False
            for tc in t.tool_calls
        )
        if had_invalid_validation:
            flagged_invalid += 1
            if t.is_valid:
                self_corrected += 1

    if flagged_invalid == 0:
        return 0.0
    return self_corrected / flagged_invalid


# ═══════════════════════════════════════════════════════════════════════
# Aggregate helper
# ═══════════════════════════════════════════════════════════════════════


_RETRY_CONDITIONS = {"C", "D", "E"}
_CRITIC_CONDITIONS = {"C"}
_TOOL_CONDITIONS = {"F"}


def compute_all_metrics(
    games: list[GameRecord],
    condition: str,
    experiment: str,
) -> dict[str, Any]:
    """Compute all applicable metrics for a condition+experiment pair.

    Automatically includes/excludes metrics based on condition and experiment.
    Returns a flat dict of ``metric_name → value``.
    """

    if not games:
        return {}

    lcpt = compute_lcpt(games)

    result: dict[str, Any] = {
        # ── RQ2 (all experiments) ──
        "ftir": compute_ftir(games),
        # ── RQ3 effectiveness (all experiments) ──
        "pfr": compute_pfr(games),
        # ── RQ3 cost (all experiments) ──
        "lcpt": lcpt,
        "tpt": compute_tpt(games),
        "latency": compute_latency_stats(games),
        # ── RQ3 error types (all experiments) ──
        "error_type_frequency": compute_error_type_frequency(games),
    }

    # ── Exp 1 only: FIR-based metrics (degenerate in full games) ──
    if experiment == "exp1":
        fir = compute_fir(games)
        result["fir"] = fir
        result["phase_stratified_fir"] = compute_phase_stratified_fir(games)
        result["cafir"] = compute_cafir(fir, lcpt)

    # ── Exp 2 & 3 only: game-level survivability ──
    if experiment in ("exp2", "exp3"):
        result["gcr"] = compute_gcr(games)
        result["mbf"] = compute_mbf(games)
        result["legality_degradation"] = compute_legality_degradation(games)
        result["error_type_by_quartile"] = compute_error_type_by_quartile(games)

    # Retry metrics (C, D, E only)
    if condition in _RETRY_CONDITIONS:
        result["rsr"] = compute_rsr(games)
        result["mrtc"] = compute_mrtc(games)
        result["mrpt"] = compute_mrpt(games)
        result["error_type_rsr"] = compute_error_type_rsr(games)

    # Critic metrics (C only)
    if condition in _CRITIC_CONDITIONS:
        result["critic_confusion_matrix"] = compute_critic_confusion_matrix(games)

    # Tool metrics (F only)
    if condition in _TOOL_CONDITIONS:
        result["tcr"] = compute_tcr(games)
        result["vta"] = compute_vta(games)
        result["tool_distribution"] = compute_tool_distribution(games)
        result["tool_stratified_fir"] = compute_tool_stratified_fir(games)
        result["avg_reasoning_steps"] = compute_avg_reasoning_steps(games)
        result["scr"] = compute_scr(games)

    return result


# ═══════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════


def _all_turns(games: list[GameRecord]) -> list[TurnRecord]:
    """Flatten all turns from a list of games."""

    return [t for g in games for t in g.turns]


def _rank(values: list[float]) -> list[float]:
    """Assign average ranks to values (for Spearman's ρ)."""

    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)

    i = 0
    while i < len(indexed):
        j = i
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1  # 1-indexed average rank
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j

    return ranks


def _spearman_rho(xs: list[float], ys: list[float]) -> float:
    """Compute Spearman's rank correlation coefficient."""

    if len(xs) != len(ys) or len(xs) < 3:
        return 0.0

    rx = _rank(xs)
    ry = _rank(ys)

    n = len(rx)
    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n

    cov = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    std_rx = (sum((r - mean_rx) ** 2 for r in rx)) ** 0.5
    std_ry = (sum((r - mean_ry) ** 2 for r in ry)) ** 0.5

    if std_rx == 0 or std_ry == 0:
        return 0.0

    return cov / (std_rx * std_ry)
