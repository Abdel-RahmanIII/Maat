"""Multi-turn consistency metrics for Experiments 2 & 3.

These functions operate on ordered sequences of
:class:`~src.metrics.definitions.TurnRecord` within individual games.
They capture error recurrence, clustering, and degradation patterns.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from src.metrics.definitions import (
    LegalityDegradationBin,
    QuartileErrorDist,
    TurnRecord,
)


# ── SERR ─────────────────────────────────────────────────────────────────


def compute_serr(turns: list[TurnRecord]) -> float:
    """Same-Error Recurrence Rate.

    Fraction of distinct error types that occur more than once in the
    game.  Returns 0.0 if no errors occurred.

    Parameters
    ----------
    turns:
        Ordered list of turn records for a single game.
    """

    error_counter: Counter[str] = Counter()
    for t in turns:
        for err in t.error_types:
            error_counter[err] += 1

    if not error_counter:
        return 0.0

    distinct_types = len(error_counter)
    recurring = sum(1 for count in error_counter.values() if count > 1)
    return recurring / distinct_types


# ── PCRR ─────────────────────────────────────────────────────────────────


def compute_pcrr(turns: list[TurnRecord]) -> float | None:
    """Post-Correction Recurrence Rate.

    Frequency of repeating the *same error type* after it was
    successfully corrected (i.e. first try invalid but final move valid).

    Returns ``None`` if no correction events occurred (e.g. conditions
    without retry or no errors).

    Parameters
    ----------
    turns:
        Ordered list of turn records for a single game.
    """

    # Find correction events: turns where first_try_valid=False, is_valid=True
    correction_events: list[tuple[int, set[str]]] = []
    for idx, t in enumerate(turns):
        if not t.first_try_valid and t.is_valid and t.error_types:
            corrected_types = set(t.error_types)
            correction_events.append((idx, corrected_types))

    if not correction_events:
        return None

    recurrences = 0
    total_corrections = 0

    for correction_idx, corrected_types in correction_events:
        total_corrections += 1
        # Check subsequent turns for same error types
        for later_turn in turns[correction_idx + 1:]:
            later_errors = set(later_turn.error_types)
            if corrected_types & later_errors:
                recurrences += 1
                break  # Count once per correction event

    return recurrences / total_corrections


# ── TTR ──────────────────────────────────────────────────────────────────


def compute_ttr(turns: list[TurnRecord]) -> list[int]:
    """Turns-to-Recovery: clean moves following each correction event.

    For each turn where a correction succeeded (first try invalid →
    final valid), count the number of consecutive *first-try valid*
    turns immediately following.

    Parameters
    ----------
    turns:
        Ordered list of turn records for a single game.

    Returns
    -------
    list[int]
        One entry per correction event.  Empty list if no corrections.
    """

    recovery_lengths: list[int] = []

    for idx, t in enumerate(turns):
        if not t.first_try_valid and t.is_valid:
            # Correction event — count clean streak after
            clean_count = 0
            for later in turns[idx + 1:]:
                if later.first_try_valid:
                    clean_count += 1
                else:
                    break
            recovery_lengths.append(clean_count)

    return recovery_lengths


# ── ECC ──────────────────────────────────────────────────────────────────


def compute_population_ftir_by_turn(
    all_games_turns: list[list[TurnRecord]],
) -> dict[int, float]:
    """Compute FTIR(t) pooled across all games at each turn index.

    This is the population-level first-try error rate at turn position
    ``t``, used as the time-varying baseline for ECC.

    Parameters
    ----------
    all_games_turns:
        List of per-game turn record lists (all from the same condition).

    Returns
    -------
    dict[int, float]
        Mapping from turn index (0-based) to FTIR at that position.
    """

    # Collect error indicators per turn index
    errors_at: dict[int, list[bool]] = {}
    for game_turns in all_games_turns:
        for idx, t in enumerate(game_turns):
            errors_at.setdefault(idx, []).append(not t.first_try_valid)

    return {
        idx: sum(flags) / len(flags) if flags else 0.0
        for idx, flags in errors_at.items()
    }


def compute_ecc(
    turns: list[TurnRecord],
    population_ftir: dict[int, float],
) -> float | None:
    """Error Clustering Coefficient for one game.

    Uses the time-varying formulation:

    .. code-block:: text

        ECC = observed_consecutive_error_pairs /
              Σ FTIR(t) × FTIR(t+1)  for t = 0..T-2

    where FTIR(t) is the *population-level* rate from
    :func:`compute_population_ftir_by_turn`.

    Returns ``None`` if the expected denominator is zero (no expected
    error pairs at all).

    Parameters
    ----------
    turns:
        Ordered list of turn records for a single game.
    population_ftir:
        Population-level FTIR by turn index.
    """

    if len(turns) < 2:
        return None

    # Observed consecutive error pairs
    observed = 0
    for i in range(len(turns) - 1):
        t_err = not turns[i].first_try_valid
        t1_err = not turns[i + 1].first_try_valid
        if t_err and t1_err:
            observed += 1

    # Expected consecutive error pairs (time-varying baseline)
    expected = 0.0
    for i in range(len(turns) - 1):
        ftir_t = population_ftir.get(i, 0.0)
        ftir_t1 = population_ftir.get(i + 1, 0.0)
        expected += ftir_t * ftir_t1

    if expected == 0.0:
        return None

    return observed / expected


# ── Legality Degradation Curve ───────────────────────────────────────────


def compute_legality_degradation(
    turns: list[TurnRecord],
    bin_size: int = 10,
) -> list[LegalityDegradationBin]:
    """FTIR plotted in N-move bins over game progress.

    Groups turns by ``move_number`` into bins of ``bin_size`` and computes
    FTIR within each bin.

    Parameters
    ----------
    turns:
        Turns from one or more games (pooled for a condition).
    bin_size:
        Width of each bin in move numbers.

    Returns
    -------
    list[LegalityDegradationBin]
        Ordered list of bins with FTIR and turn count.
    """

    if not turns:
        return []

    max_move = max(t.move_number for t in turns)
    bins: list[LegalityDegradationBin] = []

    start = 1
    while start <= max_move:
        end = start + bin_size - 1
        bin_turns = [t for t in turns if start <= t.move_number <= end]
        if bin_turns:
            ftir = sum(1 for t in bin_turns if not t.first_try_valid) / len(bin_turns)
            bins.append(
                LegalityDegradationBin(
                    bin_start=start,
                    bin_end=end,
                    ftir=ftir,
                    n_turns=len(bin_turns),
                )
            )
        start += bin_size

    return bins


# ── Input Length vs. Error ───────────────────────────────────────────────


def compute_input_length_vs_error(
    turns: list[TurnRecord],
) -> list[dict[str, Any]]:
    """Per-turn (prompt_token_count, is_error) pairs.

    Returns a list of dicts suitable for Spearman's ρ / partial
    correlation in the analysis layer.
    """

    return [
        {
            "move_number": t.move_number,
            "prompt_token_count": t.prompt_token_count,
            "is_error": not t.first_try_valid,
            "error_types": list(t.error_types),
        }
        for t in turns
    ]


# ── Error-Type Distribution over Quartiles ───────────────────────────────


def compute_error_type_over_quartiles(
    turns: list[TurnRecord],
    total_game_moves: int,
) -> list[QuartileErrorDist]:
    """Error taxonomy frequency by turn quartile (Q1–Q4).

    Assigns each turn to a quartile based on its ``move_number`` relative
    to ``total_game_moves``, then counts error-type occurrences within
    each quartile.

    Parameters
    ----------
    turns:
        Turns from one or more games.
    total_game_moves:
        Maximum move number (used to define quartile boundaries).
    """

    if total_game_moves <= 0 or not turns:
        return []

    quarter = total_game_moves / 4.0

    def _quartile_label(move_num: int) -> str:
        if move_num <= quarter:
            return "Q1"
        if move_num <= 2 * quarter:
            return "Q2"
        if move_num <= 3 * quarter:
            return "Q3"
        return "Q4"

    quartile_errors: dict[str, Counter[str]] = {
        "Q1": Counter(),
        "Q2": Counter(),
        "Q3": Counter(),
        "Q4": Counter(),
    }

    for t in turns:
        q = _quartile_label(t.move_number)
        for err in t.error_types:
            quartile_errors[q][err] += 1

    results: list[QuartileErrorDist] = []
    for q_label in ("Q1", "Q2", "Q3", "Q4"):
        counts = dict(quartile_errors[q_label])
        results.append(
            QuartileErrorDist(
                quartile=q_label,
                counts=counts,
                total_errors=sum(counts.values()),
            )
        )
    return results
