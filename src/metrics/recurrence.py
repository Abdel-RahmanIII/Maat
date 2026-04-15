"""Cross-turn recurrence and clustering metrics.

These metrics require looking *across* turns within a single game,
which is why they're separated from the single-turn aggregator.

Metrics
-------
- **SERR** — Same-Error Recurrence Rate
- **PCRR** — Post-Correction Recurrence Rate
- **ECC**  — Error Clustering Coefficient
"""

from __future__ import annotations

import statistics
from collections import Counter
from typing import Any

from src.metrics.definitions import GameRecord


# ═══════════════════════════════════════════════════════════════════════
# Per-game metrics
# ═══════════════════════════════════════════════════════════════════════


def compute_serr(game: GameRecord) -> float | None:
    """Same-Error Recurrence Rate for one game.

    ``recurrent_error_types / total_distinct_error_types``

    An error type is "recurrent" if it appears in more than one turn.
    Returns ``None`` if no errors occurred.
    """

    # Collect error types per turn, then count how many distinct types
    # appear in more than one turn.
    type_turn_counts: Counter[str] = Counter()
    for turn in game.turns:
        # Use a set to count each type at most once per turn
        unique_this_turn = set(turn.error_types)
        type_turn_counts.update(unique_this_turn)

    if not type_turn_counts:
        return None

    total_types = len(type_turn_counts)
    recurrent_types = sum(1 for count in type_turn_counts.values() if count > 1)

    return recurrent_types / total_types


def compute_pcrr(game: GameRecord) -> float | None:
    """Post-Correction Recurrence Rate for one game.

    After an error is corrected via retry (``first_try_valid=False`` and
    ``is_valid=True``), how often does the *same* error type recur in
    any subsequent turn?

    Returns ``None`` for conditions without retries (A, B) or if no
    corrections occurred.
    """

    corrections: list[tuple[int, set[str]]] = []

    for i, turn in enumerate(game.turns):
        if not turn.first_try_valid and turn.is_valid and turn.error_types:
            # This turn had an error that was corrected
            corrected_types = set(turn.error_types)
            corrections.append((i, corrected_types))

    if not corrections:
        return None

    total_corrections = 0
    recurrences = 0

    for turn_idx, corrected_types in corrections:
        for err_type in corrected_types:
            total_corrections += 1
            # Check if this error type occurs in any later turn
            for later_turn in game.turns[turn_idx + 1 :]:
                if err_type in later_turn.error_types:
                    recurrences += 1
                    break  # Count once per correction event

    if total_corrections == 0:
        return None

    return recurrences / total_corrections


def compute_ecc(game: GameRecord) -> float | None:
    """Error Clustering Coefficient for one game.

    ``observed_consecutive_error_pairs / ((total_turns − 1) × FTIR²)``

    - ECC > 1 → errors cluster (one error makes the next more likely)
    - ECC ≈ 1 → errors are independent (Bernoulli)
    - ECC < 1 → errors anti-cluster (correction effect)

    Returns ``None`` if fewer than 2 turns or FTIR is 0.
    """

    turns = game.turns
    n = len(turns)

    if n < 2:
        return None

    # FTIR for this game
    first_try_invalid = sum(1 for t in turns if not t.first_try_valid)
    ftir = first_try_invalid / n

    if ftir == 0.0:
        return None  # No errors → ECC is undefined

    # Count consecutive pairs where both turns had first-try errors
    observed_pairs = 0
    for i in range(n - 1):
        if not turns[i].first_try_valid and not turns[i + 1].first_try_valid:
            observed_pairs += 1

    expected = (n - 1) * (ftir ** 2)

    if expected == 0.0:
        return None

    return observed_pairs / expected


# ═══════════════════════════════════════════════════════════════════════
# Aggregate across games
# ═══════════════════════════════════════════════════════════════════════


def compute_recurrence_metrics(
    games: list[GameRecord],
) -> dict[str, Any]:
    """Aggregate SERR, PCRR, ECC across all games.

    Returns mean values plus per-game lists for downstream analysis.
    ``None`` values from individual games are excluded from averages.
    """

    serr_values: list[float] = []
    pcrr_values: list[float] = []
    ecc_values: list[float] = []

    for game in games:
        s = compute_serr(game)
        if s is not None:
            serr_values.append(s)

        p = compute_pcrr(game)
        if p is not None:
            pcrr_values.append(p)

        e = compute_ecc(game)
        if e is not None:
            ecc_values.append(e)

    return {
        "serr_mean": statistics.mean(serr_values) if serr_values else None,
        "serr_values": serr_values,
        "pcrr_mean": statistics.mean(pcrr_values) if pcrr_values else None,
        "pcrr_values": pcrr_values,
        "ecc_mean": statistics.mean(ecc_values) if ecc_values else None,
        "ecc_values": ecc_values,
    }
