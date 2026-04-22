"""Pydantic data models for the Maat metrics layer.

Every metric record produced by the collector and aggregator is expressed
as a Pydantic v2 model, giving typed fields, free JSON serialization
(``model_dump_json`` / ``model_validate_json``), and runtime validation.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ── Per-turn record ──────────────────────────────────────────────────────


class TurnRecord(BaseModel):
    """Structured snapshot of one turn's metrics.

    This is the Pydantic equivalent of the dict returned by
    ``base_graph.snapshot_turn_result()`` — with the addition of
    ``wall_clock_ms``, ``game_phase``, and ``board_fen`` that the
    :class:`MetricsCollector` populates.
    """

    move_number: int
    proposed_move: str = ""
    is_valid: bool = False
    first_try_valid: bool = False
    total_attempts: int = 0
    error_types: list[str] = Field(default_factory=list)
    retry_count: int = 0
    llm_calls_this_turn: int = 0
    tokens_this_turn: int = 0
    prompt_token_count: int = 0
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    critic_verdict: bool | None = None
    ground_truth_verdict: bool | None = None
    generation_strategy: str = "generator_only"
    strategic_plan: str = ""
    routed_phase: str = ""
    feedback_history: list[str] = Field(default_factory=list)

    # Fields populated by the MetricsCollector
    wall_clock_ms: float = 0.0
    game_phase: str = ""
    board_fen: str = ""


# ── Per-game record ──────────────────────────────────────────────────────


class GameRecord(BaseModel):
    """Complete record for one game (Exp 2/3) or one position (Exp 1).

    For Experiment 1, each position constitutes a single-turn "game".
    """

    game_id: str
    condition: str
    experiment: int  # 1, 2, or 3
    input_mode: str = "fen"
    turns: list[TurnRecord] = Field(default_factory=list)
    final_status: str = "ongoing"
    total_turns: int = 0
    total_llm_calls: int = 0
    total_tokens: int = 0
    starting_fen: str = ""


# ── Aggregated metric containers ─────────────────────────────────────────


class PhaseStratifiedFIR(BaseModel):
    """FIR broken down by game phase."""

    opening: float | None = None
    middlegame: float | None = None
    endgame: float | None = None


class CriticAccuracy(BaseModel):
    """Confusion matrix for the LLM Critic (Condition C).

    Ground truth is determined strictly by ``python-chess``.

    - TP: Critic says invalid AND move is actually invalid.
    - FP: Critic says invalid AND move is actually valid.
    - TN: Critic says valid AND move is actually valid.
    - FN: Critic says valid AND move is actually invalid.
    """

    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0

    tpr: float = 0.0  # sensitivity / recall
    fpr: float = 0.0
    tnr: float = 0.0  # specificity
    fnr: float = 0.0  # miss rate


class ErrorTypeRSR(BaseModel):
    """Retry Success Rate broken down by error type.

    ``rsr_by_type`` maps ``ErrorType`` name → RSR float.
    ``counts_by_type`` maps ``ErrorType`` name → (corrected, total) tuple
    for flagging low-count cells.
    """

    rsr_by_type: dict[str, float] = Field(default_factory=dict)
    counts_by_type: dict[str, tuple[int, int]] = Field(default_factory=dict)


class ToolCallDistribution(BaseModel):
    """Frequency counts per tool type (Condition F)."""

    counts: dict[str, int] = Field(default_factory=dict)
    total_tool_calls: int = 0


class DescriptiveStats(BaseModel):
    """Simple descriptive statistics container."""

    mean: float = 0.0
    median: float = 0.0
    min: float = 0.0
    max: float = 0.0
    std: float = 0.0
    n: int = 0


class LegalityDegradationBin(BaseModel):
    """One bin in a legality degradation curve."""

    bin_start: int
    bin_end: int
    ftir: float
    n_turns: int


class QuartileErrorDist(BaseModel):
    """Error-type frequency distribution for one quartile."""

    quartile: str  # Q1, Q2, Q3, Q4
    counts: dict[str, int] = Field(default_factory=dict)
    total_errors: int = 0


class FSTEntry(BaseModel):
    """One entry for Forfeit Survival Time (Kaplan-Meier input)."""

    game_id: str
    half_moves: int
    censored: bool  # True if game ended naturally (not forfeit)


class FIRDeltaEntry(BaseModel):
    """Per-position FIR delta between Exp 2 and Exp 3."""

    starting_fen: str
    condition: str
    fir_exp2: float
    fir_exp3: float
    delta: float  # fir_exp3 - fir_exp2


# ── Top-level condition metrics ──────────────────────────────────────────


class ConditionMetrics(BaseModel):
    """Aggregated metrics for one condition in one experiment.

    Fields are ``None`` when the metric is not applicable to the
    condition (e.g. RSR for conditions A/B, critic accuracy for non-C).
    """

    condition: str
    experiment: int

    # ── Rate metrics ──
    fir: float = 0.0
    ftir: float = 0.0
    mfir: float | None = None  # Only for paired comparisons
    arr: float | None = None

    # ── Phase-stratified ──
    phase_stratified_fir: PhaseStratifiedFIR | None = None

    # ── Parse failures ──
    parse_failure_count: int = 0

    # ── Retry metrics (C, D, E only) ──
    rsr: float | None = None
    mrtc: float | None = None

    # ── Cost metrics ──
    lcpt: DescriptiveStats | None = None
    tpt: DescriptiveStats | None = None
    cafir: float | None = None  # Excludes A, B

    # ── Critic accuracy (C only) ──
    critic_accuracy: CriticAccuracy | None = None

    # ── Error-type RSR (C, D, E only) ──
    error_type_rsr: ErrorTypeRSR | None = None

    # ── Condition F agent behaviour ──
    vta: float | None = None
    tcr: float | None = None
    tool_call_distribution: ToolCallDistribution | None = None
    tool_stratified_fir: dict[str, float] | None = None
    avg_reasoning_steps: float | None = None

    # ── Game-level (Exp 2/3) ──
    imfr: float | None = None
    fst_data: list[FSTEntry] | None = None

    # ── Multi-turn consistency (Exp 2/3) ──
    serr: float | None = None
    pcrr: float | None = None
    ttr_values: list[int] | None = None
    ecc: float | None = None
    legality_degradation: list[LegalityDegradationBin] | None = None
    input_length_vs_error: list[dict[str, Any]] | None = None
    error_type_over_quartiles: list[QuartileErrorDist] | None = None
