"""Pydantic models for structured turn- and game-level metric records."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TurnRecord(BaseModel):
    """One row of raw data per turn.

    Every field mirrors a value produced by the condition graphs
    (via ``snapshot_turn_result``) plus experiment-level metadata
    injected by the experiment runner.
    """

    # ── Identity ──────────────────────────────────────────────────────
    game_id: str
    condition: str
    experiment: str  # "exp1" | "exp2" | "exp3"

    # ── Position context ──────────────────────────────────────────────
    move_number: int
    game_phase: str = ""  # "opening" | "middlegame" | "endgame"

    # ── Move outcome ──────────────────────────────────────────────────
    proposed_move: str = ""
    is_valid: bool = False  # Final validity (after all retries)
    first_try_valid: bool = False
    total_attempts: int = 0
    retry_count: int = 0
    error_types: list[str] = Field(default_factory=list)

    # ── Cost & latency ────────────────────────────────────────────────
    llm_calls_this_turn: int = 0
    tokens_this_turn: int = 0
    prompt_token_count: int = 0
    wall_clock_ms: float = 0.0

    # ── Tool usage (Condition F) ──────────────────────────────────────
    tool_calls: list[dict] = Field(default_factory=list)

    # ── Critic (Condition C) ──────────────────────────────────────────
    critic_verdict: bool | None = None
    ground_truth_verdict: bool | None = None

    # ── Feedback ──────────────────────────────────────────────────────
    feedback_history: list[str] = Field(default_factory=list)

    # ── Generation metadata ───────────────────────────────────────────
    generation_strategy: str = "generator_only"
    strategic_plan: str = ""
    routed_phase: str = ""


class GameRecord(BaseModel):
    """Aggregated result for one complete game or puzzle entry.

    ``turns`` holds the ordered sequence of :class:`TurnRecord` objects.
    Game-level metrics (``fir``, ``ftir``, etc.) are populated by the
    aggregator after collection.
    """

    # ── Identity ──────────────────────────────────────────────────────
    game_id: str
    condition: str
    experiment: str

    # ── Turn data ─────────────────────────────────────────────────────
    turns: list[TurnRecord] = Field(default_factory=list)

    # ── Game-level outcome ────────────────────────────────────────────
    game_status: str = "ongoing"  # checkmate | stalemate | draw | forfeit | max_moves
    total_turns: int = 0
    total_forfeits: int = 0
    game_length: int = 0  # Number of *legal* moves played

    # ── Computed per-game metrics (populated by aggregator) ───────────
    fir: float | None = None
    ftir: float | None = None
    gcr_contributed: bool | None = None  # Did this game reach a natural end?
