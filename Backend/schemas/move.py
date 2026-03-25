from __future__ import annotations

from pydantic import BaseModel, Field


class MoveOutput(BaseModel):
    """Schema for the raw move string produced by the generator agent."""

    move: str = Field(description="A chess move in UCI (e.g. 'e2e4') or SAN (e.g. 'e4') notation.")


class ValidationResult(BaseModel):
    """Output of RuleValidator.validate_move().

    Two-stage failure is explicit: syntax errors and legality errors"""

    is_valid: bool
    normalized_move_uci: str | None = None
    # Stage the failure occurred: None when valid.
    validation_stage: str | None = Field(
        default=None,
        description="'syntax' or 'legality'. None when valid.",
    )
    error_code: str | None = None
    message: str


class MoveResult(BaseModel):
    """Combines a generator's raw output with the validator's verdict.

    This is what the GameRunner and graph nodes pass around after a
    submit_move() call, giving every downstream consumer a single object
    that contains the full attempt record.
    """

    raw_move: str
    validation: ValidationResult
    state_before_fen: str
    state_after_fen: str | None = None
    terminal_flag: bool = False