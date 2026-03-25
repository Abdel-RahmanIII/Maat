from __future__ import annotations

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.0"


class TurnLog(BaseModel):
    """Frozen log record — one object appended per attempt (valid or invalid).

    'frozen' here means immutable after construction, which is enforced by
    Pydantic. The logger writes this directly; nothing mutates it afterward.
    """

    model_config = {"frozen": True}

    schema_version: str = Field(default=SCHEMA_VERSION)
    run_id: str                      # one UUID per run_experiment.py invocation
    condition: str                   # "A" | "B" | "C" | "D"
    ply_number: int                  # half-move counter (1-indexed)
    turn_id: int                     # increments on every submit_move call
    attempt_id: int                  # increments on every attempt within a turn
    agent_role: str                  # "generator" | "monolithic" | ...
    input_move: str
    normalized_move: str | None
    validation_stage: str | None     # "syntax" | "legality" | None
    validator_result: str            # "valid" | error_code value
    validator_message: str
    state_before_fen: str
    state_after_fen: str | None
    terminal_flag: bool
    retry_count: int = 0             # how many rejections preceded this attempt