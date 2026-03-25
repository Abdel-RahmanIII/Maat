from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from core.rule_validator import RuleValidator
from core.state_manager import StateManager
from schemas.move import MoveResult, ValidationResult


class GameRunner:
    """Executes move sequences and logs every attempt.

    This is the only class that writes to the JSONL log during the
    deterministic Phase 1. In Weeks 3–5 the graph nodes will call
    submit_move() directly — the interface is intentionally identical.

    Logging here uses a minimal schema. The full TurnLog (with run_id,
    condition, agent_role, retry_count) is written by logging/logger.py
    in Week 2 onward, which wraps this class.
    """

    SCHEMA_VERSION = "1.0"

    def __init__(
        self,
        state_manager: StateManager,
        validator: RuleValidator,
        log_path: Path | None = None,
    ) -> None:
        self.state_manager = state_manager
        self.validator = validator
        self.log_path = log_path
        self.turn_id = 0
        self.attempt_id = 0

    def run_sequence(self, moves: Iterable[str]) -> None:
        """Execute a list of moves, stopping at the first terminal state."""
        for move in moves:
            if self.state_manager.snapshot().is_terminal:
                break
            self.submit_move(move)

    def submit_move(self, raw_move: str) -> MoveResult:
        """Validate and conditionally apply one move. Always logs the attempt.

        Returns a MoveResult so callers get a single object with the full 
        attempt record
        """
        self.turn_id += 1
        self.attempt_id += 1

        before_fen = self.state_manager.current_fen()
        validation: ValidationResult = self.validator.validate_move(raw_move)

        if validation.is_valid and validation.normalized_move_uci:
            self.state_manager.apply_validated_move_uci(validation.normalized_move_uci)
            after_fen = self.state_manager.current_fen()
            terminal = self.state_manager.snapshot().is_terminal

            result = MoveResult(
                raw_move=raw_move,
                validation=validation,
                state_before_fen=before_fen,
                state_after_fen=after_fen,
                terminal_flag=terminal,
            )
        else:
            result = MoveResult(
                raw_move=raw_move,
                validation=validation,
                state_before_fen=before_fen,
                state_after_fen=None,
                terminal_flag=self.state_manager.snapshot().is_terminal,
            )

        self._write_log(result)
        return result

    def terminal_status(self) -> tuple[bool, str | None]:
        snapshot = self.state_manager.snapshot()
        return snapshot.is_terminal, snapshot.outcome

    def _write_log(self, result: MoveResult) -> None:
        if not self.log_path:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "schema_version": self.SCHEMA_VERSION,
            "turn_id": self.turn_id,
            "attempt_id": self.attempt_id,
            "input_move": result.raw_move,
            "normalized_move": result.validation.normalized_move_uci,
            "validation_stage": result.validation.validation_stage,
            "validator_result": result.validation.error_code or "valid",
            "validator_message": result.validation.message,
            "state_before_fen": result.state_before_fen,
            "state_after_fen": result.state_after_fen,
            "terminal_flag": result.terminal_flag,
        }
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")