from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from core.schemas import AttemptLogRecord
from core.state_manager import StateManager
from core.validator import RuleValidator


class GameRunner:
    SCHEMA_VERSION = "1.0"

    def __init__(self, state_manager: StateManager, validator: RuleValidator, log_path: Path | None = None) -> None:
        self.state_manager = state_manager
        self.validator = validator
        self.log_path = log_path
        self.turn_id = 0
        self.attempt_id = 0

    def run_sequence(self, moves: Iterable[str]) -> None:
        for move in moves:
            if self.state_manager.snapshot().is_terminal:
                break
            self.submit_move(move)

    def submit_move(self, raw_move: str) -> bool:
        self.turn_id += 1
        self.attempt_id += 1

        before = self.state_manager.current_fen()
        validation = self.validator.validate_move(raw_move)

        if validation.is_valid and validation.normalized_move_uci:
            self.state_manager.apply_validated_move_uci(validation.normalized_move_uci)
            after = self.state_manager.current_fen()
            terminal = self.state_manager.snapshot().is_terminal
            self._write_log(
                AttemptLogRecord(
                    schema_version=self.SCHEMA_VERSION,
                    turn_id=self.turn_id,
                    attempt_id=self.attempt_id,
                    input_move=raw_move,
                    normalized_move=validation.normalized_move_uci,
                    validator_result="valid",
                    validator_message=validation.message,
                    state_before_fen=before,
                    state_after_fen=after,
                    terminal_flag=terminal,
                )
            )
            return True

        self._write_log(
            AttemptLogRecord(
                schema_version=self.SCHEMA_VERSION,
                turn_id=self.turn_id,
                attempt_id=self.attempt_id,
                input_move=raw_move,
                normalized_move=None,
                validator_result=(validation.error_code.value if validation.error_code else "unknown_error"),
                validator_message=validation.message,
                state_before_fen=before,
                state_after_fen=None,
                terminal_flag=self.state_manager.snapshot().is_terminal,
            )
        )
        return False

    def terminal_status(self) -> tuple[bool, str | None]:
        snapshot = self.state_manager.snapshot()
        return snapshot.is_terminal, snapshot.outcome

    def _write_log(self, record: AttemptLogRecord) -> None:
        if not self.log_path:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(record.as_dict()))
            log_file.write("\n")
