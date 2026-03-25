"""Stub — implemented in Week 2.

The interface is defined here so Week 2 only fills in the body,
not redesign the signature.

Usage (Week 2+):
    logger = GameLogger(run_id="abc123", condition="A", log_path=Path("logs/run.jsonl"))
    logger.log(move_result, turn_id=1, attempt_id=1, retry_count=0, agent_role="monolithic")
"""
from __future__ import annotations

from pathlib import Path

from schemas.log_entry import TurnLog


class GameLogger:
    def __init__(self, run_id: str, condition: str, log_path: Path) -> None:
        self.run_id = run_id
        self.condition = condition
        self.log_path = log_path

    def log(self, record: TurnLog) -> None:
        raise NotImplementedError("Implemented in Week 2.")