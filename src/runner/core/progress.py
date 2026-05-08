"""Thread-safe progress tracking models for the runner.

These classes are used by the orchestrator to maintain a dashboard-friendly
view of progress.

Design goals:
- Thread-safe counters (workers update concurrently).
- JSON-serializable snapshots (`to_dict`).
- No business logic beyond counting.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any


class ConditionProgress:
    """Thread-safe per-condition progress tracker."""

    def __init__(self, condition: str, total: int) -> None:
        self.condition = condition
        self.total = total
        self._lock = threading.Lock()
        self.completed = 0
        self.failed = 0
        self.in_progress = 0
        self.valid_count = 0

    def record_complete(self, is_valid: bool = True) -> None:
        with self._lock:
            self.completed += 1
            self.in_progress = max(0, self.in_progress - 1)
            if is_valid:
                self.valid_count += 1

    def record_start(self) -> None:
        with self._lock:
            self.in_progress += 1

    def record_failure(self) -> None:
        with self._lock:
            self.failed += 1
            self.in_progress = max(0, self.in_progress - 1)

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "condition": self.condition,
                "total": self.total,
                "completed": self.completed,
                "failed": self.failed,
                "in_progress": self.in_progress,
                "valid_count": self.valid_count,
            }


class ExperimentProgress:
    """Thread-safe per-experiment progress tracker."""

    def __init__(
        self,
        experiment: int,
        conditions: list[str],
        *,
        output_dir: Path | None = None,
    ) -> None:
        self.experiment = experiment
        self.conditions_progress: dict[str, ConditionProgress] = {}
        self._conditions = conditions
        self.status = "pending"
        self.started_at: str | None = None
        self.output_dir: Path | None = output_dir

    def init_condition(self, condition: str, total: int) -> None:
        self.conditions_progress[condition] = ConditionProgress(condition, total)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment": self.experiment,
            "status": self.status,
            "started_at": self.started_at,
            "conditions": {c: cp.to_dict() for c, cp in self.conditions_progress.items()},
        }
