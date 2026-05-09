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

    def __init__(self, condition: str, total: int = 0) -> None:
        self.condition = condition
        self.total = total
        self.completed = 0
        self.failed = 0
        self.in_progress = 0
        self._lock = threading.Lock()

    def record_start(self) -> None:
        with self._lock:
            self.in_progress += 1

    def record_complete(self, *, is_valid: bool = True) -> None:
        with self._lock:
            self.in_progress = max(0, self.in_progress - 1)
            self.completed += 1

    def record_failure(self) -> None:
        with self._lock:
            self.in_progress = max(0, self.in_progress - 1)
            self.failed += 1

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "condition": self.condition,
                "total": self.total,
                "completed": self.completed,
                "failed": self.failed,
                "in_progress": self.in_progress,
            }


class ExperimentProgress:
    """Thread-safe per-experiment progress tracker."""

    def __init__(
        self,
        experiment_id: int,
        conditions: list[str],
        *,
        output_dir: Path | None = None,
    ) -> None:
        self.experiment_id = experiment_id
        self.conditions_progress: dict[str, ConditionProgress] = {}
        self.status: str = "pending"
        self.started_at: str | None = None
        self.output_dir = output_dir

        for c in conditions:
            self.conditions_progress[c] = ConditionProgress(c)

    def init_condition(self, condition: str, total: int) -> None:
        """Initialize (or reinitialize) a condition with a known total."""
        if condition in self.conditions_progress:
            self.conditions_progress[condition].total = total
        else:
            self.conditions_progress[condition] = ConditionProgress(condition, total)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "status": self.status,
            "started_at": self.started_at,
            "conditions": {
                c: cp.to_dict()
                for c, cp in self.conditions_progress.items()
            },
        }
