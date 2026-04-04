from __future__ import annotations

from typing import Literal, Protocol, TypeAlias, cast, runtime_checkable

from src.state import TurnState

ConditionId = Literal["A", "B", "C", "D", "E", "F"]
ALL_CONDITIONS: tuple[ConditionId, ...] = ("A", "B", "C", "D", "E", "F")
ConditionRunResult: TypeAlias = TurnState


def normalize_condition_id(condition: str) -> ConditionId:
    """Normalize and validate a condition identifier."""

    normalized = condition.strip().upper()
    if normalized not in ALL_CONDITIONS:
        allowed = ", ".join(ALL_CONDITIONS)
        raise ValueError(f"Unknown condition '{condition}'. Expected one of: {allowed}.")

    return cast(ConditionId, normalized)


@runtime_checkable
class ConditionRunner(Protocol):
    """Common interface for condition turn executors."""

    condition_id: ConditionId

    def run_turn(self, state: TurnState) -> ConditionRunResult:
        """Process one turn and return an updated TurnState."""
