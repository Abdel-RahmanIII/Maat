from __future__ import annotations

from dataclasses import dataclass

from src.graph.contracts import ConditionId, normalize_condition_id

DEFAULT_MAX_RETRIES = 3
DEFAULT_MAX_REASONING_STEPS = 6
DEFAULT_TEMPERATURE = 0.0

CONDITION_MAX_RETRIES: dict[ConditionId, int] = {
    "A": 0,
    "B": 0,
    "C": DEFAULT_MAX_RETRIES,
    "D": DEFAULT_MAX_RETRIES,
    "E": DEFAULT_MAX_RETRIES,
    "F": 0,
}

LANGGRAPH_CONDITIONS: frozenset[ConditionId] = frozenset({"B", "C", "D", "E", "F"})
RETRY_CONDITIONS: frozenset[ConditionId] = frozenset({"C", "D", "E"})


@dataclass(frozen=True, slots=True)
class ConditionExecutionConfig:
    condition_id: ConditionId
    max_retries: int
    max_reasoning_steps: int = DEFAULT_MAX_REASONING_STEPS
    temperature: float = DEFAULT_TEMPERATURE


def max_retries_for_condition(condition: str) -> int:
    condition_id = normalize_condition_id(condition)
    return CONDITION_MAX_RETRIES[condition_id]


def default_execution_config(condition: str) -> ConditionExecutionConfig:
    condition_id = normalize_condition_id(condition)
    return ConditionExecutionConfig(
        condition_id=condition_id,
        max_retries=CONDITION_MAX_RETRIES[condition_id],
        max_reasoning_steps=DEFAULT_MAX_REASONING_STEPS,
        temperature=DEFAULT_TEMPERATURE,
    )
