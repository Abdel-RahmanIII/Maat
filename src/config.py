"""Experiment configuration for Maat."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

# Load .env from project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


class GenerationStrategy(str, Enum):
    """How the move-generation stage is structured."""

    GENERATOR_ONLY = "generator_only"
    PLANNER_ACTOR = "planner_actor"
    ROUTER_SPECIALISTS = "router_specialists"


class Condition(str, Enum):
    """Experimental conditions."""

    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"


@dataclass(frozen=True)
class ModelConfig:
    """LLM model configuration."""

    model_name: str = "gemma-4-31b-it"
    temperature: float = 0.0
    max_output_tokens: int = 1024
    api_key: str = field(default_factory=lambda: os.environ.get("GOOGLE_API_KEY", ""))


@dataclass(frozen=True)
class ConditionConfig:
    """Per-condition tuning knobs."""

    condition: Condition = Condition.A
    max_retries: int = 0
    max_react_steps: int = 6
    generation_strategy: GenerationStrategy = GenerationStrategy.GENERATOR_ONLY
    input_mode: Literal["fen", "history"] = "fen"


# ── Preset configs per condition ─────────────────────────────────────────

def config_for_condition(
    condition: Condition | str,
    *,
    generation_strategy: GenerationStrategy = GenerationStrategy.GENERATOR_ONLY,
    input_mode: Literal["fen", "history"] = "fen",
) -> ConditionConfig:
    """Return the canonical ``ConditionConfig`` for *condition*."""

    if isinstance(condition, str):
        condition = Condition(condition)

    retry_map = {
        Condition.A: 0,
        Condition.B: 0,
        Condition.C: 3,
        Condition.D: 3,
        Condition.E: 3,
        Condition.F: 0,  # ReAct uses max_react_steps, not retries
    }

    return ConditionConfig(
        condition=condition,
        max_retries=retry_map[condition],
        max_react_steps=6,
        generation_strategy=generation_strategy,
        input_mode=input_mode,
    )
