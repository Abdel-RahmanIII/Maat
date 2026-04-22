"""Experiment configuration loader.

Reads YAML experiment config files and instantiates the appropriate
:class:`PuzzleManager` or :class:`GameManager`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from src.config import ModelConfig

if TYPE_CHECKING:
    from src.engine.game_manager import GameManager
    from src.engine.puzzle_manager import PuzzleManager

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_path(raw: str) -> Path:
    """Resolve a path relative to the project root."""
    p = Path(raw)
    if p.is_absolute():
        return p
    return _PROJECT_ROOT / p


def load_experiment_config(yaml_path: str | Path) -> dict[str, Any]:
    """Load and validate an experiment YAML config file.

    Returns the parsed config dict with resolved paths and a
    constructed :class:`ModelConfig`.
    """

    path = Path(yaml_path)
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if not isinstance(raw, dict):
        raise ValueError(f"Expected a YAML mapping, got {type(raw).__name__}")

    experiment = raw.get("experiment")
    if experiment not in (1, 2, 3):
        raise ValueError(f"experiment must be 1, 2, or 3 — got {experiment!r}")

    # Build ModelConfig
    model_raw = raw.get("model", {})
    model_config = ModelConfig(
        model_name=model_raw.get("model_name", "gemma-4-31b-it"),
        temperature=model_raw.get("temperature", 0.0),
        max_output_tokens=model_raw.get("max_output_tokens", 1024),
    )

    config: dict[str, Any] = {
        "experiment": experiment,
        "conditions": raw.get("conditions", []),
        "model_config": model_config,
        "generation_strategy": raw.get("generation_strategy", "generator_only"),
        "delay_seconds": raw.get("delay_seconds", 0.0),
        "max_api_retries": raw.get("max_api_retries", 5),
        "backoff_base": raw.get("backoff_base", 2.0),
        "backoff_max": raw.get("backoff_max", 60.0),
    }

    if experiment == 1:
        puzzle_data = raw.get("puzzle_data")
        if not puzzle_data:
            raise ValueError("experiment 1 requires 'puzzle_data' path")
        config["puzzle_data"] = _resolve_path(puzzle_data)
        config["output_dir"] = _resolve_path(raw.get("output_dir", "results/exp1"))
    else:
        # Exp 2 or 3
        config["input_mode"] = raw.get("input_mode", "fen" if experiment == 2 else "history")
        positions_path = raw.get("starting_positions")
        if not positions_path:
            raise ValueError(f"experiment {experiment} requires 'starting_positions' path")
        config["starting_positions"] = _resolve_path(positions_path)
        config["output_dir"] = _resolve_path(
            raw.get("output_dir", f"results/exp{experiment}")
        )
        config["max_half_moves"] = raw.get("max_half_moves", 150)

        sf_raw = raw.get("stockfish", {})
        config["stockfish_elo"] = sf_raw.get("elo", 1000)
        config["stockfish_path"] = sf_raw.get("path")

    return config


def build_puzzle_manager_from_config(
    config: dict[str, Any],
) -> PuzzleManager:
    """Construct a :class:`PuzzleManager` from a loaded config dict."""

    from src.engine.puzzle_manager import PuzzleManager, load_puzzle_inputs

    puzzles = load_puzzle_inputs(config["puzzle_data"])

    return PuzzleManager(
        puzzles=puzzles,
        conditions=config["conditions"],
        output_dir=config["output_dir"],
        model_config=config["model_config"],
        generation_strategy=config["generation_strategy"],
        delay_seconds=config["delay_seconds"],
        max_api_retries=config["max_api_retries"],
        backoff_base=config["backoff_base"],
        backoff_max=config["backoff_max"],
    )


def build_game_manager_from_config(
    config: dict[str, Any],
) -> GameManager:
    """Construct a :class:`GameManager` from a loaded config dict."""

    from src.engine.game_manager import GameManager, load_starting_positions

    positions = load_starting_positions(config["starting_positions"])

    return GameManager(
        starting_positions=positions,
        conditions=config["conditions"],
        experiment=config["experiment"],
        output_dir=config["output_dir"],
        stockfish_elo=config.get("stockfish_elo", 1000),
        stockfish_path=config.get("stockfish_path"),
        max_half_moves=config.get("max_half_moves", 150),
        model_config=config["model_config"],
        generation_strategy=config["generation_strategy"],
        delay_seconds=config["delay_seconds"],
        max_api_retries=config["max_api_retries"],
        backoff_base=config["backoff_base"],
        backoff_max=config["backoff_max"],
    )
