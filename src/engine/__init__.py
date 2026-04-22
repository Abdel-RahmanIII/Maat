"""Chess engine wrappers and experiment orchestration.

This package provides:

- :class:`StockfishWrapper` — Thin wrapper around a UCI Stockfish binary.
- :func:`dispatch_turn` — Centralized condition router.
- :func:`dispatch_turn_with_backoff` — Dispatcher with exponential backoff.
- :class:`PuzzleManager` — Experiment 1 orchestrator.
- :class:`GameManager` — Experiments 2 & 3 orchestrator.
- Result persistence utilities in :mod:`result_store`.
"""

from src.engine.condition_dispatch import dispatch_turn, dispatch_turn_with_backoff
from src.engine.game_manager import GameManager, load_starting_positions
from src.engine.puzzle_manager import PuzzleManager, load_puzzle_inputs
from src.engine.result_store import (
    append_checkpoint,
    append_game_record,
    load_checkpoint,
    load_game_records,
    write_summary_csv,
)
from src.engine.stockfish_wrapper import StockfishWrapper

__all__ = [
    # Stockfish
    "StockfishWrapper",
    # Dispatch
    "dispatch_turn",
    "dispatch_turn_with_backoff",
    # Orchestrators
    "PuzzleManager",
    "GameManager",
    # Loaders
    "load_puzzle_inputs",
    "load_starting_positions",
    # Result store
    "append_game_record",
    "load_game_records",
    "load_checkpoint",
    "append_checkpoint",
    "write_summary_csv",
]
