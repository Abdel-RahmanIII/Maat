"""Deterministic chess layer — no LLM dependency.

Public API (frozen after Week 1):
  StateManager    — owns chess.Board, produces StateSnapshot
  RuleValidator   — two-stage validation (syntax → legality)
  GameRunner      — executes sequences, logs every attempt
  IllegalMoveError / InvalidFENError — typed exceptions
"""

from core.exceptions import IllegalMoveError, InvalidFENError
from core.game_runner import GameRunner
from core.rule_validator import RuleValidator
from core.state_manager import StateManager

__all__ = [
    "StateManager",
    "RuleValidator",
    "GameRunner",
    "IllegalMoveError",
    "InvalidFENError",
]