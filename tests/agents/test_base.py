"""Tests for agents.base utilities."""

from __future__ import annotations

import chess
import pytest

from src.agents.base import (
    build_board_representation,
    format_feedback_block,
    get_side_to_move,
    load_agent_prompt,
)


def test_load_agent_prompt_loads_existing_template() -> None:
    text = load_agent_prompt("generator", "fen", "user")
    assert "{color}" in text
    assert "UCI" in text


def test_load_agent_prompt_raises_on_missing_template() -> None:
    with pytest.raises(FileNotFoundError):
        load_agent_prompt("nonexistent_agent", "fen", "user")


def test_build_board_representation_fen_mode() -> None:
    rep = build_board_representation(chess.STARTING_FEN, "fen")
    assert "FEN:" in rep
    assert "Board:" in rep


def test_build_board_representation_history_mode() -> None:
    rep = build_board_representation(chess.STARTING_FEN, "history")
    assert "withheld" in rep.lower()
    assert "FEN:" not in rep


def test_format_feedback_block_empty() -> None:
    assert format_feedback_block([]) == ""


def test_format_feedback_block_with_entries() -> None:
    feedback = format_feedback_block(["error1", "error2"])
    assert "Attempt 1:" in feedback
    assert "Attempt 2:" in feedback
    assert "error1" in feedback
    assert "different, legal move" in feedback


def test_get_side_to_move_starting_position() -> None:
    assert get_side_to_move(chess.STARTING_FEN) == "white"


def test_get_side_to_move_after_e4() -> None:
    board = chess.Board()
    board.push_uci("e2e4")
    assert get_side_to_move(board.fen()) == "black"
