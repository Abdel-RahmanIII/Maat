import pytest
from unittest.mock import patch, MagicMock

from src.engine.puzzle_manager import PuzzleManager

@pytest.fixture
def tmp_output_dir(tmp_path):
    return tmp_path / "results"

@pytest.fixture
def manager(tmp_output_dir):
    puzzles = [
        {"puzzle_id": "p1", "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"}
    ]
    return PuzzleManager(
        puzzles=puzzles,
        conditions=["A"],
        output_dir=tmp_output_dir,
        generation_strategy="test_strategy"
    )

def test_puzzle_manager_run_single_returns_record(manager):
    with patch("src.engine.puzzle_manager.dispatch_turn") as mock_dispatch:
        mock_dispatch.return_value = {
            "game_status": "completed",
            "board_fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "move_number": 1,
            "wall_clock_ms": 12.5,
            "is_valid": True,
            "proposed_move": "e2e4",
        }

        puzzle = manager.puzzles[0]
        record = manager.run_single(
            puzzle=puzzle,
            condition="A",
        )

    assert record is not None
    assert record.total_turns == 1
    assert record.final_status == "completed"

def test_puzzle_manager_run_single_normalizes_condition(manager):
    with patch("src.engine.puzzle_manager.dispatch_turn") as mock_dispatch:
        mock_dispatch.return_value = {
            "game_status": "completed",
            "board_fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "move_number": 1,
            "wall_clock_ms": 10.0,
            "is_valid": True,
            "proposed_move": "e2e4",
        }

        puzzle = manager.puzzles[0]
        record = manager.run_single(puzzle=puzzle, condition="a")

    assert record is not None
    assert record.condition == "A"

def test_puzzle_manager_run_single_dispatch_invoked(manager):
    with patch("src.engine.puzzle_manager.dispatch_turn") as mock_dispatch:
        mock_dispatch.return_value = {
            "game_status": "completed",
            "board_fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "move_number": 1,
            "wall_clock_ms": 8.0,
            "is_valid": True,
            "proposed_move": "e2e4",
        }

        puzzle = manager.puzzles[0]
        manager.run_single(
            puzzle=puzzle,
            condition="A",
        )

    assert mock_dispatch.call_count == 1
