import json
from pathlib import Path

import pytest

from src.runner.persistence.checkpoint import (
    delete_game_state,
    list_incomplete_games,
    load_game_state,
    load_run_progress,
    save_game_state,
    save_run_progress,
)


def test_save_and_load_game_state(tmp_path: Path):
    game_id = "game_123"
    
    # Save state
    save_game_state(
        tmp_path,
        game_id=game_id,
        condition="A",
        experiment=2,
        starting_fen="start_fen",
        board_fen="current_fen",
        move_stack_uci=["e2e4"],
        half_moves_played=1,
        turn_records=[{"proposed_move": "e2e4", "is_valid": True}],
        game_status="ongoing",
    )
    
    # Check if file exists
    state_file = tmp_path / ".game_state" / f"{game_id}.json"
    assert state_file.exists()
    
    # Load state
    state = load_game_state(tmp_path, game_id)
    assert state is not None
    assert state["game_id"] == game_id
    assert state["condition"] == "A"
    assert state["experiment"] == 2
    assert state["starting_fen"] == "start_fen"
    assert state["board_fen"] == "current_fen"
    assert state["move_stack_uci"] == ["e2e4"]
    assert state["half_moves_played"] == 1
    assert len(state["turn_records"]) == 1
    assert state["game_status"] == "ongoing"


def test_load_non_existent_game_state(tmp_path: Path):
    state = load_game_state(tmp_path, "missing_game")
    assert state is None


def test_load_corrupt_game_state(tmp_path: Path):
    game_id = "corrupt_game"
    d = tmp_path / ".game_state"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{game_id}.json"
    path.write_text("invalid json", encoding="utf-8")
    
    state = load_game_state(tmp_path, game_id)
    assert state is None


def test_delete_game_state(tmp_path: Path):
    game_id = "game_to_delete"
    save_game_state(
        tmp_path,
        game_id=game_id,
        condition="A",
        experiment=2,
        starting_fen="fen",
        board_fen="fen",
        move_stack_uci=[],
        half_moves_played=0,
        turn_records=[],
        game_status="ongoing",
    )
    
    assert (tmp_path / ".game_state" / f"{game_id}.json").exists()
    
    delete_game_state(tmp_path, game_id)
    
    assert not (tmp_path / ".game_state" / f"{game_id}.json").exists()
    assert load_game_state(tmp_path, game_id) is None


def test_list_incomplete_games(tmp_path: Path):
    assert list_incomplete_games(tmp_path) == []
    
    save_game_state(
        tmp_path,
        game_id="game_1",
        condition="A", experiment=2, starting_fen="fen", board_fen="fen",
        move_stack_uci=[], half_moves_played=0, turn_records=[], game_status="ongoing",
    )
    save_game_state(
        tmp_path,
        game_id="game_2",
        condition="B", experiment=2, starting_fen="fen", board_fen="fen",
        move_stack_uci=[], half_moves_played=0, turn_records=[], game_status="ongoing",
    )
    
    games = list_incomplete_games(tmp_path)
    assert set(games) == {"game_1", "game_2"}


def test_save_and_load_run_progress(tmp_path: Path):
    progress_data = {"A": {"total": 10, "completed": 5}}
    
    save_run_progress(
        tmp_path,
        experiment=2,
        conditions=["A", "B"],
        condition_progress=progress_data,
        status="paused",
        started_at="2023-01-01T00:00:00",
        paused_at="2023-01-01T01:00:00"
    )
    
    assert (tmp_path / ".run_progress.json").exists()
    
    loaded = load_run_progress(tmp_path)
    assert loaded is not None
    assert loaded["experiment"] == 2
    assert loaded["conditions"] == ["A", "B"]
    assert loaded["condition_progress"] == progress_data
    assert loaded["status"] == "paused"
    assert loaded["started_at"] == "2023-01-01T00:00:00"
    assert loaded["paused_at"] == "2023-01-01T01:00:00"


def test_load_non_existent_run_progress(tmp_path: Path):
    assert load_run_progress(tmp_path) is None


def test_load_corrupt_run_progress(tmp_path: Path):
    path = tmp_path / ".run_progress.json"
    path.write_text("invalid json", encoding="utf-8")
    
    assert load_run_progress(tmp_path) is None

