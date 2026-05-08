import threading
from unittest.mock import MagicMock

import pytest

import src.runner.workers.puzzles as puzzles_mod
from src.metrics.definitions import GameRecord
from src.runner.workers.puzzles import run_puzzle_worker


@pytest.fixture
def puzzle():
    return {"puzzle_id": "p1", "fen": "8/8/8/8/8/8/8/8 w - - 0 1"}


def test_puzzle_worker_success(tmp_path, puzzle, monkeypatch):
    # Mock dispatch_turn_with_backoff to return a valid move
    def fake_dispatch(*args, **kwargs):
        return {
            "game_status": "ongoing",
            "is_valid": True,
            "proposed_move": "e2e4",
            "raw_llm_response": "move e4"
        }
    monkeypatch.setattr(puzzles_mod, "dispatch_turn_with_backoff", fake_dispatch)
    
    # Mock config_for_condition
    monkeypatch.setattr(puzzles_mod, "config_for_condition", lambda c: MagicMock(max_react_steps=5))
    
    events = []
    def on_progress(data):
        events.append(data)
        
    record = run_puzzle_worker(
        puzzle=puzzle,
        condition="A",
        game_id="game_1",
        output_dir=tmp_path,
        on_progress=on_progress,
    )
    
    assert isinstance(record, GameRecord)
    assert record.final_status == "completed"
    
    # Check that events were emitted
    event_types = [e["type"] for e in events]
    assert "worker_status" in event_types
    assert "puzzle_complete" in event_types
    
    # Check outputs were written
    results_path = tmp_path / "exp1_A_results.jsonl"
    assert results_path.exists()
    checkpoint_path = tmp_path / ".checkpoint"
    assert checkpoint_path.exists()


def test_puzzle_worker_stop_before_start(tmp_path, puzzle):
    stop_event = threading.Event()
    stop_event.set()
    
    events = []
    record = run_puzzle_worker(
        puzzle=puzzle,
        condition="A",
        game_id="game_1",
        output_dir=tmp_path,
        stop_event=stop_event,
        on_progress=events.append,
    )
    
    assert record is None
    # Depending on implementation, it may or may not emit "Stopped before puzzle start" if it checks twice,
    # but the first check just returns None.
    # Ah, the first check returns None without event. Let's see:
    assert len(events) == 0


def test_puzzle_worker_exception(tmp_path, puzzle, monkeypatch):
    def fake_dispatch(*args, **kwargs):
        raise ValueError("API error")
    monkeypatch.setattr(puzzles_mod, "dispatch_turn_with_backoff", fake_dispatch)
    monkeypatch.setattr(puzzles_mod, "config_for_condition", lambda c: MagicMock(max_react_steps=5))
    
    events = []
    record = run_puzzle_worker(
        puzzle=puzzle,
        condition="A",
        game_id="game_1",
        output_dir=tmp_path,
        on_progress=events.append,
    )
    
    assert record is None
    event_types = [e["type"] for e in events]
    assert "worker_error" in event_types
    
def test_puzzle_worker_pause(tmp_path, puzzle, monkeypatch):
    def fake_dispatch(*args, **kwargs):
        return {"game_status": "ongoing", "is_valid": True, "proposed_move": "e2e4"}
    monkeypatch.setattr(puzzles_mod, "dispatch_turn_with_backoff", fake_dispatch)
    monkeypatch.setattr(puzzles_mod, "config_for_condition", lambda c: MagicMock(max_react_steps=5))
    
    pause_event = threading.Event()
    
    def worker_thread():
        run_puzzle_worker(
            puzzle=puzzle, condition="A", game_id="g1",
            output_dir=tmp_path, pause_event=pause_event
        )
        
    # Start worker without setting pause_event -> it should block
    t = threading.Thread(target=worker_thread)
    t.start()
    
    t.join(timeout=0.1)
    assert t.is_alive() # Still blocked
    
    # Unpause
    pause_event.set()
    t.join(timeout=2.0)
    assert not t.is_alive()
