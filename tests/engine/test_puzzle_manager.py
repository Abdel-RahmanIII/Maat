import pytest
import threading
from unittest.mock import patch, MagicMock

from src.engine.puzzle_manager import PuzzleManager
from src.metrics.collector import MetricsCollector

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

def test_puzzle_manager_stop_event(manager):
    stop_event = threading.Event()
    stop_event.set()  # Immediate stop
    
    events_emitted = []
    def on_progress(event):
        events_emitted.append(event)
        
    puzzle = manager.puzzles[0]
    record = manager.run_single(
        puzzle=puzzle,
        condition="A",
        stop_event=stop_event,
        on_progress=on_progress
    )
    
    assert record is None
    assert events_emitted[0]["status"] == "running"
    assert events_emitted[-1]["status"] == "stopped"

def test_puzzle_manager_pause_event(manager):
    pause_event = threading.Event()
    pause_event.clear()  # Not set -> paused
    
    events_emitted = []
    def on_progress(event):
        events_emitted.append(event)
        if event.get("status") == "paused":
            pause_event.set()  # Resume so it finishes
            
    # Mock dispatch_turn so it doesn't do a real LLM call
    with patch("src.engine.puzzle_manager.dispatch_turn") as mock_dispatch:
        mock_dispatch.return_value = {"game_status": "completed"}
        
        puzzle = manager.puzzles[0]
        record = manager.run_single(
            puzzle=puzzle,
            condition="A",
            pause_event=pause_event,
            on_progress=on_progress
        )
        
    assert record is not None
    assert any(e.get("status") == "paused" for e in events_emitted)
    assert any(e.get("status") == "running" for e in events_emitted)

def test_puzzle_manager_on_progress_events(manager):
    events_emitted = []
    def on_progress(event):
        events_emitted.append(event)
        
    with patch("src.engine.puzzle_manager.dispatch_turn") as mock_dispatch:
        mock_dispatch.return_value = {"game_status": "completed"}
        
        puzzle = manager.puzzles[0]
        record = manager.run_single(
            puzzle=puzzle,
            condition="A",
            on_progress=on_progress
        )
         
    types_emitted = [e["type"] for e in events_emitted]
    assert "worker_status" in types_emitted
    assert "game_complete" in types_emitted
