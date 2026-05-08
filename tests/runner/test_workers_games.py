import threading
from unittest.mock import MagicMock

import chess
import pytest

import src.runner.workers.games as games_mod
from src.metrics.definitions import GameRecord
from src.runner.persistence.checkpoint import save_game_state
from src.runner.workers.games import run_game_worker


@pytest.fixture
def base_kwargs(tmp_path):
    return {
        "game_id": "game_1",
        "starting_fen": chess.STARTING_FEN,
        "condition": "A",
        "experiment": 2,
        "output_dir": tmp_path,
        "max_half_moves": 4, # short game
    }


def test_game_worker_success(base_kwargs, monkeypatch):
    # Mock dispatch_turn_with_backoff for White (LLM)
    def fake_dispatch(*args, **kwargs):
        board = chess.Board(kwargs["fen"])
        move = list(board.legal_moves)[0]
        return {
            "game_status": "ongoing",
            "is_valid": True,
            "proposed_move": move.uci(),
            "raw_llm_response": "move"
        }
        
    monkeypatch.setattr(games_mod, "dispatch_turn_with_backoff", fake_dispatch)
    monkeypatch.setattr(games_mod, "config_for_condition", lambda c: MagicMock(max_react_steps=5))
    
    # Mock Stockfish Wrapper for Black
    class FakeStockfish:
        def __init__(self, *args, **kwargs): pass
        def start(self): pass
        def close(self): pass
        def choose_move(self, fen):
            board = chess.Board(fen)
            return list(board.legal_moves)[0].uci()
            
    monkeypatch.setattr(games_mod, "StockfishWrapper", FakeStockfish)
    
    events = []
    
    record = run_game_worker(
        **base_kwargs,
        on_progress=events.append
    )
    
    assert isinstance(record, GameRecord)
    assert record.final_status == "max_moves" # Because we set max_half_moves=4
    assert record.total_turns == 2 # 4 half moves = 2 full turns
    
    event_types = [e["type"] for e in events]
    assert "worker_status" in event_types
    assert "game_turn" in event_types
    assert "game_complete" in event_types


def test_game_worker_resume_from_checkpoint(base_kwargs, tmp_path, monkeypatch):
    # Setup a mid-game checkpoint (2 half moves played)
    board = chess.Board()
    board.push_uci("e2e4")
    board.push_uci("e7e5")
    
    save_game_state(
        tmp_path,
        game_id=base_kwargs["game_id"],
        condition=base_kwargs["condition"],
        experiment=base_kwargs["experiment"],
        starting_fen=chess.STARTING_FEN,
        board_fen=board.fen(),
        move_stack_uci=["e2e4", "e7e5"],
        half_moves_played=2,
        turn_records=[],
        game_status="ongoing",
    )
    
    def fake_dispatch(*args, **kwargs):
        b = chess.Board(kwargs["fen"])
        move = list(b.legal_moves)[0]
        return {
            "game_status": "ongoing",
            "is_valid": True,
            "proposed_move": move.uci(),
        }
    monkeypatch.setattr(games_mod, "dispatch_turn_with_backoff", fake_dispatch)
    monkeypatch.setattr(games_mod, "config_for_condition", lambda c: MagicMock(max_react_steps=5))
    
    class FakeStockfish:
        def __init__(self, *args, **kwargs): pass
        def start(self): pass
        def close(self): pass
        def choose_move(self, fen):
            b = chess.Board(fen)
            return list(b.legal_moves)[0].uci()
    monkeypatch.setattr(games_mod, "StockfishWrapper", FakeStockfish)
    
    record = run_game_worker(**base_kwargs)
    
    assert record.final_status == "max_moves"
    # It started at half_moves_played=2, played 2 more, hit max 4.
    
    
def test_game_worker_early_termination_stop(base_kwargs, monkeypatch):
    stop_event = threading.Event()
    
    def fake_dispatch(*args, **kwargs):
        stop_event.set() # Stop after first move
        b = chess.Board(kwargs["fen"])
        return {
            "game_status": "ongoing",
            "is_valid": True,
            "proposed_move": list(b.legal_moves)[0].uci(),
        }
    monkeypatch.setattr(games_mod, "dispatch_turn_with_backoff", fake_dispatch)
    monkeypatch.setattr(games_mod, "config_for_condition", lambda c: MagicMock(max_react_steps=5))
    
    class FakeStockfish:
        def __init__(self, *args, **kwargs): pass
        def start(self): pass
        def close(self): pass
        def choose_move(self, fen):
            return "0000"
    monkeypatch.setattr(games_mod, "StockfishWrapper", FakeStockfish)
    
    events = []
    
    # Needs to be a loop that checks stop_event
    record = run_game_worker(
        **base_kwargs,
        stop_event=stop_event,
        on_progress=events.append
    )
    
    assert record is None # Because we stopped it
    
    # It should have saved state
    state_file = base_kwargs["output_dir"] / ".game_state" / f"{base_kwargs['game_id']}.json"
    assert state_file.exists()


def test_game_worker_exception(base_kwargs, monkeypatch):
    def fake_dispatch(*args, **kwargs):
        raise ValueError("API failed")
    monkeypatch.setattr(games_mod, "dispatch_turn_with_backoff", fake_dispatch)
    monkeypatch.setattr(games_mod, "config_for_condition", lambda c: MagicMock(max_react_steps=5))
    
    class FakeStockfish:
        def __init__(self, *args, **kwargs): pass
        def start(self): pass
        def close(self): pass
    monkeypatch.setattr(games_mod, "StockfishWrapper", FakeStockfish)
    
    events = []
    record = run_game_worker(
        **base_kwargs,
        on_progress=events.append
    )
    
    assert record is None
    event_types = [e["type"] for e in events]
    assert "worker_error" in event_types

def test_game_worker_natural_termination_checkmate(base_kwargs, monkeypatch):
    base_kwargs["starting_fen"] = "4k3/R7/8/8/8/8/8/4K2R w K - 0 1" # Mate in 1 for white (Rh8)
    
    def fake_dispatch(*args, **kwargs):
        return {
            "game_status": "ongoing",
            "is_valid": True,
            "proposed_move": "h1h8", # Checkmate
        }
    monkeypatch.setattr(games_mod, "dispatch_turn_with_backoff", fake_dispatch)
    monkeypatch.setattr(games_mod, "config_for_condition", lambda c: MagicMock(max_react_steps=5))
    
    class FakeStockfish:
        def __init__(self, *args, **kwargs): pass
        def start(self): pass
        def close(self): pass
    monkeypatch.setattr(games_mod, "StockfishWrapper", FakeStockfish)
    
    record = run_game_worker(**base_kwargs)
    assert record.final_status == "checkmate"
