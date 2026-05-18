import json
import pytest
import threading
from unittest.mock import MagicMock, patch
import chess

from src.engine.game_manager import GameManager
from src.metrics.collector import MetricsCollector
from src.context import ConversationContext

@pytest.fixture
def tmp_output_dir(tmp_path):
    return tmp_path / "results"

@pytest.fixture
def manager(tmp_output_dir):
    return GameManager(
        starting_positions=["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"],
        conditions=["A"],
        experiment=2,
        output_dir=tmp_output_dir,
        generation_strategy="test_strategy",
        max_half_moves=4,
    )

def test_checkpointing_directory_and_cleanup(manager, tmp_output_dir):
    checkpoints_seen = []

    def mock_llm_turn(*args, **kwargs):
        board = kwargs["board"]
        board.push(list(board.legal_moves)[0])
        return "ongoing"

    def mock_sf_turn(board, stockfish):
        # The checkpoint from White's turn should exist now
        gid = "exp2_A_game000"
        cond = "A"
        strat = manager.generation_strategy
        expected_path = tmp_output_dir / f"{strat}_{cond}" / "checkpoints" / f"{gid}.jsonl"
        
        if expected_path.exists():
            checkpoints_seen.append(expected_path)
        
        board.push(list(board.legal_moves)[0])

    with patch.object(manager, "_llm_turn", side_effect=mock_llm_turn), \
         patch.object(manager, "_stockfish_turn", side_effect=mock_sf_turn):
         
         sf_mock = MagicMock()
         manager._play_game(
             starting_fen=manager.starting_positions[0],
             condition="A",
             game_index=0,
             stockfish=sf_mock,
         )
         
    # Should have seen the checkpoint during Black's turn
    assert len(checkpoints_seen) > 0
    
    # Checkpoint must be deleted after max_half_moves reached
    gid = "exp2_A_game000"
    cond = "A"
    strat = manager.generation_strategy
    final_checkpoint = tmp_output_dir / f"{strat}_{cond}" / "checkpoints" / f"{gid}.jsonl"
    assert not final_checkpoint.exists()

def test_resume_from_state(manager, tmp_output_dir):
    starting_fen = manager.starting_positions[0]
    board = chess.Board(starting_fen)
    
    # Play 2 half-moves
    m1 = list(board.legal_moves)[0]
    board.push(m1)
    m2 = list(board.legal_moves)[0]
    board.push(m2)
    
    move_history = [m1.uci(), m2.uci()]
    
    collector = MetricsCollector(
        game_id="exp2_A_game000",
        condition="A",
        experiment=2,
        starting_fen=starting_fen,
    )
    context = ConversationContext()
    context.add_turn_messages("generator", [{"role": "user", "content": "hello"}])
    
    resume_state = {
        "starting_fen": starting_fen,
        "condition": "A",
        "game_id": "exp2_A_game000",
        "move_history": move_history,
        "collector": collector.to_dict(),
        "context": context.to_dict(),
    }
    
    call_counts = {"llm": 0, "sf": 0}
    
    def mock_llm_turn(*args, **kwargs):
        call_counts["llm"] += 1
        b = kwargs["board"]
        ctx = kwargs["context"]
        
        # Verify the context was restored
        assert len(ctx.get_history("generator")) == 1
        # Verify board state matches the resumed moves
        assert len(b.move_stack) >= 2
        
        b.push(list(b.legal_moves)[0])
        return "ongoing"

    def mock_sf_turn(b, stockfish):
        call_counts["sf"] += 1
        b.push(list(b.legal_moves)[0])

    with patch.object(manager, "_llm_turn", side_effect=mock_llm_turn), \
         patch.object(manager, "_stockfish_turn", side_effect=mock_sf_turn):
         
         sf_mock = MagicMock()
         manager._play_game(
             starting_fen=starting_fen,
             condition="A",
             game_index=0,
             stockfish=sf_mock,
             resume_state=resume_state
         )
         
    assert call_counts["llm"] == 1
    assert call_counts["sf"] == 1

def test_game_manager_stop_event(manager, tmp_output_dir):
    stop_event = threading.Event()
    stop_event.set()  # Immediate stop
    
    events_emitted = []
    def on_progress(event):
        events_emitted.append(event)
        
    sf_mock = MagicMock()
    record = manager._play_game(
        starting_fen=manager.starting_positions[0],
        condition="A",
        game_index=0,
        stockfish=sf_mock,
        stop_event=stop_event,
        on_progress=on_progress
    )
    
    assert record is None
    assert events_emitted[0]["status"] == "running"
    assert events_emitted[-1]["status"] == "stopped"
    
    gid = "exp2_A_game000"
    cond = "A"
    strat = manager.generation_strategy
    checkpoint_path = tmp_output_dir / f"{strat}_{cond}" / "checkpoints" / f"{gid}.jsonl"
    assert checkpoint_path.exists()

def test_game_manager_pause_event(manager):
    pause_event = threading.Event()
    pause_event.clear()  # Not set -> paused
    
    events_emitted = []
    def on_progress(event):
        events_emitted.append(event)
        if event.get("status") == "paused":
            pause_event.set()  # Resume so it finishes
            
    sf_mock = MagicMock()
    manager.max_half_moves = 0
    record = manager._play_game(
        starting_fen=manager.starting_positions[0],
        condition="A",
        game_index=0,
        stockfish=sf_mock,
        pause_event=pause_event,
        on_progress=on_progress
    )
    
    assert record is not None
    assert any(e.get("status") == "paused" for e in events_emitted)

def test_game_manager_on_progress_events(manager):
    events_emitted = []
    def on_progress(event):
        events_emitted.append(event)
        
    def mock_llm_turn(*args, **kwargs):
        board = kwargs["board"]
        board.push(list(board.legal_moves)[0])
        return "ongoing"

    def mock_sf_turn(board, stockfish):
        board.push(list(board.legal_moves)[0])
        
    with patch.object(manager, "_llm_turn", side_effect=mock_llm_turn), \
         patch.object(manager, "_stockfish_turn", side_effect=mock_sf_turn):
         
         sf_mock = MagicMock()
         manager._play_game(
             starting_fen=manager.starting_positions[0],
             condition="A",
             game_index=0,
             stockfish=sf_mock,
             on_progress=on_progress
         )
         
    types_emitted = [e["type"] for e in events_emitted]
    assert "worker_status" in types_emitted
    assert "game_turn" in types_emitted
    assert "game_complete" in types_emitted
