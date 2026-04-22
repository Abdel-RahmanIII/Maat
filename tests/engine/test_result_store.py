import json
from pathlib import Path

import pytest

from src.metrics.definitions import GameRecord, TurnRecord
from src.engine.result_store import (
    append_checkpoint,
    append_game_record,
    load_checkpoint,
    load_game_records,
    write_summary_csv,
)

def test_result_store_roundtrip(tmp_path: Path):
    tr = TurnRecord(
        move_number=1,
        proposed_move="e2e4",
        is_valid=True,
        first_try_valid=True,
        total_attempts=1,
        llm_calls_this_turn=1,
        tokens_this_turn=100,
        prompt_token_count=50,
        wall_clock_ms=123.4,
        game_phase="opening",
        board_fen="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
    )
    gr = GameRecord(
        game_id="test_001",
        condition="A",
        experiment=1,
        turns=[tr],
        final_status="completed",
        total_turns=1,
        total_llm_calls=1,
        total_tokens=100,
        starting_fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    )

    fp = tmp_path / "test.jsonl"
    
    # Write and append
    append_game_record(gr, fp)
    append_game_record(gr, fp)
    
    # Read back
    loaded = load_game_records(fp)
    assert len(loaded) == 2
    assert loaded[0].game_id == "test_001"
    assert loaded[0].turns[0].proposed_move == "e2e4"

def test_checkpointing(tmp_path: Path):
    cp = tmp_path / "ckpt.txt"
    
    assert load_checkpoint(cp) == set()
    
    append_checkpoint("g1", cp)
    append_checkpoint("g2", cp)
    
    completed = load_checkpoint(cp)
    assert completed == {"g1", "g2"}

def test_write_summary_csv(tmp_path: Path):
    gr = GameRecord(
        game_id="test_001",
        condition="A",
        experiment=1,
        turns=[],
        final_status="completed",
        total_turns=2,
        total_llm_calls=2,
        total_tokens=200,
        starting_fen="some_fen",
    )
    
    csv_path = tmp_path / "summary.csv"
    write_summary_csv([gr], csv_path)
    
    with csv_path.open() as f:
        content = f.read()
        
    assert "game_id" in content
    assert "test_001" in content
    assert "A" in content
