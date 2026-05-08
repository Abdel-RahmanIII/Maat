from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import src.runner.core.orchestrator as orch_mod
from src.config import ModelConfig
from src.runner.core.orchestrator import Orchestrator


@pytest.fixture
def mock_deps(monkeypatch, tmp_path):
    # Mock configs and paths
    def fake_experiment_config_path(exp_id):
        p = tmp_path / f"experiment_{exp_id}.yaml"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("dummy")
        return p
    monkeypatch.setattr(orch_mod, "experiment_config_path", fake_experiment_config_path)
    
    def fake_load_experiment_config(path):
        return {
            "output_dir": tmp_path / "out",
            "puzzle_data": "dummy.json",
            "starting_positions": "dummy.txt",
            "model_config": ModelConfig(api_key="key"),
            "generation_strategy": "generator_only",
            "max_api_retries": 1,
            "backoff_base": 0.0,
            "backoff_max": 0.0,
            "input_mode": "fen",
            "max_half_moves": 2,
            "stockfish_elo": 1000,
            "stockfish_path": None,
        }
    monkeypatch.setattr(orch_mod, "load_experiment_config", fake_load_experiment_config)
    
    # Mock inputs
    monkeypatch.setattr(orch_mod, "load_puzzle_inputs", lambda p: [{"puzzle_id": "p1", "fen": "fen1"}, {"puzzle_id": "p2", "fen": "fen2"}])
    monkeypatch.setattr(orch_mod, "load_starting_positions", lambda p: ["fen1", "fen2"])
    
    # Mock checkpoints
    monkeypatch.setattr(orch_mod, "load_checkpoint", lambda p: [])
    monkeypatch.setattr(orch_mod, "list_incomplete_games", lambda p: [])
    
    # Mock rate limiter
    mock_rl = MagicMock()
    mock_rl.get_status.return_value = {"rpm_limit": 10}
    monkeypatch.setattr(orch_mod, "get_rate_limiter", lambda: mock_rl)


def test_orchestrator_initialization():
    orch = Orchestrator()
    assert orch.status == "idle"
    assert orch.get_full_status()["status"] == "idle"


def test_orchestrator_start_already_running():
    orch = Orchestrator()
    orch._status = "running"
    with pytest.raises(RuntimeError, match="Already running"):
        orch.start([{"id": 1, "conditions": ["A"]}])


def test_orchestrator_pause_resume_stop():
    orch = Orchestrator()
    
    # Should not do anything if not running
    orch.pause()
    assert orch.status == "idle"
    
    orch.resume()
    assert orch.status == "idle"
    
    orch.stop()
    assert orch.status == "idle"
    
    # Set to running
    orch._status = "running"
    
    orch.pause()
    assert orch.status == "paused"
    assert not orch._pause_event.is_set()
    
    orch.resume()
    assert orch.status == "running"
    assert orch._pause_event.is_set()
    
    orch.stop()
    assert orch.status == "stopping"
    assert orch._stop_event.is_set()
    assert orch._pause_event.is_set()


def test_orchestrator_get_full_status(mock_deps):
    orch = Orchestrator()
    orch.add_api_log_entry({"call": 1})
    orch._recent_errors.append({"error": "test"})
    
    status = orch.get_full_status()
    assert status["status"] == "idle"
    assert "rate_limits" in status
    assert len(status["api_log"]) == 1
    assert len(status["recent_errors"]) == 1


def test_orchestrator_run_puzzle_experiment_sequential(mock_deps, monkeypatch):
    events = []
    orch = Orchestrator(on_event=events.append)
    
    def fake_run_puzzle(*args, **kwargs):
        class FakeRecord:
            final_status = "completed"
        return FakeRecord()
        
    monkeypatch.setattr(orch_mod, "run_puzzle_worker", fake_run_puzzle)
    
    orch.start([{"id": 1, "conditions": ["A"]}], parallel_experiments=False)
    orch._run_thread.join()
    
    assert orch.status == "completed"
    
    event_types = [e["type"] for e in events]
    assert "run_started" in event_types
    assert "condition_started" in event_types
    assert "experiment_finished" in event_types
    assert "run_finished" in event_types
    
    full_status = orch.get_full_status()
    cond_a = full_status["experiments"][1]["conditions"]["A"]
    assert cond_a["total"] == 2
    assert cond_a["completed"] == 2


def test_orchestrator_run_game_experiment_parallel(mock_deps, monkeypatch):
    orch = Orchestrator()
    
    def fake_run_game(*args, **kwargs):
        class FakeRecord:
            final_status = "completed"
        return FakeRecord()
        
    monkeypatch.setattr(orch_mod, "run_game_worker", fake_run_game)
    
    orch.start([{"id": 2, "conditions": ["A"]}, {"id": 3, "conditions": ["B"]}], parallel_experiments=True)
    orch._run_thread.join()
    
    assert orch.status == "completed"
    
    full_status = orch.get_full_status()
    assert full_status["experiments"][2]["conditions"]["A"]["completed"] == 2
    assert full_status["experiments"][3]["conditions"]["B"]["completed"] == 2


def test_orchestrator_missing_config(mock_deps, monkeypatch, tmp_path):
    orch = Orchestrator()
    # Mock to return a non-existent path
    monkeypatch.setattr(orch_mod, "experiment_config_path", lambda exp_id: tmp_path / "does_not_exist.yaml")
    
    orch.start([{"id": 1, "conditions": ["A"]}], parallel_experiments=False)
    orch._run_thread.join()
    
    # Experiment doesn't get created in status
    assert 1 not in orch.get_full_status()["experiments"]


def test_orchestrator_worker_exception(mock_deps, monkeypatch):
    orch = Orchestrator()
    
    def fake_run_puzzle(*args, **kwargs):
        return None # Worker returns None on error
        
    monkeypatch.setattr(orch_mod, "run_puzzle_worker", fake_run_puzzle)
    
    orch.start([{"id": 1, "conditions": ["A"]}], parallel_experiments=False)
    orch._run_thread.join()
    
    # Exception is caught, experiment finishes
    assert orch.status == "completed"
    
    full_status = orch.get_full_status()
    cond_a = full_status["experiments"][1]["conditions"]["A"]
    assert cond_a["failed"] == 2


def test_orchestrator_handle_worker_event():
    orch = Orchestrator()
    orch._handle_worker_event({"type": "worker_error", "error": "test"})
    assert len(orch._recent_errors) == 1
    
    for _ in range(150):
        orch._handle_worker_event({"type": "worker_error", "error": "test"})
    assert len(orch._recent_errors) == 100 # capped at 100


def test_orchestrator_add_api_log_entry():
    orch = Orchestrator()
    for i in range(600):
        orch.add_api_log_entry({"call": i})
    
    logs = orch._get_recent_api_log(10)
    assert len(logs) == 10
    assert logs[-1]["call"] == 599
    
    # Total capped at 500
    with orch._api_log_lock:
        assert len(orch._api_log) == 500


def test_orchestrator_stop_interrupts_loop(mock_deps, monkeypatch):
    orch = Orchestrator()
    
    def fake_run_puzzle(*args, **kwargs):
        orch.stop()
        class FakeRecord:
            final_status = "completed"
        return FakeRecord()
        
    monkeypatch.setattr(orch_mod, "run_puzzle_worker", fake_run_puzzle)
    
    orch.start([{"id": 1, "conditions": ["A", "B"]}], parallel_experiments=False)
    orch._run_thread.join()
    
    assert orch.status == "stopped"
