"""Tests for the rewritten single-experiment Orchestrator."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import src.runner.core.orchestrator as orch_mod
from src.config import ModelConfig
from src.runner.core.orchestrator import Orchestrator


@pytest.fixture
def mock_deps(monkeypatch, tmp_path):
    """Mock external dependencies so the orchestrator runs in-process."""

    # Mock experiment config path
    def fake_experiment_config_path(exp_id):
        p = tmp_path / f"experiment_{exp_id}.yaml"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("dummy")
        return p

    monkeypatch.setattr(orch_mod, "experiment_config_path", fake_experiment_config_path)

    # Mock config loader
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

    # Mock puzzle/position loading
    monkeypatch.setattr(
        orch_mod, "load_puzzle_inputs",
        lambda p: [{"puzzle_id": "p1", "fen": "fen1"}, {"puzzle_id": "p2", "fen": "fen2"}],
    )
    monkeypatch.setattr(
        orch_mod, "load_starting_positions",
        lambda p: ["fen1", "fen2"],
    )

    # Mock RequestsManager so no real API calls are made
    mock_rm = MagicMock()
    mock_rm.get_status.return_value = {"paused": False, "queue_size": 0}

    def fake_rm_constructor(*args, **kwargs):
        return mock_rm

    monkeypatch.setattr(orch_mod, "RequestsManager", fake_rm_constructor)
    monkeypatch.setattr(orch_mod, "set_global_manager", lambda m: None)

    return mock_rm


def test_orchestrator_initialization():
    orch = Orchestrator()
    assert orch.status == "idle"
    status = orch.get_full_status()
    assert status["status"] == "idle"
    assert status["experiment"] == 1
    assert status["condition"] == "a"


def test_orchestrator_start_already_running():
    orch = Orchestrator()
    orch._status = "running"
    with pytest.raises(RuntimeError, match="Already running"):
        orch.start(experiment=1, condition="A")


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


def test_orchestrator_get_full_status():
    orch = Orchestrator()
    status = orch.get_full_status()
    assert status["status"] == "idle"
    assert "rate_limits" in status
    assert "progress" in status


def test_orchestrator_run_puzzles(mock_deps, monkeypatch, tmp_path):
    """Test that puzzles run to completion through _evaluate_puzzle."""

    events = []
    orch = Orchestrator(on_event=events.append)

    # Mock _evaluate_puzzle on PuzzleManager
    class FakeRecord:
        final_status = "completed"

        def model_dump_json(self):
            return json.dumps({"game_id": "test", "final_status": "completed"})

    def fake_evaluate_puzzle(self, puzzle, condition, game_id=None):
        return FakeRecord()

    monkeypatch.setattr(
        "src.engine.puzzle_manager.PuzzleManager._evaluate_puzzle",
        fake_evaluate_puzzle,
    )

    # Mock persistence (don't write to disk in tests)
    monkeypatch.setattr(orch_mod, "append_game_record", lambda r, p: None)
    monkeypatch.setattr(orch_mod, "append_checkpoint", lambda g, p: None)
    monkeypatch.setattr(orch_mod, "load_completed_game_ids", lambda *a, **kw: set())

    # Mock runner config
    monkeypatch.setattr(Orchestrator, "_load_runner_config", staticmethod(lambda: {}))

    orch.start(experiment=1, condition="A", n_runners=2)
    orch._run_thread.join(timeout=10)

    assert orch.status == "completed"

    event_types = [e["type"] for e in events]
    assert "run_started" in event_types
    assert "condition_started" in event_types
    assert "run_finished" in event_types


def test_orchestrator_skips_completed_puzzles(mock_deps, monkeypatch, tmp_path):
    """Test that already-completed puzzles are skipped."""

    events = []
    orch = Orchestrator(on_event=events.append)

    # Pretend p1 is already completed
    monkeypatch.setattr(
        orch_mod, "load_completed_game_ids",
        lambda *a, **kw: {"exp1_p1_A"},
    )

    call_count = 0

    class FakeRecord:
        final_status = "completed"

        def model_dump_json(self):
            return "{}"

    def fake_evaluate_puzzle(self, puzzle, condition, game_id=None):
        nonlocal call_count
        call_count += 1
        return FakeRecord()

    monkeypatch.setattr(
        "src.engine.puzzle_manager.PuzzleManager._evaluate_puzzle",
        fake_evaluate_puzzle,
    )
    monkeypatch.setattr(orch_mod, "append_game_record", lambda r, p: None)
    monkeypatch.setattr(orch_mod, "append_checkpoint", lambda g, p: None)
    monkeypatch.setattr(Orchestrator, "_load_runner_config", staticmethod(lambda: {}))

    orch.start(experiment=1, condition="A", n_runners=1)
    orch._run_thread.join(timeout=10)

    # Only p2 should have been evaluated
    assert call_count == 1


def test_orchestrator_stop_interrupts_runners(mock_deps, monkeypatch, tmp_path):
    """Test that stop event causes runners to exit."""

    orch = Orchestrator()

    class FakeRecord:
        final_status = "completed"

        def model_dump_json(self):
            return "{}"

    def fake_evaluate_puzzle(self, puzzle, condition, game_id=None):
        # Stop during first puzzle
        orch.stop()
        return FakeRecord()

    monkeypatch.setattr(
        "src.engine.puzzle_manager.PuzzleManager._evaluate_puzzle",
        fake_evaluate_puzzle,
    )
    monkeypatch.setattr(orch_mod, "append_game_record", lambda r, p: None)
    monkeypatch.setattr(orch_mod, "append_checkpoint", lambda g, p: None)
    monkeypatch.setattr(orch_mod, "load_completed_game_ids", lambda *a, **kw: set())
    monkeypatch.setattr(Orchestrator, "_load_runner_config", staticmethod(lambda: {}))

    orch.start(experiment=1, condition="A", n_runners=1)
    orch._run_thread.join(timeout=10)

    assert orch.status == "stopped"


def test_orchestrator_missing_config(monkeypatch, tmp_path):
    """Test that a missing YAML config stops gracefully."""

    orch = Orchestrator()

    monkeypatch.setattr(
        orch_mod, "experiment_config_path",
        lambda exp_id: tmp_path / "does_not_exist.yaml",
    )
    monkeypatch.setattr(Orchestrator, "_load_runner_config", staticmethod(lambda: {}))

    orch.start(experiment=1, condition="A")
    orch._run_thread.join(timeout=5)

    assert orch.status == "stopped"


def test_orchestrator_rpd_triggers_pause(mock_deps, monkeypatch, tmp_path):
    """Test that on_rpd_limit callback pauses the orchestrator."""

    orch = Orchestrator()
    orch._status = "running"

    orch._on_rpd_limit()

    assert orch.status == "paused"
    assert not orch._pause_event.is_set()
