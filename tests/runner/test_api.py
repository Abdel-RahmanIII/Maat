"""Tests for the FastAPI runner application."""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import src.runner.api.app as app_mod


@pytest.fixture
def mock_orchestrator():
    mock_orch = MagicMock()
    return mock_orch


@pytest.fixture
def client(mock_orchestrator, monkeypatch):
    monkeypatch.setattr(app_mod, "Orchestrator", lambda *args, **kwargs: mock_orchestrator)

    app = app_mod.create_app()
    return TestClient(app)


def test_dashboard_html(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_get_status(client, mock_orchestrator):
    mock_orchestrator.get_full_status.return_value = {"status": "running"}

    response = client.get("/api/status")
    assert response.status_code == 200
    assert response.json() == {"status": "running"}


def test_start_experiment(client, mock_orchestrator):
    response = client.post("/api/start", json={
        "experiment": 1,
        "condition": "A",
        "generation_strategy": "generator_only",
        "n_runners": 5,
    })

    assert response.status_code == 200
    assert response.json() == {"status": "started"}
    mock_orchestrator.start.assert_called_once_with(
        experiment=1,
        condition="A",
        generation_strategy="generator_only",
        n_runners=5,
    )


def test_start_missing_params(client, mock_orchestrator):
    response = client.post("/api/start", json={})
    assert response.status_code == 400
    assert "experiment and condition are required" in response.json()["error"]


def test_start_conflict(client, mock_orchestrator):
    mock_orchestrator.start.side_effect = RuntimeError("Already running")

    response = client.post("/api/start", json={
        "experiment": 1,
        "condition": "A",
    })
    assert response.status_code == 409
    assert response.json() == {"error": "Already running"}


def test_pause_resume_stop(client, mock_orchestrator):
    assert client.post("/api/pause").json() == {"status": "paused"}
    mock_orchestrator.pause.assert_called_once()

    assert client.post("/api/resume").json() == {"status": "resumed"}
    mock_orchestrator.resume.assert_called_once()

    assert client.post("/api/stop").json() == {"status": "stopping"}
    mock_orchestrator.stop.assert_called_once()


def test_rate_limits_no_manager(client, monkeypatch):
    monkeypatch.setattr(
        "src.runner.requests.manager.get_global_manager", lambda: None,
    )
    response = client.get("/api/rate-limits")
    assert response.status_code == 200
    assert response.json() == {"status": "Stopped"}


def test_rate_limits_with_manager(client, monkeypatch):
    mock_rm = MagicMock()
    mock_rm.get_status.return_value = {"paused": False, "queue_size": 0}
    monkeypatch.setattr(
        "src.runner.requests.manager.get_global_manager", lambda: mock_rm,
    )
    response = client.get("/api/rate-limits")
    assert response.status_code == 200
    assert response.json()["paused"] is False


def test_list_experiments(client, monkeypatch):
    from pathlib import Path
    monkeypatch.setattr(app_mod, "project_root", lambda: Path("/fake/root"))

    def fake_exists(self):
        return True

    monkeypatch.setattr(Path, "exists", fake_exists)

    response = client.get("/api/experiments")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 3
    assert data[0]["id"] == 1


def test_get_update_config(client):
    response = client.get("/api/config")
    assert response.status_code == 200

    response = client.post("/api/config", json={
        "rate_limits": {"rpm": 100, "rpd": 2000},
    })
    assert response.status_code == 200


def test_websocket(client, mock_orchestrator):
    mock_orchestrator.get_full_status.return_value = {"status": "initial"}
    with client.websocket_connect("/ws") as websocket:
        # Should receive initial status
        data = websocket.receive_json()
        assert data["type"] == "initial_status"
        assert data["status"] == "initial"

        # Test ping-pong
        websocket.send_json({"type": "ping"})
        data = websocket.receive_json()
        assert data["type"] == "pong"
