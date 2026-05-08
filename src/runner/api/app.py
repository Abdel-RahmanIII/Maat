"""FastAPI application for the Maat experiment runner.

This module is the primary *entrypoint* for the runner server:

- `create_app()` builds the FastAPI app used by Uvicorn.
- `main()` runs Uvicorn and serves the dashboard.

The API is intentionally thin: it delegates experiment execution to
`src.runner.core.orchestrator.Orchestrator`.

Contract note
-------------
The dashboard HTML and JS (see `src/runner/dashboard.html`) expects stable
route paths and specific WebSocket event types. This module preserves that
contract.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

from src.runner.api.ws import ConnectionManager
from src.runner.core.orchestrator import Orchestrator
from src.runner.limiting.rate_limiter import get_rate_limiter
from src.runner.paths import dashboard_path, project_root, runner_config_path

logger = logging.getLogger(__name__)


def _load_runner_config() -> dict[str, Any]:
    """Load configs/runner.yaml, returning defaults if missing."""

    path = runner_config_path()
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    return {}


_manager = ConnectionManager()


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Capture the app event loop so worker threads can broadcast.
    _manager.set_event_loop(asyncio.get_event_loop())
    yield


def create_app() -> FastAPI:
    """FastAPI application factory."""

    app = FastAPI(
        title="Maat Experiment Runner",
        version="1.0.0",
        lifespan=_lifespan,
    )

    # Load configuration
    runner_cfg = _load_runner_config()
    rate_cfg = runner_cfg.get("rate_limits", {})
    concurrency_cfg = runner_cfg.get("concurrency", {})

    # Configure rate limiter
    limiter = get_rate_limiter()
    limiter.configure(
        rpm=rate_cfg.get("rpm", 15),
        rpd=rate_cfg.get("rpd", 1500),
        tpm=rate_cfg.get("tpm"),
        project_root=project_root(),
    )

    # Create orchestrator
    orchestrator = Orchestrator(
        max_concurrent_per_condition=concurrency_cfg.get(
            "max_concurrent_per_condition", 5
        ),
        on_event=_manager.broadcast_sync,
    )

    # Store in app state
    app.state.orchestrator = orchestrator
    app.state.limiter = limiter

    # ── Dashboard ────────────────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    async def dashboard() -> HTMLResponse:
        html = dashboard_path().read_text(encoding="utf-8")
        return HTMLResponse(html)

    # ── REST API ─────────────────────────────────────────────────────

    @app.get("/api/status")
    async def get_status() -> JSONResponse:
        return JSONResponse(orchestrator.get_full_status())

    @app.post("/api/start")
    async def start_experiments(body: dict[str, Any]) -> JSONResponse:
        """Start experiments.

        Body::

            {
                "experiments": [
                    {"id": 1, "conditions": ["A", "B", "C", "D", "E", "F"]},
                    {"id": 2, "conditions": ["A", "B"]}
                ],
                "parallel_experiments": false,
                "generation_strategy": "generator_only"
            }
        """

        try:
            experiments = body.get("experiments", [])
            parallel = body.get("parallel_experiments", False)
            gen_strategy = body.get("generation_strategy", "generator_only")

            if not experiments:
                return JSONResponse({"error": "No experiments specified"}, status_code=400)

            orchestrator.start(
                experiments,
                parallel_experiments=parallel,
                generation_strategy=gen_strategy,
            )
            return JSONResponse({"status": "started"})
        except RuntimeError as e:
            return JSONResponse({"error": str(e)}, status_code=409)

    @app.post("/api/pause")
    async def pause() -> JSONResponse:
        orchestrator.pause()
        return JSONResponse({"status": "paused"})

    @app.post("/api/resume")
    async def resume() -> JSONResponse:
        orchestrator.resume()
        return JSONResponse({"status": "resumed"})

    @app.post("/api/stop")
    async def stop() -> JSONResponse:
        orchestrator.stop()
        return JSONResponse({"status": "stopping"})

    @app.get("/api/rate-limits")
    async def rate_limits() -> JSONResponse:
        return JSONResponse(limiter.get_status())

    @app.get("/api/experiments")
    async def list_experiments() -> JSONResponse:
        """List available experiments from YAML configs.

        Always returns all 6 conditions (A-F) for every experiment
        so the user can freely choose any combination from the UI.
        """

        _ALL_CONDITIONS = ["A", "B", "C", "D", "E", "F"]
        configs_dir = project_root() / "configs"
        experiments = []
        for i in (1, 2, 3):
            path = configs_dir / f"experiment_{i}.yaml"
            if path.exists():
                experiments.append({
                    "id": i,
                    "conditions": _ALL_CONDITIONS,
                    "description": _experiment_description(i),
                })
        return JSONResponse(experiments)

    @app.get("/api/config")
    async def get_config() -> JSONResponse:
        return JSONResponse(runner_cfg)

    @app.post("/api/config")
    async def update_config(body: dict[str, Any]) -> JSONResponse:
        """Update runner configuration at runtime.

        Only rate limits are applied dynamically.
        """

        if "rate_limits" in body:
            rl = body["rate_limits"]
            limiter.configure(
                rpm=rl.get("rpm", rate_cfg.get("rpm", 15)),
                rpd=rl.get("rpd", rate_cfg.get("rpd", 1500)),
                tpm=rl.get("tpm", rate_cfg.get("tpm")),
                project_root=project_root(),
            )
        return JSONResponse({"status": "updated"})

    # ── WebSocket ────────────────────────────────────────────────────

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket) -> None:
        await _manager.connect(ws)
        try:
            # Initial full status snapshot
            await ws.send_json({
                "type": "initial_status",
                **orchestrator.get_full_status(),
            })

            # Keep alive — read messages (future interactive control)
            while True:
                try:
                    data = await asyncio.wait_for(ws.receive_text(), timeout=30)
                    msg = json.loads(data)
                    if msg.get("type") == "ping":
                        await ws.send_json({"type": "pong"})
                except asyncio.TimeoutError:
                    # Periodic status updates
                    try:
                        await ws.send_json({
                            "type": "status_update",
                            **orchestrator.get_full_status(),
                        })
                    except Exception:
                        break
        except WebSocketDisconnect:
            pass
        except Exception:
            pass
        finally:
            _manager.disconnect(ws)

    return app


def _experiment_description(exp_id: int) -> str:
    return {
        1: "Isolated Position Evaluation (Puzzles)",
        2: "Full Games with Board State (FEN)",
        3: "Full Games with Move History Only",
    }.get(exp_id, f"Experiment {exp_id}")


def main() -> None:
    """Run the server via Uvicorn."""

    # Ensure project root is on sys.path (helps when invoked from odd CWDs)
    root = str(project_root())
    if root not in sys.path:
        sys.path.insert(0, root)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    runner_cfg = _load_runner_config()
    server_cfg = runner_cfg.get("server", {})

    host = server_cfg.get("host", "localhost")
    port = server_cfg.get("port", 8420)

    app = create_app()

    logger.info("Starting Maat Experiment Runner on http://%s:%d", host, port)

    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
