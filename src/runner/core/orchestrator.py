"""Experiment orchestrator — manages worker threads and experiment lifecycle.

The orchestrator is the runner's *control-plane brain*:

- Reads experiment YAML configs and runner config.
- Spawns worker threads via `ThreadPoolExecutor`.
- Manages pause / resume / stop events.
- Tracks per-experiment, per-condition progress.
- Pushes real-time events to the WebSocket via callbacks.

It deliberately does *not* know anything about FastAPI/WebSockets directly.
It only emits dictionaries to an `on_event` callback.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from src.config import ModelConfig
from src.engine.config_loader import load_experiment_config
from src.engine.game_manager import load_starting_positions
from src.engine.puzzle_manager import load_puzzle_inputs
from src.engine.result_store import load_completed_game_ids
from src.runner.core.progress import ExperimentProgress
from src.runner.limiting.rate_limiter import get_rate_limiter
from src.runner.paths import experiment_config_path, project_root
from src.runner.persistence.checkpoint import (
    list_incomplete_games,
    save_run_progress,
)
from src.runner.workers.games import run_game_worker
from src.runner.workers.puzzles import run_puzzle_worker
from src.state import InputMode

logger = logging.getLogger(__name__)


class Orchestrator:
    """Manages the lifecycle of experiment runs."""

    def __init__(
        self,
        *,
        max_concurrent_per_condition: int = 5,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self._max_concurrent = max_concurrent_per_condition
        self._on_event = on_event

        # Lifecycle events shared across all workers
        self._pause_event = threading.Event()
        self._pause_event.set()  # Initially NOT paused
        self._stop_event = threading.Event()

        # State
        self._status = "idle"  # idle | running | paused | stopping | stopped | completed
        self._futures: list[Future] = []
        self._experiments: dict[int, ExperimentProgress] = {}
        self._run_thread: threading.Thread | None = None
        self._started_at: str | None = None
        self._active_workers: dict[str, dict[str, Any]] = {}
        self._active_lock = threading.Lock()
        self._recent_errors: list[dict[str, Any]] = []
        self._api_log: list[dict[str, Any]] = []
        self._api_log_lock = threading.Lock()

    # ── Public API ───────────────────────────────────────────────────

    @property
    def status(self) -> str:
        return self._status

    def start(
        self,
        experiments: list[dict[str, Any]],
        parallel_experiments: bool = False,
        generation_strategy: str = "generator_only",
    ) -> None:
        """Start running experiments.

        Parameters
        ----------
        experiments:
            List of `{"id": 1, "conditions": ["A", "B", ...]}`.
        parallel_experiments:
            If True, run all experiments concurrently.
            If False, run them sequentially.
        generation_strategy:
            Overrides the YAML config (`generator_only`, `planner_actor`, `threat_analyst`).
        """

        if self._status == "running":
            raise RuntimeError("Already running")

        self._stop_event.clear()
        self._pause_event.set()
        self._status = "running"
        self._started_at = datetime.now().isoformat()
        self._generation_strategy = generation_strategy
        self._futures.clear()
        self._experiments.clear()

        self._emit({"type": "run_started", "experiments": experiments})

        # Run in a background thread so API can return immediately
        self._run_thread = threading.Thread(
            target=self._run_experiments,
            args=(experiments, parallel_experiments),
            daemon=True,
        )
        self._run_thread.start()

    def pause(self) -> None:
        """Pause all workers."""

        if self._status != "running":
            return
        self._status = "paused"
        self._pause_event.clear()
        self._save_all_progress()
        self._emit({"type": "run_paused"})

    def resume(self) -> None:
        """Resume paused workers."""

        if self._status != "paused":
            return
        self._status = "running"
        self._pause_event.set()
        self._emit({"type": "run_resumed"})

    def stop(self) -> None:
        """Graceful stop — workers finish current unit and exit."""

        if self._status not in ("running", "paused"):
            return
        self._status = "stopping"
        self._stop_event.set()
        self._pause_event.set()  # Unblock paused workers
        self._save_all_progress()
        self._emit({"type": "run_stopping"})

    def get_full_status(self) -> dict[str, Any]:
        """Return comprehensive status for the dashboard."""

        rate_limiter = get_rate_limiter()

        with self._active_lock:
            active = list(self._active_workers.values())

        return {
            "status": self._status,
            "started_at": self._started_at,
            "rate_limits": rate_limiter.get_status(),
            "experiments": {exp_id: ep.to_dict() for exp_id, ep in self._experiments.items()},
            "active_workers": active,
            "recent_errors": self._recent_errors[-20:],
            "api_log": self._get_recent_api_log(50),
        }

    # ── Internal run logic ───────────────────────────────────────────

    def _run_experiments(self, exp_configs: list[dict[str, Any]], parallel: bool) -> None:
        """Main orchestration loop (runs in background thread)."""

        try:
            if parallel:
                threads: list[threading.Thread] = []
                for exp_cfg in exp_configs:
                    t = threading.Thread(
                        target=self._run_single_experiment,
                        args=(exp_cfg,),
                        daemon=True,
                    )
                    threads.append(t)
                    t.start()
                for t in threads:
                    t.join()
            else:
                for exp_cfg in exp_configs:
                    if self._stop_event.is_set():
                        break
                    self._run_single_experiment(exp_cfg)

        except Exception:
            logger.exception("Orchestrator error")
        finally:
            self._status = "completed" if not self._stop_event.is_set() else "stopped"
            self._save_all_progress()
            self._emit({"type": "run_finished", "status": self._status})

    def _run_single_experiment(self, exp_cfg: dict[str, Any]) -> None:
        """Run one experiment across all its conditions."""

        exp_id = exp_cfg["id"]
        conditions = [c.upper() for c in exp_cfg["conditions"]]

        yaml_path = experiment_config_path(exp_id)
        if not yaml_path.exists():
            logger.error("Config not found: %s", yaml_path)
            return

        config = load_experiment_config(yaml_path)
        output_dir: Path = config["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)

        model_config: ModelConfig = config["model_config"]

        gen_strategy = getattr(self, "_generation_strategy", None) or config["generation_strategy"]
        max_api_retries = config["max_api_retries"]
        backoff_base = config["backoff_base"]
        backoff_max = config["backoff_max"]

        exp_progress = ExperimentProgress(exp_id, conditions, output_dir=output_dir)
        exp_progress.status = "running"
        exp_progress.started_at = datetime.now().isoformat()
        self._experiments[exp_id] = exp_progress

        if exp_id == 1:
            self._run_puzzle_experiment(
                conditions=conditions,
                config=config,
                output_dir=output_dir,
                model_config=model_config,
                gen_strategy=gen_strategy,
                max_api_retries=max_api_retries,
                backoff_base=backoff_base,
                backoff_max=backoff_max,
                exp_progress=exp_progress,
            )
        else:
            self._run_game_experiment(
                experiment=exp_id,
                conditions=conditions,
                config=config,
                output_dir=output_dir,
                model_config=model_config,
                gen_strategy=gen_strategy,
                max_api_retries=max_api_retries,
                backoff_base=backoff_base,
                backoff_max=backoff_max,
                exp_progress=exp_progress,
            )

        exp_progress.status = "stopped" if self._stop_event.is_set() else "completed"
        self._emit({"type": "experiment_finished", "experiment": exp_id, "status": exp_progress.status})

    # ── Puzzle experiment (Exp 1) ───────────────────────────────────

    def _run_puzzle_experiment(
        self,
        *,
        conditions: list[str],
        config: dict[str, Any],
        output_dir: Path,
        model_config: ModelConfig,
        gen_strategy: str,
        max_api_retries: int,
        backoff_base: float,
        backoff_max: float,
        exp_progress: ExperimentProgress,
    ) -> None:
        puzzles = load_puzzle_inputs(config["puzzle_data"])

        max_workers = self._max_concurrent * len(conditions)
        executor = ThreadPoolExecutor(max_workers=max_workers)
        futures: list[tuple[Future, str, str]] = []

        try:
            for condition in conditions:
                if self._stop_event.is_set():
                    break

                results_path = output_dir / f"exp1_{condition}_results.jsonl"
                completed_ids = load_completed_game_ids(results_path, output_dir / ".checkpoint")

                remaining: list[tuple[dict[str, Any], int, str]] = []
                for idx, puzzle in enumerate(puzzles):
                    pid = puzzle.get("puzzle_id", f"pos_{idx}")
                    gid = f"exp1_{pid}_{condition}"
                    if gid not in completed_ids:
                        remaining.append((puzzle, idx, gid))

                exp_progress.init_condition(condition, total=len(puzzles))

                cp = exp_progress.conditions_progress[condition]
                cp.completed = len(puzzles) - len(remaining)

                self._emit({
                    "type": "condition_started",
                    "experiment": 1,
                    "condition": condition,
                    "total": len(puzzles),
                    "remaining": len(remaining),
                })

                for puzzle, _idx, gid in remaining:
                    if self._stop_event.is_set():
                        break

                    cp.record_start()
                    f = executor.submit(
                        self._tracked_puzzle_worker,
                        puzzle=puzzle,
                        condition=condition,
                        game_id=gid,
                        output_dir=output_dir,
                        model_config=model_config,
                        gen_strategy=gen_strategy,
                        max_api_retries=max_api_retries,
                        backoff_base=backoff_base,
                        backoff_max=backoff_max,
                        exp_progress=exp_progress,
                    )
                    futures.append((f, condition, gid))

            for f, _cond, gid in futures:
                try:
                    f.result()
                except Exception:
                    logger.exception("Future error for %s", gid)

        finally:
            executor.shutdown(wait=True)

    def _tracked_puzzle_worker(
        self,
        *,
        puzzle: dict[str, Any],
        condition: str,
        game_id: str,
        output_dir: Path,
        model_config: ModelConfig,
        gen_strategy: str,
        max_api_retries: int,
        backoff_base: float,
        backoff_max: float,
        exp_progress: ExperimentProgress,
    ) -> None:
        """Wrapper that updates progress tracking after puzzle completes."""

        with self._active_lock:
            self._active_workers[game_id] = {
                "game_id": game_id,
                "condition": condition,
                "experiment": 1,
                "status": "running",
                "started": time.time(),
            }

        try:
            record = run_puzzle_worker(
                puzzle=puzzle,
                condition=condition,
                game_id=game_id,
                output_dir=output_dir,
                model_config=model_config,
                generation_strategy=gen_strategy,
                max_api_retries=max_api_retries,
                backoff_base=backoff_base,
                backoff_max=backoff_max,
                pause_event=self._pause_event,
                stop_event=self._stop_event,
                on_progress=self._handle_worker_event,
            )

            cp = exp_progress.conditions_progress.get(condition)
            if cp:
                if record:
                    is_valid = record.final_status != "forfeit"
                    cp.record_complete(is_valid=is_valid)
                else:
                    cp.record_failure()

        finally:
            with self._active_lock:
                self._active_workers.pop(game_id, None)

    # ── Game experiments (Exp 2/3) ───────────────────────────────────

    def _run_game_experiment(
        self,
        *,
        experiment: int,
        conditions: list[str],
        config: dict[str, Any],
        output_dir: Path,
        model_config: ModelConfig,
        gen_strategy: str,
        max_api_retries: int,
        backoff_base: float,
        backoff_max: float,
        exp_progress: ExperimentProgress,
    ) -> None:
        positions = load_starting_positions(config["starting_positions"])
        _incomplete_ids = set(list_incomplete_games(output_dir))

        input_mode: InputMode = config.get("input_mode", "fen" if experiment == 2 else "history")
        max_half_moves = config.get("max_half_moves", 150)
        sf_elo = config.get("stockfish_elo", 1000)
        sf_path = config.get("stockfish_path")

        max_workers = self._max_concurrent * len(conditions)
        executor = ThreadPoolExecutor(max_workers=max_workers)
        futures: list[tuple[Future, str, str]] = []

        try:
            for condition in conditions:
                if self._stop_event.is_set():
                    break

                results_path = output_dir / f"exp{experiment}_{condition}_results.jsonl"
                completed_ids = load_completed_game_ids(results_path, output_dir / ".checkpoint")

                remaining: list[tuple[str, int, str]] = []
                for idx, fen in enumerate(positions):
                    gid = f"exp{experiment}_{condition}_game{idx:03d}"
                    if gid not in completed_ids:
                        remaining.append((fen, idx, gid))

                exp_progress.init_condition(condition, total=len(positions))
                cp = exp_progress.conditions_progress[condition]
                cp.completed = len(positions) - len(remaining)

                self._emit({
                    "type": "condition_started",
                    "experiment": experiment,
                    "condition": condition,
                    "total": len(positions),
                    "remaining": len(remaining),
                })

                for fen, _idx, gid in remaining:
                    if self._stop_event.is_set():
                        break

                    cp.record_start()
                    f = executor.submit(
                        self._tracked_game_worker,
                        game_id=gid,
                        starting_fen=fen,
                        condition=condition,
                        experiment=experiment,
                        output_dir=output_dir,
                        input_mode=input_mode,
                        model_config=model_config,
                        gen_strategy=gen_strategy,
                        max_half_moves=max_half_moves,
                        sf_elo=sf_elo,
                        sf_path=sf_path,
                        max_api_retries=max_api_retries,
                        backoff_base=backoff_base,
                        backoff_max=backoff_max,
                        exp_progress=exp_progress,
                    )
                    futures.append((f, condition, gid))

            for f, _cond, gid in futures:
                try:
                    f.result()
                except Exception:
                    logger.exception("Future error for %s", gid)

        finally:
            executor.shutdown(wait=True)

    def _tracked_game_worker(
        self,
        *,
        game_id: str,
        starting_fen: str,
        condition: str,
        experiment: int,
        output_dir: Path,
        input_mode: InputMode,
        model_config: ModelConfig,
        gen_strategy: str,
        max_half_moves: int,
        sf_elo: int,
        sf_path: str | None,
        max_api_retries: int,
        backoff_base: float,
        backoff_max: float,
        exp_progress: ExperimentProgress,
    ) -> None:
        """Wrapper that updates progress tracking after game completes."""

        with self._active_lock:
            self._active_workers[game_id] = {
                "game_id": game_id,
                "condition": condition,
                "experiment": experiment,
                "status": "running",
                "started": time.time(),
            }

        try:
            record = run_game_worker(
                game_id=game_id,
                starting_fen=starting_fen,
                condition=condition,
                experiment=experiment,
                output_dir=output_dir,
                input_mode=input_mode,
                model_config=model_config,
                generation_strategy=gen_strategy,
                max_half_moves=max_half_moves,
                stockfish_elo=sf_elo,
                stockfish_path=sf_path,
                max_api_retries=max_api_retries,
                backoff_base=backoff_base,
                backoff_max=backoff_max,
                pause_event=self._pause_event,
                stop_event=self._stop_event,
                on_progress=self._handle_worker_event,
            )

            cp = exp_progress.conditions_progress.get(condition)
            if cp:
                if record:
                    cp.record_complete(is_valid=record.final_status != "forfeit")
                else:
                    cp.record_failure()

        finally:
            with self._active_lock:
                self._active_workers.pop(game_id, None)

    # ── Event handling ───────────────────────────────────────────────

    def _handle_worker_event(self, event: dict[str, Any]) -> None:
        """Process events from workers."""

        if event.get("type") == "worker_error":
            self._recent_errors.append({**event, "timestamp": datetime.now().isoformat()})
            if len(self._recent_errors) > 100:
                self._recent_errors = self._recent_errors[-100:]

        self._emit(event)

    def _emit(self, event: dict[str, Any]) -> None:
        """Push event to the API layer via callback."""

        if self._on_event:
            try:
                self._on_event(event)
            except Exception:
                pass

    # ── Persistence for dashboard resume ─────────────────────────────

    def _save_all_progress(self) -> None:
        for exp_id, ep in self._experiments.items():
            output_dir = ep.output_dir or (project_root() / "results" / f"exp{exp_id}")
            save_run_progress(
                output_dir,
                experiment=exp_id,
                conditions=list(ep.conditions_progress.keys()),
                condition_progress={c: cp.to_dict() for c, cp in ep.conditions_progress.items()},
                status=self._status,
                started_at=self._started_at,
                paused_at=datetime.now().isoformat() if self._status == "paused" else None,
            )

    def _get_recent_api_log(self, n: int = 50) -> list[dict[str, Any]]:
        with self._api_log_lock:
            return self._api_log[-n:]

    def add_api_log_entry(self, entry: dict[str, Any]) -> None:
        """Add an API call log entry (reserved for future hooks)."""

        with self._api_log_lock:
            self._api_log.append(entry)
            if len(self._api_log) > 500:
                self._api_log = self._api_log[-500:]
