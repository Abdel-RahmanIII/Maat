"""Experiment orchestrator — manages worker threads and experiment lifecycle.

The :class:`Orchestrator` is the central brain of the runner.  It:

1. Reads experiment YAML configs and runner config.
2. Spawns worker threads via ``ThreadPoolExecutor``.
3. Manages pause / resume / stop events.
4. Tracks per-experiment, per-condition progress.
5. Pushes real-time events to the WebSocket via callbacks.
"""

from __future__ import annotations

import json
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
from src.engine.result_store import load_checkpoint
from src.runner.checkpoint import (
    list_incomplete_games,
    load_run_progress,
    save_run_progress,
)
from src.runner.rate_limiter import get_rate_limiter
from src.runner.worker import run_game_worker, run_puzzle_worker
from src.state import InputMode

logger = logging.getLogger(__name__)

# Project root for resolving paths
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ── Progress tracking data structures ────────────────────────────────────


class ConditionProgress:
    """Thread-safe per-condition progress tracker."""

    def __init__(self, condition: str, total: int) -> None:
        self.condition = condition
        self.total = total
        self._lock = threading.Lock()
        self.completed = 0
        self.failed = 0
        self.in_progress = 0
        self.valid_count = 0

    def record_complete(self, is_valid: bool = True) -> None:
        with self._lock:
            self.completed += 1
            self.in_progress = max(0, self.in_progress - 1)
            if is_valid:
                self.valid_count += 1

    def record_start(self) -> None:
        with self._lock:
            self.in_progress += 1

    def record_failure(self) -> None:
        with self._lock:
            self.failed += 1
            self.in_progress = max(0, self.in_progress - 1)

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "condition": self.condition,
                "total": self.total,
                "completed": self.completed,
                "failed": self.failed,
                "in_progress": self.in_progress,
                "valid_count": self.valid_count,
            }


class ExperimentProgress:
    """Thread-safe per-experiment progress tracker."""

    def __init__(self, experiment: int, conditions: list[str], output_dir: Path | None = None) -> None:
        self.experiment = experiment
        self.conditions_progress: dict[str, ConditionProgress] = {}
        self._conditions = conditions
        self.status = "pending"
        self.started_at: str | None = None
        self.output_dir: Path | None = output_dir

    def init_condition(self, condition: str, total: int) -> None:
        self.conditions_progress[condition] = ConditionProgress(condition, total)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment": self.experiment,
            "status": self.status,
            "started_at": self.started_at,
            "conditions": {
                c: cp.to_dict() for c, cp in self.conditions_progress.items()
            },
        }


# ── Orchestrator ─────────────────────────────────────────────────────────


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

        # Lifecycle events
        self._pause_event = threading.Event()
        self._pause_event.set()  # Initially NOT paused
        self._stop_event = threading.Event()

        # State
        self._status = "idle"  # idle | running | paused | stopping | stopped
        self._executor: ThreadPoolExecutor | None = None
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
            List of ``{"id": 1, "conditions": ["A", "B", ...]}``.
        parallel_experiments:
            If ``True``, run all experiments concurrently.
            If ``False``, run them sequentially.
        generation_strategy:
            One of ``generator_only``, ``planner_actor``,
            ``threat_analyst``.  Overrides the YAML config.
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

        # Run in a background thread so the API endpoint returns immediately
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
        self._pause_event.set()  # Unblock any paused workers
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
            "experiments": {
                exp_id: ep.to_dict()
                for exp_id, ep in self._experiments.items()
            },
            "active_workers": active,
            "recent_errors": self._recent_errors[-20:],
            "api_log": self._get_recent_api_log(50),
        }

    # ── Internal run logic ───────────────────────────────────────────

    def _run_experiments(
        self,
        exp_configs: list[dict[str, Any]],
        parallel: bool,
    ) -> None:
        """Main orchestration loop (runs in background thread)."""

        try:
            if parallel:
                threads = []
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
            if not self._stop_event.is_set():
                self._status = "completed"
            else:
                self._status = "stopped"
            self._save_all_progress()
            self._emit({"type": "run_finished", "status": self._status})

    def _run_single_experiment(self, exp_cfg: dict[str, Any]) -> None:
        """Run one experiment across all its conditions."""

        exp_id = exp_cfg["id"]
        conditions = [c.upper() for c in exp_cfg["conditions"]]

        # Load experiment YAML config
        yaml_path = _PROJECT_ROOT / "configs" / f"experiment_{exp_id}.yaml"
        if not yaml_path.exists():
            logger.error("Config not found: %s", yaml_path)
            return

        config = load_experiment_config(yaml_path)
        output_dir = config["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)

        model_config = config["model_config"]
        # Use the UI-selected generation strategy, falling back to YAML config
        gen_strategy = getattr(self, "_generation_strategy", None) or config["generation_strategy"]
        max_api_retries = config["max_api_retries"]
        backoff_base = config["backoff_base"]
        backoff_max = config["backoff_max"]

        # Build experiment progress tracker
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
        self._emit({
            "type": "experiment_finished",
            "experiment": exp_id,
            "status": exp_progress.status,
        })

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
        """Run Experiment 1 (puzzles) across conditions."""

        puzzles = load_puzzle_inputs(config["puzzle_data"])
        completed_ids = load_checkpoint(output_dir / ".checkpoint")

        # Total workers = conditions × puzzles
        max_workers = self._max_concurrent * len(conditions)
        executor = ThreadPoolExecutor(max_workers=max_workers)
        futures: list[tuple[Future, str, str]] = []

        try:
            for condition in conditions:
                if self._stop_event.is_set():
                    break

                # Count remaining puzzles
                remaining = []
                for idx, puzzle in enumerate(puzzles):
                    pid = puzzle.get("puzzle_id", f"pos_{idx}")
                    gid = f"exp1_{pid}_{condition}"
                    if gid not in completed_ids:
                        remaining.append((puzzle, idx, gid))

                exp_progress.init_condition(
                    condition,
                    total=len(puzzles),
                )
                # Mark already-completed count
                cp = exp_progress.conditions_progress[condition]
                cp.completed = len(puzzles) - len(remaining)

                self._emit({
                    "type": "condition_started",
                    "experiment": 1,
                    "condition": condition,
                    "total": len(puzzles),
                    "remaining": len(remaining),
                })

                for puzzle, idx, gid in remaining:
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

            # Wait for all futures
            for f, cond, gid in futures:
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
        """Run Experiment 2 or 3 (full games) across conditions."""

        positions = load_starting_positions(config["starting_positions"])
        completed_ids = load_checkpoint(output_dir / ".checkpoint")
        incomplete_ids = set(list_incomplete_games(output_dir))

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

                remaining = []
                for idx, fen in enumerate(positions):
                    gid = f"exp{experiment}_{condition}_game{idx:03d}"
                    if gid not in completed_ids:
                        remaining.append((fen, idx, gid))

                exp_progress.init_condition(
                    condition,
                    total=len(positions),
                )
                cp = exp_progress.conditions_progress[condition]
                cp.completed = len(positions) - len(remaining)

                self._emit({
                    "type": "condition_started",
                    "experiment": experiment,
                    "condition": condition,
                    "total": len(positions),
                    "remaining": len(remaining),
                })

                for fen, idx, gid in remaining:
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

            for f, cond, gid in futures:
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

        event_type = event.get("type", "")

        if event_type == "worker_error":
            self._recent_errors.append({
                **event,
                "timestamp": datetime.now().isoformat(),
            })
            # Keep only last 100 errors
            if len(self._recent_errors) > 100:
                self._recent_errors = self._recent_errors[-100:]

        self._emit(event)

    def _emit(self, event: dict[str, Any]) -> None:
        """Push event to WebSocket via callback."""
        if self._on_event:
            try:
                self._on_event(event)
            except Exception:
                pass

    def _save_all_progress(self) -> None:
        """Persist progress for all experiments."""

        for exp_id, ep in self._experiments.items():
            output_dir = ep.output_dir or (_PROJECT_ROOT / "results" / f"exp{exp_id}")
            save_run_progress(
                output_dir,
                experiment=exp_id,
                conditions=list(ep.conditions_progress.keys()),
                condition_progress={
                    c: cp.to_dict()
                    for c, cp in ep.conditions_progress.items()
                },
                status=self._status,
                started_at=self._started_at,
                paused_at=datetime.now().isoformat() if self._status == "paused" else None,
            )

    def _get_recent_api_log(self, n: int = 50) -> list[dict[str, Any]]:
        with self._api_log_lock:
            return self._api_log[-n:]

    def add_api_log_entry(self, entry: dict[str, Any]) -> None:
        """Add an API call log entry (called from llm_client hook)."""
        with self._api_log_lock:
            self._api_log.append(entry)
            if len(self._api_log) > 500:
                self._api_log = self._api_log[-500:]
