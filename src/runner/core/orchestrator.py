"""Experiment orchestrator — single experiment/condition/strategy runner.

The orchestrator is the runner's *control-plane brain*:

- Accepts exactly one ``(experiment, condition, generation_strategy)`` per run.
- Initialises the :class:`RequestsManager` for rate-limited LLM dispatch.
- Loads data from YAML configs and detects existing results/checkpoints.
- Spawns N concurrent runners via ``ThreadPoolExecutor``.
- Manages pause / resume / stop lifecycle.
- Derives all progress from disk (results files + checkpoint files).
- Pushes real-time events to the WebSocket via callbacks.
- Tracks active workers in-memory for live dashboard visibility.
- Handles terminated requests: saves to failed file, frees workers.

It deliberately does *not* know anything about FastAPI/WebSockets directly.
It only emits dictionaries to an ``on_event`` callback.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from src.config import ModelConfig
from src.engine.config_loader import load_experiment_config
from src.engine.game_manager import GameManager, load_starting_positions
from src.engine.puzzle_manager import PuzzleManager, load_puzzle_inputs
from src.engine.result_store import (
    append_checkpoint,
    append_failed_item,
    append_game_record,
    load_completed_game_ids,
    load_failed_item_ids,
)
from src.runner.core.progress import get_progress_from_disk
from src.runner.paths import experiment_config_path
from src.runner.requests.manager import (
    RequestsManager,
    RequestsPausedError,
    RequestTerminatedError,
    set_global_manager,
)

logger = logging.getLogger(__name__)

# How long a single puzzle/game request can block before we consider it
# failed and move the worker on to the next item.
_REQUEST_TIMEOUT_SECONDS = 300.0  # 5 minutes


class Orchestrator:
    """Manages the lifecycle of a single experiment/condition/strategy run."""

    def __init__(
        self,
        *,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self._on_event = on_event

        # Lifecycle events shared across all runners
        self._pause_event = threading.Event()
        self._pause_event.set()  # Initially NOT paused
        self._stop_event = threading.Event()

        # State
        self._status = "idle"  # idle | running | paused | stopping | stopped | completed
        self._run_thread: threading.Thread | None = None
        self._started_at: str | None = None
        self._requests_manager: RequestsManager | None = None

        # Current run config (set on start)
        self._experiment: int | int = 1
        self._condition: str = "a"  # Default value
        self._generation_strategy: str | None = None
        self._output_dir: Path = Path("results/")
        self._n_runners: int = 5
        self._total_items: int = 0

        # Live worker registry: game_id → worker info dict
        self._workers: dict[str, dict[str, Any]] = {}
        self._workers_lock = threading.Lock()

    # ── Public API ───────────────────────────────────────────────────

    @property
    def status(self) -> str:
        return self._status

    def start(
        self,
        experiment: int,
        condition: str,
        generation_strategy: str = "generator_only",
        n_runners: int = 5,
    ) -> None:
        """Start running a single experiment + condition + strategy.

        Parameters
        ----------
        experiment:
            Experiment ID (1, 2, or 3).
        condition:
            Single condition letter (e.g. ``"A"``).
        generation_strategy:
            ``"generator_only"``, ``"planner_actor"``, ``"observer_executor"``, or
            ``"observer_strategist_tactician"``.
        n_runners:
            Number of concurrent runner threads.
        """

        if self._status == "running":
            raise RuntimeError("Already running")

        self._stop_event.clear()
        self._pause_event.set()
        self._status = "running"
        self._started_at = datetime.now().isoformat()
        self._experiment = experiment
        self._condition = condition.upper()
        self._generation_strategy = generation_strategy
        self._n_runners = n_runners

        # Clear worker registry from any previous run
        with self._workers_lock:
            self._workers.clear()

        self._emit({
            "type": "run_started",
            "experiment": experiment,
            "condition": self._condition,
            "generation_strategy": generation_strategy,
        })

        # Run in a background thread so API can return immediately
        self._run_thread = threading.Thread(
            target=self._run,
            daemon=True,
        )
        self._run_thread.start()

    def pause(self) -> None:
        """Pause all runners and the requests manager."""

        if self._status not in ("running",):
            return
        self._status = "paused"
        self._pause_event.clear()

        # Pause the requests manager — drains its queue
        if self._requests_manager:
            self._requests_manager.pause()

        self._emit({"type": "run_paused"})

    def resume(self) -> None:
        """Resume paused runners — starts the next item in the pipeline."""

        if self._status != "paused":
            return
        self._status = "running"

        # Resume requests manager first
        if self._requests_manager:
            self._requests_manager.resume()

        self._pause_event.set()
        self._emit({"type": "run_resumed"})

    def stop(self) -> None:
        """Graceful stop — runners finish current unit and exit."""

        if self._status not in ("running", "paused"):
            return
        self._status = "stopping"
        self._stop_event.set()
        self._pause_event.set()  # Unblock paused runners

        # Stop the requests manager
        if self._requests_manager:
            self._requests_manager.stop()

        self._emit({"type": "run_stopping"})

    def get_full_status(self) -> dict[str, Any]:
        """Return comprehensive status for the dashboard.

        Progress is derived from disk — never stale.
        Workers list is from in-memory registry — always live.
        """

        with self._workers_lock:
            workers_snapshot = list(self._workers.values())

        result: dict[str, Any] = {
            "status": self._status,
            "started_at": self._started_at,
            "experiment": self._experiment,
            "condition": self._condition,
            "generation_strategy": self._generation_strategy,
            "rate_limits": {},
            "progress": {},
            "workers": workers_snapshot,
        }

        # Rate-limit status from RequestsManager
        if self._requests_manager:
            result["rate_limits"] = self._requests_manager.get_status()

        # Progress from disk
        if self._experiment and self._condition and self._output_dir:
            result["progress"] = get_progress_from_disk(
                experiment=self._experiment,
                condition=self._condition,
                generation_strategy=self._generation_strategy or "generator_only",
                output_dir=self._output_dir,
                total_items=self._total_items,
            )

        return result

    # ── Worker registry ──────────────────────────────────────────────

    def _register_worker(
        self,
        game_id: str,
        experiment: int,
        condition: str,
        detail: str = "",
    ) -> None:
        """Register an active worker in the live registry."""

        with self._workers_lock:
            self._workers[game_id] = {
                "game_id": game_id,
                "experiment": experiment,
                "condition": condition,
                "status": "running",
                "detail": detail,
                "started": time.time(),
            }

    def _update_worker(self, game_id: str, **fields: Any) -> None:
        """Update fields on a registered worker."""

        with self._workers_lock:
            if game_id in self._workers:
                self._workers[game_id].update(fields)

    def _unregister_worker(self, game_id: str) -> None:
        """Remove a worker from the live registry."""

        with self._workers_lock:
            self._workers.pop(game_id, None)

    # ── Internal run logic ───────────────────────────────────────────

    def _run(self) -> None:
        """Main orchestration loop (runs in background thread)."""

        try:
            # 1. Load experiment config
            yaml_path = experiment_config_path(self._experiment)
            if not yaml_path.exists():
                logger.error("Config not found: %s", yaml_path)
                self._status = "stopped"
                return

            config = load_experiment_config(yaml_path)
            output_dir: Path = config["output_dir"]
            output_dir.mkdir(parents=True, exist_ok=True)
            self._output_dir = output_dir

            model_config: ModelConfig = config["model_config"]
            gen_strategy = self._generation_strategy or config["generation_strategy"]

            # 2. Create and start RequestsManager
            runner_cfg = self._load_runner_config()
            rate_cfg = runner_cfg.get("rate_limits", {})

            self._requests_manager = RequestsManager(
                model_config=model_config,
                rpm_limit=rate_cfg.get("rpm", 15),
                rpd_limit=rate_cfg.get("rpd", 1500),
                max_retries=rate_cfg.get("max_retries", 5),
                backoff_base=rate_cfg.get("backoff_base", 2.0),
                backoff_max=rate_cfg.get("backoff_max", 60.0),
                request_timeout=rate_cfg.get("request_timeout", _REQUEST_TIMEOUT_SECONDS),
                on_global_rpd_limit=self._on_rpd_limit,
            )
            set_global_manager(self._requests_manager)
            self._requests_manager.start()

            # 3. Dispatch to puzzle or game runner
            if self._experiment == 1:
                self._run_puzzles(config, model_config, gen_strategy)
            else:
                self._run_games(config, model_config, gen_strategy)

        except Exception:
            logger.exception("Orchestrator error")
        finally:
            # Cleanup
            if self._requests_manager:
                self._requests_manager.stop()
                set_global_manager(None)

            # Clear worker registry
            with self._workers_lock:
                self._workers.clear()

            final_status = "completed" if not self._stop_event.is_set() else "stopped"
            # If we were paused and stop wasn't set, we completed via pause
            if self._status == "paused":
                final_status = "paused"
            # If status was already set to "stopped" by an early exit, preserve it
            elif self._status == "stopped":
                final_status = "stopped"
            self._status = final_status
            self._emit({"type": "run_finished", "status": self._status})

    def _on_rpd_limit(self) -> None:
        """Called by RequestsManager when all API keys exhaust RPD."""

        logger.warning("All API keys exhausted RPD — pausing orchestrator.")
        self.pause()

    # ── Puzzle experiment (Exp 1) ────────────────────────────────────

    def _run_puzzles(
        self,
        config: dict[str, Any],
        model_config: ModelConfig,
        gen_strategy: str,
    ) -> None:
        """Run puzzles for the current condition."""

        condition = self._condition
        output_dir = self._output_dir

        # Load all puzzles
        puzzles = load_puzzle_inputs(config["puzzle_data"])
        self._total_items = len(puzzles)

        # Detect completed puzzles from results file
        results_path = self._output_dir / f"{self._generation_strategy}_{self._condition}" / "results.jsonl"
        completed_ids = load_completed_game_ids(results_path)

        # Detect previously failed puzzles
        failed_path = self._output_dir / f"{self._generation_strategy}_{self._condition}" / "fail.jsonl"
        failed_ids = load_failed_item_ids(failed_path)

        # Build work queue of remaining puzzles (skip completed AND failed)
        work_queue: list[tuple[dict[str, Any], str]] = []
        for idx, puzzle in enumerate(puzzles):
            pid = puzzle.get("puzzle_id", f"pos_{idx}")
            gid = f"exp1_{pid}_{condition}"
            if gid not in completed_ids and gid not in failed_ids:
                work_queue.append((puzzle, gid))

        remaining = len(work_queue)
        logger.info(
            "[Orchestrator] Exp1/%s: %d total, %d completed, %d failed, %d remaining",
            condition, len(puzzles), len(puzzles) - remaining - len(failed_ids),
            len(failed_ids), remaining,
        )

        self._emit({
            "type": "condition_started",
            "experiment": 1,
            "condition": condition,
            "total": len(puzzles),
            "remaining": remaining,
        })

        if not work_queue:
            return

        # Create PuzzleManager (used by runners to call _evaluate_puzzle)
        puzzle_manager = PuzzleManager(
            puzzles=puzzles,
            conditions=[condition],
            output_dir=output_dir,
            model_config=model_config,
            generation_strategy=gen_strategy,
        )

        # Use a thread-safe queue for work distribution
        import queue
        q: queue.Queue[tuple[dict[str, Any], str]] = queue.Queue()
        for item in work_queue:
            q.put(item)

        # Spawn N runners
        executor = ThreadPoolExecutor(max_workers=self._n_runners)
        futures: list[Future] = []

        for _ in range(self._n_runners):
            f = executor.submit(
                self._puzzle_runner,
                work_queue=q,
                puzzle_manager=puzzle_manager,
                condition=condition,
                output_dir=output_dir,
            )
            futures.append(f)

        # Wait for all runners to complete
        try:
            for f in futures:
                try:
                    f.result()
                except Exception:
                    logger.exception("Puzzle runner error")
        finally:
            executor.shutdown(wait=True)

    def _puzzle_runner(
        self,
        *,
        work_queue: Any,  # queue.Queue
        puzzle_manager: PuzzleManager,
        condition: str,
        output_dir: Path,
    ) -> None:
        """Runner thread: pulls puzzles from queue and evaluates them.

        Lifecycle per puzzle:
        1. Pull from queue → register worker → emit worker_status(running)
        2. Call _evaluate_puzzle
        3. On success: persist → emit puzzle_complete → unregister → next
        4. On terminated/timeout: save to failed file → unregister → next
        5. On other error: save to failed file → unregister → next
        6. Queue empty or stop signaled → exit
        """

        import queue

        while not self._stop_event.is_set():
            # ── Check pause ──
            if not self._pause_event.is_set():
                self._pause_event.wait()
                if self._stop_event.is_set():
                    break
                continue

            # ── Get next puzzle ──
            try:
                puzzle, game_id = work_queue.get_nowait()
            except queue.Empty:
                break  # No more work

            # ── Register worker ──
            puzzle_id = puzzle.get("puzzle_id", "?")
            self._register_worker(game_id, 1, condition, detail=f"Puzzle {puzzle_id}")

            self._emit({
                "type": "worker_status",
                "game_id": game_id,
                "condition": condition,
                "experiment": 1,
                "status": "running",
                "detail": f"Puzzle {puzzle_id}",
            })

            try:
                record = puzzle_manager._evaluate_puzzle(
                    puzzle=puzzle,
                    condition=condition,
                    game_id=game_id,
                )

                # ── Success: persist results ──
                results_path = self._output_dir / f"{self._generation_strategy}_{self._condition}" / "results.jsonl"
                append_game_record(record, results_path)

                # Extract move data and puzzle info for dashboard
                proposed_move = ""
                is_valid = False
                puzzle_difficulty = puzzle.get("difficulty", "unknown")
                game_phase = ""
                if record.turns:
                    last_turn = record.turns[-1]
                    proposed_move = last_turn.proposed_move
                    is_valid = last_turn.is_valid
                    game_phase = last_turn.game_phase

                self._emit({
                    "type": "puzzle_complete",
                    "game_id": game_id,
                    "condition": condition,
                    "experiment": 1,
                    "status": record.final_status,
                    "proposed_move": proposed_move,
                    "is_valid": is_valid,
                    "puzzle_difficulty": puzzle_difficulty,
                    "game_phase": game_phase,
                })

            except (RequestsPausedError, RequestTerminatedError) as exc:
                # Request was terminated by the manager — skip and log
                error_msg = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "[Orchestrator] Puzzle %s terminated — saving to failed file: %s",
                    game_id, error_msg,
                )
                append_failed_item(game_id, error_msg, self._output_dir / f"{self._generation_strategy}_{self._condition}" / "fail.jsonl")
                self._emit({
                    "type": "worker_error",
                    "game_id": game_id,
                    "condition": condition,
                    "experiment": 1,
                    "error": error_msg,
                })
                # Worker is free — continue to next puzzle

            except Exception as exc:
                error_msg = f"{type(exc).__name__}: {exc}"
                logger.exception("Puzzle runner error on %s", game_id)
                append_failed_item(game_id, error_msg, self._output_dir / f"{self._generation_strategy}_{self._condition}" / "fail.jsonl")
                self._emit({
                    "type": "worker_error",
                    "game_id": game_id,
                    "condition": condition,
                    "experiment": 1,
                    "error": error_msg,
                })
                # Worker is free — continue to next puzzle

            finally:
                self._unregister_worker(game_id)

    # ── Game experiments (Exp 2/3) ───────────────────────────────────

    def _run_games(
        self,
        config: dict[str, Any],
        model_config: ModelConfig,
        gen_strategy: str,
    ) -> None:
        """Run games for the current condition."""

        condition = self._condition
        output_dir = self._output_dir
        experiment = self._experiment

        # Load starting positions
        positions = load_starting_positions(config["starting_positions"])
        self._total_items = len(positions)

        # Detect completed games from results
        results_path = self._output_dir / f"{gen_strategy}_{condition}" / "results.jsonl"
        completed_ids = load_completed_game_ids(results_path)

        # Detect previously failed games
        failed_path = self._output_dir / f"{gen_strategy}_{condition}" / "fail.jsonl"
        failed_ids = load_failed_item_ids(failed_path)

        # Detect mid-game checkpoints (paused games)
        midgame_dir = output_dir / f"{gen_strategy}_{condition}" / "checkpoints"
        checkpoint_glob = "*.jsonl"
        paused_game_ids: set[str] = set()
        if midgame_dir.exists():
            paused_game_ids = {p.stem for p in midgame_dir.glob(checkpoint_glob)}

        # Build work queue: paused games first, then remaining new games
        import json
        work_queue_paused: list[tuple[str, int, str, dict[str, Any] | None]] = []
        work_queue_new: list[tuple[str, int, str, dict[str, Any] | None]] = []

        for idx, fen in enumerate(positions):
            gid = f"exp{experiment}_{condition}_game{idx:03d}"
            if gid in completed_ids or gid in failed_ids:
                continue

            if gid in paused_game_ids: # type: ignore
                # Load resume state
                cp_path = midgame_dir / f"{gid}.jsonl"
                try:
                    resume_state = json.loads(cp_path.read_text(encoding="utf-8"))
                except Exception:
                    logger.warning("Corrupt checkpoint for %s, treating as new", gid)
                    resume_state = None
                work_queue_paused.append((fen, idx, gid, resume_state))
            else:
                work_queue_new.append((fen, idx, gid, None))

        # Paused games first, then new games
        work_items = work_queue_paused + work_queue_new
        remaining = len(work_items)

        logger.info(
            "[Orchestrator] Exp%d/%s: %d total, %d completed, %d failed, %d paused, %d new",
            experiment, condition,
            len(positions), len(positions) - remaining - len(failed_ids),
            len(failed_ids), len(work_queue_paused), len(work_queue_new),
        )

        self._emit({
            "type": "condition_started",
            "experiment": experiment,
            "condition": condition,
            "total": len(positions),
            "remaining": remaining,
            "paused_games": len(work_queue_paused),
        })

        if not work_items:
            return

        # Create GameManager (used by runners to call _play_game)
        game_manager = GameManager(
            starting_positions=positions,
            conditions=[condition],
            experiment=experiment,
            output_dir=output_dir,
            stockfish_elo=config.get("stockfish_elo", 1000),
            stockfish_path=config.get("stockfish_path"),
            max_half_moves=config.get("max_half_moves", 150),
            model_config=model_config,
            generation_strategy=gen_strategy,
        )

        # Use a thread-safe queue for work distribution
        import queue
        q: queue.Queue[tuple[str, int, str, dict[str, Any] | None]] = queue.Queue()
        for item in work_items:
            q.put(item)

        # Spawn N runners
        executor = ThreadPoolExecutor(max_workers=self._n_runners)
        futures: list[Future] = []

        for _ in range(self._n_runners):
            f = executor.submit(
                self._game_runner,
                work_queue=q,
                game_manager=game_manager,
                condition=condition,
                experiment=experiment,
                output_dir=output_dir,
            )
            futures.append(f)

        # Wait for all runners to complete
        try:
            for f in futures:
                try:
                    f.result()
                except Exception:
                    logger.exception("Game runner error")
        finally:
            executor.shutdown(wait=True)

    def _game_runner(
        self,
        *,
        work_queue: Any,  # queue.Queue
        game_manager: GameManager,
        condition: str,
        experiment: int,
        output_dir: Path,
    ) -> None:
        """Runner thread: pulls games from queue and plays them.

        Lifecycle per game:
        1. Pull from queue → register worker → emit worker_status(running)
        2. Call _play_game
        3. On record returned: persist → emit game_complete → unregister → next
        4. On record=None (pause/stop): checkpoint saved → unregister → next
        5. On terminated/timeout: save checkpoint + failed file → unregister → next
        6. On other error: save to failed file → unregister → next
        7. Queue empty or stop signaled → exit
        """

        import queue

        from src.engine.stockfish_wrapper import StockfishWrapper

        # Each runner gets its own Stockfish instance (not thread-safe)
        sf = StockfishWrapper(
            engine_path=game_manager.stockfish_path,
            elo=game_manager.stockfish_elo,
        )

        try:
            sf.start()

            while not self._stop_event.is_set():
                # ── Check pause ──
                if not self._pause_event.is_set():
                    self._pause_event.wait()
                    if self._stop_event.is_set():
                        break
                    continue

                # ── Get next game ──
                try:
                    fen, game_index, game_id, resume_state = work_queue.get_nowait()
                except queue.Empty:
                    break  # No more work

                # ── Register worker ──
                self._register_worker(
                    game_id, experiment, condition,
                    detail=f"Game {game_index}",
                )

                self._emit({
                    "type": "worker_status",
                    "game_id": game_id,
                    "condition": condition,
                    "experiment": experiment,
                    "status": "running",
                    "detail": f"Game {game_index}",
                })

                try:
                    record = game_manager._play_game(
                        starting_fen=fen,
                        condition=condition,
                        game_index=game_index,
                        stockfish=sf,
                        game_id=game_id,
                        resume_state=resume_state,
                        pause_event=self._pause_event,
                        stop_event=self._stop_event,
                        on_progress=self._handle_worker_event,
                    )

                    if record is not None:
                        # ── Game completed — persist results ──
                        results_path = self._output_dir / f"{self._generation_strategy}_{self._condition}" / "results.jsonl"
                        append_game_record(record, results_path)

                        self._emit({
                            "type": "game_complete",
                            "game_id": game_id,
                            "condition": condition,
                            "experiment": experiment,
                            "status": record.final_status,
                            "total_turns": record.total_turns,
                        })
                    else:
                        # Game was paused/stopped — checkpoint saved by _play_game
                        logger.info("Game %s paused/stopped — checkpoint saved.", game_id)

                except (RequestsPausedError, RequestTerminatedError) as exc:
                    # Request terminated by the manager — save checkpoint + log
                    error_msg = f"{type(exc).__name__}: {exc}"
                    logger.warning(
                        "[Orchestrator] Game %s terminated — saving failed: %s",
                        game_id, error_msg,
                    )
                    # _play_game may have saved a checkpoint already on exception;
                    # we additionally record it as failed so it's skipped on re-run
                    append_failed_item(game_id, error_msg, self._output_dir / f"{self._generation_strategy}_{self._condition}" / "fail.jsonl")
                    self._emit({
                        "type": "worker_error",
                        "game_id": game_id,
                        "condition": condition,
                        "experiment": experiment,
                        "error": error_msg,
                    })
                    # Worker is free — continue to next game

                except Exception as exc:
                    error_msg = f"{type(exc).__name__}: {exc}"
                    logger.exception("Game runner error on %s", game_id)
                    append_failed_item(game_id, error_msg, self._output_dir / f"{self._generation_strategy}_{self._condition}" / "fail.jsonl")
                    self._emit({
                        "type": "worker_error",
                        "game_id": game_id,
                        "condition": condition,
                        "experiment": experiment,
                        "error": error_msg,
                    })
                    # Worker is free — continue to next game

                finally:
                    self._unregister_worker(game_id)

        finally:
            sf.close()

    # ── Event handling ───────────────────────────────────────────────

    def _handle_worker_event(self, event: dict[str, Any]) -> None:
        """Process events from GameManager._play_game on_progress callback.

        Also updates the worker registry detail for live dashboard display.
        """

        game_id = event.get("game_id")
        if game_id:
            # Update worker detail with latest turn info
            detail = event.get("detail", "")
            if event.get("type") == "game_turn":
                detail = f"Turn {event.get('half_moves', '?')}"
            elif event.get("type") == "worker_status":
                detail = event.get("detail", "")
            if detail:
                self._update_worker(game_id, detail=detail, status=event.get("status", "running"))

        self._emit(event)

    def _emit(self, event: dict[str, Any]) -> None:
        """Push event to the API layer via callback."""

        if self._on_event:
            try:
                self._on_event(event)
            except Exception:
                pass

    # ── Config loading ───────────────────────────────────────────────

    @staticmethod
    def _load_runner_config() -> dict[str, Any]:
        """Load configs/runner.yaml, returning defaults if missing."""

        import yaml
        from src.runner.paths import runner_config_path

        path = runner_config_path()
        if path.exists():
            with path.open("r", encoding="utf-8") as fh:
                return yaml.safe_load(fh) or {}
        return {}
