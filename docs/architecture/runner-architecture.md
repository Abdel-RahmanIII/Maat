# Experiment Runner Architecture (Refactored)

This document explains the *runner* subsystem (the web dashboard + the experiment orchestration code). It’s written to help you quickly answer:

- “Where do I change X?”
- “What calls what?”
- “What must stay stable so the dashboard and saved results keep working?”

The core idea is to separate **control-plane** code (HTTP/WebSocket + orchestration) from **data-plane** code (workers that actually run puzzles/games).

---

## Mental model (two planes)

### Control plane
Responsible for:

- Serving the dashboard UI.
- Exposing REST endpoints to start/pause/resume/stop runs.
- Broadcasting real-time events to the browser over WebSocket.
- Tracking aggregate progress and worker status.

### Data plane
Responsible for:

- Running one unit of work (one puzzle, or one full game).
- Calling into the condition dispatcher (which calls the LLM).
- Persisting results/checkpoints.
- Emitting fine-grained events (turns, completion, errors).

---

## Directory layout (target structure)

The refactor organizes code by responsibility:

```
src/runner/
  __main__.py                 # `python -m src.runner` entrypoint
  paths.py                    # project root + runner file locations

  api/
    app.py                    # FastAPI app factory + route wiring
    ws.py                     # WebSocket connection manager + broadcast helper

  core/
    orchestrator.py           # Orchestrator: run lifecycle + thread pools
    progress.py               # ConditionProgress/ExperimentProgress (thread-safe)

  workers/
    puzzles.py                # Experiment 1 worker: run_puzzle_worker
    games.py                  # Experiment 2/3 worker: run_game_worker

  persistence/
    checkpoint.py             # .game_state + .run_progress persistence

  limiting/
    rate_limiter.py           # global RPM/RPD/TPM limiter (singleton)
```

Notes:

- This layout intentionally mirrors the control/data-plane split.
- Modules are small and named after the domain concept they own.

---

## Public contracts (what must remain stable)

### 1) REST API routes
The dashboard depends on these endpoints (paths + payload shape):

- `GET /` → returns the HTML dashboard
- `GET /api/status` → returns the full status object
- `POST /api/start` → starts a run
- `POST /api/pause` → pauses a run
- `POST /api/resume` → resumes a run
- `POST /api/stop` → stops a run
- `GET /api/experiments` → lists experiments/conditions (for the UI selector)
- `GET /api/config` → returns runner config
- `POST /api/config` → updates runtime config (rate limits)
- `GET /api/rate-limits` → returns rate limiter status

Keeping these stable means you can rearrange internal files freely without breaking the UI.

### 2) WebSocket path + event types
The dashboard connects to:

- `GET /ws`

Events are JSON objects with a `type` field.

The UI currently expects:

- Status snapshots:
  - `initial_status`
  - `status_update`

- Lifecycle:
  - `run_started`
  - `run_paused`
  - `run_resumed`
  - `run_stopping`
  - `run_finished`

- Worker stream:
  - `worker_status`
  - `game_turn`
  - `game_complete`
  - `puzzle_complete`
  - `worker_error`

You can add new event types later, but changing or removing these requires updating the dashboard.

### 3) Persisted checkpoint files
For resumability and continuity across runs, these locations and formats should remain stable:

- `output_dir/.checkpoint` (existing engine-level checkpointing)
- `output_dir/.game_state/{game_id}.json` (turn-level resume state)
- `output_dir/.run_progress.json` (dashboard progress state)
- `results/.rpd_state.json` (rate limiter daily counter)

---

## What calls what (high level)

### Start → Orchestrate → Workers

1. The browser calls `POST /api/start`.
2. API layer validates the body and calls `Orchestrator.start(...)`.
3. The orchestrator spawns a background thread which:
   - loads experiment YAML,
   - creates a `ThreadPoolExecutor` per experiment,
   - schedules worker functions for each condition/unit.
4. Workers write results and emit progress events.
5. Orchestrator forwards worker events to the WebSocket broadcaster.

---

## Components in detail

### `paths.py` (runner environment)
Problem it solves: internal modules need to locate `configs/` and `results/` reliably even after files move.

Responsibilities:

- Find `project_root` (e.g., by walking up until `pyproject.toml` or `configs/` exists).
- Provide helper paths (dashboard HTML path, runner.yaml path).

Design rule:

- Runner code should **not** hardcode `Path(__file__).parent.parent.parent` in multiple places.

---

### API layer (`api/app.py`, `api/ws.py`)
Responsibilities:

- Create the FastAPI app.
- Load `configs/runner.yaml`.
- Configure the global rate limiter.
- Create a single `Orchestrator` instance and store it in `app.state`.
- Manage WebSocket connections and broadcast events.

Key invariant:

- API endpoints stay thin; they should not contain orchestration logic.

---

### Core layer (`core/orchestrator.py`, `core/progress.py`)

#### Orchestrator
The orchestrator is the “brain” of the runner.

Responsibilities:

- Own run lifecycle (`idle`, `running`, `paused`, `stopping`, `stopped`, `completed`).
- Own the pause/stop events shared by all workers.
- For each experiment:
  - load YAML config,
  - create a thread pool sized by `max_concurrent_per_condition * len(conditions)`,
  - submit workers.
- Maintain a “full status” view for the dashboard.

Design rules:

- Orchestrator knows *what* to run and *how to schedule it*.
- Workers know *how to execute one unit of work*.

#### Progress models
Thread-safe trackers:

- `ConditionProgress`: completed/failed/in_progress/valid_count/total
- `ExperimentProgress`: per-condition map + experiment-level status

This data is what the dashboard renders.

---

### Workers layer (`workers/puzzles.py`, `workers/games.py`)

#### Puzzle worker (Experiment 1)
`run_puzzle_worker(...)` executes a single puzzle:

- Takes a puzzle FEN.
- Calls the condition dispatcher once.
- Produces a single `GameRecord`.
- Appends results to `exp1_{condition}_results.jsonl`.
- Writes the `.checkpoint` entry.

#### Game worker (Experiments 2 & 3)
`run_game_worker(...)` executes a full game:

- Uses per-game Stockfish instance for Black.
- Uses the condition dispatcher for White.
- Writes turn-level checkpoints after each White (LLM) turn.
- Resumes from `.game_state/{game_id}.json` when present.
- Finalizes the game record and deletes per-game state file.

Important design rules:

- Workers must **periodically check** `stop_event` and honor `pause_event`.
- Workers emit progress/error events but do not know about WebSockets.

---

### Persistence (`persistence/checkpoint.py`)

Responsibilities:

- Save/load/delete per-game state files under `.game_state/`.
- Save/load experiment run progress in `.run_progress.json`.

Design rule:

- Persistence functions are side-effect utilities: keep them pure and predictable.

---

### Rate limiting (`limiting/rate_limiter.py`)

Responsibilities:

- Provide a global, thread-safe limiter for:
  - RPM (requests/minute)
  - RPD (requests/day)
  - TPM (tracked, not blocking)
- Persist the RPD count to disk so restarts don’t reset the daily budget.

Integration point:

- The LLM client lazily imports the limiter and gates each call with `acquire()`.

---

## End-to-end flows

### A) Starting a run

- UI → `POST /api/start` with:

```json
{
  "experiments": [
    {"id": 1, "conditions": ["A", "B", "C"]},
    {"id": 2, "conditions": ["A"]}
  ],
  "parallel_experiments": false,
  "generation_strategy": "generator_only"
}
```

- API calls `Orchestrator.start(...)`.
- Orchestrator emits `run_started`.
- Workers start emitting `worker_status`, then `game_turn`/`puzzle_complete`, etc.

### B) Pausing/resuming

- Pause sets `pause_event.clear()`.
- Workers save state and block on `pause_event.wait()`.
- Resume sets `pause_event.set()` and workers continue.

### C) Stopping

- Stop sets `stop_event.set()` and unpauses workers.
- Workers finish their current unit (or save mid-game) and exit.
- Orchestrator emits `run_finished` and persists progress.

---

## Import conventions

- Prefer absolute imports from `src.*` across runner modules.
- Standalone scripts should add the project root to `sys.path` before importing `src.*`.

---

## Extending the runner (common tasks)

### Add a new experiment type
- Add a new branch in the orchestrator’s “run single experiment” logic.
- Implement a worker function for the new unit of work.
- Ensure result/checkpoint naming is consistent.

### Add a new dashboard panel
- Add new fields to `Orchestrator.get_full_status()`.
- Update the HTML/JS to render them.

### Add a new event
- Emit a new `type` from workers or orchestrator.
- Update the dashboard `handleWSMessage` to consume it.

---

## Migration map (old → new)

This refactor mostly moves code; behavior remains the same.

- `src/runner/server.py` → `src/runner/api/app.py` (+ `src/runner/api/ws.py`)
- `src/runner/orchestrator.py` → `src/runner/core/orchestrator.py` (+ `src/runner/core/progress.py`)
- `src/runner/worker.py` → `src/runner/workers/puzzles.py` and `src/runner/workers/games.py`
- `src/runner/checkpoint.py` → `src/runner/persistence/checkpoint.py`
- `src/runner/rate_limiter.py` → `src/runner/limiting/rate_limiter.py`
- `src/runner/__main__.py` remains the entrypoint, but imports the new `main()`.
