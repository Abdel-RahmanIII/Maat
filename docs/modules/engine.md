# Engine Module Reference

## Overview

The `src/engine/` package is the orchestration layer that runs experiments end to end.
It loads experiment configs, dispatches condition graphs, persists results/checkpoints, and
coordinates Stockfish where needed.

## File Map

| File | Responsibility |
|------|----------------|
| `stockfish_wrapper.py` | UCI wrapper around Stockfish |
| `puzzle_manager.py` | Experiment 1 orchestrator (single-turn puzzle evaluation) |
| `game_manager.py` | Experiment 2/3 orchestrator (full-game simulation) |
| `config_loader.py` | YAML config loading + validation + manager construction |
| `condition_dispatch.py` | Condition A-F routing + retry/backoff wrapper |
| `result_store.py` | JSONL persistence, checkpoint I/O, CSV summaries |

## `config_loader.py`

Loads YAML experiment files and converts them into a normalized config dictionary with
resolved filesystem paths and a constructed `ModelConfig` instance.

### Public API

```python
def load_experiment_config(yaml_path: str | Path) -> dict[str, Any]
def build_puzzle_manager_from_config(config: dict[str, Any]) -> PuzzleManager
def build_game_manager_from_config(config: dict[str, Any]) -> GameManager
```

### Validation and Defaults

- `experiment` must be one of `1`, `2`, `3`.
- `model` block defaults:
  - `model_name="gemma-4-31b-it"`
  - `temperature=0.0`
  - `max_output_tokens=1024`
- Runtime defaults:
  - `generation_strategy="generator_only"`
  - `delay_seconds=0.0`
  - `max_api_retries=5`
  - `backoff_base=2.0`
  - `backoff_max=60.0`

### Experiment-Specific Requirements

- Experiment 1 requires `puzzle_data`.
- Experiments 2 and 3 require `starting_positions`.
- Experiment 2 defaults `input_mode="fen"`; experiment 3 defaults `input_mode="history"`.

### Path Resolution

Relative paths are resolved against the project root via `_resolve_path`, so YAML files can
use repository-relative paths.

## `condition_dispatch.py`

Central condition router that maps condition letters to their `run_condition_X` entry points.
Imports are lazy to avoid circular imports and unnecessary startup cost.

### Public API

```python
def dispatch_turn(
    condition: str,
    *,
    fen: str,
    move_history: list[str] | None = None,
    move_number: int = 1,
    game_id: str = "",
    input_mode: InputMode = "fen",
    generation_strategy: str = "generator_only",
    model_config: ModelConfig | None = None,
    max_react_steps: int = 6,
) -> TurnState

def dispatch_turn_with_backoff(
    condition: str,
    *,
    fen: str,
    move_history: list[str] | None = None,
    move_number: int = 1,
    game_id: str = "",
    input_mode: InputMode = "fen",
    generation_strategy: str = "generator_only",
    model_config: ModelConfig | None = None,
    max_react_steps: int = 6,
    max_api_retries: int = 5,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
) -> TurnState
```

### Dispatch Notes

- Conditions `A` to `E` receive `generation_strategy`.
- Condition `F` uses a different signature and receives `max_steps=max_react_steps`.
- Unknown condition letters raise `ValueError`.

### Backoff Behavior

`dispatch_turn_with_backoff` retries on likely transient API failures using exponential backoff.
Transient matching is string-based (`429`, `rate`, `quota`, `500/502/503/504`, `timeout`, `unavailable`).
Non-transient errors are raised immediately.

## `result_store.py`

Persistence helpers for run artifacts.

### Public API

```python
def append_game_record(record: GameRecord, filepath: Path) -> None
def load_game_records(filepath: Path) -> list[GameRecord]
def load_checkpoint(filepath: Path) -> set[str]
def append_checkpoint(game_id: str, filepath: Path) -> None
def write_summary_csv(records: list[GameRecord], filepath: Path) -> None
```

### Storage Format

- **Results**: JSONL, one `GameRecord` per line (`model_dump_json`).
- **Checkpoint**: plain text, one completed `game_id` per line.
- **Summary**: CSV with per-game rollup columns including `fir` and `ftir`.

Malformed JSONL lines are skipped with a warning to keep long runs robust.

## `puzzle_manager.py`

Runs Experiment 1 (single-turn puzzle evaluation per condition).

### Class API

```python
class PuzzleManager:
    def __init__(
        self,
        puzzles: list[dict[str, Any]],
        conditions: list[str],
        output_dir: Path | str,
        *,
        model_config: ModelConfig | None = None,
        generation_strategy: str = "generator_only",
        delay_seconds: float = 0.0,
        max_api_retries: int = 5,
        backoff_base: float = 2.0,
        backoff_max: float = 60.0,
    ) -> None

    def run_all(self) -> list[GameRecord]
    def run_condition(self, condition: str) -> list[GameRecord]
    def run_single(self, puzzle: dict[str, Any], condition: str) -> GameRecord
```

### Execution Flow

1. Load completed IDs from `.checkpoint`.
2. For each `(puzzle, condition)` pair, call `dispatch_turn_with_backoff`.
3. Record metrics via `MetricsCollector`.
4. Append JSONL record and checkpoint line.
5. Write `exp1_summary.csv` at end.

## `game_manager.py`

Runs experiments 2 and 3 (full games, LLM as White, Stockfish as Black).

### Class API

```python
class GameManager:
    def __init__(
        self,
        starting_positions: list[str],
        conditions: list[str],
        experiment: int,
        output_dir: Path | str,
        *,
        stockfish_elo: int = 1000,
        stockfish_path: str | None = None,
        max_half_moves: int = 150,
        model_config: ModelConfig | None = None,
        generation_strategy: str = "generator_only",
        delay_seconds: float = 0.0,
        max_api_retries: int = 5,
        backoff_base: float = 2.0,
        backoff_max: float = 60.0,
    ) -> None

    def run_all(self) -> list[GameRecord]
    def run_single_game(
        self,
        starting_fen: str,
        condition: str,
        game_index: int = 0,
        stockfish: StockfishWrapper | None = None,
    ) -> GameRecord
```

### Experiment Modes

- `experiment=2` -> `input_mode="fen"`
- `experiment=3` -> `input_mode="history"`

### Full-Game Loop

For each turn:
- White turn: `dispatch_turn_with_backoff` and apply validated move.
- Black turn: `StockfishWrapper.choose_move` and apply engine move.
- Stop on forfeit, natural termination (`checkmate`, `stalemate`, draw), or `max_half_moves`.

## `stockfish_wrapper.py`

Wraps a UCI-compatible Stockfish binary behind a small interface.

### Class API

```python
class StockfishWrapper:
    def __init__(self, engine_path: str | Path | None = None, elo: int = 1320) -> None
    def start(self) -> None
    def close(self) -> None
    def set_elo(self, elo: int) -> None
    def choose_move(self, fen: str, time_limit: float = 0.1) -> str
```

Context-manager usage:

```python
with StockfishWrapper() as engine:
    move = engine.choose_move(fen)
```

### Engine Path Resolution

1. Explicit `engine_path` argument
2. `STOCKFISH_PATH` environment variable
3. `stockfish` in system `PATH`
4. otherwise `FileNotFoundError`

### Error Handling Summary

- Invalid FEN in `choose_move` -> `ValueError`
- Engine unavailable -> `FileNotFoundError`
- Engine returns no move -> `RuntimeError`
