# Engine Wrapper Module Reference

## File

- `src/engine/stockfish_wrapper.py`

## Purpose

Wraps a UCI-compatible Stockfish binary behind a small, testable Python interface.

## Class API

```python
class StockfishWrapper:
    def __init__(self, engine_path: str | Path | None = None, elo: int = 1320) -> None
    def start(self) -> None
    def close(self) -> None
    def set_elo(self, elo: int) -> None
    def choose_move(self, fen: str, time_limit: float = 0.1) -> str
```

Context manager support:

```python
with StockfishWrapper() as engine:
    move = engine.choose_move(fen)
```

## Engine Path Resolution Order

1. Explicit `engine_path` constructor argument
2. `STOCKFISH_PATH` environment variable
3. `stockfish` discovered in system PATH
4. Otherwise `FileNotFoundError`

## ELO Configuration Behavior

- Stores the requested ELO in wrapper state.
- When engine options are available, sets:
  - `UCI_LimitStrength = True`
  - `UCI_Elo = <elo>`
- Safely skips options absent in a specific binary build.

## Error Handling

- Invalid FEN in `choose_move` -> `ValueError`
- No engine binary found -> `FileNotFoundError`
- Engine returns no move -> `RuntimeError`
