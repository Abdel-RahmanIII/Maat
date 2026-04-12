# Testing Guide

## Test Framework

- `pytest`

## Run Full Suite

```powershell
python -m pytest -q
```

Expected: 68 passed (Stockfish tests may skip if engine binary is unavailable).

## Run by Area

```powershell
python -m pytest -q tests/state
python -m pytest -q tests/validators
python -m pytest -q tests/tools
python -m pytest -q tests/data
python -m pytest -q tests/engine
python -m pytest -q tests/agents
python -m pytest -q tests/graph
```

## Test Categories

| Category | Tests | Description |
|----------|-------|-------------|
| State | 3 | TurnState field completeness, defaults, copy safety |
| Validators | 16 | Parser (6) + symbolic validator (10) |
| Tools | 7 | All tool contracts via `.invoke()` |
| Data | 15 | Quality filtering, difficulty bands, phase+difficulty sampling, per-collection outputs, logging/pause flow |
| Engine | 4 | Path resolution, FEN validation, move generation |
| Agents | 8 | Base utilities: prompt loading, board repr, feedback |
| Graphs | 15 | parse_and_validate, conditions A/B/D (mocked LLM), config |

## Mocking Strategy

Graph tests mock `src.graph.base_graph.generate_move` to avoid real LLM calls. This tests the full pipeline (parsing, validation, retry logic, state updates) without API dependencies.

## Current Known Skip Condition

- Engine integration test in `tests/engine/test_stockfish_wrapper.py` skips when no Stockfish binary is available via PATH or `STOCKFISH_PATH`.

## Tooling Configuration

Defined in `pyproject.toml`:

- pytest path configuration
- ruff lint profile
- black formatting profile

If local tooling is missing, install dev dependencies from `requirements.txt`.
