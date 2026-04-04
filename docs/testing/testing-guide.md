# Testing Guide

## Test Framework

- `pytest`

## Run Full Suite

```powershell
python -m pytest -q
```

## Run by Area

```powershell
python -m pytest -q tests/state
python -m pytest -q tests/validators
python -m pytest -q tests/tools
python -m pytest -q tests/data
python -m pytest -q tests/engine
```

## Current Known Skip Condition

- Engine integration test in `tests/engine/test_stockfish_wrapper.py` skips when no Stockfish binary is available via PATH or `STOCKFISH_PATH`.

## Tooling Configuration

Defined in `pyproject.toml`:

- pytest path configuration
- ruff lint profile
- black formatting profile

If local tooling is missing, install dev dependencies from `requirements.txt`.
