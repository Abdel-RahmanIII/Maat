# Maat Backend

Deterministic chess backend for rule-governed agent workflows.

## Overview

This project wraps `python-chess` with deterministic modules so move handling is predictable and testable:

- `StateManager` owns the board and snapshots.
- `RuleValidator` performs two-stage validation (syntax, then legality).
- `GameRunner` executes attempts and writes JSONL logs.

Core guarantee: malformed or illegal moves never mutate game state.

## Current Status

- Python package metadata and test config are in `pyproject.toml`.
- Core contract is documented in `docs/phase1_contract.md`.
- Test suite currently passes (`18 passed`).

## Project Structure

```text
Backend/
	core/
		__init__.py
		exceptions.py
		game_runner.py
		rule_validator.py
		run_game.py
		state_manager.py
	docs/
		phase1_contract.md
	schemas/
		__init__.py
		game.py
		log_entry.py
		move.py
	tests/
		agents/
		core/
			test_game_runner.py
			test_rule_validator.py
			test_state_manager.py
	pyproject.toml
	requirements.txt
	README.md
```

## Requirements

- Python 3.11+

Runtime dependencies:

- `python-chess>=1.999`
- `pydantic>=2.6.0`

Development/testing dependency:

- `pytest>=8.0.0`

## Setup

From `Backend/`:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Running Tests

From `Backend/`:

```bash
pytest -q
```

Core-only tests:

```bash
pytest -q tests/core
```

## Running The CLI

`run_game.py` supports:

- `--fen` optional starting position
- `--moves` zero or more moves in UCI or SAN
- `--log` optional JSONL output path

Because the current file imports use both `core.*` and `Backend.*` module paths, use this invocation from `Backend/`:

```powershell
python -m core.run_game --moves e2e4 e7e5 g1f3 b8c6
```

Example with custom FEN and logging:

```powershell
python -m core.run_game --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" --moves e4 e5 Nf3 Nc6 --log logs/attempts.jsonl
```

CLI prints:

- Final FEN
- Terminal flag
- Outcome (`1-0`, `0-1`, `1/2-1/2`, or `None`)

## Core API Summary

### `StateManager`

- Initializes from default start position or explicit FEN.
- Exposes `current_fen()` and immutable `snapshot()` data.
- Applies validated UCI moves through `apply_validated_move_uci(...)`.
- Raises `InvalidFENError` for bad FEN input.
- Raises `IllegalMoveError` for illegal UCI moves.

Snapshot fields include:

- `fen`, `side_to_move`, move counters
- terminal state and `GameStatus`
- `outcome` and `legal_move_count`

### `RuleValidator`

Validation sequence:

1. Reject if game is already terminal (`game_already_terminal`).
2. Reject empty input (`syntax_error`).
3. Parse as UCI or SAN, else `unsupported_format`.
4. If parsed but not legal on board, return `illegal_move` with stage `legality`.
5. Return normalized UCI for valid moves.

`ValidationResult` includes:

- `is_valid`
- `normalized_move_uci`
- `validation_stage` (`syntax`, `legality`, or `None`)
- `error_code`
- `message`

### `GameRunner`

- `submit_move(raw_move)` validates, conditionally applies, logs, and returns `MoveResult`.
- `run_sequence(moves)` stops early if state becomes terminal.
- `terminal_status()` returns `(is_terminal, outcome)`.

Counters:

- `turn_id` increments on every `submit_move` call.
- `attempt_id` increments on every `submit_move` call.

## JSONL Logging Schema (Current)

Each attempt appends one JSON object. Current keys written by `GameRunner`:

- `schema_version`
- `turn_id`
- `attempt_id`
- `input_move`
- `normalized_move`
- `validation_stage`
- `validator_result`
- `validator_message`
- `state_before_fen`
- `state_after_fen`
- `terminal_flag`

Example valid attempt:

```json
{
	"schema_version": "1.0",
	"turn_id": 1,
	"attempt_id": 1,
	"input_move": "e2e4",
	"normalized_move": "e2e4",
	"validation_stage": null,
	"validator_result": "valid",
	"validator_message": "Move is valid.",
	"state_before_fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
	"state_after_fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
	"terminal_flag": false
}
```

## Related Docs

- `docs/phase1_contract.md`: deterministic module contracts and Phase 1 exit gates.
