# Maat Backend

Deterministic chess core for experimenting with role-separated LLM workflows.

## Overview

This backend implements a deterministic game loop around python-chess with clear separation of concerns:

- State manager owns board state and snapshots.
- Validator normalizes and validates move attempts.
- Runner applies only validated moves and writes per-attempt JSONL logs.

The goal is reproducible behavior: illegal or malformed moves must never mutate game state.

## Features

- Deterministic board state transitions
- UCI as canonical move format
- SAN input support (normalized to UCI)
- Structured validation errors
- Append-only JSONL move-attempt logging
- Terminal detection (checkmate/stalemate/draw claims)

## Project Structure

```text
Backend/
	core/
		schemas.py         # Shared dataclasses and error codes
		state_manager.py   # Board ownership and snapshots
		validator.py       # Move parsing/normalization/legality checks
		game_runner.py     # Sequence runner + JSONL logging
		run_game.py        # CLI entrypoint
	docs/
		phase1_contract.md # Phase 1 contract and log schema
	tests/
		core/
			test_state_manager.py
			test_validator.py
			test_game_runner.py
	pyproject.toml
	requirements.txt
```

## Requirements

- Python 3.11+

Dependencies:

- python-chess>=1.999
- pydantic>=2.6.0
- pytest>=8.0.0 (dev/test)

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run Tests

From the `Backend` directory:

```bash
pytest
```

Run only core tests:

```bash
pytest tests/core
```

## Run A Deterministic Sequence

From the `Backend` directory:

```bash
python -m core.run_game --moves e2e4 e7e5 g1f3 b8c6
```

With custom starting FEN and JSONL log output:

```bash
python -m core.run_game --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" --moves e4 e5 Nf3 Nc6 --log logs/attempts.jsonl
```

CLI arguments:

- `--fen`: starting FEN (defaults to standard starting position)
- `--moves`: move list in UCI or SAN
- `--log`: optional JSONL output file path

CLI output:

- Final FEN
- Terminal flag
- Outcome (`1-0`, `0-1`, `1/2-1/2`, or `None`)

## Core Module Contracts

### StateManager

Responsibilities:

- Own a single chess board instance
- Initialize from optional FEN
- Return current FEN
- Produce state snapshots
- Apply validated UCI moves

Important behavior:

- `apply_validated_move_uci` raises `ValueError` if the move is not legal in the current position.

### RuleValidator

Validation flow:

1. Reject move if game is already terminal.
2. Reject empty move string.
3. Parse as UCI (length 4 or 5) or SAN.
4. Reject unsupported format.
5. Reject illegal move.
6. Return normalized UCI for valid moves.

Error codes:

- `syntax_error`
- `unsupported_format`
- `illegal_move`
- `game_already_terminal`
- `wrong_turn` (reserved for future role-level checks)

### GameRunner

Responsibilities:

- Execute move sequences in order
- Stop sequence if state is terminal
- Log every move attempt (valid or invalid)
- Expose terminal status and outcome

Logging behavior:

- `schema_version` is fixed to `1.0`
- Each `submit_move` increments `turn_id` and `attempt_id`
- Invalid attempts keep `state_after_fen` as `null` in JSON

## JSONL Log Record Schema

Each line is one JSON object with keys:

- `schema_version`
- `turn_id`
- `attempt_id`
- `input_move`
- `normalized_move`
- `validator_result`
- `validator_message`
- `state_before_fen`
- `state_after_fen`
- `terminal_flag`

Example (valid move):

```json
{
	"schema_version": "1.0",
	"turn_id": 1,
	"attempt_id": 1,
	"input_move": "e2e4",
	"normalized_move": "e2e4",
	"validator_result": "valid",
	"validator_message": "Move is valid.",
	"state_before_fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
	"state_after_fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
	"terminal_flag": false
}
```

## Phase 1 Contract

See `docs/phase1_contract.md` for the Week 1 deterministic contract, accepted formats, error taxonomy, and exit gates.
