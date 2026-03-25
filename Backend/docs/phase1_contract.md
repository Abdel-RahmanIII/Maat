# Phase 1 Contract (Week 1)

## Deterministic Modules
- State Manager: single source of truth for board state and snapshots.
- Rule Validator: move normalization + legality enforcement with machine-readable error codes.
- Game Runner: submit sequence and mutate state only after validated moves.
- Logger Contract: append-only JSONL per attempt.

## Canonical Formats
- Canonical move format: UCI.
- Accepted input formats: UCI and SAN (normalized to UCI).
- State snapshot fields: FEN, turn, move counters, terminal flags, outcome.

## Validator Error Taxonomy
- syntax_error: empty or malformed move string.
- unsupported_format: move cannot be parsed as UCI or SAN.
- illegal_move: parsed move is not legal for current state.
- wrong_turn: reserved for future explicit role-level checks.
- game_already_terminal: any move attempted after terminal state.

## JSONL Log Schema v1.0
Each attempt writes one JSON record with keys:
- schema_version
- turn_id
- attempt_id
- input_move
- normalized_move
- validator_result
- validator_message
- state_before_fen
- state_after_fen
- terminal_flag

## Phase 1 Exit Gates
- Deterministic module interfaces implemented and tested.
- Illegal moves never mutate game state.
- Minimal runner executes scripted move lists from optional FEN.
- Logging schema versioned from day one.
