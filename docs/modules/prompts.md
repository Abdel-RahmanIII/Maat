# Prompts Module Reference

## Directory

- `src/prompts/`

## Purpose

All prompt templates used by agents. Prompts are YAML files loaded by agent id.

Each agent prompt file provides both input variants:

- `fen` variant (board context available)
- `history` variant (reasoning from move history)

Each variant has two prompt roles:

- `system`
- `user`

Runtime loading is YAML-only (no `.txt` fallback).

Legacy `.txt` files may exist in `src/prompts/` for prompt-lab/manual workflows,
but production runtime agents load YAML templates only.

## Template Inventory

| File | Agent |
|------|-------|
| `generator.yaml` | Generator |
| `critic.yaml` | Critic |
| `explainer.yaml` | Explainer |
| `react.yaml` | ReAct Agent |
| `strategist.yaml` | Strategist |
| `router.yaml` | Router |
| `tactician.yaml` | Tactician |
| `opening_specialist.yaml` | Opening Specialist |
| `middlegame_specialist.yaml` | Middlegame Specialist |
| `endgame_specialist.yaml` | Endgame Specialist |

## YAML Shape

```yaml
agent: generator
variants:
	fen:
		system: |
			...
		user: |
			...
	history:
		system: |
			...
		user: |
			...
```

## Common Placeholder Values

| Placeholder | Source | Description |
|-------------|--------|-------------|
| `{color}` | `base.get_side_to_move()` | `"white"` or `"black"` |
| `{fen}` | current state | Current FEN |
| `{ascii_board}` | `str(chess.Board(fen))` | 8x8 ASCII board |
| `{move_history}` | Joined UCI moves | e.g. `"e2e4 e7e5 g1f3"` or `"(none)"` |
| `{feedback_block}` | `base.format_feedback_block()` | Empty string or numbered error list |

## Design Notes

- Generator prompt asks for **only** UCI move output — no explanation — to minimize parsing complexity.
- Critic and Explainer prompts require **JSON-only** responses for structured parsing.
- Specialist prompts include phase-specific chess principles to guide move selection.
- Strategist prompt explicitly prohibits UCI move output (plan only).
