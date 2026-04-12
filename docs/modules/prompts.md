# Prompts Module Reference

## Directory

- `src/prompts/`

## Purpose

All prompt templates used by agents. Templates are plain `.txt` files with Python `str.format()` placeholders.

## Template Inventory

| File | Agent | Placeholders |
|------|-------|-------------|
| `generator.txt` | Generator | `{color}`, `{board_representation}`, `{move_history}`, `{feedback_block}` |
| `critic.txt` | Critic | `{fen}`, `{board_ascii}`, `{proposed_move}` |
| `explainer.txt` | Explainer | `{fen}`, `{board_ascii}`, `{proposed_move}`, `{error_type}`, `{error_reason}` |
| `react.txt` | ReAct Agent | `{color}`, `{board_representation}`, `{move_history}` |
| `strategist.txt` | Strategist | `{color}`, `{board_representation}`, `{move_history}` |
| `router.txt` | Router | `{board_representation}`, `{move_history}` |
| `opening_specialist.txt` | Specialist (opening) | `{color}`, `{board_representation}`, `{move_history}`, `{feedback_block}` |
| `middlegame_specialist.txt` | Specialist (middlegame) | `{color}`, `{board_representation}`, `{move_history}`, `{feedback_block}` |
| `endgame_specialist.txt` | Specialist (endgame) | `{color}`, `{board_representation}`, `{move_history}`, `{feedback_block}` |

## Common Placeholder Values

| Placeholder | Source | Description |
|-------------|--------|-------------|
| `{color}` | `base.get_side_to_move()` | `"white"` or `"black"` |
| `{board_representation}` | `base.build_board_representation()` | Full FEN + ASCII (fen mode) or withheld message (history mode) |
| `{move_history}` | Joined UCI moves | e.g. `"e2e4 e7e5 g1f3"` or `"(none)"` |
| `{feedback_block}` | `base.format_feedback_block()` | Empty string or numbered error list |

## Design Notes

- Generator prompt asks for **only** UCI move output — no explanation — to minimize parsing complexity.
- Critic prompt requires **JSON-only** response for structured parsing.
- Specialist prompts include phase-specific chess principles to guide move selection.
- Strategist prompt explicitly prohibits UCI move output (plan only).
