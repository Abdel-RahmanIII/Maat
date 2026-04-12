# Tools Module Reference

## File

- `src/tools/chess_tools.py`

## Purpose

Provides chess analysis tools for use by the ReAct agent (Condition F). All functions are decorated with `@tool` and are bound dynamically based on input mode:

- `fen` mode (Experiments 1 and 2): full tool catalog.
- `history` mode (Experiment 3): safe subset only (no board-revealing tools).

## Tool APIs

Tools are invoked via LangChain's tool-calling protocol.

| Tool | Signature | Returns |
|------|-----------|---------|
| `validate_move` | `(fen: str, move_uci: str)` | JSON `{"legal": bool, "reason": str, "rule_ref": str, "error_type": str | null}` |
| `is_in_check` | `(fen: str)` | JSON `{"in_check": bool, "checking_squares": list[str]}` |
| `get_game_phase` | `(move_history: list[str])` | String: `"opening" | "middlegame" | "endgame"` |
| `get_move_history_pgn` | `(move_history: list[str])` | PGN move text string |
| `get_board_visual` | `(fen: str)` | ASCII board string |
| `get_piece_at` | `(fen: str, square: str)` | Piece code (`"wN"`, `"bQ"`) or `"empty"` |
| `get_attackers` | `(fen: str, square: str)` | JSON list of `{square, piece, color}` |
| `get_defenders` | `(fen: str, square: str)` | JSON list of defenders of the square occupant |
| `is_square_safe` | `(fen: str, square: str, color: str)` | JSON `{"safe": bool, "threats": list[str]}` |
| `get_position_after_moves` | `(fen: str, moves: list[str])` | Resulting FEN string |
| `submit_move` | `(uci_move: str)` | Sentinel string `"SUBMIT:<move>"` |

## Convenience Export

```python
from src.tools.chess_tools import ALL_TOOLS, get_tools_for_input_mode
```

- `ALL_TOOLS`: full catalog for `fen` mode.
- `get_tools_for_input_mode(input_mode)`: returns mode-appropriate tools (`fen` vs `history`).

## `submit_move` Semantics

`submit_move` is a sentinel tool. When the ReAct agent calls it, the loop terminates and the submitted move undergoes ground-truth validation. It does not perform any validation itself.

## Error Cases

- Invalid FEN in board-dependent tools → raises `ValueError`.
- Invalid square string in square-dependent tools → raises `ValueError`.
- Invalid color string in `is_square_safe` → raises `ValueError`.
- Invalid or illegal sequence in `get_move_history_pgn` / `get_position_after_moves` → raises `ValueError`.
