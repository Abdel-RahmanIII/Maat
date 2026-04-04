# Tools Module Reference

## File

- `src/tools/chess_tools.py`

## Purpose

Provides helper tools aligned with Condition F (ReAct + tool calling) requirements.

## APIs

```python
validate_move(fen: str, uci_move: str) -> dict[str, bool | str]
get_board_state(fen: str) -> str
get_legal_moves(fen: str) -> list[str]
get_piece_moves(fen: str, square: str) -> list[str]
get_attacked_squares(fen: str, color: str) -> list[str]
```

## Behavior Notes

- `validate_move` delegates to symbolic validator and returns `{valid, reason}`.
- `get_board_state` returns a multi-line prompt-friendly snapshot including FEN, side-to-move, counters, and ASCII board.
- `get_legal_moves` returns all legal moves in UCI format.
- `get_piece_moves` validates square format and filters legal moves from that source square.
- `get_attacked_squares` accepts color aliases (`white`, `black`, `w`, `b`) and returns attacked algebraic squares.

## Error Cases

- Invalid FEN in any tool -> raises `ValueError`.
- Invalid square string in `get_piece_moves` -> raises `ValueError`.
- Invalid color string in `get_attacked_squares` -> raises `ValueError`.
