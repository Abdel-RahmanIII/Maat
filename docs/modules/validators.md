# Validators Module Reference

## Files

- `src/validators/move_parser.py`
- `src/validators/symbolic.py`

## Move Parser

Primary API:

```python
parse_uci_move(raw_output: str | None) -> ParseResult
```

`ParseResult` fields:

- `is_valid: bool`
- `move_uci: str | None`
- `error_type: str | None`
- `reason: str | None`
- `used_fallback: bool`

Behavior summary:

1. Empty/whitespace/None output -> invalid with `NO_OUTPUT`
2. Strict UCI parse path via `chess.Move.from_uci`
3. Regex fallback search: `[a-h][1-8][a-h][1-8][qrbn]?`
4. If fallback candidate parses, returns valid with `used_fallback=True`
5. Otherwise invalid with `PARSE_ERROR`

## Symbolic Validator

Primary API:

```python
validate_move(fen: str, move_uci: str | None) -> ValidationResult
```

`ValidationResult` fields:

- `valid: bool`
- `error_type: str | None`
- `reason: str`

Key semantics:

- Uses `python-chess` as legality ground truth.
- Detects and classifies core rule violations into taxonomy categories.
- Returns deterministic machine-readable category and human-readable reason.

Important edge handling:

- Invalid or missing move input
- Invalid FEN
- Null move rejection
- Wrong piece/wrong side source
- Promotion misuse
- Castling and en passant specific violations
- Pseudo-legal but king-exposing moves -> `LEAVES_IN_CHECK`
