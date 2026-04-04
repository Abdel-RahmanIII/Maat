# Test Coverage Matrix

## Coverage by Module

| Area | Source File | Test File | Covered Behaviors |
|---|---|---|---|
| State | `src/state.py` | `tests/state/test_state.py` | TurnState field completeness, default payload creation, move history copy safety |
| Parser | `src/validators/move_parser.py` | `tests/validators/test_move_parser.py` | strict UCI parse, regex fallback parse, empty output handling, parse errors, normalization |
| Symbolic validator | `src/validators/symbolic.py` | `tests/validators/test_symbolic_validator.py` | legal move acceptance, taxonomy categories, parse/no-output handling |
| Tools | `src/tools/chess_tools.py` | `tests/tools/test_chess_tools.py` | tool contracts for validate/board/legal/piece/attacked + invalid color handling |
| Puzzle data | `src/data/puzzle_sampler.py` | `tests/data/test_puzzle_sampler.py` | phase classification, invalid row filtering, per-phase stratified counts |
| Engine wrapper | `src/engine/stockfish_wrapper.py` | `tests/engine/test_stockfish_wrapper.py` | invalid FEN handling, path resolution rules, legal engine move generation when binary exists |

## Coverage Notes

- Current tests are focused on correctness of core infrastructure and edge cases.
- No graph-level integration tests exist yet because graph modules are not implemented.
