# Test Coverage Matrix

## Coverage by Module

| Area | Source File(s) | Test File | Tests | Covered Behaviors |
|------|---------------|-----------|-------|-------------------|
| State | `src/state.py` | `tests/state/test_state.py` | 3 | TurnState field completeness (including generation strategy fields), default payload creation, move history copy safety |
| Parser | `src/validators/move_parser.py` | `tests/validators/test_move_parser.py` | 6 | strict UCI parse, regex fallback parse, empty output handling, parse errors, normalization, null move rejection |
| Symbolic validator | `src/validators/symbolic.py` | `tests/validators/test_symbolic_validator.py` | 10 | legal move acceptance, all 8 taxonomy categories, parse/no-output handling |
| Tools | `src/tools/chess_tools.py` | `tests/tools/test_chess_tools.py` | 7 | All tool contracts via `.invoke()`: validate, board state, legal moves, piece moves, attacked squares |
| Puzzle data | `scripts/puzzle_sampler.py` (re-exported via `src/data/__init__.py`) | `tests/data/test_puzzle_sampler.py` | 15 | quality filtering thresholds, difficulty bands, theme-first phase+difficulty sampling with heuristic top-up, output writers, runtime logging/pause behavior |
| Engine wrapper | `src/engine/stockfish_wrapper.py` | `tests/engine/test_stockfish_wrapper.py` | 4 | invalid FEN handling, path resolution rules, legal engine move generation |
| Agent base | `src/agents/base.py` | `tests/agents/test_base.py` | 8 | prompt loading (success + failure), board repr (fen + history modes), feedback formatting (empty + entries), side-to-move |
| Graphs | `src/graph/base_graph.py`, `condition_a.py`, `condition_b.py`, `condition_d.py` | `tests/graph/test_conditions.py` | 15 | parse_and_validate (valid, illegal, unparseable, empty, promotion), Condition A (valid + forfeit), Condition B (valid + forfeit, no-retry verification), Condition D (first-try success, retry-then-succeed, exhaust retries), config factory |
| Metrics | `src/metrics/definitions.py`, `collector.py`, `aggregator.py`, `recurrence.py` | `tests/metrics/test_definitions.py`, `tests/metrics/test_collector.py`, `tests/metrics/test_aggregator.py`, `tests/metrics/test_recurrence.py` | 99 | Pydantic model validation/serialization, collector timing + phase inference, metric formulas/edge cases, recurrence and clustering metrics |

## Coverage Notes

- Graph tests use mocked LLM calls — they test the full pipeline (parsing, validation, retry, state) without API keys.
- Metrics tests are deterministic and use synthetic TurnRecord/GameRecord fixtures to validate formulas and edge-case handling.
- Conditions C/E are structurally similar to D, differing only in the feedback agent (Critic/Explainer). Their agent modules can be tested independently with mocked LLM calls.
- Condition F's ReAct loop is not unit-tested due to complex ToolNode interactions — it requires integration testing with a real or mock LLM.
- Agents (generator, critic, explainer, etc.) are not individually unit-tested yet because they require LLM mocking at the LangChain level. Their logic is exercised indirectly through graph tests.
