# Graphs Module Reference

## Directory

- `src/graph/`

## Purpose

Condition implementations for the six experimental architectures. Conditions A-E share
the same `TurnState` contract and core helper pipeline in `base_graph.py`.

## Files

| File | Description |
|------|-------------|
| `base_graph.py` | Shared generation dispatch, parse+validate, turn snapshot helpers |
| `condition_a.py` | Direct function baseline (no LangGraph, no retries) |
| `condition_b.py` | LangGraph control baseline (same behavior as A) |
| `condition_c.py` | Generator + LLM Critic + ground-truth check + retries |
| `condition_d.py` | Generator + symbolic validation + terse retry feedback |
| `condition_e.py` | Generator + symbolic validation + LLM explainer feedback |
| `condition_f.py` | ReAct loop with tools and post-submission validation |

## Shared Utilities (`base_graph.py`)

### Generation Dispatch

```python
run_generation(state: TurnState, model_config: ModelConfig | None = None) -> dict[str, Any]
```

Strategy routing:

- `generator_only` -> `generate_move(...)`
- `planner_actor` -> `create_plan(...)` then `execute_plan(...)`
- `router_specialists` -> `classify_phase(...)` then `generate_specialist_move(...)`

Return payload includes:

- `raw_output`
- `prompt_tokens`, `completion_tokens`
- `strategic_plan` (planner-actor only)
- `routed_phase` (router-specialists only)
- `extra_llm_calls` (0 or 1)

### Parse and Validate

```python
parse_and_validate(raw_output: str, fen: str) -> dict[str, Any]
```

Pipeline:

1. `parse_uci_move(raw_output)`
2. If parse succeeds: `validate_move(fen, move_uci)`

Returns normalized fields used by all conditions:

- `proposed_move`
- `is_valid`
- `error_type`
- `error_reason`
- `used_fallback`

### Turn Snapshot

```python
snapshot_turn_result(state: TurnState) -> dict[str, Any]
```

Builds a serializable turn payload stored in `turn_results`, including:

- attempt metadata (`total_attempts`, `retry_count`, `first_try_valid`)
- resource metadata (`llm_calls_this_turn`, `tokens_this_turn`, `prompt_token_count`)
- strategy metadata (`generation_strategy`, `strategic_plan`, `routed_phase`)
- context metadata (`board_fen`, `game_phase`, `wall_clock_ms`)

## State Mutation Patterns

Every generation attempt in conditions A-E updates:

- `total_attempts += 1`
- `llm_calls_this_turn += 1 + extra_llm_calls`
- `tokens_this_turn += prompt_tokens + completion_tokens`
- `prompt_token_count += prompt_tokens`
- validation outputs (`proposed_move`, `is_valid`, `error_types`)

Retry-capable loops (C-E) additionally update:

- `retry_count += 1`
- `feedback_history.append(...)`

Finalization appends one snapshot to `turn_results` and sets `game_status` to either
`ongoing` or `forfeit`.

## Condition Entry Points

Each condition exposes a runtime entry function:

```python
from src.graph.condition_a import run_condition_a
from src.graph.condition_b import run_condition_b
from src.graph.condition_c import run_condition_c
from src.graph.condition_d import run_condition_d
from src.graph.condition_e import run_condition_e
from src.graph.condition_f import run_condition_f
```

Notes:

- Conditions B-E also expose `build_graph(model_config)`.
- Condition F is function-driven (`run_react_loop`) and does not expose a `StateGraph` builder.
- Conditions A-E accept `generation_strategy`; condition F does not.

## Minimal Example

```python
from src.graph.condition_d import run_condition_d

result = run_condition_d(
    fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    game_id="demo",
    generation_strategy="planner_actor",
)
print(result["is_valid"], result["proposed_move"], result["retry_count"])
```

## Deep Dive

For node-by-node flow, retry edges, and full state transition examples for conditions A-F,
see `docs/architecture/graph-implementation.md`.
