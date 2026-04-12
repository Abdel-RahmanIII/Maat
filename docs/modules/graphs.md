# Graphs Module Reference

## Directory

- `src/graph/`

## Purpose

LangGraph graph definitions for each experimental condition. All graphs use the shared `TurnState` contract and `base_graph.py` utilities.

## Files

| File | Description |
|------|-------------|
| `base_graph.py` | Generation dispatch, parse+validate, turn snapshots |
| `condition_a.py` | Direct LLM call (no LangGraph) |
| `condition_b.py` | LangGraph StateGraph, no retries |
| `condition_c.py` | Generator + LLM Critic + ground-truth check, 3 retries |
| `condition_d.py` | Generator + symbolic validator + terse feedback, 3 retries |
| `condition_e.py` | Generator + symbolic validator + LLM explainer, 3 retries |
| `condition_f.py` | ReAct + tool calling, max 6 steps |

## Shared Utilities (`base_graph.py`)

### Generation Dispatch

```python
run_generation(state: TurnState, model_config) -> dict
```

Routes to the correct generation strategy based on `state["generation_strategy"]`:

- `generator_only` → `generate_move()`
- `planner_actor` → `create_plan()` → `execute_plan()`
- `router_specialists` → `classify_phase()` → `generate_specialist_move()`

Returns `raw_output`, token counts, `strategic_plan`, `routed_phase`, and `extra_llm_calls`.

### Parse and Validate

```python
parse_and_validate(raw_output: str, fen: str) -> dict
```

Chains `parse_uci_move` → `validate_move`. Returns `proposed_move`, `is_valid`, `error_type`, `error_reason`, `used_fallback`.

### Turn Snapshot

```python
snapshot_turn_result(state: TurnState) -> dict
```

Creates a per-turn metrics record from the current state for accumulation in `turn_results`.

## Usage Pattern

Each condition exposes:

1. `build_graph(model_config)` → compiled `StateGraph` (conditions B–E)
2. `run_condition_X(*, fen, ...)` → completed `TurnState`

```python
from src.graph.condition_d import run_condition_d

result = run_condition_d(
    fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    game_id="test",
    generation_strategy="planner_actor",
)
print(result["is_valid"], result["proposed_move"])
```
