# Graph Implementation Details

## Scope

This document describes concrete implementation behavior for condition runners in:

- `src/graph/condition_a.py`
- `src/graph/condition_b.py`
- `src/graph/condition_c.py`
- `src/graph/condition_d.py`
- `src/graph/condition_e.py`
- `src/graph/condition_f.py`

For topology-only diagrams, see `docs/architecture/condition-graphs.md`.

## Shared Initialization

All condition runners start by calling `create_initial_turn_state(...)` with:

- `board_fen`
- `game_id`
- `condition`
- `input_mode`
- `move_history`
- `move_number`
- `max_retries` (0 for A/B/F, 3 for C/D/E)

Conditions A-E then set:

- `state["generation_strategy"]` to one of:
  - `generator_only`
  - `planner_actor`
   - `observer_executor`
   - `observer_strategist_tactician`

Condition F always sets:

- `state["generation_strategy"] = "generator_only"`

## Condition A

File: `src/graph/condition_a.py`

### Execution

1. Run `run_generation(state, model_config)`.
2. Parse and validate using `parse_and_validate(raw_output, fen)`.
3. Set final state directly (`ongoing` or `forfeit`).
4. Append one `snapshot_turn_result(state)`.

### Notable State Updates

- `llm_calls_this_turn = 1 + extra_llm_calls`
- `tokens_this_turn = prompt_tokens + completion_tokens`
- `total_attempts = 1`
- `first_try_valid = is_valid`

No retry loop and no LangGraph graph object.

## Condition B

File: `src/graph/condition_b.py`

### Nodes

- `generate`
- `accept`
- `forfeit`

### Routing

- `generate -> accept` if `is_valid`
- `generate -> forfeit` otherwise

### Notes

- Behavior is intentionally equivalent to A, but wrapped in `StateGraph`.
- Useful as a framework-overhead control condition.

## Condition C

File: `src/graph/condition_c.py`

### Nodes

- `generate`
- `critic`
- `retry_generate`
- `ground_truth`
- `accept`
- `forfeit`

### Flow

1. `generate` produces and parses/validates candidate move.
2. If parse failed:
   - retry via `retry_generate` if retries remain
   - else `forfeit`
3. If parse succeeded:
   - `critic` (LLM legality judgment)
4. If critic rejects:
   - retry via `retry_generate` if retries remain
   - else `forfeit`
5. If critic approves:
   - `ground_truth` uses symbolic validation (`validate_move`)
6. Ground truth valid -> `accept`; invalid -> `forfeit`

### Retry Semantics

`retry_generate` increments `retry_count` and appends a generic rejection message to
`feedback_history`.

## Condition D

File: `src/graph/condition_d.py`

### Nodes

- `generate`
- `terse_feedback`
- `accept`
- `forfeit`

### Flow

1. `generate` runs generation + parse/symbolic validation.
2. If valid -> `accept`.
3. If invalid and retries remain -> `terse_feedback` then back to `generate`.
4. If invalid and retries exhausted -> `forfeit`.

### Retry Feedback

`terse_feedback` adds machine-style text:

- `Illegal move <move>: <reason>`

This keeps feedback deterministic and low-token.

## Condition E

File: `src/graph/condition_e.py`

### Nodes

- `generate`
- `explainer`
- `accept`
- `forfeit`

### Flow

1. `generate` runs generation + parse/symbolic validation.
2. If valid -> `accept`.
3. If invalid and retries remain -> `explainer` then back to `generate`.
4. If invalid and retries exhausted -> `forfeit`.

### Retry Feedback

`explainer` calls `explain_error(...)` and appends rich pedagogical feedback text to
`feedback_history`.

Compared to D, E trades token cost for potentially higher correction quality.

## Condition F

File: `src/graph/condition_f.py`

### Nodes

- `agent_reason`
- `execute_tools`
- `extract_submit`
- `ground_truth`
- `accept`
- `forfeit`

### Flow

1. `agent_reason` invokes the tool-bound LLM with the current message history.
   Increments `react_steps_taken`.
2. Routing after `agent_reason`:
   - If tool calls include `submit_move` → `extract_submit` → `ground_truth`
   - If tool calls present (non-submit) → `execute_tools`
   - If text-fallback submit found → `ground_truth`
   - If `react_steps_taken < max_react_steps` → `agent_reason` (continue reasoning)
   - Otherwise → `forfeit`
3. `execute_tools` executes tool calls directly and appends `ToolMessage`s to messages.
4. Routing after `execute_tools`:
   - If `submit_move` was among calls → `ground_truth`
   - If `react_steps_taken < max_react_steps` → `agent_reason`
   - Otherwise → `forfeit`
5. `ground_truth` extracts the submitted UCI move and runs `parse_and_validate`.
6. Valid → `accept`; invalid → `forfeit`.

### State Fields

- `react_steps_taken` — current step count (incremented by `agent_reason`)
- `max_react_steps` — maximum allowed steps (default 6, set by `run_condition_f`)
- `messages` — accumulates `SystemMessage`, `HumanMessage`, `AIMessage`, `ToolMessage`
- `tool_calls` — log of all tool calls with args, step, and results

## Cross-Condition Retry and Validation Matrix

| Condition | Retry Loop | Max Retries | Validation Gates |
|-----------|------------|-------------|------------------|
| A | No | 0 | parse + symbolic |
| B | No | 0 | parse + symbolic |
| C | Yes | 3 | parse -> critic -> symbolic |
| D | Yes | 3 | parse + symbolic |
| E | Yes | 3 | parse + symbolic + explainer feedback |
| F | Step-bounded (ReAct) | max_steps | parse + symbolic after submit |

## Example State Transition (Condition C)

```json
{
  "before": {
    "retry_count": 0,
    "feedback_history": [],
    "total_attempts": 0
  },
  "after_generate_invalid": {
    "retry_count": 0,
    "total_attempts": 1,
    "is_valid": false,
    "error_types": ["ILLEGAL_MOVE"]
  },
  "after_retry_generate": {
    "retry_count": 1,
    "feedback_history": [
      "Move e2e5 was rejected. Please try a different legal move."
    ]
  }
}
```
