# Agents Module Reference

## Directory

- `src/agents/`

## Purpose

Agent implementations for each experimental role. Each agent encapsulates prompt formatting, LLM invocation, and response parsing for a specific function in the pipeline.

## Agent Summary

| Agent | File | Used By | Input | Output |
|-------|------|---------|-------|--------|
| **Base utilities** | `base.py` | All agents | — | Prompt loading, board repr, feedback formatting |
| **Generator** | `generator.py` | Conditions A–E | FEN + history + feedback | Raw UCI text + token counts |
| **Critic** | `critic.py` | Condition C | FEN + proposed move | `CriticResult` (valid, reasoning, suggestion) |
| **Explainer** | `explainer.py` | Condition E | FEN + move + error | `ExplainerResult` (explanation text) |
| **ReAct Agent** | `react_agent.py` | Condition F | FEN + history | Submitted move + tool call log |
| **Strategist** | `strategist.py` | Planner-Actor ext. | FEN + history | NL strategic plan |
| **Tactician** | `tactician.py` | Planner-Actor ext. | FEN + history + plan | Raw UCI text |
| **Router** | `router.py` | Router-Specialists ext. | FEN + history | Phase classification |
| **Specialists** | `specialists.py` | Router-Specialists ext. | FEN + history + phase | Raw UCI text |

## Base Utilities (`base.py`)

```python
load_prompt(template_name: str) -> str
build_board_representation(fen, input_mode, move_history) -> str
format_feedback_block(feedback_history: list[str]) -> str
get_side_to_move(fen: str) -> str
```

- `build_board_representation` returns full FEN + ASCII board in `fen` mode or a withheld-state message in `history` mode.
- `format_feedback_block` returns empty string when no feedback exists; otherwise builds a numbered list with retry instructions.

## Generator (`generator.py`)

```python
generate_move(*, fen, move_history, feedback_history, input_mode, model_config, prompt_template) -> dict
```

Returns `{"raw_output": str, "prompt_tokens": int, "completion_tokens": int}`.

The `prompt_template` parameter allows specialist prompts to reuse the same generation logic.

## Critic (`critic.py`)

```python
critique_move(*, fen, proposed_move, model_config) -> CriticResult
```

Parses JSON from the LLM response. Falls back to `valid=False` if JSON parsing fails (conservative — unparseable responses are treated as rejections).

## Explainer (`explainer.py`)

```python
explain_error(*, fen, proposed_move, error_type, error_reason, model_config) -> ExplainerResult
```

Translates a symbolic validator error into pedagogical feedback.

## ReAct Agent (`react_agent.py`)

```python
run_react_loop(*, fen, move_history, input_mode, max_steps, model_config) -> dict
```

Returns `submitted_move`, `tool_calls_log`, `steps_taken`, token usage, and `forfeited` flag.

The loop binds tools dynamically via `get_tools_for_input_mode(input_mode)`:

- `fen` mode: full tool catalog.
- `history` mode: restricted safe subset.

Tool calls are executed directly in the loop (not via `ToolNode`) to avoid runtime compatibility issues and to keep explicit per-call logging. `submit_move` is a sentinel tool that terminates the loop.

## Strategist / Tactician (Planner-Actor Extension)

- **Strategist** produces a natural-language strategic plan (does NOT output UCI).
- **Tactician** receives the plan + board state and produces a UCI move.

## Router / Specialists (Router-Specialists Extension)

- **Router** classifies game phase ("opening" / "middlegame" / "endgame"). Falls back to "middlegame" on unparseable responses.
- **Specialists** are phase-specific generators using tailored prompt templates.
