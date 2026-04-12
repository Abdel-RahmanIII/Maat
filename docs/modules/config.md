# Config Module Reference

## File

- `src/config.py`

## Purpose

Central experiment configuration: model settings, condition presets, and generation strategy definitions.

## Types

### `GenerationStrategy` (Enum)

```python
GENERATOR_ONLY = "generator_only"
PLANNER_ACTOR = "planner_actor"
ROUTER_SPECIALISTS = "router_specialists"
```

### `Condition` (Enum)

```python
A, B, C, D, E, F
```

### `ModelConfig` (Dataclass)

| Field | Type | Default |
|-------|------|---------|
| `model_name` | `str` | `"gemma-4-31b-it"` |
| `temperature` | `float` | `0.0` |
| `max_output_tokens` | `int` | `1024` |
| `api_key` | `str` | from `GOOGLE_API_KEY` env var |

### `ConditionConfig` (Dataclass)

| Field | Type | Default |
|-------|------|---------|
| `condition` | `Condition` | `Condition.A` |
| `max_retries` | `int` | `0` |
| `max_react_steps` | `int` | `6` |
| `generation_strategy` | `GenerationStrategy` | `GENERATOR_ONLY` |
| `input_mode` | `Literal["fen", "history"]` | `"fen"` |

## Factory

```python
config_for_condition(condition: Condition | str, *, generation_strategy, input_mode) -> ConditionConfig
```

Returns canonical config with pre-set retry counts:

| Condition | `max_retries` |
|-----------|---------------|
| A, B | 0 |
| C, D, E | 3 |
| F | 0 (uses `max_react_steps`) |

## Environment

Loads `.env` from project root via `python-dotenv` at import time.
