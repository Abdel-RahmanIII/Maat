# Configuration Reference

## Project Metadata

Source: `pyproject.toml`

- package name: `maat`
- python requirement: `>=3.11`
- build backend: `setuptools.build_meta`

## Runtime Dependencies

- `python-chess>=1.999`
- `pydantic>=2.6`
- `langgraph>=0.2.0`
- `google-genai>=0.7.0`
- `langchain-google-genai>=2.1.0`
- `langchain-core>=0.3.0`
- `python-dotenv>=1.0.0`
- `PyYAML>=6.0`
- `stockfish>=3.28.0`

## Dev Dependencies

- `pytest>=8.0`
- `pytest-cov>=5.0`
- `ruff>=0.6.0`
- `black>=24.0`

## Test Configuration

`pyproject.toml`:

- `pythonpath = ["src"]`
- `testpaths = ["tests"]`

## Lint/Format Configuration

- Ruff target: `py311`
- Ruff line length: `100`
- Black line length: `100`

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GOOGLE_API_KEY` | Yes (for LLM calls) | Google AI Studio API key for Gemma 4 31B |
| `STOCKFISH_PATH` | Optional | Explicit path to Stockfish executable |

## Experiment Configuration (`src/config.py`)

| Setting | Default | Description |
|---------|---------|-------------|
| `model_name` | `gemma-4-31b-it` | Google AI Studio model identifier |
| `temperature` | `0.0` | Greedy decoding for reproducibility |
| `max_output_tokens` | `1024` | Maximum response length |
| `max_retries` | Condition-dependent (0 or 3) | Max retry attempts for validation loops |
| `max_react_steps` | `6` | Max think-act cycles for Condition F |
| `generation_strategy` | `generator_only` | One of: `generator_only`, `planner_actor`, `router_specialists` |
| `input_mode` | `fen` | `fen` (full board) or `history` (move list only) |

## Important Root Files

- `.env`: local environment values (API keys, engine path)
- `.gitignore`: ignore patterns including `.env`, virtual env, and caches
- `requirements.txt`: direct install list
