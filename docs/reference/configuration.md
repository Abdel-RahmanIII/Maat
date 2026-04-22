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

## Prompt Configuration

Prompt templates are stored in `src/prompts/` as per-agent YAML files.

- Loader entry point: `src.agents.base.load_agent_prompt(agent_id, input_mode, role)`
- Input mode variants: `fen` and `history`
- Message roles: `system` and `user`
- Loading policy: YAML only (legacy txt prompt loading is not used at runtime)

Expected YAML shape:

```yaml
agent: <agent_id>
variants:
  fen:
    system: |
      ...
    user: |
      ...
  history:
    system: |
      ...
    user: |
      ...
```

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

## Experiment YAML Configurations

Configuring experiments (Experiments 1, 2, and 3) is done via YAML files located in `configs/`. These YAML files are read and processed by `src/engine/config_loader.py`.

### Keys Setup for YAML Configs

| Key | Example / Type | Description |
|-----|----------------|-------------|
| `experiment` | `1`, `2`, `3` | The experiment identifier. |
| `conditions` | `["A", "B"]` | List of condition letters to run. |
| `puzzle_data` | `"src/data/...jsonl"` | (Exp 1 only) Path to the JSONL dataset containing puzzles. |
| `starting_positions`| `"configs/...txt"` | (Exp 2 & 3 only) Path to a file containing FENs or setup data for starting positions. |
| `input_mode` | `"fen"` or `"history"` | The format of board state provided to the LLM. |
| `games_per_condition`| `50` | (Exp 2 & 3 only) Number of full games to play per condition. |
| `max_half_moves` | `150` | (Exp 2 & 3 only) Hard limit on the length of the game before triggering a draw. |
| `output_dir` | `"results/exp1"` | The directory where result JSONL and checkpoints are written. |
| `generation_strategy`| `"generator_only"` | Selected generation strategy for the LLM prompts. |
| `delay_seconds` | `0.5` | Sleep delay between API calls to prevent rate-limiting. |
| `stockfish` | Dict (`elo`, `path`) | Configuration for Stockfish engine (used by Exp 2 & 3). |
| `model` | Dict | Options such as `model_name`, `temperature`, `max_output_tokens`. |

### Example: Experiment 1 Config (`configs/experiment_1.yaml`)

```yaml
experiment: 1
conditions: ["A", "B", "C", "D", "E", "F"]
puzzle_data: "src/data/Experiment 1/experiment_inputs.jsonl"
output_dir: "results/exp1"
generation_strategy: "generator_only"
delay_seconds: 0.5
model:
  model_name: "gemma-4-31b-it"
  temperature: 0.0
  max_output_tokens: 1024
```

### Example: Experiment 2 Config (`configs/experiment_2.yaml`)

```yaml
experiment: 2
conditions: ["A", "B"]
input_mode: "fen"
starting_positions: "configs/starting_positions.txt"
games_per_condition: 50
max_half_moves: 150
output_dir: "results/exp2"
delay_seconds: 0.5
stockfish:
  elo: 1000
  path: null
model:
  model_name: "gemma-4-31b-it"
  temperature: 0.0
  max_output_tokens: 1024
```

## Important Root Files

- `.env`: local environment values (API keys, engine path)
- `.gitignore`: ignore patterns including `.env`, virtual env, and caches
- `requirements.txt`: direct install list
