# File Index

This index reflects the current workspace files (excluding `.git/` internals and local caches).

## Root

- `.env`
- `.gitattributes`
- `.gitignore`
- `pyproject.toml`
- `requirements.txt`

## Scripts (`scripts/`)

- `scripts/prompt_lab.py`
- `scripts/prompt_lab_ui.html`
- `scripts/puzzle_sampler.py`
- `scripts/smoke_test.py`

## Source (`src/`)

### Core

- `src/__init__.py`
- `src/config.py`
- `src/error_taxonomy.py`
- `src/state.py`

### LLM Client (`src/llm/`)

- `src/llm/__init__.py`
- `src/llm/llm_client.py`

### Agents (`src/agents/`)

- `src/agents/__init__.py`
- `src/agents/base.py`
- `src/agents/generator.py`
- `src/agents/critic.py`
- `src/agents/explainer.py`
- `src/agents/react_agent.py`
- `src/agents/strategist.py`
- `src/agents/tactician.py`
- `src/agents/router.py`
- `src/agents/specialists.py`

### Prompts (`src/prompts/`)

- `src/prompts/generator.txt`
- `src/prompts/critic.txt`
- `src/prompts/explainer.txt`
- `src/prompts/react.txt`
- `src/prompts/strategist.txt`
- `src/prompts/router.txt`
- `src/prompts/opening_specialist.txt`
- `src/prompts/middlegame_specialist.txt`
- `src/prompts/endgame_specialist.txt`

### Graphs (`src/graph/`)

- `src/graph/__init__.py`
- `src/graph/base_graph.py`
- `src/graph/condition_a.py`
- `src/graph/condition_b.py`
- `src/graph/condition_c.py`
- `src/graph/condition_d.py`
- `src/graph/condition_e.py`
- `src/graph/condition_f.py`

### Validators (`src/validators/`)

- `src/validators/__init__.py`
- `src/validators/move_parser.py`
- `src/validators/symbolic.py`

### Tools (`src/tools/`)

- `src/tools/__init__.py`
- `src/tools/chess_tools.py`

### Data (`src/data/`)

- `src/data/__init__.py`
- `src/data/puzzle_sampler.py`

### Engine (`src/engine/`)

- `src/engine/__init__.py`
- `src/engine/stockfish_wrapper.py`

## Tests (`tests/`)

- `tests/__init__.py`
- `tests/agents/__init__.py`
- `tests/agents/test_base.py`
- `tests/data/test_puzzle_sampler.py`
- `tests/engine/test_stockfish_wrapper.py`
- `tests/graph/__init__.py`
- `tests/graph/test_conditions.py`
- `tests/state/test_state.py`
- `tests/tools/test_chess_tools.py`
- `tests/validators/test_move_parser.py`
- `tests/validators/test_symbolic_validator.py`

## Documentation (`docs/`)

- `docs/README.md`
- `docs/getting-started/overview.md`
- `docs/getting-started/installation.md`
- `docs/getting-started/quickstart.md`
- `docs/architecture/system-overview.md`
- `docs/architecture/state-and-errors.md`
- `docs/architecture/condition-graphs.md`
- `docs/modules/validators.md`
- `docs/modules/tools.md`
- `docs/modules/data.md`
- `docs/modules/engine.md`
- `docs/modules/llm-client.md`
- `docs/modules/agents.md`
- `docs/modules/graphs.md`
- `docs/modules/config.md`
- `docs/modules/prompts.md`
- `docs/testing/testing-guide.md`
- `docs/testing/test-coverage-matrix.md`
- `docs/reference/configuration.md`
- `docs/reference/file-index.md`
- `docs/reference/implementation_plan.md`
- `docs/status/implementation-status.md`
