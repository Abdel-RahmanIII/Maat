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

- `STOCKFISH_PATH`: optional explicit path to stockfish executable used by `StockfishWrapper`.

## Important Root Files

- `.env`: local environment values
- `.gitignore`: ignore patterns including virtual env and caches
- `requirements.txt`: direct install list
