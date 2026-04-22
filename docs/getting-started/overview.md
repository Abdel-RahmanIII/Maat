# Project Overview

## What Maat Is

Maat is a Python codebase for research experiments on chess-playing LLM architectures. It investigates whether explicit architectural structure — role separation, structured validation, and rule enforcement — can reduce rule violations in LLM-based chess play.

## Implemented Components

### Phase 1 — Core Infrastructure

- Shared state contract for turn processing (`src/state.py`)
- Error taxonomy with 8 error categories (`src/error_taxonomy.py`)
- Symbolic legality validation and error classification (`src/validators/`)
- Robust move parsing from model output (`src/validators/move_parser.py`)
- Chess analysis tools with LangGraph tool integration (`src/tools/chess_tools.py`)
- Puzzle data sampling utilities (`scripts/puzzle_sampler.py`, re-exported by `src/data/__init__.py`)
- Stockfish wrapper for engine-based move generation (`src/engine/`)

### Phase 2 — Conditions & Graphs

- Central experiment configuration (`src/config.py`)
- LLM client factory for Gemma 4 31B (`src/llm/llm_client.py`)
- 9 prompt templates for all agent roles (`src/prompts/`)
- 10 agent modules: generator, critic, explainer, ReAct, strategist, tactician, router, specialists (`src/agents/`)
- 6 condition graphs implementing all experimental conditions A–F (`src/graph/`)
- 3 swappable generation strategies: generator_only, planner_actor, router_specialists

### Phase 3 — Metrics

- Metrics schema models (`src/metrics/definitions.py`)
- Real-time turn/game collection (`src/metrics/collector.py`)
- Experiment-level metric aggregation (`src/metrics/aggregator.py`)
- Multi-turn recurrence metrics (`src/metrics/recurrence.py`)
- Metrics package exports (`src/metrics/__init__.py`)
- Metrics test suite (`tests/metrics/`)

## What Is Not Implemented Yet

- Experiment runners and orchestration (puzzle_manager, game_manager)
- Analysis scripts and reporting pipeline
- Experiment YAML configs

## Design Principles

- Deterministic core behavior where possible.
- Explicit typed return payloads for module boundaries.
- Reproducible sampling through fixed seeds.
- Temperature 0 for LLM calls (greedy decoding).
- All conditions share the same TurnState contract for apples-to-apples comparison.
- Generation strategies are orthogonal to validation pipelines.
- Test-first coverage of key error branches.
