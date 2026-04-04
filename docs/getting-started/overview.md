# Project Overview

## What Maat Is

Maat is a Python codebase for research experiments on chess-playing LLM architectures. The current implementation focuses on **Phase 1 infrastructure**:

- shared state contract for turn processing
- symbolic legality validation and error classification
- robust move parsing from model output
- ReAct support tools for board analysis
- puzzle data sampling utilities
- Stockfish wrapper for engine-based move generation

## Current Implemented Components

- `src/state.py`
- `src/error_taxonomy.py`
- `src/validators/move_parser.py`
- `src/validators/symbolic.py`
- `src/tools/chess_tools.py`
- `src/data/puzzle_sampler.py`
- `src/engine/stockfish_wrapper.py`

## What Is Not Implemented Yet

The following are planned but currently absent from implementation:

- graph conditions (A-F)
- experiment runners and orchestration
- metric aggregation pipeline
- analysis scripts and reporting pipeline

## Design Principles Used So Far

- Deterministic core behavior where possible.
- Explicit typed return payloads for module boundaries.
- Reproducible sampling through fixed seeds.
- Test-first coverage of key error branches.
