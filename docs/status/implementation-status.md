# Implementation Status

## Scope of This Status Page

Status is based on what is currently implemented in the repository.

Last verified: 2026-04-19.

## Phase 1 Checklist (Core Infrastructure)

- Project scaffolding, dependencies, linting config: done
- TurnState schema and shared utilities: done
- Symbolic validator + error classifier: done
- Move parser with fallback regex: done
- Chess tools (Condition F tool set): done
- Puzzle sampler and stratified sampling support: done
- Stockfish wrapper: done
- Unit tests for implemented modules: done

## Phase 2 Checklist (Conditions & Graphs)

- Config module (`src/config.py`): done
- LLM client module (`src/llm/llm_client.py`): done
- State schema extended with generation strategy fields: done
- Prompt templates (9 files in `src/prompts/`): done
- Agent modules (10 files in `src/agents/`): done
- Chess tools rewritten with `@tool` decorators: done
- Condition A graph (direct call, no LangGraph): done
- Condition B graph (LangGraph, no retries): done
- Condition C graph (LLM Critic + ground-truth, 3 retries): done
- Condition D graph (Symbolic validator + terse feedback, 3 retries): done
- Condition E graph (Symbolic + LLM Explainer, 3 retries): done
- Condition F graph (ReAct + tool calling, max 6 steps): done
- Generation strategies (generator_only, planner_actor, router_specialists): done
- Unit tests for agents and graphs: done

## Phase 3 Checklist (Metrics)

- Metrics data models (`src/metrics/definitions.py`): done
- Real-time collector (`src/metrics/collector.py`): done
- Aggregate metrics (`src/metrics/aggregator.py`): done
- Recurrence metrics (`src/metrics/recurrence.py`): done
- Metrics package exports (`src/metrics/__init__.py`): done
- Metrics unit tests (`tests/metrics/`): done

## Phase 4 Checklist (Orchestration & Config)

- Experiment runners and orchestration layers (puzzle_manager, game_manager): done
- Condition dispatcher (`condition_dispatch.py`): done
- Result serialization store (`result_store.py`): done
- Experiment YAML configs (`experiment_1.yaml`, `experiment_2.yaml`, `experiment_3.yaml`): done

## Current Test Health

- Full suite passes: `python -m pytest -q` → `182 passed`.
- One Stockfish integration test skips when engine binary is unavailable.

## Not Implemented Yet

- Analysis/reporting scripts

## Practical Next Milestone

Implement analysis/reporting scripts to parse the JSONL output from the experiments.
