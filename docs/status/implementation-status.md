# Implementation Status

## Scope of This Status Page

Status is based on what is currently implemented in the repository.

Last verified: 2026-04-10.

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

## Current Test Health

- Full suite passes: `python -m pytest -q` → `68 passed`.
- One Stockfish integration test skips when engine binary is unavailable.

## Not Implemented Yet

- Experiment runners and orchestration layers (puzzle_manager, game_manager)
- Metrics aggregation pipeline (collector, aggregator, recurrence)
- Analysis/reporting scripts
- Experiment YAML configs

## Practical Next Milestone

Start Phase 3 by implementing experiment orchestration (puzzle_manager for Exp 1, game_manager for Exp 2 & 3), the metrics pipeline, and YAML-based experiment configs.
