"""Generation strategy subgraphs.

Three strategies are available:

- **Generator (G)**: Single LLM baseline — one call per attempt.
- **Planner-Actor (PA)**: Strategy → Tactics decomposition — two calls.
- **Observer-Strategist-Tactician (OST)**: Observation → Strategy → Tactics decomposition — three calls.
- **Observer-Executor (OE)**: Observation → Execution decomposition — two calls.
  The Observer describes the board; the Executor selects a move based on
  that description.

Each strategy is a compiled LangGraph StateGraph that takes TurnState as
input and outputs TurnState with ``proposed_move``, ``is_valid``,
``error_types``, and token counts populated.
"""

from src.graph.generation.factory import build_generation_subgraph

__all__ = ["build_generation_subgraph"]
