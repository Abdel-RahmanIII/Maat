"""Factory for generation strategy subgraphs."""

from __future__ import annotations

from langgraph.graph.state import CompiledStateGraph

from src.config import GenerationStrategy, ModelConfig
from src.context import ConversationContext


def build_generation_subgraph(
    strategy: str | GenerationStrategy,
    model_config: ModelConfig | None = None,
    context: ConversationContext | None = None,
) -> CompiledStateGraph:
    """Return a compiled subgraph for the requested generation strategy.

    The returned subgraph:

    - Accepts ``TurnState`` as input
    - Populates: ``proposed_move``, ``is_valid``, ``error_types``,
      ``first_try_valid``, ``total_attempts``, ``llm_calls_this_turn``,
      ``tokens_this_turn``, ``prompt_token_count``, ``strategic_plan``,
      ``observation_summary``, ``raw_llm_response``

    Parameters
    ----------
    context:
        Optional :class:`ConversationContext` for multi-turn memory.
        Forwarded to strategy-specific builders that support it.
    """

    if isinstance(strategy, GenerationStrategy):
        strategy = strategy.value

    if strategy == GenerationStrategy.PLANNER_ACTOR.value:
        from src.graph.generation.planner_actor import build_planner_actor_subgraph

        return build_planner_actor_subgraph(model_config, context)

    if strategy == GenerationStrategy.OBSERVER_STRATEGIST_TACTICIAN.value:
        from src.graph.generation.observer_strategist_tactician import build_observer_strategist_tactician_subgraph

        return build_observer_strategist_tactician_subgraph(model_config, context)

    if strategy == GenerationStrategy.OBSERVER_EXECUTOR.value:
        from src.graph.generation.observer_executor import build_observer_executor_subgraph

        return build_observer_executor_subgraph(model_config, context)

    # Default: generator_only
    from src.graph.generation.generator_only import build_generator_only_subgraph

    return build_generator_only_subgraph(model_config, context)
