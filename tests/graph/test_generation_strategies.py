"""Tests for generation strategy subgraphs."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import chess

from src.config import GenerationStrategy
from src.context import ConversationContext
from src.graph.condition_d import run_condition_d
from src.graph.generation.factory import build_generation_subgraph
from src.state import create_initial_turn_state


class TestObserverStrategistTacticianGraph:

    @patch("src.graph.generation.observer_strategist_tactician.execute_plan_from_observation")
    @patch("src.graph.generation.observer_strategist_tactician.create_plan_from_observation")
    @patch("src.graph.generation.observer_strategist_tactician.observe_position")
    def test_flow_passes_observation_into_strategy_and_tactics(
        self,
        mock_observer: MagicMock,
        mock_strategist: MagicMock,
        mock_tactician: MagicMock,
    ) -> None:
        mock_observer.return_value = {
            "summary": "Board summary from observer.",
            "prompt_tokens": 11,
            "completion_tokens": 7,
            "elapsed_ms": 1.5,
            "turn_messages": [],
        }
        mock_strategist.return_value = {
            "plan": "Advance the queenside majority.",
            "prompt_tokens": 9,
            "completion_tokens": 5,
            "elapsed_ms": 1.2,
            "turn_messages": [],
        }
        mock_tactician.return_value = {
            "raw_output": "e2e4",
            "prompt_tokens": 13,
            "completion_tokens": 4,
            "elapsed_ms": 2.1,
            "turn_messages": [],
        }

        graph = build_generation_subgraph(GenerationStrategy.OBSERVER_STRATEGIST_TACTICIAN)
        state = create_initial_turn_state(
            board_fen=chess.STARTING_FEN,
            game_id="test-ost",
            condition="B",
        )

        result = graph.invoke(state)

        assert result["is_valid"] is True
        assert result["proposed_move"] == "e2e4"
        assert result["observation_summary"] == "Board summary from observer."
        assert result["strategic_plan"] == "Advance the queenside majority."
        assert mock_observer.call_count == 1
        assert mock_strategist.call_count == 1
        assert mock_tactician.call_count == 1

        strategist_call = mock_strategist.call_args.kwargs
        tactician_call = mock_tactician.call_args.kwargs

        assert strategist_call["observation_summary"] == "Board summary from observer."
        assert tactician_call["observation_summary"] == "Board summary from observer."
        assert tactician_call["strategic_plan"] == "Advance the queenside majority."
        assert tactician_call["fen"] == chess.STARTING_FEN

    @patch("src.graph.generation.observer_strategist_tactician.execute_plan_from_observation")
    @patch("src.graph.generation.observer_strategist_tactician.create_plan_from_observation")
    @patch("src.graph.generation.observer_strategist_tactician.observe_position")
    def test_retry_reruns_observer_and_strategist(
        self,
        mock_observer: MagicMock,
        mock_strategist: MagicMock,
        mock_tactician: MagicMock,
    ) -> None:
        mock_observer.return_value = {
            "summary": "Fresh observer summary.",
            "prompt_tokens": 11,
            "completion_tokens": 7,
            "elapsed_ms": 1.5,
            "turn_messages": [],
        }
        mock_strategist.return_value = {
            "plan": "Fresh strategic plan.",
            "prompt_tokens": 9,
            "completion_tokens": 5,
            "elapsed_ms": 1.2,
            "turn_messages": [],
        }
        mock_tactician.return_value = {
            "raw_output": "e2e4",
            "prompt_tokens": 13,
            "completion_tokens": 4,
            "elapsed_ms": 2.1,
            "turn_messages": [],
        }

        graph = build_generation_subgraph(GenerationStrategy.OBSERVER_STRATEGIST_TACTICIAN)
        state = create_initial_turn_state(
            board_fen=chess.STARTING_FEN,
            game_id="test-ost-retry",
            condition="B",
        )

        graph.invoke(state)
        graph.invoke({**state, "feedback_history": ["Invalid move, try again."]})

        assert mock_observer.call_count == 2
        assert mock_strategist.call_count == 2
        assert mock_tactician.call_count == 2


class TestTurnContextPersistence:

    @patch("src.graph.generation.planner_actor.execute_plan")
    @patch("src.graph.generation.planner_actor.create_plan")
    def test_only_successful_attempt_messages_are_persisted(
        self,
        mock_create_plan: MagicMock,
        mock_execute_plan: MagicMock,
    ) -> None:
        mock_create_plan.return_value = {
            "plan": "Improve piece activity.",
            "prompt_tokens": 7,
            "completion_tokens": 5,
            "elapsed_ms": 1.0,
            "turn_messages": ["strategist-success"],
        }
        mock_execute_plan.side_effect = [
            {
                "raw_output": "e2e5",
                "prompt_tokens": 8,
                "completion_tokens": 4,
                "elapsed_ms": 1.0,
                "turn_messages": ["tactician-failed-attempt"],
            },
            {
                "raw_output": "e2e4",
                "prompt_tokens": 8,
                "completion_tokens": 4,
                "elapsed_ms": 1.0,
                "turn_messages": ["tactician-successful-attempt"],
            },
        ]

        context = ConversationContext()

        result = run_condition_d(
            fen=chess.STARTING_FEN,
            game_id="ctx-persist",
            generation_strategy=GenerationStrategy.PLANNER_ACTOR.value,
            context=context,
        )

        assert result["is_valid"] is True
        assert result["retry_count"] == 1
        assert mock_create_plan.call_count == 1
        assert mock_execute_plan.call_count == 2

        # Retries within the same turn should not consume newly generated context.
        first_call_history = mock_execute_plan.call_args_list[0].kwargs["conversation_history"]
        second_call_history = mock_execute_plan.call_args_list[1].kwargs["conversation_history"]
        assert first_call_history == []
        assert second_call_history == []

        # Next-turn context receives only strategist history for the next turn.
        assert context.get_history("strategist") == ["strategist-success"]
        assert context.get_history("tactician") == []

    @patch("src.graph.generation.observer_strategist_tactician.execute_plan_from_observation")
    @patch("src.graph.generation.observer_strategist_tactician.create_plan_from_observation")
    @patch("src.graph.generation.observer_strategist_tactician.observe_position")
    def test_observer_and_tactician_do_not_keep_cross_turn_context(
        self,
        mock_observer: MagicMock,
        mock_strategist: MagicMock,
        mock_tactician: MagicMock,
    ) -> None:
        mock_observer.return_value = {
            "summary": "Fresh observer summary.",
            "prompt_tokens": 11,
            "completion_tokens": 7,
            "elapsed_ms": 1.5,
            "turn_messages": [],
        }
        mock_strategist.return_value = {
            "plan": "Fresh strategic plan.",
            "prompt_tokens": 9,
            "completion_tokens": 5,
            "elapsed_ms": 1.2,
            "turn_messages": [],
        }
        mock_tactician.return_value = {
            "raw_output": "e2e4",
            "prompt_tokens": 13,
            "completion_tokens": 4,
            "elapsed_ms": 2.1,
            "turn_messages": [],
        }

        context = ConversationContext()
        graph = build_generation_subgraph(GenerationStrategy.OBSERVER_STRATEGIST_TACTICIAN, context=context)
        state = create_initial_turn_state(
            board_fen=chess.STARTING_FEN,
            game_id="test-ost-context",
            condition="B",
        )

        graph.invoke(state)
        graph.invoke({**state, "feedback_history": ["Retry from prior turn."]})

        assert mock_observer.call_count == 2
        assert mock_strategist.call_count == 2
        assert mock_tactician.call_count == 2

        assert context.get_history("observer") == []
        assert context.get_history("tactician") == []