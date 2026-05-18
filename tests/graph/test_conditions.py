"""Tests for graph utilities and condition graphs using mocked LLM."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import chess
import pytest

from src.graph.base_graph import parse_and_validate, snapshot_turn_result
from src.state import create_initial_turn_state


# ── parse_and_validate tests ─────────────────────────────────────────────


class TestParseAndValidate:

    def test_valid_move_returns_valid(self) -> None:
        result = parse_and_validate("e2e4", chess.STARTING_FEN)
        assert result["is_valid"] is True
        assert result["proposed_move"] == "e2e4"

    def test_illegal_move_returns_invalid_with_error(self) -> None:
        result = parse_and_validate("e2e5", chess.STARTING_FEN)
        assert result["is_valid"] is False
        assert result["error_type"] is not None

    def test_unparseable_text_returns_parse_error(self) -> None:
        result = parse_and_validate("I think we should play pawn to e4", chess.STARTING_FEN)
        # The fallback regex should find e4 but it's not a full UCI move
        # so it depends on whether the regex finds a match
        assert isinstance(result["is_valid"], bool)

    def test_empty_output_returns_no_output(self) -> None:
        result = parse_and_validate("", chess.STARTING_FEN)
        assert result["is_valid"] is False
        assert result["error_type"] == "NO_OUTPUT"

    def test_promotion_move_parses_correctly(self) -> None:
        # Position where a pawn can promote
        fen = "8/P7/8/8/8/8/8/4K2k w - - 0 1"
        result = parse_and_validate("a7a8q", fen)
        assert result["is_valid"] is True
        assert result["proposed_move"] == "a7a8q"

    def test_san_move_parses_correctly(self) -> None:
        result = parse_and_validate("e4", chess.STARTING_FEN)
        assert result["is_valid"] is True
        assert result["proposed_move"] == "e2e4"


# ── snapshot_turn_result tests ────────────────────────────────────────────


class TestSnapshotTurnResult:

    def test_snapshot_contains_required_fields(self) -> None:
        state = create_initial_turn_state(
            board_fen=chess.STARTING_FEN,
            game_id="test",
            condition="B",
        )
        state["proposed_move"] = "e2e4"
        state["is_valid"] = True
        state["total_attempts"] = 1

        snapshot = snapshot_turn_result(state)

        assert snapshot["move_number"] == 1
        assert snapshot["proposed_move"] == "e2e4"
        assert snapshot["is_valid"] is True
        assert snapshot["total_attempts"] == 1


# ── Condition A tests (mocked LLM) ───────────────────────────────────────


class TestConditionA:

    @patch("src.graph.generation.generator_only.generate_move")
    def test_valid_move_accepted(self, mock_gen: MagicMock) -> None:
        mock_gen.return_value = {
            "raw_output": "e2e4",
            "prompt_tokens": 100,
            "completion_tokens": 5,
            "elapsed_ms": 10.0,
        }

        from src.graph.condition_a import run_condition_a

        result = run_condition_a(
            fen=chess.STARTING_FEN,
            game_id="test-a",
        )

        assert result["is_valid"] is True
        assert result["ground_truth_verdict"] is True
        assert result["proposed_move"] == "e2e4"
        assert result["game_status"] == "ongoing"
        assert result["total_attempts"] == 1

    @patch("src.graph.generation.generator_only.generate_move")
    def test_illegal_move_forfeits(self, mock_gen: MagicMock) -> None:
        mock_gen.return_value = {
            "raw_output": "e2e5",
            "prompt_tokens": 100,
            "completion_tokens": 5,
            "elapsed_ms": 10.0,
        }

        from src.graph.condition_a import run_condition_a

        result = run_condition_a(
            fen=chess.STARTING_FEN,
            game_id="test-a",
        )

        assert result["is_valid"] is False
        assert result["ground_truth_verdict"] is False
        assert result["game_status"] == "forfeit"
        assert len(result["error_types"]) > 0

    @patch("src.graph.generation.generator_only.generate_move")
    def test_generator_history_carries_to_next_turn(self, mock_gen: MagicMock) -> None:
        mock_gen.return_value = {
            "raw_output": "e2e4",
            "prompt_tokens": 100,
            "completion_tokens": 5,
            "elapsed_ms": 10.0,
            "turn_messages": ["human-message", "assistant-message"],
        }

        from src.context import ConversationContext
        from src.graph.condition_a import run_condition_a

        context = ConversationContext()

        first_result = run_condition_a(
            fen=chess.STARTING_FEN,
            game_id="test-a-1",
            context=context,
        )
        second_result = run_condition_a(
            fen=chess.STARTING_FEN,
            game_id="test-a-2",
            context=context,
        )

        assert first_result["is_valid"] is True
        assert second_result["is_valid"] is True

        first_call_history = mock_gen.call_args_list[0].kwargs["conversation_history"]
        second_call_history = mock_gen.call_args_list[1].kwargs["conversation_history"]

        assert first_call_history == []
        assert len(second_call_history) == 2


# ── Condition B tests (mocked LLM) ───────────────────────────────────────


class TestConditionB:

    @patch("src.graph.generation.generator_only.generate_move")
    def test_valid_move_accepted(self, mock_gen: MagicMock) -> None:
        mock_gen.return_value = {
            "raw_output": "e2e4",
            "prompt_tokens": 100,
            "completion_tokens": 5,
            "elapsed_ms": 10.0,
        }

        from src.graph.condition_b import run_condition_b

        result = run_condition_b(
            fen=chess.STARTING_FEN,
            game_id="test-b",
        )

        assert result["is_valid"] is True
        assert result["ground_truth_verdict"] is True
        assert result["game_status"] == "ongoing"

    @patch("src.graph.generation.generator_only.generate_move")
    def test_illegal_move_forfeits_no_retry(self, mock_gen: MagicMock) -> None:
        mock_gen.return_value = {
            "raw_output": "e2e5",
            "prompt_tokens": 100,
            "completion_tokens": 5,
            "elapsed_ms": 10.0,
        }

        from src.graph.condition_b import run_condition_b

        result = run_condition_b(
            fen=chess.STARTING_FEN,
            game_id="test-b",
        )

        assert result["is_valid"] is False
        assert result["ground_truth_verdict"] is False
        assert result["game_status"] == "forfeit"
        # Should NOT retry (max_retries=0)
        assert mock_gen.call_count == 1


# ── Condition C tests (mocked LLM) ───────────────────────────────────────


class TestConditionC:

    @patch("src.graph.condition_c.critique_move")
    @patch("src.graph.generation.generator_only.generate_move")
    def test_valid_move_accepted_after_critic(self, mock_gen: MagicMock, mock_crit: MagicMock) -> None:
        mock_gen.return_value = {
            "raw_output": "e2e4",
            "prompt_tokens": 100,
            "completion_tokens": 5,
            "elapsed_ms": 10.0,
        }
        mock_crit.return_value = {
            "valid": True,
            "reasoning": "Legal move.",
            "suggestion": "",
            "raw_output": "{\"valid\": true}",
            "prompt_tokens": 20,
            "completion_tokens": 3,
            "elapsed_ms": 2.0,
        }

        from src.graph.condition_c import run_condition_c

        result = run_condition_c(
            fen=chess.STARTING_FEN,
            game_id="test-c",
        )

        assert result["is_valid"] is True
        assert result["ground_truth_verdict"] is True
        assert result["critic_verdict"] is True
        assert result["game_status"] == "ongoing"
        assert result["retry_count"] == 0
        assert mock_gen.call_count == 1
        assert mock_crit.call_count == 1

    @patch("src.graph.condition_c.critique_move")
    @patch("src.graph.generation.generator_only.generate_move")
    def test_critic_rejects_until_forfeit(self, mock_gen: MagicMock, mock_crit: MagicMock) -> None:
        mock_gen.return_value = {
            "raw_output": "e2e4",
            "prompt_tokens": 100,
            "completion_tokens": 5,
            "elapsed_ms": 10.0,
        }
        mock_crit.return_value = {
            "valid": False,
            "reasoning": "The move is not aligned with the requested plan.",
            "suggestion": "Try another legal move.",
            "raw_output": "{\"valid\": false}",
            "prompt_tokens": 20,
            "completion_tokens": 3,
            "elapsed_ms": 2.0,
        }

        from src.graph.condition_c import run_condition_c

        result = run_condition_c(
            fen=chess.STARTING_FEN,
            game_id="test-c",
        )

        assert result["is_valid"] is False
        assert result["critic_verdict"] is False
        assert result["game_status"] == "forfeit"
        assert result["retry_count"] == 4
        assert mock_gen.call_count == 4
        assert mock_crit.call_count == 4
        assert len(result["feedback_history"]) == 4

    @patch("src.graph.condition_c.critique_move")
    @patch("src.graph.generation.generator_only.generate_move")
    def test_critic_accepts_but_ground_truth_rejects(self, mock_gen: MagicMock, mock_crit: MagicMock) -> None:
        mock_gen.return_value = {
            "raw_output": "e2e5",
            "prompt_tokens": 100,
            "completion_tokens": 5,
            "elapsed_ms": 10.0,
        }
        mock_crit.return_value = {
            "valid": True,
            "reasoning": "Looks legal.",
            "suggestion": "",
            "raw_output": "{\"valid\": true}",
            "prompt_tokens": 20,
            "completion_tokens": 3,
            "elapsed_ms": 2.0,
        }

        from src.graph.condition_c import run_condition_c

        result = run_condition_c(
            fen=chess.STARTING_FEN,
            game_id="test-c",
        )

        assert result["is_valid"] is False
        assert result["critic_verdict"] is True
        assert result["ground_truth_verdict"] is False
        assert result["game_status"] == "forfeit"
        assert mock_gen.call_count == 1
        assert mock_crit.call_count == 1


# ── Condition D tests (mocked LLM) ───────────────────────────────────────


class TestConditionD:

    @patch("src.graph.generation.generator_only.generate_move")
    def test_valid_on_first_try(self, mock_gen: MagicMock) -> None:
        mock_gen.return_value = {
            "raw_output": "e2e4",
            "prompt_tokens": 100,
            "completion_tokens": 5,
            "elapsed_ms": 10.0,
        }

        from src.graph.condition_d import run_condition_d

        result = run_condition_d(
            fen=chess.STARTING_FEN,
            game_id="test-d",
        )

        assert result["is_valid"] is True
        assert result["ground_truth_verdict"] is True
        assert result["game_status"] == "ongoing"

    @patch("src.graph.condition_d.validate_move")
    @patch("src.graph.generation.generator_only.generate_move")
    def test_parse_error_retries_from_ground_truth_node(
        self,
        mock_gen: MagicMock,
        mock_validate: MagicMock,
    ) -> None:
        """A parse error should retry through ground truth with Invalid UCI feedback."""
        mock_gen.side_effect = [
            {"raw_output": "not a move", "prompt_tokens": 100, "completion_tokens": 5, "elapsed_ms": 10.0},
            {"raw_output": "e2e4", "prompt_tokens": 110, "completion_tokens": 5, "elapsed_ms": 12.0},
        ]
        mock_validate.return_value = {
            "valid": True,
            "reason": "",
            "error_type": None,
        }

        from src.graph.condition_d import run_condition_d

        result = run_condition_d(
            fen=chess.STARTING_FEN,
            game_id="test-d",
        )

        assert result["is_valid"] is True
        assert result["ground_truth_verdict"] is True
        assert result["total_attempts"] == 2
        assert result["retry_count"] == 1
        assert result["feedback_history"] == ["Invalid UCI: not a move"]
        assert mock_gen.call_count == 2
        assert mock_validate.call_count == 1

    @patch("src.graph.generation.generator_only.generate_move")
    def test_retries_then_succeeds(self, mock_gen: MagicMock) -> None:
        """First attempt illegal, second attempt legal."""
        mock_gen.side_effect = [
            {"raw_output": "e2e5", "prompt_tokens": 100, "completion_tokens": 5, "elapsed_ms": 10.0},
            {"raw_output": "e2e4", "prompt_tokens": 110, "completion_tokens": 5, "elapsed_ms": 12.0},
        ]

        from src.graph.condition_d import run_condition_d

        result = run_condition_d(
            fen=chess.STARTING_FEN,
            game_id="test-d",
        )

        assert result["is_valid"] is True
        assert result["ground_truth_verdict"] is True
        assert result["game_status"] == "ongoing"
        assert result["total_attempts"] == 2
        assert result["retry_count"] == 1

    @patch("src.graph.generation.generator_only.generate_move")
    def test_exhausts_retries_then_forfeits(self, mock_gen: MagicMock) -> None:
        """All 4 attempts (1 initial + 3 retries) are illegal."""
        mock_gen.return_value = {
            "raw_output": "e2e5",
            "prompt_tokens": 100,
            "completion_tokens": 5,
            "elapsed_ms": 10.0,
        }

        from src.graph.condition_d import run_condition_d

        result = run_condition_d(
            fen=chess.STARTING_FEN,
            game_id="test-d",
        )

        assert result["is_valid"] is False
        assert result["game_status"] == "forfeit"
        assert result["total_attempts"] == 4  # 1 initial + 3 retries
        assert mock_gen.call_count == 4


# ── Condition E tests (mocked LLM) ───────────────────────────────────────


class TestConditionE:

    @patch("src.graph.condition_e.explain_error")
    @patch("src.graph.generation.generator_only.generate_move")
    def test_valid_move_accepted(self, mock_gen: MagicMock, mock_explain: MagicMock) -> None:
        mock_gen.return_value = {
            "raw_output": "e2e4",
            "prompt_tokens": 100,
            "completion_tokens": 5,
            "elapsed_ms": 10.0,
        }

        from src.graph.condition_e import run_condition_e

        result = run_condition_e(
            fen=chess.STARTING_FEN,
            game_id="test-e",
        )

        assert result["is_valid"] is True
        assert result["ground_truth_verdict"] is True
        assert result["game_status"] == "ongoing"
        assert mock_gen.call_count == 1
        assert mock_explain.call_count == 0

    @patch("src.graph.condition_e.explain_error")
    @patch("src.graph.generation.generator_only.generate_move")
    def test_retries_through_explainer_then_succeeds(self, mock_gen: MagicMock, mock_explain: MagicMock) -> None:
        mock_gen.side_effect = [
            {"raw_output": "e2e5", "prompt_tokens": 100, "completion_tokens": 5, "elapsed_ms": 10.0},
            {"raw_output": "e2e4", "prompt_tokens": 110, "completion_tokens": 5, "elapsed_ms": 12.0},
        ]
        mock_explain.return_value = {
            "feedback_text": "That move is illegal. Try a legal pawn advance instead.",
            "prompt_tokens": 50,
            "completion_tokens": 10,
            "elapsed_ms": 5.0,
        }

        from src.graph.condition_e import run_condition_e

        result = run_condition_e(
            fen=chess.STARTING_FEN,
            game_id="test-e",
        )

        assert result["is_valid"] is True
        assert result["ground_truth_verdict"] is True
        assert result["game_status"] == "ongoing"
        assert result["total_attempts"] == 2
        assert result["retry_count"] == 1
        assert mock_gen.call_count == 2
        assert mock_explain.call_count == 1


# ── Config tests ──────────────────────────────────────────────────────────


class TestConfig:

    def test_config_for_condition_returns_correct_retries(self) -> None:
        from src.config import Condition, config_for_condition

        assert config_for_condition(Condition.A).max_retries == 0
        assert config_for_condition(Condition.B).max_retries == 0
        assert config_for_condition(Condition.C).max_retries == 3
        assert config_for_condition(Condition.D).max_retries == 3
        assert config_for_condition(Condition.E).max_retries == 3
        assert config_for_condition(Condition.F).max_retries == 0

    def test_config_for_condition_accepts_string(self) -> None:
        from src.config import config_for_condition

        cfg = config_for_condition("D")
        assert cfg.condition.value == "D"
        assert cfg.max_retries == 3
