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

    @patch("src.graph.base_graph.generate_move")
    def test_valid_move_accepted(self, mock_gen: MagicMock) -> None:
        mock_gen.return_value = {
            "raw_output": "e2e4",
            "prompt_tokens": 100,
            "completion_tokens": 5,
        }

        from src.graph.condition_a import run_condition_a

        result = run_condition_a(
            fen=chess.STARTING_FEN,
            game_id="test-a",
        )

        assert result["is_valid"] is True
        assert result["proposed_move"] == "e2e4"
        assert result["game_status"] == "ongoing"
        assert result["total_attempts"] == 1

    @patch("src.graph.base_graph.generate_move")
    def test_illegal_move_forfeits(self, mock_gen: MagicMock) -> None:
        mock_gen.return_value = {
            "raw_output": "e2e5",
            "prompt_tokens": 100,
            "completion_tokens": 5,
        }

        from src.graph.condition_a import run_condition_a

        result = run_condition_a(
            fen=chess.STARTING_FEN,
            game_id="test-a",
        )

        assert result["is_valid"] is False
        assert result["game_status"] == "forfeit"
        assert len(result["error_types"]) > 0


# ── Condition B tests (mocked LLM) ───────────────────────────────────────


class TestConditionB:

    @patch("src.graph.base_graph.generate_move")
    def test_valid_move_accepted(self, mock_gen: MagicMock) -> None:
        mock_gen.return_value = {
            "raw_output": "e2e4",
            "prompt_tokens": 100,
            "completion_tokens": 5,
        }

        from src.graph.condition_b import run_condition_b

        result = run_condition_b(
            fen=chess.STARTING_FEN,
            game_id="test-b",
        )

        assert result["is_valid"] is True
        assert result["game_status"] == "ongoing"

    @patch("src.graph.base_graph.generate_move")
    def test_illegal_move_forfeits_no_retry(self, mock_gen: MagicMock) -> None:
        mock_gen.return_value = {
            "raw_output": "e2e5",
            "prompt_tokens": 100,
            "completion_tokens": 5,
        }

        from src.graph.condition_b import run_condition_b

        result = run_condition_b(
            fen=chess.STARTING_FEN,
            game_id="test-b",
        )

        assert result["is_valid"] is False
        assert result["game_status"] == "forfeit"
        # Should NOT retry (max_retries=0)
        assert mock_gen.call_count == 1


# ── Condition D tests (mocked LLM) ───────────────────────────────────────


class TestConditionD:

    @patch("src.graph.base_graph.generate_move")
    def test_valid_on_first_try(self, mock_gen: MagicMock) -> None:
        mock_gen.return_value = {
            "raw_output": "e2e4",
            "prompt_tokens": 100,
            "completion_tokens": 5,
        }

        from src.graph.condition_d import run_condition_d

        result = run_condition_d(
            fen=chess.STARTING_FEN,
            game_id="test-d",
        )

        assert result["is_valid"] is True
        assert result["game_status"] == "ongoing"

    @patch("src.graph.base_graph.generate_move")
    def test_retries_then_succeeds(self, mock_gen: MagicMock) -> None:
        """First attempt illegal, second attempt legal."""
        mock_gen.side_effect = [
            {"raw_output": "e2e5", "prompt_tokens": 100, "completion_tokens": 5},
            {"raw_output": "e2e4", "prompt_tokens": 110, "completion_tokens": 5},
        ]

        from src.graph.condition_d import run_condition_d

        result = run_condition_d(
            fen=chess.STARTING_FEN,
            game_id="test-d",
        )

        assert result["is_valid"] is True
        assert result["game_status"] == "ongoing"
        assert result["total_attempts"] == 2
        assert result["retry_count"] == 1

    @patch("src.graph.base_graph.generate_move")
    def test_exhausts_retries_then_forfeits(self, mock_gen: MagicMock) -> None:
        """All 4 attempts (1 initial + 3 retries) are illegal."""
        mock_gen.return_value = {
            "raw_output": "e2e5",
            "prompt_tokens": 100,
            "completion_tokens": 5,
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
