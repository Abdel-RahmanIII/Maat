import json
from pathlib import Path

from core.game_runner import GameRunner
from core.rule_validator import RuleValidator
from core.state_manager import StateManager


def test_submit_move_returns_move_result(tmp_path: Path) -> None:
    """submit_move() must return a MoveResult — graph nodes will consume it."""
    manager = StateManager()
    validator = RuleValidator(manager)
    runner = GameRunner(manager, validator)

    result = runner.submit_move("e2e4")

    assert result.validation.is_valid is True
    assert result.state_after_fen is not None
    assert result.terminal_flag is False


def test_invalid_move_result_has_no_after_fen(tmp_path: Path) -> None:
    manager = StateManager()
    validator = RuleValidator(manager)
    runner = GameRunner(manager, validator)

    result = runner.submit_move("e2e5")

    assert result.validation.is_valid is False
    assert result.state_after_fen is None


def test_sequence_skips_illegal_move_without_mutating_state(tmp_path: Path) -> None:
    log_path = tmp_path / "log.jsonl"
    manager = StateManager()
    validator = RuleValidator(manager)
    runner = GameRunner(manager, validator, log_path)

    runner.run_sequence(["e2e4", "e7e5", "e2e5", "g1f3"])

    assert manager.board.fullmove_number >= 2

    lines = log_path.read_text().strip().splitlines()
    records = [json.loads(l) for l in lines]
    illegal_record = records[2]
    assert illegal_record["validator_result"] == "illegal_move"
    assert illegal_record["validation_stage"] == "legality"
    assert illegal_record["state_after_fen"] is None


def test_log_includes_validation_stage(tmp_path: Path) -> None:
    """validation_stage must be written to JSONL for Week 8 analysis."""
    log_path = tmp_path / "log.jsonl"
    manager = StateManager()
    validator = RuleValidator(manager)
    runner = GameRunner(manager, validator, log_path)

    runner.submit_move("hello")   # syntax failure
    runner.submit_move("e2e5")    # legality failure
    runner.submit_move("e2e4")    # valid

    records = [json.loads(l) for l in log_path.read_text().strip().splitlines()]
    assert records[0]["validation_stage"] == "syntax"
    assert records[1]["validation_stage"] == "legality"
    assert records[2]["validation_stage"] is None


def test_checkmate_detected(tmp_path: Path) -> None:
    log_path = tmp_path / "log.jsonl"
    manager = StateManager()
    validator = RuleValidator(manager)
    runner = GameRunner(manager, validator, log_path)

    runner.run_sequence(["f2f3", "e7e5", "g2g4", "d8h4"])

    terminal, outcome = runner.terminal_status()
    assert terminal is True
    assert outcome == "0-1"