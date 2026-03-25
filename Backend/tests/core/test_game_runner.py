import json
from pathlib import Path

from core.game_runner import GameRunner
from core.state_manager import StateManager
from core.validator import RuleValidator


def test_game_runner_sequence_applies_only_valid_moves(tmp_path: Path) -> None:
    log_path = tmp_path / "logs" / "attempts.jsonl"
    manager = StateManager()
    validator = RuleValidator(manager)
    runner = GameRunner(manager, validator, log_path)

    runner.run_sequence(["e2e4", "e7e5", "e2e5", "g1f3"])

    # Illegal move must not change board state.
    assert manager.board.fullmove_number >= 2

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 4
    records = [json.loads(line) for line in lines]
    assert records[2]["validator_result"] == "illegal_move"


def test_game_runner_detects_terminal_checkmate(tmp_path: Path) -> None:
    log_path = tmp_path / "logs" / "attempts.jsonl"
    manager = StateManager()
    validator = RuleValidator(manager)
    runner = GameRunner(manager, validator, log_path)

    # Fool's mate sequence
    runner.run_sequence(["f2f3", "e7e5", "g2g4", "d8h4"])

    terminal, outcome = runner.terminal_status()
    assert terminal is True
    assert outcome == "0-1"
