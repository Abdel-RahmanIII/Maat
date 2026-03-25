from __future__ import annotations

import argparse
from pathlib import Path

from core.game_runner import GameRunner
from core.state_manager import StateManager
from core.validator import RuleValidator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a deterministic chess move sequence.")
    parser.add_argument("--fen", type=str, default=None, help="Starting FEN. Defaults to standard start.")
    parser.add_argument("--moves", nargs="*", default=[], help="Move list in UCI or SAN.")
    parser.add_argument("--log", type=Path, default=None, help="Optional JSONL log output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    state_manager = StateManager(args.fen)
    validator = RuleValidator(state_manager)
    runner = GameRunner(state_manager, validator, args.log)

    runner.run_sequence(args.moves)
    terminal, outcome = runner.terminal_status()

    print("Final FEN:", state_manager.current_fen())
    print("Terminal:", terminal)
    print("Outcome:", outcome)


if __name__ == "__main__":
    main()
