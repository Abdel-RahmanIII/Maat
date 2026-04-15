"""Per-turn metric recording and JSONL persistence.

The :class:`MetricsCollector` converts completed ``TurnState`` dicts into
:class:`TurnRecord` objects, accumulates them into :class:`GameRecord`
containers, and writes both to JSONL files for downstream analysis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.metrics.definitions import GameRecord, TurnRecord
from src.state import TurnState


class MetricsCollector:
    """Records per-turn and per-game metrics to JSONL files.

    Usage::

        collector = MetricsCollector(output_dir=Path("results/exp1"), experiment="exp1")

        # After each turn completes:
        turn_rec = collector.record_turn(state, experiment="exp1")

        # After a game/puzzle finishes:
        game_rec = collector.record_game(
            game_id="puzzle_001",
            condition="B",
            turns=accumulated_turns,
            game_status="ongoing",
        )

        # Persist everything:
        collector.flush()
    """

    def __init__(self, output_dir: Path, experiment: str) -> None:
        self._output_dir = Path(output_dir)
        self._experiment = experiment
        self._turn_buffer: list[TurnRecord] = []
        self._game_buffer: list[GameRecord] = []

    # ── Recording ─────────────────────────────────────────────────────

    def record_turn(
        self,
        state: TurnState,
        *,
        experiment: str | None = None,
    ) -> TurnRecord:
        """Convert a completed ``TurnState`` into a :class:`TurnRecord`.

        The ``experiment`` parameter overrides the collector-level default
        if provided.
        """

        exp = experiment or self._experiment

        record = TurnRecord(
            game_id=state["game_id"],
            condition=state["condition"],
            experiment=exp,
            move_number=state["move_number"],
            game_phase=state.get("game_phase", ""),
            proposed_move=state.get("proposed_move", ""),
            is_valid=state["is_valid"],
            first_try_valid=state["first_try_valid"],
            total_attempts=state["total_attempts"],
            retry_count=state["retry_count"],
            error_types=list(state["error_types"]),
            llm_calls_this_turn=state["llm_calls_this_turn"],
            tokens_this_turn=state["tokens_this_turn"],
            prompt_token_count=state["prompt_token_count"],
            wall_clock_ms=state.get("wall_clock_ms", 0.0),
            tool_calls=list(state["tool_calls"]),
            critic_verdict=state.get("critic_verdict"),
            ground_truth_verdict=state.get("ground_truth_verdict"),
            feedback_history=list(state["feedback_history"]),
            generation_strategy=state.get("generation_strategy", "generator_only"),
            strategic_plan=state.get("strategic_plan", ""),
            routed_phase=state.get("routed_phase", ""),
        )

        self._turn_buffer.append(record)
        return record

    def record_game(
        self,
        game_id: str,
        condition: str,
        turns: list[TurnRecord],
        game_status: str,
        *,
        experiment: str | None = None,
    ) -> GameRecord:
        """Finalize a game from its accumulated turns.

        Computes basic game-level fields (total_turns, total_forfeits,
        game_length, gcr_contributed).  Higher-level metrics (fir, ftir)
        are set later by the aggregator.
        """

        exp = experiment or self._experiment

        total_turns = len(turns)
        total_forfeits = sum(1 for t in turns if not t.is_valid)
        game_length = sum(1 for t in turns if t.is_valid)

        # gcr: game reached a *natural* termination (not forfeit)
        gcr_contributed = game_status in ("checkmate", "stalemate", "draw", "max_moves")

        record = GameRecord(
            game_id=game_id,
            condition=condition,
            experiment=exp,
            turns=turns,
            game_status=game_status,
            total_turns=total_turns,
            total_forfeits=total_forfeits,
            game_length=game_length,
            gcr_contributed=gcr_contributed,
        )

        self._game_buffer.append(record)
        return record

    # ── Persistence ───────────────────────────────────────────────────

    def flush(self) -> None:
        """Write all buffered records to JSONL files.

        Creates ``turns.jsonl`` and ``games.jsonl`` in the output directory.
        Appends to existing files so that interrupted runs can resume.
        """

        self._output_dir.mkdir(parents=True, exist_ok=True)

        turns_path = self._output_dir / "turns.jsonl"
        games_path = self._output_dir / "games.jsonl"

        if self._turn_buffer:
            with open(turns_path, "a", encoding="utf-8") as f:
                for rec in self._turn_buffer:
                    f.write(rec.model_dump_json() + "\n")
            self._turn_buffer.clear()

        if self._game_buffer:
            with open(games_path, "a", encoding="utf-8") as f:
                for rec in self._game_buffer:
                    f.write(rec.model_dump_json() + "\n")
            self._game_buffer.clear()

    # ── Loading ───────────────────────────────────────────────────────

    @staticmethod
    def load_turns(path: Path) -> list[TurnRecord]:
        """Load :class:`TurnRecord` objects from a JSONL file."""

        records: list[TurnRecord] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(TurnRecord.model_validate_json(line))
        return records

    @staticmethod
    def load_games(path: Path) -> list[GameRecord]:
        """Load :class:`GameRecord` objects from a JSONL file."""

        records: list[GameRecord] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(GameRecord.model_validate_json(line))
        return records
