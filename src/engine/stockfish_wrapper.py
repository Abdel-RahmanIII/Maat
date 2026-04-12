from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import chess
import chess.engine


class StockfishWrapper:
    """Thin wrapper around a UCI-compatible Stockfish binary."""

    def __init__(self, engine_path: str | Path | None = None, elo: int = 1320) -> None:
        self._engine_path = str(engine_path) if engine_path is not None else None
        self._engine: chess.engine.SimpleEngine | None = None
        self._elo = elo

    def _resolve_engine_path(self) -> str:
        if self._engine_path:
            return self._engine_path

        env_path = os.getenv("STOCKFISH_PATH")
        if env_path:
            return env_path

        detected = shutil.which("stockfish")
        if detected:
            return detected

        raise FileNotFoundError(
            "Stockfish binary not found. Provide engine_path or set STOCKFISH_PATH."
        )

    def start(self) -> None:
        if self._engine is not None:
            return

        resolved_path = self._resolve_engine_path()
        self._engine = chess.engine.SimpleEngine.popen_uci(resolved_path)
        self.set_elo(self._elo)

    def close(self) -> None:
        if self._engine is not None:
            self._engine.quit()
            self._engine = None

    def __enter__(self) -> StockfishWrapper:
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def set_elo(self, elo: int) -> None:
        self._elo = elo
        if self._engine is not None:
            options = self._engine.options
            config: dict[str, int | bool] = {}
            if "UCI_LimitStrength" in options:
                config["UCI_LimitStrength"] = True
            if "UCI_Elo" in options:
                config["UCI_Elo"] = elo

            if config:
                self._engine.configure(config)

    def choose_move(self, fen: str, time_limit: float = 0.1) -> str:
        try:
            board = chess.Board(fen)
        except ValueError as exc:
            raise ValueError("Invalid FEN provided.") from exc

        self.start()
        assert self._engine is not None

        result = self._engine.play(board, chess.engine.Limit(time=time_limit))
        if result.move is None:
            raise RuntimeError("Engine did not return a move.")

        return result.move.uci()

    @staticmethod
    def _format_centipawn(cp: int) -> str:
        return f"{cp / 100:+.2f}"

    @staticmethod
    def _score_payload(score: chess.engine.PovScore, perspective: chess.Color) -> dict[str, Any]:
        relative = score.pov(perspective)
        cp_for_loss = relative.score(mate_score=100000)
        if cp_for_loss is None:
            raise RuntimeError("Engine score is unavailable.")

        mate = relative.mate()
        if mate is not None:
            if mate > 0:
                display = f"M{mate}"
            elif mate < 0:
                display = f"-M{abs(mate)}"
            else:
                display = "M0"
            return {
                "kind": "mate",
                "cp": cp_for_loss,
                "mate": mate,
                "display": display,
            }

        cp = relative.score()
        if cp is None:
            raise RuntimeError("Engine score is unavailable.")

        return {
            "kind": "cp",
            "cp": cp,
            "mate": None,
            "display": StockfishWrapper._format_centipawn(cp),
        }

    @staticmethod
    def classify_move_quality(score_loss_cp: int) -> str:
        if score_loss_cp <= 25:
            return "Best"
        if score_loss_cp <= 90:
            return "Good"
        if score_loss_cp <= 200:
            return "Inaccuracy"
        if score_loss_cp <= 350:
            return "Mistake"
        return "Blunder"

    def analyze_move(self, fen: str, move_uci: str, time_limit: float = 0.12) -> dict[str, Any]:
        try:
            board_before = chess.Board(fen)
        except ValueError as exc:
            raise ValueError("Invalid FEN provided.") from exc

        try:
            move = chess.Move.from_uci(move_uci)
        except ValueError as exc:
            raise ValueError("Invalid UCI move provided.") from exc

        if move not in board_before.legal_moves:
            raise ValueError("Move is not legal for the provided FEN.")

        perspective = board_before.turn

        self.start()
        assert self._engine is not None

        limit = chess.engine.Limit(time=time_limit)
        info_before = self._engine.analyse(board_before, limit)
        score_before = info_before.get("score")
        if score_before is None:
            raise RuntimeError("Engine did not provide a score for the initial position.")

        best = self._engine.play(board_before, limit)
        best_move_uci = best.move.uci() if best.move is not None else None

        board_after = board_before.copy(stack=False)
        board_after.push(move)
        info_after = self._engine.analyse(board_after, limit)
        score_after = info_after.get("score")
        if score_after is None:
            raise RuntimeError("Engine did not provide a score for the resulting position.")

        score_before_payload = self._score_payload(score_before, perspective)
        score_after_payload = self._score_payload(score_after, perspective)
        score_loss_cp = int(score_before_payload["cp"]) - int(score_after_payload["cp"])

        return {
            "move_uci": move_uci,
            "best_move": best_move_uci,
            "is_best_move": bool(best_move_uci and best_move_uci == move_uci),
            "quality": self.classify_move_quality(score_loss_cp),
            "score_loss_cp": score_loss_cp,
            "score_before": score_before_payload,
            "score_after": score_after_payload,
        }
