from __future__ import annotations

import os
import shutil
from pathlib import Path

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
