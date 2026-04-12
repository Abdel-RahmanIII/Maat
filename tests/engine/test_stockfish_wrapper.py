from __future__ import annotations

import os
import shutil

import chess
import chess.engine
import pytest

from src.engine.stockfish_wrapper import StockfishWrapper


def test_choose_move_rejects_invalid_fen() -> None:
    wrapper = StockfishWrapper()

    with pytest.raises(ValueError):
        wrapper.choose_move("not-a-fen")


def test_resolve_engine_path_prefers_constructor_value() -> None:
    wrapper = StockfishWrapper(engine_path="C:/custom/stockfish")
    assert wrapper._resolve_engine_path() == "C:/custom/stockfish"


def test_resolve_engine_path_uses_environment_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STOCKFISH_PATH", "C:/env/stockfish")
    wrapper = StockfishWrapper(engine_path=None)
    assert wrapper._resolve_engine_path() == "C:/env/stockfish"


def test_choose_move_returns_legal_move_when_engine_available() -> None:
    engine_path = os.getenv("STOCKFISH_PATH") or shutil.which("stockfish")
    if not engine_path:
        pytest.skip("Stockfish binary is not available in this environment.")

    wrapper = StockfishWrapper(engine_path=engine_path)
    try:
        move = wrapper.choose_move(chess.STARTING_FEN, time_limit=0.01)
        board = chess.Board(chess.STARTING_FEN)
        assert move in {candidate.uci() for candidate in board.legal_moves}
    finally:
        wrapper.close()


def test_score_payload_formats_centipawn_score() -> None:
    score = chess.engine.PovScore(chess.engine.Cp(57), chess.WHITE)
    payload = StockfishWrapper._score_payload(score, chess.WHITE)

    assert payload["kind"] == "cp"
    assert payload["cp"] == 57
    assert payload["mate"] is None
    assert payload["display"] == "+0.57"


def test_score_payload_formats_mate_score() -> None:
    score = chess.engine.PovScore(chess.engine.Mate(3), chess.WHITE)
    payload = StockfishWrapper._score_payload(score, chess.WHITE)

    assert payload["kind"] == "mate"
    assert payload["mate"] == 3
    assert payload["display"] == "M3"
    assert isinstance(payload["cp"], int)


@pytest.mark.parametrize(
    ("loss", "expected"),
    [
        (-20, "Best"),
        (0, "Best"),
        (40, "Good"),
        (120, "Inaccuracy"),
        (260, "Mistake"),
        (500, "Blunder"),
    ],
)
def test_classify_move_quality_thresholds(loss: int, expected: str) -> None:
    assert StockfishWrapper.classify_move_quality(loss) == expected


def test_analyze_move_rejects_invalid_move_without_engine() -> None:
    wrapper = StockfishWrapper()
    with pytest.raises(ValueError):
        wrapper.analyze_move(chess.STARTING_FEN, "e2e9")


def test_analyze_move_rejects_illegal_move_without_engine() -> None:
    wrapper = StockfishWrapper()
    with pytest.raises(ValueError):
        wrapper.analyze_move(chess.STARTING_FEN, "e2e5")
