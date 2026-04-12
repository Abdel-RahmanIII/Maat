from __future__ import annotations

import json

import chess

from src.tools.chess_tools import (
    get_attackers,
    get_board_visual,
    get_defenders,
    get_game_phase,
    get_move_history_pgn,
    get_piece_at,
    get_position_after_moves,
    get_tools_for_input_mode,
    is_in_check,
    is_square_safe,
    validate_move,
)


def test_tool_validate_move_returns_valid_for_legal_move() -> None:
    result = json.loads(validate_move.invoke({"fen": chess.STARTING_FEN, "move_uci": "e2e4"}))
    assert result["legal"] is True
    assert result["rule_ref"] == "LEGAL"
    assert result["error_type"] is None


def test_tool_validate_move_returns_reason_for_illegal_move() -> None:
    result = json.loads(validate_move.invoke({"fen": chess.STARTING_FEN, "move_uci": "e2e5"}))
    assert result["legal"] is False
    assert result["rule_ref"] == result["error_type"]
    assert "destination" in result["reason"].lower() or "legal" in result["reason"].lower()


def test_is_in_check_detects_checking_square() -> None:
    fen = "4k3/8/8/8/8/8/4R3/4K3 b - - 0 1"
    result = json.loads(is_in_check.invoke({"fen": fen}))
    assert result["in_check"] is True
    assert result["checking_squares"] == ["e2"]


def test_get_game_phase_thresholds() -> None:
    assert get_game_phase.invoke({"move_history": []}) == "opening"
    assert get_game_phase.invoke({"move_history": ["e2e4"] * 20}) == "opening"
    assert get_game_phase.invoke({"move_history": ["e2e4"] * 21}) == "middlegame"
    assert get_game_phase.invoke({"move_history": ["e2e4"] * 80}) == "middlegame"
    assert get_game_phase.invoke({"move_history": ["e2e4"] * 81}) == "endgame"


def test_get_move_history_pgn_formats_san_sequence() -> None:
    pgn = get_move_history_pgn.invoke({"move_history": ["e2e4", "e7e5", "g1f3"]})
    assert pgn == "1. e4 e5 2. Nf3"


def test_get_board_visual_returns_ascii_board() -> None:
    board_visual = get_board_visual.invoke({"fen": chess.STARTING_FEN})
    assert "r n b q k b n r" in board_visual


def test_get_piece_at_returns_piece_or_empty() -> None:
    assert get_piece_at.invoke({"fen": chess.STARTING_FEN, "square": "e1"}) == "wK"
    assert get_piece_at.invoke({"fen": chess.STARTING_FEN, "square": "e4"}) == "empty"


def test_get_attackers_returns_both_colors() -> None:
    fen = "4k3/8/8/8/8/8/4r3/4K3 w - - 0 1"
    attackers = json.loads(get_attackers.invoke({"fen": fen, "square": "e1"}))
    assert attackers == [{"square": "e2", "piece": "bR", "color": "black"}]


def test_get_defenders_returns_occupant_defenders() -> None:
    fen = "8/8/8/8/8/5k2/4r3/K7 w - - 0 1"
    defenders = json.loads(get_defenders.invoke({"fen": fen, "square": "e2"}))
    assert defenders == [{"square": "f3", "piece": "bK", "color": "black"}]


def test_is_square_safe_returns_threats() -> None:
    fen = "4k3/8/8/8/8/8/4r3/4K3 w - - 0 1"
    unsafe = json.loads(is_square_safe.invoke({"fen": fen, "square": "e1", "color": "white"}))
    safe = json.loads(is_square_safe.invoke({"fen": fen, "square": "a1", "color": "white"}))
    assert unsafe == {"safe": False, "threats": ["e2"]}
    assert safe == {"safe": True, "threats": []}


def test_get_position_after_moves_returns_resulting_fen() -> None:
    result_fen = get_position_after_moves.invoke(
        {"fen": chess.STARTING_FEN, "moves": ["e2e4", "e7e5"]}
    )
    board = chess.Board(chess.STARTING_FEN)
    board.push_uci("e2e4")
    board.push_uci("e7e5")
    assert result_fen == board.fen()


def test_get_tools_for_input_mode_gates_fen_only_tools() -> None:
    fen_tools = {tool.name for tool in get_tools_for_input_mode("fen")}
    history_tools = {tool.name for tool in get_tools_for_input_mode("history")}

    assert "submit_move" in fen_tools
    assert "submit_move" in history_tools
    assert "get_board_visual" in fen_tools
    assert "get_board_visual" not in history_tools
