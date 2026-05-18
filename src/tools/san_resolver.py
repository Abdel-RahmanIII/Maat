from __future__ import annotations

import re

import chess

SAN_MOVE_RE = re.compile(
    r"^(?P<piece>[KQRBN])?"
    r"(?P<file>[a-h])?"
    r"(?P<rank>[1-8])?"
    r"(?P<capture>x)?"
    r"(?P<to>[a-h][1-8])"
    r"(?:=(?P<promo>[QRBN]))?$",
    re.IGNORECASE,
)

PIECE_TYPES = {
    "K": chess.KING,
    "Q": chess.QUEEN,
    "R": chess.ROOK,
    "B": chess.BISHOP,
    "N": chess.KNIGHT,
}


def _strip_san_suffix(token: str) -> str:
    return re.sub(r"[+#]+$", "", token)


def _castle_uci(board: chess.Board, token: str) -> str:
    if board.turn == chess.WHITE:
        return "e1g1" if token == "O-O" else "e1c1"
    return "e8g8" if token == "O-O" else "e8c8"


def _path_clear(board: chess.Board, from_square: int, to_square: int) -> bool:
    from_file = chess.square_file(from_square)
    from_rank = chess.square_rank(from_square)
    to_file = chess.square_file(to_square)
    to_rank = chess.square_rank(to_square)

    file_step = (to_file > from_file) - (to_file < from_file)
    rank_step = (to_rank > from_rank) - (to_rank < from_rank)

    cur_file = from_file + file_step
    cur_rank = from_rank + rank_step

    while (cur_file, cur_rank) != (to_file, to_rank):
        square = chess.square(cur_file, cur_rank)
        if board.piece_at(square) is not None:
            return False
        cur_file += file_step
        cur_rank += rank_step

    return True


def _pawn_can_reach(
    board: chess.Board,
    from_square: int,
    to_square: int,
    capture: bool,
) -> bool:
    from_file = chess.square_file(from_square)
    to_file = chess.square_file(to_square)
    from_rank = chess.square_rank(from_square)
    to_rank = chess.square_rank(to_square)

    direction = 1 if board.turn == chess.WHITE else -1
    start_rank = 1 if board.turn == chess.WHITE else 6

    file_delta = to_file - from_file
    rank_delta = to_rank - from_rank

    if capture:
        return abs(file_delta) == 1 and rank_delta == direction

    if file_delta != 0:
        return False

    if rank_delta == direction:
        return True

    if rank_delta == 2 * direction and from_rank == start_rank:
        intermediate = chess.square(from_file, from_rank + direction)
        return board.piece_at(intermediate) is None

    return False


def _piece_can_reach(
    board: chess.Board,
    piece: chess.Piece,
    from_square: int,
    to_square: int,
    capture: bool,
) -> bool:
    from_file = chess.square_file(from_square)
    from_rank = chess.square_rank(from_square)
    to_file = chess.square_file(to_square)
    to_rank = chess.square_rank(to_square)

    file_delta = to_file - from_file
    rank_delta = to_rank - from_rank
    abs_file = abs(file_delta)
    abs_rank = abs(rank_delta)

    if piece.piece_type == chess.PAWN:
        return _pawn_can_reach(board, from_square, to_square, capture)

    if piece.piece_type == chess.KNIGHT:
        return (abs_file, abs_rank) in ((1, 2), (2, 1))

    if piece.piece_type == chess.BISHOP:
        return abs_file == abs_rank and _path_clear(board, from_square, to_square)

    if piece.piece_type == chess.ROOK:
        if file_delta != 0 and rank_delta != 0:
            return False
        return _path_clear(board, from_square, to_square)

    if piece.piece_type == chess.QUEEN:
        if abs_file == abs_rank or file_delta == 0 or rank_delta == 0:
            return _path_clear(board, from_square, to_square)
        return False

    if piece.piece_type == chess.KING:
        return max(abs_file, abs_rank) == 1

    return False


def _resolve_san_to_uci(candidate: str, board: chess.Board) -> str | None:
    token = _strip_san_suffix(candidate)

    if token in ("O-O", "O-O-O"):
        return _castle_uci(board, token)

    match = SAN_MOVE_RE.fullmatch(token)
    if not match:
        return None

    piece_symbol = match.group("piece")
    piece_type = PIECE_TYPES.get(piece_symbol) if piece_symbol else chess.PAWN
    dis_file = match.group("file")
    dis_rank = match.group("rank")
    capture = match.group("capture") is not None
    to_square = match.group("to")
    promo_symbol = match.group("promo")
    promotion = PIECE_TYPES.get(promo_symbol) if promo_symbol else None

    if promotion is not None and piece_type != chess.PAWN:
        return None

    try:
        to_square_index = chess.parse_square(to_square)
    except ValueError:
        return None

    if dis_file is not None:
        dis_file_index = ord(dis_file.lower()) - ord("a")
    else:
        dis_file_index = None

    if dis_rank is not None:
        dis_rank_index = int(dis_rank) - 1
    else:
        dis_rank_index = None

    candidates: list[int] = []
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None or piece.color != board.turn:
            continue
        if piece.piece_type != piece_type:
            continue
        if dis_file_index is not None and chess.square_file(square) != dis_file_index:
            continue
        if dis_rank_index is not None and chess.square_rank(square) != dis_rank_index:
            continue
        if not _piece_can_reach(board, piece, square, to_square_index, capture):
            continue
        candidates.append(square)

    if len(candidates) != 1:
        return None

    uci_move = chess.square_name(candidates[0]) + to_square
    if promotion is not None:
        uci_move += chess.piece_symbol(promotion)
    return uci_move
