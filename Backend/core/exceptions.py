from __future__ import annotations
 
 
class IllegalMoveError(Exception):
    """Raised when a UCI move string is legal in syntax but illegal on the board."""
 
    def __init__(self, move: str, fen: str) -> None:
        self.move = move
        self.fen = fen
        super().__init__(f"Illegal move '{move}' in position: {fen}")
 
 
class InvalidFENError(Exception):
    """Raised when a FEN string cannot be parsed into a valid board position."""
 
    def __init__(self, fen: str) -> None:
        self.fen = fen
        super().__init__(f"Invalid FEN string: '{fen}'")