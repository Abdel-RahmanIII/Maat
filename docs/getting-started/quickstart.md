# Quickstart

## 1. Run Test Suite

```powershell
python -m pytest -q
```

## 2. Parse a Model Move

```python
from src.validators.move_parser import parse_uci_move

print(parse_uci_move("e2e4"))
print(parse_uci_move("I play e7e8q now"))
```

## 3. Validate Move Legality

```python
import chess
from src.validators.symbolic import validate_move

print(validate_move(chess.STARTING_FEN, "e2e4"))
print(validate_move(chess.STARTING_FEN, "e2e5"))
```

## 4. Use Analysis Tools

```python
import chess
from src.tools.chess_tools import get_legal_moves, get_board_state

print(get_board_state(chess.STARTING_FEN))
print(get_legal_moves(chess.STARTING_FEN)[:5])
```

## 5. Sample Puzzle Data

```python
from src.data.puzzle_sampler import sample_from_csv

sample = sample_from_csv("path/to/puzzles.csv", per_phase=10, seed=42)
print(len(sample))
```

## 6. Query Stockfish (Optional)

```python
import chess
from src.engine.stockfish_wrapper import StockfishWrapper

with StockfishWrapper() as engine:
    move = engine.choose_move(chess.STARTING_FEN, time_limit=0.05)
    print(move)
```
