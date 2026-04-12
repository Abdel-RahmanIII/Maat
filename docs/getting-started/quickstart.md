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
import json
from src.tools.chess_tools import get_board_visual, get_piece_at, validate_move
import chess

print(get_board_visual.invoke({"fen": chess.STARTING_FEN}))
print(get_piece_at.invoke({"fen": chess.STARTING_FEN, "square": "e1"}))
result = json.loads(validate_move.invoke({"fen": chess.STARTING_FEN, "move_uci": "e2e4"}))
print(result)
```

## 5. Run a Single Condition (Mocked — No API Key Needed)

```python
# Condition A with mocked LLM (for testing)
from unittest.mock import patch

with patch("src.graph.base_graph.generate_move") as mock_gen:
    mock_gen.return_value = {"raw_output": "e2e4", "prompt_tokens": 100, "completion_tokens": 5}
    from src.graph.condition_a import run_condition_a
    import chess

    result = run_condition_a(fen=chess.STARTING_FEN, game_id="demo")
    print(f"Valid: {result['is_valid']}, Move: {result['proposed_move']}")
```

## 6. Run a Condition with Real LLM (API Key Required)

```python
import chess
from src.graph.condition_d import run_condition_d

result = run_condition_d(
    fen=chess.STARTING_FEN,
    game_id="demo",
    generation_strategy="generator_only",
)
print(f"Valid: {result['is_valid']}, Move: {result['proposed_move']}")
print(f"Attempts: {result['total_attempts']}, Status: {result['game_status']}")
```

## 7. Sample Puzzle Data

```python
from src.data.puzzle_sampler import sample_from_csv

sample = sample_from_csv("path/to/puzzles.csv", per_phase=10, seed=42)
print(len(sample))
```

## 8. Query Stockfish (Optional)

```python
import chess
from src.engine.stockfish_wrapper import StockfishWrapper

with StockfishWrapper() as engine:
    move = engine.choose_move(chess.STARTING_FEN, time_limit=0.05)
    print(move)
```
