# Data Sampler Module Reference

## File

- `src/data/puzzle_sampler.py`

## Purpose

Loads and samples chess puzzle positions for experiment datasets with phase-aware stratification.

## Core Types

```python
@dataclass(frozen=True)
class PuzzleRecord:
    puzzle_id: str
    fen: str
    rating: int
    phase: str
    fullmove_number: int
```

## Public APIs

```python
classify_phase(fen: str) -> str
download_puzzle_csv(destination: str | Path, url: str = LICHESS_PUZZLE_CSV_URL) -> Path
load_puzzles(csv_path: str | Path) -> list[PuzzleRecord]
stratified_sample(records: list[PuzzleRecord], per_phase: int = 100, seed: int = 42, rating_buckets: int = 4) -> list[PuzzleRecord]
sample_from_csv(csv_path: str | Path, per_phase: int = 100, seed: int = 42, rating_buckets: int = 4) -> list[PuzzleRecord]
```

## Phase Classification Rules

- Opening: `fullmove_number <= 15`
- Endgame: `fullmove_number > 35` or non-pawn material <= 13
- Middlegame: otherwise

Non-pawn material points count:

- knight = 3
- bishop = 3
- rook = 5
- queen = 9

## Sampling Strategy

1. Partition records by phase.
2. For each phase, sort by rating and bucketize into quartiles (default 4 buckets).
3. Sample near-evenly from each rating bucket.
4. If some buckets are short, backfill from leftovers.
5. Shuffle final combined sample deterministically using seed.

## Input Tolerance

- Accepts common CSV key variants (for id, fen, rating).
- Skips rows with missing identifiers or invalid FEN.
- Invalid/missing ratings default to 0.
