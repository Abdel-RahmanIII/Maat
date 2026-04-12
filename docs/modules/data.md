# Data Sampler Module Reference

## File

- scripts/puzzle_sampler.py

Re-exported through:

- src/data/__init__.py

## Purpose

Loads Lichess puzzle CSV data, applies quality filtering, labels each puzzle by phase and difficulty, performs stratified sampling, and writes experiment artifacts.

## Puzzle Record Model

`PuzzleRecord` stores both raw CSV fields and derived metadata used for filtering/sampling:

- puzzle_id, fen, moves, rating
- rating_deviation, popularity, nb_plays, themes
- fullmove_number, piece_count, major_pieces
- phase, phase_source, heuristic_phase, difficulty

## Filtering Pipeline (Current Behavior)

The end-to-end pipeline is implemented in `prepare_experiment_dataset(...)` and runs in this order:

1. Load and parse rows
- Reads CSV via `load_puzzles(...)`.
- Drops rows with missing `PuzzleId`/`FEN`.
- Drops rows with invalid FEN.
- Extracts FEN features:
    - fullmove
    - piece_count
    - major_pieces (piece_count minus pawns)

2. Phase labeling (theme-first)
- If themes contain `opening`, phase is opening.
- If themes contain `middlegame`, phase is middlegame.
- If themes contain `endgame` or endgame hints (`rookendgame`, `queenendgame`, `pawnendgame`, `bishopendgame`, `knightendgame`, `queenvsrook`), phase is endgame.
- Otherwise, fallback heuristic phase is used:
    - opening if fullmove <= 12
    - endgame if piece_count <= 10 or fullmove >= 40
    - middlegame otherwise
- `phase_source` is set to `theme` or `heuristic`.

3. Difficulty labeling
- easy if rating < 1300
- medium if 1300 <= rating < 1700
- hard if rating >= 1700

4. Quality filter
- Applied by `apply_quality_filters(...)`.
- Keeps only rows satisfying all thresholds:
    - RatingDeviation < 75
    - Popularity > 50
    - NbPlays >= 100

5. Phase x difficulty stratified sampling
- Implemented by `stratified_sample_phase_difficulty(...)`.
- For each of 9 cells (phase x difficulty):
    - sample from themed pool first
    - if themed sample is below target_per_cell, top up from heuristic pool in the same cell
    - stop when target_per_cell reached or pools exhausted
- Final sampled list is shuffled deterministically by seed.
- If `final_target > 0`, the shuffled list is truncated to that size.

6. Optional sanity screening during sampling
- If `enforce_sanity=True`:
    - FEN position must be non-terminal (`board.is_game_over()` must be false)
    - first move in `Moves` must be legal from that FEN

7. Experiment input conversion
- `build_experiment_inputs(...)` converts sampled records into dict payloads containing:
    - puzzle_id, fen, phase, difficulty, rating
    - board_ascii
    - solution_uci (first move from `Moves`)
    - legal_moves

## Runtime Logging and Phase Pauses

`prepare_experiment_dataset(...)` supports terminal progress tracking and interactive stepping:

- enable_logs (default true): emits `[phase i/5]` timing/count logs.
- pause_between_phases (default true): prompts `Press Enter to continue...` after phases 1 through 4.
- In non-interactive terminals, pauses are skipped automatically.

## Output Writers

- `write_sampled_csv(...)`: writes sampled records to one CSV.
- `write_experiment_inputs_jsonl(...)`: writes experiment dicts as JSONL.
- `write_phase_difficulty_collections(...)`: writes one CSV per phase/difficulty collection using the naming pattern:
    - `<prefix>_<phase>_<difficulty>.csv`

## Public APIs (Data Prep)

- download_puzzle_csv(...)
- load_puzzles(...)
- apply_quality_filters(...)
- stratified_sample_phase_difficulty(...)
- build_experiment_inputs(...)
- prepare_experiment_dataset(...)
- write_sampled_csv(...)
- write_experiment_inputs_jsonl(...)
- write_phase_difficulty_collections(...)
