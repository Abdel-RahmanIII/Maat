# Metrics Module Reference

## Directory

- `src/metrics/`

## Purpose

Provides typed metric records, real-time collection around condition execution, and pure aggregation functions for Experiments 1, 2, and 3.

Statistical significance testing is intentionally out of scope for this package and belongs in analysis scripts.

## Files

| File | Description |
|------|-------------|
| `definitions.py` | Pydantic models for per-turn/per-game records and aggregate metric outputs |
| `collector.py` | `MetricsCollector` lifecycle and game-phase inference |
| `aggregator.py` | Pure aggregate functions and condition-aware metric dispatch |
| `recurrence.py` | Multi-turn consistency and clustering metrics for Exp 2/3 |
| `__init__.py` | Public exports for collector and model classes |

## Core Data Models (`definitions.py`)

- `TurnRecord`: structured per-turn snapshot (validity, retries, error types, token/call counts, tool calls, timing, phase, and board FEN).
- `GameRecord`: per-game container with metadata, ordered turns, final status, and totals.
- `ConditionMetrics`: top-level aggregate result container with optional fields depending on experiment and condition.
- Helper containers include:
  `PhaseStratifiedFIR`, `CriticAccuracy`, `ErrorTypeRSR`, `ToolCallDistribution`, `DescriptiveStats`, `LegalityDegradationBin`, `QuartileErrorDist`, `FSTEntry`, `FIRDeltaEntry`.

## Collector Lifecycle (`collector.py`)

### API

```python
collector = MetricsCollector(game_id, condition, experiment, input_mode="fen")
collector.start_turn()
turn_record = collector.end_turn(state)
game_record = collector.finalize_game(final_status="checkmate")
```

### Behavior

- `start_turn()` records wall-clock start time.
- `end_turn(state)` computes `wall_clock_ms`, infers `game_phase`, and validates a `TurnRecord` from final state/snapshot data.
- `finalize_game(...)` computes totals and returns a validated `GameRecord`.

### Phase Inference

`infer_game_phase(fen, move_number)` rules:

- opening: `move_number <= 15`
- endgame: `move_number > 35` or non-pawn material `<= 13`
- middlegame: otherwise

## Aggregation (`aggregator.py`)

All functions are pure and side-effect free.

### Core rates

- `compute_fir`, `compute_ftir`, `compute_mfir`, `compute_arr`
- `compute_phase_stratified_fir`
- `compute_parse_failure_counts`

### Retry/cost metrics

- `compute_rsr`, `compute_mrtc`
- `compute_lcpt`, `compute_tpt`, `compute_cafir`

### Condition-specific metrics

- Condition C: `compute_critic_accuracy`
- Conditions C/D/E: `compute_error_type_rsr`
- Condition F: `compute_vta`, `compute_tcr`, `compute_tool_call_distribution`, `compute_tool_stratified_fir`, `compute_avg_reasoning_steps`

### Game-level metrics (Exp 2/3)

- `compute_game_fir`, `compute_imfr`, `compute_fst_data`, `compute_fir_cross_experiment_delta`

### Dispatcher entry points

- `compute_all_exp1_metrics(turn_records, condition)`
- `compute_all_game_metrics(game_records, condition)`

Both dispatchers are condition-aware and only populate metrics applicable to the given condition.

## Recurrence Metrics (`recurrence.py`)

Sequence-aware functions for full-game analysis:

- `compute_serr`: same-error recurrence rate
- `compute_pcrr`: post-correction recurrence rate
- `compute_ttr`: turns-to-recovery values
- `compute_population_ftir_by_turn`: pooled per-turn FTIR baseline
- `compute_ecc`: error clustering coefficient using time-varying baseline
- `compute_legality_degradation`: FTIR by move bins
- `compute_input_length_vs_error`: per-turn correlation dataset
- `compute_error_type_over_quartiles`: error distribution across Q1-Q4

## Package Exports

The package exports the collector and key models through `src/metrics/__init__.py` to keep imports stable for orchestrators and analysis code.