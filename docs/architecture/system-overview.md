# System Overview

## Implemented Runtime Components

- State contract layer (`src/state.py`)
- Error taxonomy layer (`src/error_taxonomy.py`)
- Validation layer (`src/validators/*`)
- Tooling layer (`src/tools/chess_tools.py`)
- Data preparation layer (`src/data/puzzle_sampler.py`)
- Engine interface layer (`src/engine/stockfish_wrapper.py`)

## Current Data Flow

```mermaid
flowchart LR
  A[Raw model output] --> B[move_parser.parse_uci_move]
  B -->|valid UCI| C[symbolic.validate_move]
  B -->|parse error| D[ErrorType PARSE_ERROR or NO_OUTPUT]
  C -->|valid| E[Tooling or downstream graph use]
  C -->|invalid| F[Taxonomy error type + reason]

  G[FEN position] --> C
  G --> H[chess_tools.get_legal_moves]
  G --> I[chess_tools.get_board_state]
  G --> J[chess_tools.get_piece_moves]
  G --> K[chess_tools.get_attacked_squares]

  L[Puzzle CSV] --> M[puzzle_sampler.load_puzzles]
  M --> N[puzzle_sampler.stratified_sample]

  O[Stockfish binary] --> P[StockfishWrapper]
  G --> P
  P --> Q[Engine UCI move]
```

## Implementation Boundaries

- Modules currently provide deterministic behavior and typed return payloads.
- LangGraph orchestration and multi-condition execution are not implemented yet.
- Analysis and experiment runners are not implemented yet.
