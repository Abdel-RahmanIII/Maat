# System Overview

## Implemented Runtime Components

- State contract layer (`src/state.py`)
- Error taxonomy layer (`src/error_taxonomy.py`)
- Configuration layer (`src/config.py`, `src/engine/config_loader.py`)
- LLM client layer (`src/llm/llm_client.py`)
- Agent layer (`src/agents/*`)
- Prompt template layer (`src/prompts/*`)
- Graph layer (`src/graph/*`)
- Validation layer (`src/validators/*`)
- Tooling layer (`src/tools/chess_tools.py`)
- Data preparation layer (`scripts/puzzle_sampler.py`, re-exported by `src/data/__init__.py`)
- Engine interface layer (`src/engine/stockfish_wrapper.py`)
- Orchestration layer (`src/engine/puzzle_manager.py`, `src/engine/game_manager.py`, `src/engine/condition_dispatch.py`, `src/engine/result_store.py`)
- Metrics layer (`src/metrics/*`)

## High-Level Architecture

```mermaid
flowchart TB
  subgraph "Orchestration Layer"
    PM["PuzzleManager\n(Exp 1)"]
    GM["GameManager\n(Exp 2 & 3)"]
    CONF["config_loader.py"]
  end

  subgraph "Condition Dispatch"
    CD{Condition Selector}
    CA["Condition A\n(direct call)"]
    CB["Condition B\n(LangGraph)"]
    CC["Condition C\n(+ LLM Critic)"]
    CD2["Condition D\n(+ Symbolic Validator)"]
    CE["Condition E\n(+ Symbolic + Explainer)"]
    CF["Condition F\n(ReAct + Tools)"]
  end

  subgraph "Generation Strategies"
    GS{Strategy}
    GO["Generator Only"]
    PA["Planner → Actor"]
    RS["Router → Specialists"]
  end

  subgraph "Shared Infrastructure"
    LLM["Gemma 4 31B\n(Google AI Studio)"]
    VAL["python-chess\nValidator"]
    PARSE["UCI Move Parser"]
    TOOLS["Chess Analysis\nTools"]
    METRICS["Metrics Collector +\nAggregators"]
    STOCKFISH["StockfishWrapper"]
  end

  CONF --> PM & GM
  PM & GM --> CD
  GM <--> STOCKFISH
  CD --> CA & CB & CC & CD2 & CE & CF
  CA & CB & CC & CD2 & CE --> GS
  GS --> GO & PA & RS
  GO & PA & RS --> LLM
  CF --> LLM
  CF --> TOOLS
  CA & CB & CC & CD2 & CE & CF --> VAL
  CA & CB & CC & CD2 & CE & CF --> PARSE
  CA & CB & CC & CD2 & CE & CF --> METRICS
```

## Data Flow

```mermaid
flowchart LR
  A[Raw model output] --> B[move_parser.parse_uci_move]
  B -->|valid UCI| C[symbolic.validate_move]
  B -->|parse error| D[ErrorType PARSE_ERROR or NO_OUTPUT]
  C -->|valid| E[Accept move]
  C -->|invalid| F[Taxonomy error type + reason]
  F -->|retries remain| G[Feedback to Generator]
  F -->|retries exhausted| H[Forfeit]

  I[FEN position] --> C
  I --> J[chess_tools.*]

  K[Puzzle CSV] --> L[puzzle_sampler.load_puzzles]
  L --> M[puzzle_sampler.stratified_sample]

  N[Stockfish binary] --> O[StockfishWrapper]
  I --> O
  O --> P[Engine UCI move]
```

## Implementation Boundaries

- Phases 1, 2, 3, and 4 are complete: core infrastructure, all 6 condition graphs, metrics package, and orchestration (managers and configs).
- Analysis/reporting pipelines are not implemented yet.
