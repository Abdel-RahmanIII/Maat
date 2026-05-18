# Maat — Research Plan & System Architecture
## 1. Research Objectives
This project investigates whether explicit architectural structure — role separation, structured validation, and rule enforcement — can reduce rule violations and improve multi-turn consistency in LLM-based chess play.
### Research Questions
| ID | Question | Primary Metrics | Experiments |
|----|----------|-----------------|-------------|
| **RQ1** | Can explicit role separation reduce rule violations? | FIR, MFIR, ARR, Phase-Stratified FIR, IMFR, FST | Exp 1 (all conditions), Exp 2 (selected conditions) |
| **RQ2** | Does structured validation improve multi-turn consistency? | SERR, PCRR, TTR, Legality Degradation Curve, ECC, Error-Type over Quartiles | Exp 2 |
| **RQ3** | How should rule enforcement be embedded inside an agentic workflow? | FIR, RSR, MRTC, LCPT, TPT, CAFIR, Critic Accuracy, Error-Type × Condition RSR | Exp 1 (all conditions) |
### Fixed Variables
| Variable | Value |
|----------|-------|
| LLM | Gemma 4 31B via Google AI Studio API |
| Framework | LangGraph (Python) |
| Rule Engine | python-chess |
| Move Format | UCI (e.g. e2e4, e7e8q) |
| Board Format | FEN |
| Opponent (full games) | Stockfish at ELO 800–1000 |
---
## 2. Experimental Conditions
### 2.1 Experiment Design: Two Independent Variables
The study manipulates two independent variables:
1. **Generation Strategy** — how the move is produced:
   - **Generator (G)**: A single LLM generates the UCI move directly.
   - **Planner-Actor (PA)**: A Strategist LLM produces a natural-language plan; a Tactician LLM converts it to a UCI move.
   - **Observer-Executor (OE)**: An Observer LLM produces a comprehensive board description; an Executor LLM selects a move based entirely on that description.
2. **Enforcement Strategy** — how the move is validated / corrected:
   - **None**: No validation; forfeit on first illegal move.
   - **LLM Critic (C)**: A second LLM pass evaluates legality.
   - **Symbolic Validator (D)**: python-chess deterministic check with terse feedback.
   - **Symbolic + Explainer (E)**: python-chess check + LLM-generated pedagogical feedback.
Condition **F** (ReAct + Tool Calling) is a standalone paradigm that does not decompose into the generation × enforcement matrix.
### 2.2 Experiment Arms (13 Total)
mermaid
graph TB
    subgraph "Non-MAS Baseline"
        A["A: Single Generator<br/>(no enforcement, no framework)"]
    end
    subgraph "MAS Generation Baselines (no enforcement)"
        B1["B1: Planner-Actor<br/>(no enforcement)"]
        B2["B2: Observer-Executor<br/>(no enforcement)"]
    end
    subgraph "Enforcement × Generation Matrix (9 arms)"
        CG["C-G: Critic + Generator"]
        CPA["C-PA: Critic + Planner-Actor"]
        COE["C-OE: Critic + Observer-Executor"]
        DG["D-G: Symbolic + Generator"]
        DPA["D-PA: Symbolic + Planner-Actor"]
        DOE["D-OE: Symbolic + Observer-Executor"]
        EG["E-G: Symbolic+Explainer + Generator"]
        EPA["E-PA: Symbolic+Explainer + Planner-Actor"]
        EOE["E-OE: Symbolic+Explainer + Observer-Executor"]
    end
    subgraph "Autonomous Paradigm"
        F["F: ReAct + Tool Calling"]
    end
    A -.->|"adds MAS generation"| B1
    A -.->|"adds MAS generation"| B2
    A -.->|"adds enforcement"| CG
    B1 -.->|"adds enforcement"| CPA
    B2 -.->|"adds enforcement"| COE

| Arm | Generation Strategy | Enforcement Strategy | LLM Calls (min–max) |
|-----|--------------------|--------------------|---------------------|
| **A** | Generator | None | 1 |
| **B1** | Planner-Actor | None | 2 |
| **B2** | Observer-Executor | None | 2 |
| **C-G** | Generator | LLM Critic | 2 – 2+2N |
| **C-PA** | Planner-Actor | LLM Critic | 3 – 3+2N |
| **C-OE** | Observer-Executor | LLM Critic | 3 – 3+2N |
| **D-G** | Generator | Symbolic Validator | 1 – 1+N |
| **D-PA** | Planner-Actor | Symbolic Validator | 2 – 2+N |
| **D-OE** | Observer-Executor | Symbolic Validator | 2 – 2+N |
| **E-G** | Generator | Symbolic + Explainer | 1 – 1+2N |
| **E-PA** | Planner-Actor | Symbolic + Explainer | 2 – 2+2N |
| **E-OE** | Observer-Executor | Symbolic + Explainer | 2 – 2+2N |
| **F** | ReAct (integrated) | Tool-based (autonomous) | 1 – M |
> [!NOTE]
> **Condition A** is the true non-MAS baseline: a single Generator agent with no enforcement and no LangGraph framework. It establishes the raw LLM capability floor.
>
> **Conditions B1 and B2** are MAS generation baselines: they test whether splitting the generation stage (Planner-Actor or Observer-Executor) improves first-try legality *without* any enforcement. They forfeit on first illegal move, just like A.
### 2.3 Condition Details
#### Condition A — Single Generator Baseline
- **Architecture**: Direct API call, no LangGraph, no MAS.
- **Generation**: Single Generator LLM.
- **Enforcement**: None. If illegal, **forfeit**.
- **Purpose**: Establishes the raw LLM capability floor — what does a single LLM do when it bears sole responsibility for generating a legal move?
#### Condition B1 — Planner-Actor Baseline
- **Architecture**: Two-LLM pipeline in LangGraph, no enforcement.
- **Generation**: Strategist LLM → natural-language plan → Tactician LLM → UCI move.
- **Enforcement**: None. If illegal, **forfeit**.
- **Purpose**: Tests whether separating strategic planning from tactical move selection reduces first-try violations, independent of any enforcement mechanism.
#### Condition B2 — Observer-Executor Baseline
- **Architecture**: Observer + Executor pipeline in LangGraph, no enforcement.
- **Generation**: Observer LLM produces a comprehensive natural-language board description (piece positions, material balance, pawn structure, king safety, key square control, tactical features, piece activity). Executor LLM receives that description as the authoritative board representation and selects a UCI move based entirely on it, explicitly discouraged from re-interpreting the raw board independently.
- **Enforcement**: None. If illegal, **forfeit**.
- **Purpose**: Tests whether separating board observation from move execution — forcing the move-generator to rely on a structured description rather than raw FEN — reduces violations, independent of enforcement.
#### Conditions C (C-G, C-PA, C-OE) — LLM Critic Enforcement
- **Architecture**: Generation pipeline + Critic agent.
- **Generation**: Varies by arm (Generator, Planner-Actor, or Observer-Executor).
- **Enforcement**: Critic LLM (same model, different system prompt) evaluates legality → if valid, accept; if invalid, sends detailed natural-language feedback → Generator retries → up to **N** times → **forfeit**.
- **Critic Prompt Design**: The Critic receives the FEN and proposed move. Returns a structured verdict: {valid: bool, reasoning: str, suggestion: str}.
- **Ground-Truth Check**: After Critic approval, python-chess performs a ground-truth check. If the Critic approved an illegal move (false positive), the move forfeits with no retry.
- **Purpose**: Tests whether a second LLM pass (without ground-truth access) can catch errors.
#### Conditions D (D-G, D-PA, D-OE) — Symbolic Validator Enforcement
- **Architecture**: Generation pipeline + python-chess validator node.
- **Generation**: Varies by arm (Generator, Planner-Actor, or Observer-Executor).
- **Enforcement**: python-chess checks legality → if valid, accept; if invalid, return a **terse machine-generated error** → Generator retries → up to **N** times → **forfeit**.
- **Purpose**: Tests ground-truth rule enforcement with minimal feedback.
#### Conditions E (E-G, E-PA, E-OE) — Symbolic + Explainer Enforcement
- **Architecture**: Generation pipeline + python-chess validator + Explainer agent.
- **Generation**: Varies by arm (Generator, Planner-Actor, or Observer-Executor).
- **Enforcement**: python-chess checks → if invalid, Explainer LLM translates error into **rich pedagogical feedback** → Generator retries → up to **N** times → **forfeit**.
- **Purpose**: Tests whether combining ground-truth detection with LLM-generated explanation outperforms either alone.
#### Condition F — ReAct + Tool Calling
- **Architecture**: Multi-agent ReAct loop in LangGraph. The orchestrator agent reasons, selects actions (tool calls or final move submission), observes results, and iterates.
- **Available Tools** (all optional — the agent decides whether to call them):
Always available (all experiments):
| Tool | Signature | Description |
|------|-----------|-------------|
| validate_move | (fen, move_uci) → {legal, reason, rule_ref} | Only rule-enforcement tool |
| is_in_check | (fen) → {in_check, checking_squares} | Check status without board reveal |
| get_game_phase | (move_history) → {opening/middlegame/endgame} | Phase inference from ply count |
| get_move_history_pgn | (move_history) → str | Converts UCI history to PGN text |
Additional tools for Experiment 1 and 2 (fen mode):
| Tool | Signature | Description |
|------|-----------|-------------|
| get_board_visual | (fen) → str | ASCII 8x8 board |
| get_piece_at | (fen, square) → str | Piece lookup (wN, bQ, empty) |
| get_attackers | (fen, square) → list[dict] | Attackers of a target square |
| get_defenders | (fen, square) → list[dict] | Defenders of square occupant |
| is_square_safe | (fen, square, color) → {safe, threats} | Destination safety helper |
| get_position_after_moves | (fen, moves) → str | Forward simulation returning FEN |
- **Flow**: Agent thinks → optionally calls tools → submits final move → ground-truth validation → if illegal, **forfeit**. Maximum **M** reasoning steps to prevent infinite loops.
- **Key Covariate**: Log every tool call (which tool, when, result) for stratified analysis.
- **Purpose**: Tests autonomous rule-seeking behaviour — does the LLM learn to validate before committing?
### 2.4 Retry & Termination Policy
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Max retries (N) for C, D, E arms | **3** | Balances giving correction opportunity vs. fair comparison; common in prior work |
| Max reasoning steps (M) for F | **6** | Prevents runaway loops; allows think → tool → think → tool → think → submit |
| Termination on final invalid | **Forfeit** (all conditions) | Uniform treatment; game counts as loss |
---
## 3. System Architecture
### 3.1 High-Level Architecture
mermaid
graph TB
    subgraph "Experiment Runner"
        ER[Experiment<br/>Orchestrator]
        PM[Puzzle Manager<br/>Exp 1]
        GM[Game Manager<br/>Exp 2]
        SF[Stockfish<br/>Opponent]
    end
    subgraph "Arm Dispatcher"
        CD{Arm<br/>Selector}
        GA["A: Generator<br/>(direct call)"]
        GB1["B1: Planner-Actor"]
        GB2["B2: Observer-Executor"]
        GCx["C-G / C-PA / C-OE"]
        GDx["D-G / D-PA / D-OE"]
        GEx["E-G / E-PA / E-OE"]
        GF["F: ReAct"]
    end
    subgraph "Shared Infrastructure"
        LLM["Gemma 4 31B<br/>(Google AI Studio)"]
        VAL["python-chess<br/>Validator"]
        MC[Metrics<br/>Collector]
        LOG[Structured<br/>Logger]
    end
    ER --> PM
    ER --> GM
    GM --> SF
    ER --> CD
    CD --> GA & GB1 & GB2 & GCx & GDx & GEx & GF
    GA & GB1 & GB2 & GCx & GDx & GEx & GF --> LLM
    GA & GB1 & GB2 & GCx & GDx & GEx & GF --> VAL
    GA & GB1 & GB2 & GCx & GDx & GEx & GF --> MC
    MC --> LOG

### 3.2 Shared LangGraph State Schema
All LangGraph arms (B1, B2, C-\*, D-\*, E-\*, F) share a common state schema. This ensures consistent metric collection and enables apples-to-apples comparison.
python
from typing import TypedDict, Literal
from langgraph.graph import MessagesState
class TurnState(TypedDict):
    # ── Position ──
    board_fen: str                          # Current FEN
    move_history: list[str]                 # All UCI moves so far
    move_number: int                        # Current full-move number
    # ── Current Turn ──
    proposed_move: str                      # LLM's proposed UCI move
    is_valid: bool                          # Ground-truth validity
    retry_count: int                        # Attempts this turn
    max_retries: int                        # N (3 for C/D/E arms, 0 for A/B1/B2)
    feedback_history: list[str]             # Feedback messages this turn
    # ── Messages ──
    messages: list                          # LangGraph message list
    # ── Turn Metrics ──
    first_try_valid: bool                   # Was the very first attempt legal?
    error_types: list[str]                  # Error classifications this turn
    tool_calls: list[dict]                  # Tool call log (Condition F)
    total_attempts: int                     # Total attempts this turn
    llm_calls_this_turn: int               # LLM API calls this turn (for LCPT)
    tokens_this_turn: int                  # Total tokens (in+out) this turn (for TPT)
    prompt_token_count: int                # Input prompt tokens this turn (for RQ2b)
    wall_clock_ms: float                   # Wall-clock time for this turn (for Latency Per Turn)
    game_phase: str                        # Opening / Middlegame / Endgame (for Phase-Stratified FIR)
    # ── Critic-Specific (Condition C) ──
    critic_verdict: bool | None            # Critic's validity judgment (None if N/A)
    ground_truth_verdict: bool | None      # python-chess ground truth
    # ── Game-Level (accumulated) ──
    game_id: str
    condition: str                          # Arm label (e.g. "A", "B1", "C-PA", "D-OE", "F")
    generation_strategy: str               # "generator_only", "planner_actor", or "observer_executor"
    enforcement_strategy: str              # "none", "critic", "symbolic", "symbolic_explainer", "react"
    turn_results: list[dict]               # Accumulated per-turn metric records
    game_status: Literal[
        "ongoing", "checkmate", "stalemate",
        "draw", "forfeit", "max_moves"
    ]

### 3.3 LangGraph Graph Topologies
The generation stage in the diagrams below is shown as GEN for simplicity. In practice, GEN is one of:
- **Generator**: single LLM call (arms A, C-G, D-G, E-G)
- **Planner-Actor**: Strategist → Tactician (arms B1, C-PA, D-PA, E-PA)
- **Observer-Executor**: Observer → Executor (arms B2, C-OE, D-OE, E-OE)
#### Arms A, B1, B2 — No Enforcement
mermaid
graph LR
    S((START)) --> GEN["Generation Stage<br/>(G / PA / OE)"]
    GEN --> VAL{Valid?}
    VAL -->|Yes| ACC[Accept<br/>Move]
    VAL -->|No| FORFEIT[Forfeit]
    ACC --> E((END))
    FORFEIT --> E

#### Arms C-G, C-PA, C-OE — LLM Critic Loop
mermaid
graph LR
    S((START)) --> GEN[Generate<br/>Move]
    GEN --> CRITIC[LLM Critic<br/>Validate]
    CRITIC -->|Valid| GT{Ground-Truth<br/>Check}
    CRITIC -->|"Invalid &<br/>retries < N"| GEN
    CRITIC -->|"Invalid &<br/>retries ≥ N"| FORFEIT[Forfeit]
    GT -->|Valid| ACC[Accept]
    GT -->|Invalid| FORFEIT
    ACC --> E((END))
    FORFEIT --> E

> [!IMPORTANT]
> For Condition C, the Critic is an LLM — it can be wrong. A **ground-truth check** after the Critic approves is essential. If the Critic says "valid" but python-chess disagrees, the move is still recorded as invalid (but the game does NOT get a retry — the Critic already passed it). This captures the Critic's false-positive rate as a secondary metric.
#### Arms D-G, D-PA, D-OE — Symbolic Validator Loop
mermaid
graph LR
    S((START)) --> GEN[Generate<br/>Move]
    GEN --> SYM["Symbolic<br/>Validator<br/>(python-chess)"]
    SYM -->|Valid| ACC[Accept]
    SYM -->|"Invalid &<br/>retries < N"| FB[Terse<br/>Feedback] --> GEN
    SYM -->|"Invalid &<br/>retries ≥ N"| FORFEIT[Forfeit]
    ACC --> E((END))
    FORFEIT --> E

#### Arms E-G, E-PA, E-OE — Symbolic + Explainer Loop
mermaid
graph LR
    S((START)) --> GEN[Generate<br/>Move]
    GEN --> SYM["Symbolic<br/>Validator"]
    SYM -->|Valid| ACC[Accept]
    SYM -->|"Invalid &<br/>retries < N"| EXP[LLM<br/>Explainer] --> GEN
    SYM -->|"Invalid &<br/>retries ≥ N"| FORFEIT[Forfeit]
    ACC --> E((END))
    FORFEIT --> E

#### Condition F — ReAct + Tools
mermaid
graph LR
    S((START)) --> THINK[Agent<br/>Reason]
    THINK -->|"Use Tool"| TOOL[Execute<br/>Tool] --> OBS[Observe<br/>Result] --> THINK
    THINK -->|"Submit Move"| GT{Ground-Truth<br/>Check}
    THINK -->|"Max steps<br/>reached"| FORFEIT[Forfeit]
    GT -->|Valid| ACC[Accept]
    GT -->|Invalid| FORFEIT
    ACC --> E((END))
    FORFEIT --> E

### 3.4 Agent Prompt Architecture
All agents share a **base context block** injected at each turn:

You are playing chess as {color}.
{board_representation}
Move history (UCI): {move_history}

Where {board_representation} is always **Full FEN + ASCII board diagram** (used in both Exp 1 and Exp 2).
Agent-specific system prompts:
| Agent | Key Instructions |
|-------|-----------------|
| **Generator** | "Output exactly one move in UCI format. Respond with ONLY the UCI move, no explanation." |
| **Critic** (C) | "You are a chess rules expert. Given the board position (FEN) and a proposed move (UCI), determine if the move is legal. Respond with JSON: {valid, reasoning, suggestion}." |
| **Explainer** (E) | "A chess move was rejected by the rule engine. Translate the following error into a clear, pedagogical explanation that helps the player understand why the move is illegal and what alternatives exist." |
| **ReAct Agent** (F) | "You are a chess player with access to analysis tools. Generate candidates yourself, optionally validate, and call submit_move(uci) to play. All board analysis tools are available." |
> [!NOTE]
> The Generator prompt deliberately asks for **only** the UCI move to minimize parsing complexity. A regex extractor r'[a-h][1-8][a-h][1-8][qrbn]?' serves as a fallback parser.
### 3.5 Generation Strategy Pipelines
The generation stage is a modular component shared across all enforcement conditions. The validation pipeline downstream remains unchanged regardless of which generation strategy is used.
#### Generator (G) — Single LLM
A single LLM call produces the UCI move. Used in arms A, C-G, D-G, E-G.
#### Planner-Actor (PA)
mermaid
graph LR
    S((START)) --> STRAT[Strategist<br/>LLM] -->|"NL Plan"| TACT[Tactician<br/>LLM] -->|"UCI Move"| VAL["Enforcement<br/>Pipeline<br/>(none / C / D / E)"]
    VAL --> E((END))

- **Strategist**: Receives the board state. Outputs a natural-language strategic plan (e.g., *"Develop the knight to f3 to control the center and prepare kingside castling"*).
- **Tactician**: Receives the plan + board state. Selects the best UCI move implementing the strategy.
- Used in arms B1, C-PA, D-PA, E-PA.
#### Observer-Executor (OE)
mermaid
graph LR
    S((START)) --> OBS[Observer<br/>LLM] -->|"Board Description"| EXE[Executor<br/>LLM] -->|"UCI Move"| VAL["Enforcement<br/>Pipeline<br/>(none / C / D / E)"]
    VAL --> E((END))

- **Observer**: Receives the FEN, ASCII board, move history, and past feedback. Produces a comprehensive natural-language description of the board state — piece positions, control of key squares, pawn structure, material balance, king safety, and tactical features. Does **not** suggest moves, evaluate options, or express intent. Only describes what is true about the position.
- **Executor**: Receives the Observer's natural-language summary alongside the game context. Treats the Observer's summary as the authoritative representation of the board and bases its decision entirely on that description. Explicitly discouraged from re-interpreting the raw board independently. Outputs a single UCI move.
- Used in arms B2, C-OE, D-OE, E-OE.
> [!WARNING]
> MAS generation strategies (PA, OE) double LLM calls per turn before enforcement even begins. Budget API costs carefully. With Gemma 4 31B on AI Studio, check rate limits and quotas before committing to full-matrix runs.
---
## 4. Experiments
### 4.1 Experiment 1 — Isolated Position Evaluation
| Parameter | Value |
|-----------|-------|
| **Objective** | Measure single-move legality across all arms |
| **Answers** | RQ1, RQ3 |
| **Data Source** | Lichess puzzle database |
| **Sample Size** | 300 positions |
| **Stratification** | 100 opening × 100 middlegame × 100 endgame |
| **Difficulty** | Equally distributed across Lichess rating buckets within each phase |
| **Board Input** | Full FEN + ASCII board |
| **Task** | Play any legal move (not necessarily the puzzle solution) |
| **Arms** | All 13 (A, B1, B2, C-G, C-PA, C-OE, D-G, D-PA, D-OE, E-G, E-PA, E-OE, F) |
| **Runs** | 1 pass per position per arm = 3,900 total evaluations |
**Puzzle Sampling Strategy**:
1. Download the [Lichess puzzle CSV](https://database.lichess.org/#puzzles)
2. Classify phase by move number and material count:
   - **Opening**: move ≤ 15
   - **Middlegame**: 15 < move ≤ 35 AND total material > endgame threshold
   - **Endgame**: move > 35 OR total material ≤ endgame threshold (≤ 13 non-pawn material points)
3. Within each phase, stratified random sample across Lichess rating quartiles
4. Each position is presented as: the FEN from the puzzle (after the opponent's last move)
### 4.2 Experiment 2 — Full Games with Board State
| Parameter | Value |
|-----------|-------|
| **Objective** | Measure multi-turn legality and consistency with full observability |
| **Answers** | RQ1, RQ3 (multi-turn), baseline for RQ2 |
| **Opponent** | Stockfish at ELO 800–1000 |
| **Sample Size** | 50 games per arm |
| **Board Input** | Full FEN + ASCII board sent **every turn** |
| **Max Moves** | 150 half-moves (75 full moves), then adjudicate as draw |
| **Arms (preferred)** | All 13 arms |
| **Arms (fallback)** | A + B1 + B2 + best C/D/E arm from Exp 1 + F (5 arms) |
| **Total Games (preferred)** | 650 (13 arms × 50) |
| **Total Games (fallback)** | 250 (5 arms × 50) |
### 4.3 Experiment 2 Execution Policy
> [!IMPORTANT]
> **Full matrix (preferred)**: Run all 13 arms in Exp 2 if time and API budget allow. This enables the complete generation × enforcement interaction analysis.
>
> **Fallback (reduced set)**: If time or resource limits prevent the full matrix, run only:
> - **A** (non-MAS baseline)
> - **B1** (Planner-Actor baseline)
> - **B2** (Observer-Executor baseline)
> - **Best C/D/E arm** from Exp 1 (the single arm with the lowest FIR)
> - **F** (ReAct)
>
> The fallback set preserves the key comparisons (baseline vs. best enforcement vs. ReAct) while reducing total games from 650 to 250.
>
> **Decision point**: The execution policy is decided after Exp 1 results are analyzed. If Exp 1 completes within 40% of the total time budget, proceed with the full matrix.
**Controlled Variables for Exp 2**:
- Same Stockfish level
- Same opening positions (use a set of 50 fixed starting positions, reused across arms)
- Same LLM, same temperature, same prompts
> [!IMPORTANT]
> **Reproducibility**: Set a fixed random seed for Stockfish and use deterministic sampling for puzzles. Save all FENs, prompts, and raw LLM outputs for post-hoc analysis.
---
## 5. Evaluation Framework
### 5.1 Error Taxonomy
Every illegal move is classified into exactly one error type. This taxonomy is referenced by all metrics involving error categorization (RSR heatmaps, recurrence analysis, quartile distributions).
| Error Type | Description | Detection |
|------------|-------------|-----------|
| INVALID_PIECE | No piece on source square, or wrong color | python-chess |
| ILLEGAL_DESTINATION | Piece cannot reach target square | python-chess |
| LEAVES_IN_CHECK | Move leaves own king in check | python-chess |
| CASTLING_VIOLATION | Illegal castling (through check, rights lost) | python-chess |
| EN_PASSANT_VIOLATION | Invalid en passant attempt | python-chess |
| PROMOTION_ERROR | Pawn reaches 8th rank without specifying promotion, or non-pawn promotes | python-chess |
| PARSE_ERROR | Output cannot be parsed as UCI | Regex parser |
| NO_OUTPUT | LLM returned empty or irrelevant text | Parser |
### 5.2 Experiment 1 Metrics — Isolated Positions
Each data point is a single puzzle position evaluated once per arm. N=300 positions × 13 arms = 3,900 evaluations. All metrics are computed from this data.
| Metric | Definition | Formula | Arms |
|--------|------------|---------|------|
| **Final Invalid Rate (FIR)** | Fraction of positions ending in forfeit after all retries | forfeits / total_positions | All |
| **First-Try Invalid Rate (FTIR)** | Fraction of positions where the first attempt was illegal | first_try_invalid / total_positions | All |
| **Marginal FIR Reduction (MFIR)** | Percentage reduction in FIR when adding a role | (FIR_X − FIR_Y) / FIR_X for paired arms X→Y | Paired |
| **Absolute Risk Reduction (ARR)** | Absolute drop in FIR between arms | FIR_X − FIR_Y for paired arms X→Y | Paired |
| **Phase-Stratified FIR** | FIR computed within Opening / Middlegame / Endgame | FIR per phase bucket | All |
| **Parse Failure Counts** | Raw count of PARSE_ERROR + NO_OUTPUT | Count | All |
| **Retry Success Rate (RSR)** | Of initially-invalid moves, fraction corrected within N retries | corrected / initially_invalid | C-\*, D-\*, E-\* |
| **Mean Retries to Correct (MRTC)** | Average retries when correction succeeds | sum(retries_for_corrected) / count(corrected) | C-\*, D-\*, E-\* |
| **LLM Calls Per Turn (LCPT)** | Total API calls per position | Count per position | All |
| **Tokens Per Turn (TPT)** | Total tokens (input + output) per position | Count per position | All |
| **Cost-Adjusted FIR (CAFIR)** | FIR penalized by compute cost | FIR × LCPT | C-\*, D-\*, E-\*, F |
| **Critic Accuracy (TPR, FPR, TNR, FNR)** | Confusion matrix vs. python-chess ground truth | Standard rates | C-\* only |
| **Error-Type × Arm RSR** | RSR per error type per arm (heatmap) | RSR within each error category | C-\*, D-\*, E-\* |
| **Validation Tool Adoption (VTA)** | Fraction of turns using validate_move before submission | validate_turns / total_turns | F only |
| **Tool Call Rate (TCR)** | Fraction of turns with any tool call | tool_turns / total_turns | F only |
| **Tool-Call Distribution** | Frequency breakdown per tool type | Histogram | F only |
| **Tool-Stratified FIR** | FIR split by whether tools were used | FIR per tool-usage group | F only |
| **Avg. Reasoning Steps** | Mean think/act cycles per turn | total_steps / total_turns | F only |
> [!NOTE]
> **MFIR edge case**: If the baseline condition has FIR = 0, MFIR is undefined (division by zero). In this case, ARR must be reported exclusively. Both MFIR and ARR are tested via the existing McNemar pairs.
MFIR is computed for each chained pair to show the marginal value of each architectural change:
| Pair | What It Isolates |
|------|------------------|
| A → B1 | Adding Planner-Actor generation (no enforcement) |
| A → B2 | Adding Observer-Executor generation (no enforcement) |
| A → C-G | Adding Critic enforcement (same generator) |
| A → D-G | Adding Symbolic Validator enforcement (same generator) |
| D-G → E-G | Adding Explainer on top of Symbolic Validator |
| A → F | Adding autonomous tool access |
| C-G → C-PA | Effect of Planner-Actor within Critic enforcement |
| C-G → C-OE | Effect of Observer-Executor within Critic enforcement |
| D-G → D-PA | Effect of Planner-Actor within Symbolic enforcement |
| D-G → D-OE | Effect of Observer-Executor within Symbolic enforcement |
Expected LCPT by arm (see Section 2.2 for full table):
| Arm Group | Min LCPT | Max LCPT | Notes |
|-----------|----------|----------|-------|
| A | 1 | 1 | Single call, no retries |
| B1, B2 | 2 | 2 | Generation pipeline, no retries |
| C-G | 2 | 2 + 2N | Generator + Critic per attempt |
| C-PA, C-OE | 3 | 3 + 2N | Gen pipeline + Critic per attempt |
| D-G | 1 | 1 + N | Generator only; validator is symbolic (free) |
| D-PA, D-OE | 2 | 2 + N | Gen pipeline; validator is symbolic (free) |
| E-G | 1 | 1 + 2N | Generator + Explainer per failed attempt |
| E-PA, E-OE | 2 | 2 + 2N | Gen pipeline + Explainer per failed attempt |
| F | 1 | M | 1 to M reasoning steps, each may call LLM |
> [!IMPORTANT]
> **CAFIR exclusion**: Arms A, B1, and B2 are strictly excluded from CAFIR ranking because they have no enforcement mechanism and cannot trade cost for quality. They are reported separately as fixed baselines.
> [!IMPORTANT]
> **Critic FNR**: The Critic False Negative Rate measures how often the Critic *approves* an illegal move. Since the ground-truth check after Critic approval causes a forfeit (not a retry), a high FNR means the Critic is a liability — it provides false confidence.
### 5.3 Experiment 2 Metrics — Full Games
Each data point is a full game against Stockfish. Under the full-matrix policy: N=50 games × 13 arms = 650 games. Under the fallback policy: N=50 games × 5 arms = 250 games. Experiment 2 provides the full FEN every turn.
| Metric | Definition | Formula | Conditions |
|--------|------------|---------|------------|
| **Final Invalid Rate (FIR)** | Fraction of turns ending in forfeit | forfeits / total_turns | All |
| **First-Try Invalid Rate (FTIR)** | Fraction of turns where first attempt was illegal | first_try_invalid / total_turns | All |
| **Illegal-Move Forfeit Rate (IMFR)** | Fraction of games lost specifically to a rule violation | forfeit_games / total_games | All |
| **Forfeit Survival Time (FST)** | Half-moves played before a rule-violation forfeit | Kaplan-Meier estimand | All |
| **Same-Error Recurrence Rate (SERR)** | Fraction of error types that occur more than once in a game | Per-game rate | All |
| **Post-Correction Recurrence (PCRR)** | Frequency of repeating the same error type after correction | Per-game rate | Retry only ¹ |
| **Turns-to-Recovery (TTR)** | Clean moves following a corrected error | Count per correction event | Retry only ¹ |
| **Legality Degradation Curve** | FTIR plotted in 10-move bins over game progress | Visual + regression | All |
| **Error Clustering Coefficient (ECC)** | Ratio of observed consecutive error pairs to expected pairs | Per-game ratio | All |
| **Error-Type Dist. over Quartiles** | Error taxonomy frequency by turn quartile (Q1–Q4) | Chi-squared table | All |
| **Parse Failure Counts** | Raw count of PARSE_ERROR + NO_OUTPUT | Count | All |
| **LLM Calls Per Turn (LCPT)** | Average API calls per turn | total_llm_calls / total_turns | All |
| **Tokens Per Turn (TPT)** | Average tokens (input + output) per turn | total_tokens / total_turns | All |
> ¹ PCRR and TTR are only computable for arms with a retry mechanism. Arms A, B1, and B2 forfeit on first illegal move — there is no correction event and therefore no recovery to measure. These metrics are reported as N/A for A, B1, and B2.
**ECC operationalization**:

ECC = (observed pairs of errors in consecutive turns) /
      (Σ FTIR(t) × FTIR(t+1)  for t = 1 … total_turns − 1)

A flat Bernoulli baseline (i.e. (total_turns − 1) × FTIR²) is explicitly rejected here because the Legality Degradation Curve shows that error probability increases with move number. Using a single global FTIR would underestimate expected late-game error pairs and artificially inflate ECC in the endgame. The time-varying formulation computes expected pairs turn-by-turn using the empirical FTIR at each position.
> [!IMPORTANT]
> **FTIR(t) definition**: FTIR(t) is the empirical first-try error rate at turn t **pooled across all games in that condition** — not a per-game quantity. Within a single game, each turn is binary (error or not), so a per-game FTIR(t) would be 0 or 1, which is meaningless as a probability baseline. Each game's observed pairs are divided by the population-level expected pairs, yielding a per-game ECC ratio suitable for bootstrapping.
- ECC > 1 → errors cluster (one error makes the next more likely)
- ECC ≈ 1 → errors are independent
- ECC < 1 → errors anti-cluster (correction effect — error makes next turn *less* likely to error)
**Key rationales**:
- **FST**: Replaces the prior Moves Before Forfeit (MBF) metric with a proper survival-analysis estimand. Games reaching natural termination (checkmate/draw) **and** games hitting the 150 half-move limit are right-censored at their final move, not excluded.
- **Error-Type over Quartiles**: The Legality Degradation Curve shows aggregate trends but not *which* error types drive degradation. This metric breaks down the taxonomy by game progress quartile (Q1: turns 1–25%, Q2: 25–50%, Q3: 50–75%, Q4: 75–100%).
> [!NOTE]
> **Scope limitation (fallback only)**: If the fallback execution policy is used, Experiment 2 runs only A, B1, B2, the best C/D/E arm from Exp 1, and F. Metrics requiring retry mechanisms (PCRR, TTR, RSR, CAFIR) or tool-calling (TCR, VTA, Tool-Stratified FIR) are only computed for arms that include those mechanisms. Under the full-matrix policy, all metrics are computed for all applicable arms.
---
## 6. Research Question Analysis
### 6.1 RQ1: Can explicit role separation reduce rule violations?
**Core question**: Does adding distinct agent roles (Critic, Validator, Explainer, ReAct tools) or MAS generation strategies (Planner-Actor, Observer-Executor) to the pipeline reduce the rate of illegal moves?
#### 6.1.1 Metrics
| Metric | Role | Source | What It Reveals |
|--------|------|--------|-----------------|
| FIR | Primary | Exp 1 | Main comparison axis — ultimate failure rate after all retries across all 13 arms |
| MFIR & ARR | Primary | Exp 1 | Marginal contribution of each added role (quantifies the value of upgrading A→B1/B2, A→C-G/D-G, D-G→E-G, A→F, and generation strategy effects within enforcement) |
| Phase-Stratified FIR | Primary | Exp 1 | Whether architecture effectiveness depends on game phase (opening vs. middlegame vs. endgame) |
| FTIR | Primary | Exp 1 | Directly relevant: MAS generation strategies (PA, OE) change the generation stage, making FTIR differ across arms even before enforcement fires |
| IMFR | Primary | Exp 2 | Game-level: fraction of games lost specifically to a rule violation |
| FST | Primary | Exp 2 | Game-level: how long the agent survives before a fatal illegal move |
> [!NOTE]
> **FTIR is now a first-class RQ1 metric**: Unlike the previous 6-condition design where the same generator was shared across B–E, the 13-arm matrix uses three different generation strategies. Arms using Planner-Actor or Observer-Executor may produce different first-try legality rates than the plain Generator, making FTIR a meaningful differentiator across all arms.
#### 6.1.2 Statistical Tests
| Test | Target Metric | Question Answered | Execution Details |
|------|--------------|-------------------|-------------------|
| **Cochran's Q** + **McNemar's** (post-hoc) | FIR (all arms) | Do the architectures perform differently? Which specifically beats another? | N=300 matched pairs across all 13 arms. Parse failures count as forfeits, not missing data. McNemar with Bonferroni correction for confirmatory significance; Benjamini-Hochberg (FDR) flags exploratory signals. ARR is directly derived from these McNemar pairs. |
| **95% Bootstrapped CI** | MFIR & ARR | What is the exact magnitude of upgrading the architecture? | Handles the ratio instability of MFIR. If FIR_baseline = 0, MFIR is undefined and ARR must be reported exclusively. |
| **Logistic Regression** | Phase-Stratified FIR | Does architecture effectiveness depend on game phase? | Primary estimand: interaction term in P(illegal) ~ arm + phase + arm × phase. |
| **Two-Way ANOVA / Logistic Regression** | FIR | Is there a generation × enforcement interaction? | P(illegal) ~ generation_strategy + enforcement_strategy + generation_strategy × enforcement_strategy. Tests whether the best enforcement strategy depends on which generation strategy is used. |
| **Fisher's Exact Test** | IMFR | Do complex architectures reduce games lost to illegal moves? | Two-sided. Run pairwise across 3 conditions in Exp 2/3 (3 pairs per experiment) with Bonferroni correction. Required over Chi-squared because N=50 games makes expected cell counts < 5 highly likely. |
| **Log-Rank Test** | FST (Kaplan-Meier) | Does the architecture significantly extend the agent's lifespan before a fatal error? | Games reaching natural termination (checkmate/draw) and games hitting the 150 half-move cap are strictly right-censored at their final move, not excluded. |
### 6.2 RQ2: Does structured validation improve multi-turn consistency?
**Core question**: Does the agent maintain legality over the course of a full game, and does structured validation slow the rate of degradation?
> [!IMPORTANT]
> **FDR Correction**: RQ2 runs multiple inferential tests on the same Exp 2 dataset. A Benjamini-Hochberg False Discovery Rate correction (q < 0.05) must be applied across all p-values before interpreting significance.
#### 6.2.1 Metrics
| Metric | Role | Comparison Axis |
|--------|------|-----------------|
| SERR | Primary | Cross-condition within Exp 2 |
| PCRR | Primary | Cross-condition within Exp 2 (retry conditions only) |
| TTR | Primary | Cross-condition within Exp 2 (retry conditions only) |
| Legality Degradation Curve | Primary | Cross-condition — does failure rate spike over time? |
| ECC | Secondary | Cross-condition — do errors trigger cascading failure loops? |
| Error-Type Dist. over Quartiles | Secondary | Cross-condition — which error types drive degradation? |
| FTIR (per-game, over time) | Supporting | Cross-condition — time-series of raw error rate |
#### 6.2.2 Statistical Tests
| Test | Target Metric | Question Answered | Execution Details |
|------|--------------|-------------------|-------------------|
| **Wilcoxon Signed-Rank** | SERR & PCRR | Does structured enforcement prevent repeating mistakes better than no enforcement? | Paired by starting position across arms within Exp 2. |
| **Mixed-Effects Logistic Regression** | Legality Degradation | Does the agent's failure rate spike as the game progresses? | Must include random intercept per game (1|game_id) for intra-game correlation. |
| **Bootstrap 95% CI** | ECC | Do errors cause cascading failure loops? | Time-varying baseline FTIR(t) × FTIR(t+1). Aggregation: sum(observed)/sum(expected) per game. CI on game-level ECC values. |
| **Kruskal-Wallis** | SERR & PCRR | Which architecture best prevents hallucination loops? | Across arms within Experiment 2. With k=13 arms (full matrix) or k=5 (fallback), report effect sizes alongside p-values. |
| **Mann-Whitney U** or **Kaplan-Meier** | TTR | Which architecture returns to clean play fastest after a mistake? | Retry conditions only. Default: Mann-Whitney U. If censoring is substantial (>20% of corrections result in clean play until game end), switch to Kaplan-Meier. |
| **Chi-Squared of Homogeneity** | Error-Type over Quartiles | Does the *type* of mistake change as the agent gets fatigued? | Pre-analysis rule: error types with expected cell frequency < 5 must be collapsed into an "Other" category. |
### 6.3 RQ3: How should rule enforcement be embedded?
**Core question**: Comparing enforcement strategies — LLM Critic (C) vs. Symbolic Validator (D) vs. Symbolic+Explainer (E) vs. ReAct+Tools (F) — which provides the best reliability, at what cost, and with what failure modes?
#### 6.3.1 Metrics
**Effectiveness**:
| Metric | Role | Conditions | What It Reveals |
|--------|------|------------|-----------------|
| FIR | Primary | All | Ultimate failure rate — which strategy lets the fewest errors through? |
| Phase-Stratified FIR | Primary | All | Does enforcement effectiveness vary by game phase? |
| RSR | Primary | C, D, E | Of initially-invalid moves, what fraction gets corrected? |
| MRTC | Primary | C, D, E | How many retries does correction take on average? |
**Cost & Efficiency**:
| Metric | Role | Conditions | What It Reveals |
|--------|------|------------|-----------------|
| LCPT | Primary | All | API cost per move — how expensive is each strategy? |
| TPT | Primary | All | Token consumption — compute footprint |
| CAFIR | Secondary | C-\*, D-\*, E-\*, F | ROI metric — is the cost worth the error reduction? A, B1, B2 excluded (no enforcement). |
| Parse Failure Counts | Diagnostic | All | Raw formatting failures (PARSE_ERROR + NO_OUTPUT) — logged across all experiments |
**Critic Accuracy** (C-\* arms only):
| Metric | Definition |
|--------|------------|
| TPR | P(Critic says invalid ∣ move is actually invalid) |
| FPR | P(Critic says invalid ∣ move is actually valid) |
| TNR | P(Critic says valid ∣ move is actually valid) |
| FNR | P(Critic says valid ∣ move is actually invalid) |
Ground-truth classifications determined strictly by python-chess. The puzzle's tactical solution is irrelevant.
**Error-Type Recovery**: RSR per error type per arm (C-\*, D-\*, E-\*), presented as a heatmap. Reveals which enforcement strategy excels at which error types. RSR heatmaps must be presented alongside base-rate frequency — high RSR for rare error types (small cell counts) must be flagged.
**Condition F Agent Behavior**:
| Metric | Definition |
|--------|------------|
| TCR | Fraction of turns where at least one tool was called |
| Tool-Call Distribution | Frequency breakdown per tool type |
| VTA | Fraction of turns where validate_move was called before submission |
| Tool-Stratified FIR | FIR split by whether tools were used |
| Avg. Reasoning Steps | Mean think/act cycles per turn |
#### 6.3.2 Statistical Tests
| Test | Target Metric | Question Answered | Execution Details |
|------|--------------|-------------------|-------------------|
| **Cochran's Q** + **McNemar's** (post-hoc) | FIR (C-\*, D-\*, E-\*, F) | Which enforcement strategy is the ultimate winner for reliability? | Same constraints as RQ1: strict matched pairs, parse errors = forfeits. Bonferroni correction. |
| **Fisher's Exact Test** | RSR | Which strategy fixes the most errors? | Preferred over Chi-squared: 9 enforcement arms (C-\*, D-\*, E-\*) with potentially rare error types make expected cells < 5 likely. |
| **Chi-Squared of Proportions** | Parse Failure Counts | Which strategy formats output best? | Numerator combines PARSE_ERROR + NO_OUTPUT. |
| **Kruskal-Wallis** | MRTC, LCPT, TPT | Which strategy uses least compute and recovers fastest? | Required: API calls, tokens, and retry steps are non-normally distributed counts. |
| **Bootstrap Ranking (95% CI)** | CAFIR | Which strategy offers the best ROI factoring in API costs? | A, B1, B2 strictly excluded from ranking (no enforcement). Reported separately as fixed baselines. |
| **Chi-Squared of Homogeneity** + **Fisher's Exact** | Error-Type × Arm RSR | Do architectures have blind spots for specific error types? | Fisher's Exact when expected cell counts < 5. High RSR for rare error types must be flagged. |
| **Wilson 95% CI** | TPR, FPR, TNR, FNR (Critic) | How trustworthy is the LLM Critic? | Wilson intervals handle small denominators correctly. |
#### 6.3.3 Condition F Stratified Analysis
| Analysis | Target | Purpose | Execution Details |
|----------|--------|---------|-------------------|
| Tool-use tercile split | Tool-Stratified FIR | Does tool use causally reduce errors? | Split by TCR terciles (low/med/high); compare FIR within each. |
| Logistic Regression | FIR ~ tool usage | Controlled test of tool effect | P(illegal) ~ tool_used + position_difficulty + game_phase. Report odds ratios with 95% CI. |
| Descriptive statistics | TCR, VTA, Tool-Call Dist., Avg. Reasoning Steps | Agent behavior profile | Medians with IQR. VTA as predictor of FIR. Tool-Call Distribution as frequency histogram per tool type. |
### 6.4 Complete RQ → Metric → Experiment Mapping
| Metric | RQ1 | RQ2 | RQ3 | Exp 1 | Exp 2 |
|--------|-----|-----|-----|-------|-------|
| FIR | ✅ Primary | | ✅ Primary | ✅ | ✅ |
| FTIR | ✅ ¹ | ✅ Primary | | ✅ | ✅ |
| MFIR | ✅ Primary | | | ✅ | |
| ARR | ✅ Primary | | | ✅ | |
| Phase-Stratified FIR | ✅ Primary | | ✅ | ✅ | |
| IMFR | ✅ Primary | | | | ✅ |
| FST | ✅ Primary | | | | ✅ |
| SERR | | ✅ Primary | | | ✅ |
| PCRR | | ✅ Primary | | | ✅ ² |
| TTR | | ✅ Primary | | | ✅ ² |
| Legality Degradation | | ✅ Primary | | | ✅ |
| ECC | | ✅ Secondary | | | ✅ |
| Error-Type over Quartiles | | ✅ Secondary | | | ✅ |
| FTIR (over time) | | ✅ Supporting | | | ✅ |
| Parse Failure Counts | | | ✅ Diagnostic | ✅ | ✅ |
| RSR | | | ✅ Primary | ✅ | ³ |
| MRTC | | | ✅ Primary | ✅ | ³ |
| LCPT | | | ✅ Primary | ✅ | ✅ |
| TPT | | | ✅ Primary | ✅ | ✅ |
| CAFIR | | | ✅ Secondary | ✅ | ³ |
| Critic Accuracy | | | ✅ Secondary | ✅ | |
| Error-Type × Condition RSR | | | ✅ Secondary | ✅ | |
| VTA (Cond. F) | | | ✅ F-specific | ✅ | ³ |
| TCR (Cond. F) | | | ✅ F-specific | ✅ | ³ |
| Tool-Call Distribution (F) | | | ✅ F-specific | ✅ | ³ |
| Tool-Stratified FIR (F) | | | ✅ F-specific | ✅ | ³ |
| Avg. Reasoning Steps (F) | | | ✅ F-specific | ✅ | ³ |
> ¹ FTIR is now a first-class RQ1 metric because MAS generation strategies (Planner-Actor, Observer-Executor) produce different first-try legality rates than the plain Generator. FTIR differences across arms A, B1, B2, and the generation-strategy variants of C/D/E are directly meaningful.
>
> ² Only computable for arms with retry mechanisms. Arms A, B1, and B2 forfeit on first error.
>
> ³ Under the full-matrix policy, computed for all applicable arms in Exp 2. Under the fallback policy, computed only if the selected best arm includes the relevant mechanism.
### 6.5 Effect Size & Power
- Report **odds ratios** with 95% CI for all rate-based metrics
- Report **Cohen's d** for continuous metrics (MRTC, LCPT, TPT)
- With N=300 per condition (Exp 1), a 10-percentage-point difference in FIR (e.g., 40% vs 30%) is detectable at α=0.05, power=0.80
- With N=50 games (Exp 2), per-game IMFR differences of ~20 percentage points are detectable
- Bonferroni correction for all pairwise comparisons within RQ1 and RQ3
- Benjamini-Hochberg FDR correction (q < 0.05) for the RQ2 family of tests
---
## 7. Project Structure

Maat/
├── src/
│   ├── agents/
│   │   ├── [base.py](http://base.py)               # Base agent class, prompt builder
│   │   ├── [generator.py](http://generator.py)          # Move generator agent
│   │   ├── [critic.py](http://critic.py)             # LLM critic (Condition C)
│   │   ├── [explainer.py](http://explainer.py)          # LLM explainer (Condition E)
│   │   ├── react_[agent.py](http://agent.py)        # ReAct orchestrator (Condition F)
│   │   ├── [strategist.py](http://strategist.py)         # Planner (Planner-Actor)
│   │   ├── [tactician.py](http://tactician.py)          # Actor (Planner-Actor)
│   │   ├── [observer.py](http://observer.py)            # Board describer (Observer-Executor)
│   │   └── [executor.py](http://executor.py)            # Move selector (Observer-Executor)
│   ├── graphs/
│   │   ├── base_[graph.py](http://graph.py)         # Shared graph utilities
│   │   ├── condition_[a.py](http://a.py)        # Direct LLM baseline (no LangGraph)
│   │   ├── condition_[b.py](http://b.py)        # Generator Only
│   │   ├── condition_[c.py](http://c.py)        # + LLM Critic
│   │   ├── condition_[d.py](http://d.py)        # + Symbolic Validator
│   │   ├── condition_[e.py](http://e.py)        # + Symbolic + Explainer
│   │   └── condition_[f.py](http://f.py)        # ReAct + Tools
│   ├── validators/
│   │   ├── [symbolic.py](http://symbolic.py)           # python-chess validation + error classification
│   │   └── move_[parser.py](http://parser.py)        # UCI output parser with fallback regex
│   ├── tools/
│   │   └── chess_[tools.py](http://tools.py)        # Tool implementations for Condition F
│   ├── engine/
│   │   ├── game_[manager.py](http://manager.py)       # Full-game orchestration (Exp 2)
│   │   ├── puzzle_[manager.py](http://manager.py)     # Single-position evaluation (Exp 1)
│   │   └── stockfish_[wrapper.py](http://wrapper.py)  # Stockfish interface
│   ├── metrics/
│   │   ├── [collector.py](http://collector.py)          # Per-turn metric recording
│   │   ├── [aggregator.py](http://aggregator.py)         # Game-level & condition-level aggregation
│   │   ├── [recurrence.py](http://recurrence.py)         # SERR, PCRR computation
│   │   └── [definitions.py](http://definitions.py)        # Metric enums and schemas
│   ├── data/
│   │   ├── puzzle_[sampler.py](http://sampler.py)     # Lichess puzzle download & stratified sampling
│   │   └── opening_[positions.py](http://positions.py)  # Fixed starting positions for Exp 2
│   ├── prompts/
│   │   ├── generator.txt         # Generator system prompt
│   │   ├── critic.txt            # Critic system prompt
│   │   ├── explainer.txt         # Explainer system prompt
│   │   └── react.txt             # ReAct agent system prompt
│   ├── [state.py](http://state.py)                  # TurnState TypedDict definition
│   └── [config.py](http://config.py)                 # Experiment configuration loader
├── experiments/
│   ├── run_experiment_[1.py](http://1.py)       # Exp 1 runner
│   └── run_experiment_[2.py](http://2.py)       # Exp 2 runner
├── analysis/
│   ├── analyze_[exp1.py](http://exp1.py)           # Exp 1 statistical analysis
│   ├── analyze_[exp2.py](http://exp2.py)           # Exp 2 analysis
│   ├── plot_[results.py](http://results.py)           # Visualization
│   └── [tables.py](http://tables.py)                 # LaTeX table generation
├── configs/
│   ├── experiment_1.yaml
│   └── experiment_2.yaml
├── results/                      # Output directory (gitignored)
│   ├── exp1/
│   └── exp2/
├── tests/
│   ├── test_[validator.py](http://validator.py)
│   ├── test_[parser.py](http://parser.py)
│   ├── test_[graphs.py](http://graphs.py)
│   └── test_[metrics.py](http://metrics.py)
├── requirements.txt
├── .env.example                  # API key template
└── [README.md](http://README.md)

---
## 8. Implementation Roadmap
### Phase 1 — Core Infrastructure (Week 1–2)
1. Set up project scaffolding, dependencies, linting
2. Implement TurnState schema and shared utilities
3. Implement [symbolic.py](http://symbolic.py) validator + error classifier
4. Implement move_[parser.py](http://parser.py) (UCI extraction with fallback regex)
5. Implement chess_[tools.py](http://tools.py) (all 5 tools for Condition F)
6. Implement puzzle_[sampler.py](http://sampler.py) (download + stratified sampling from Lichess)
7. Implement stockfish_[wrapper.py](http://wrapper.py)
8. Write unit tests for all of the above
### Phase 2 — Arms & Graphs (Week 2–3)
1. Implement Condition A (direct LLM call, no LangGraph)
2. Implement Generation Strategy modules:
   - Generator (single LLM)
   - Planner-Actor (Strategist → Tactician)
   - Observer-Executor (Observer → Executor)
3. Implement Enforcement Pipelines:
   - Condition C (LLM Critic loop)
   - Condition D (Symbolic Validator loop)
   - Condition E (Symbolic + Explainer loop)
4. Implement Condition F (ReAct + Tools graph)
5. Wire generation × enforcement combinations (9 C/D/E arms + B1 + B2)
6. End-to-end integration test: run each of the 13 arms on 5 positions, verify metric collection
### Phase 3 — Experiment Infrastructure (Week 3–4)
1. Implement metrics/[collector.py](http://collector.py) and metrics/[aggregator.py](http://aggregator.py)
2. Implement puzzle_[manager.py](http://manager.py) (Exp 1 orchestrator)
3. Implement game_[manager.py](http://manager.py) (Exp 2 orchestrator)
4. Implement experiment YAML configs (with arm selection and execution policy support)
5. Implement result serialization (JSON lines per turn, CSV summaries)
6. Dry-run: 10 positions × 13 arms, verify all outputs and metrics
### Phase 4 — Run Experiments (Week 4–6)
1. **Experiment 1**: 300 positions × 13 arms (3,900 evaluations)
2. Analyze Exp 1 results, identify best-performing arm, decide execution policy for Exp 2
3. **Experiment 2** (full matrix or fallback): 50 games × 13 arms (650 games) or 50 games × 5 arms (250 games)
### Phase 5 — Analysis & Writing (Week 6–8)
1. Run statistical tests per analysis plan (including generation × enforcement interaction analysis)
2. Generate figures: FTIR bar charts, degradation curves, recurrence heatmaps, generation × enforcement interaction plots
3. Write results and discussion sections
---
## 9. Risk & Mitigation
| Risk | Impact | Mitigation |
|------|--------|------------|
| API rate limits on Google AI Studio | Slows experiments | Implement exponential backoff + checkpointing; resume from last completed position |
| Gemma 4 can't parse FEN at all | Very high FTIR, uninformative results | Add ASCII board to all FEN prompts; run a pilot with 20 positions first |
| Stockfish games too long | Exceeds token limits | Cap at 150 half-moves; use Stockfish ELO 800 for shorter games |
| Critic agent (C) has very high false-positive rate | C-* arms look worse than A | This IS a valid finding (LLM-only validation is unreliable) — report it |
| Gemma 4 refuses to play chess | Blocking | Pilot test; if needed, adjust system prompt or consider model switch |
| 13-arm matrix exceeds API budget for Exp 2 | Cannot run full matrix | Fall back to reduced set: A + B1 + B2 + best C/D/E arm + F (5 arms). Decision made after Exp 1. |
---
## Verification Plan
### Automated Tests
- Unit tests for symbolic validator (all 8 error types)
- Unit tests for UCI parser (valid moves, invalid strings, edge cases like promotions)
- Integration test: each of the 13 arms runs on 5 fixed FENs, produces expected state transitions
- Metric computation tests: mock data → verify FTIR, SERR, PCRR formulas
### Pilot Run
- Before full experiments: 10 positions × 13 arms pilot (130 evaluations)
- Verify: prompts produce parseable output, metrics collect correctly, API throughput is sustainable
- Estimate total runtime and cost for full experiments
- Use pilot results to confirm or revise the execution policy for Exp 2
### Reproducibility
- All raw LLM outputs saved as JSONL
- Random seeds for puzzle sampling and Stockfish fixed and documented
- Full experiment configs committed to git  