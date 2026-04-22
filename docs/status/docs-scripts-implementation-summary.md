# Maat Implementation Summary (Source-Verified)

Last analyzed: 2026-04-18
Analysis scope: docs/, scripts/, src/, tests/ (implementation-first pass)

## 1. Executive Snapshot

This summary was updated by reading the actual implementation files, not only the docs.

Codebase snapshot (current tree):

- 38 Python source files in src/
- 16 Python test files in tests/
- 9 prompt templates in src/prompts/
- 6 condition graph modules (condition_a.py to condition_f.py)
- 10 agent modules in src/agents/

Runtime/test status validated on this workspace:

- python -m pytest -q -> 182 passed

Current high-level state:

- Core chess legality pipeline is implemented and stable.
- Condition A-F turn-level logic is implemented.
- Planner-Actor and Router-Specialists generation strategies are wired into shared generation dispatch.
- Local prompt engineering and smoke-testing utilities are implemented in scripts/.
- Experiment-scale orchestration and analysis/reporting modules are still absent from src/.

## 2. Implemented System (From Actual Source Files)

### 2.1 State contract and initialization behavior

Actual implementation in src/state.py:

- TurnState is the single shared payload used by all conditions.
- It includes position, input mode, current turn state, metrics, critic metadata, generation-strategy metadata, and game-level fields.
- create_initial_turn_state() guarantees deterministic defaults:
  proposed_move="", is_valid=False, retry_count=0, game_status="ongoing".
- move_history is defensively copied (list(move_history or [])), preventing caller-side mutation leaks.

Notable design detail:

- generation_strategy, strategic_plan, and routed_phase are included directly in the shared state, allowing strategy-specific metadata to be preserved in condition snapshots.

### 2.2 Configuration and environment behavior

Actual implementation in src/config.py:

- .env is loaded at import time from project root via python-dotenv.
- ModelConfig defaults:
  model_name="gemma-4-31b-it", temperature=0.0, max_output_tokens=1024.
- API key is read from GOOGLE_API_KEY into ModelConfig.api_key.
- config_for_condition() enforces retry policy:
  A/B/F -> 0 retries, C/D/E -> 3 retries.
- Condition F uses max_react_steps=6 (retries remain 0).

### 2.3 LLM client layer

Actual implementation in src/llm/llm_client.py:

- _build_model() is the single constructor path for ChatGoogleGenerativeAI.
- Missing GOOGLE_API_KEY raises ValueError with actionable message.
- get_model() returns a plain chat model.
- get_model_with_tools() binds tools through model.bind_tools(tools), used by ReAct.

### 2.4 Parsing and symbolic legality classification

Actual implementation in src/validators/move_parser.py and src/validators/symbolic.py:

Parser behavior:

- Strict path first: chess.Move.from_uci(normalized_input)
- Fallback path: regex extraction [a-h][1-8][a-h][1-8][qrbn]?
- Null move 0000 is explicitly rejected.
- Empty text -> NO_OUTPUT, unparseable text -> PARSE_ERROR.

Validator behavior:

- Uses python-chess for legality ground truth.
- Detects source-square and side-to-move violations as INVALID_PIECE.
- Handles promotion misuse (missing pawn promotion, invalid promotion piece/rank) as PROMOTION_ERROR.
- Detects illegal castling patterns as CASTLING_VIOLATION.
- Detects invalid en passant attempts as EN_PASSANT_VIOLATION.
- If move is pseudo-legal but illegal due to king safety -> LEAVES_IN_CHECK.
- Remaining illegal movement falls to ILLEGAL_DESTINATION.

### 2.5 ReAct tool layer

Actual implementation in src/tools/chess_tools.py:

- All tools are @tool-decorated for LangChain/LangGraph tool calling.
- Tool set includes always-available tools (validate_move, is_in_check, get_game_phase, get_move_history_pgn, submit_move)
  and fen-only tools (get_board_visual, get_piece_at, get_attackers, get_defenders, is_square_safe, get_position_after_moves).
- ALL_TOOLS exports the full fen-mode catalog; get_tools_for_input_mode() applies mode-based gating.
- Input hardening exists for invalid FEN, color, square format, and illegal/invalid move sequences.
- submit_move returns sentinel string SUBMIT:<move>; final legality is validated by condition logic, not by submit_move itself.

### 2.6 Puzzle sampling subsystem

Actual implementation in scripts/puzzle_sampler.py (re-exported by src/data/__init__.py):

- PuzzleRecord stores core and derived fields used for filtering/sampling (including moves, rating_deviation, popularity, nb_plays, themes, phase_source, heuristic_phase, difficulty).
- classify_phase() fallback heuristic uses fullmove/piece-count thresholds (opening <= 12; endgame if piece_count <= 10 or fullmove >= 40).
- assign_difficulty() uses current boundaries: easy < 1300, medium < 1700, hard >= 1700.
- load_puzzles() accepts key variants, parses FEN features, applies theme-first phase assignment, and retains heuristic phase metadata.
- apply_quality_filters() keeps puzzles with RatingDeviation < 75, Popularity > 50, NbPlays >= 100.
- stratified_sample_phase_difficulty() samples themed rows first, then tops up from heuristic rows in the same phase+difficulty cell.
- prepare_experiment_dataset() supports runtime logs and optional pause-between-phases prompts in interactive terminals.
- Output writers include sampled CSV, JSONL experiment inputs, and per phase/difficulty collection CSV files.

### 2.7 Stockfish engine wrapper

Actual implementation in src/engine/stockfish_wrapper.py:

- Engine path resolution priority:
  constructor engine_path -> STOCKFISH_PATH -> PATH lookup -> FileNotFoundError.
- Context-manager support is implemented.
- set_elo() configures UCI_LimitStrength and UCI_Elo only if options exist in the engine build.
- choose_move() validates FEN, auto-starts engine, and returns UCI move string.
- Missing move from engine raises RuntimeError.

### 2.8 Agent layer details

Actual implementation in src/agents/:

- base.py:
  prompt loader, board representation builder for fen/history modes, feedback block formatter, side-to-move resolver.
- generator.py:
  generic UCI move generation with pluggable prompt template and token accounting.
- critic.py:
  JSON-structured legality critic with best-effort JSON extraction and conservative fallback valid=False.
- explainer.py:
  symbolic-error-to-pedagogical-feedback transformer.
- strategist.py:
  natural-language planning (explicitly not UCI output).
- tactician.py:
  consumes strategic_plan and emits concrete UCI move via inline tactician template.
- router.py:
  phase classifier with parser fallback to middlegame for unparseable outputs.
- specialists.py:
  phase-to-prompt dispatch; unknown phase falls back to generator prompt.
- react_agent.py:
  iterative think-act loop with bound tools, direct tool execution, per-step logging, token accumulation, and fallback SUBMIT extraction from plain text.

### 2.9 Graph and condition execution details

Actual implementation in src/graph/:

Shared utilities (base_graph.py):

- run_generation() dispatches strategy-specific generation:
  generator_only, planner_actor, router_specialists.
- planner_actor and router_specialists each add one extra LLM call per attempt via extra_llm_calls=1.
- parse_and_validate() runs parser then symbolic validator and returns normalized fields for graph nodes.
- snapshot_turn_result() serializes per-turn metrics and strategy metadata into turn_results.
- snapshot_turn_result() includes wall_clock_ms, game_phase, and board_fen for downstream metric analysis.

Condition A (condition_a.py):

- Direct function execution (no StateGraph).
- One generation attempt only.
- Invalid result immediately sets game_status="forfeit".

Condition B (condition_b.py):

- LangGraph StateGraph nodes: generate -> accept/forfeit.
- No retry path.
- first_try_valid and metrics are updated in generate node.

Condition C (condition_c.py):

- Nodes: generate -> critic -> ground_truth/ retry_generate / forfeit.
- Parse failures skip critic and go straight to retry/forfeit route.
- Critic rejection can trigger retry loop until max_retries.
- Critic approval always goes through symbolic ground truth.
- If ground truth rejects after critic approval, condition forfeits (no extra retry branch after ground_truth).

Condition D (condition_d.py):

- Nodes: generate -> accept / terse_feedback / forfeit.
- terse_feedback node appends machine-style feedback string and increments retry_count.

Condition E (condition_e.py):

- Nodes: generate -> accept / explainer / forfeit.
- explainer node invokes LLM feedback, appends explanation text to feedback_history, increments retry_count.

Condition F (condition_f.py):

- Runs run_react_loop() outside StateGraph (single orchestrating function).
- If no submitted move within max_steps, forfeit with NO_OUTPUT.
- If submitted move exists, parse_and_validate() performs final ground-truth legality check.
- tool_calls from ReAct are persisted into TurnState.tool_calls and snapshot_turn_result.

### 2.10 Metrics package details

Actual implementation in src/metrics/:

- definitions.py:
  Pydantic v2 models for TurnRecord, GameRecord, ConditionMetrics, and helper containers
  (PhaseStratifiedFIR, CriticAccuracy, ErrorTypeRSR, ToolCallDistribution,
  DescriptiveStats, LegalityDegradationBin, QuartileErrorDist, FSTEntry, FIRDeltaEntry).
- collector.py:
  MetricsCollector lifecycle (start_turn, end_turn, finalize_game) and infer_game_phase()
  with opening/middlegame/endgame rules using move number and non-pawn material.
- aggregator.py:
  pure aggregate functions for Exp 1 and Exp 2/3 including FIR, FTIR, MFIR, ARR,
  RSR, MRTC, LCPT, TPT, CAFIR, critic accuracy, tool metrics, IMFR, FST,
  and condition-aware dispatch via compute_all_exp1_metrics/compute_all_game_metrics.
- recurrence.py:
  multi-turn consistency metrics (SERR, PCRR, TTR, ECC), legality degradation,
  input-length/error extraction, and quartile error distributions.
- __init__.py:
  explicit public exports for collector and model classes.

### 2.10 Prompt implementation details

Actual templates in src/prompts/:

- generator and specialists enforce strict UCI-only output.
- strategist explicitly forbids UCI output and asks for 2-3 sentence plan.
- router requests exactly one token among opening/middlegame/endgame.
- critic requires JSON-only verdict payload.
- explainer requires concise 3-5 sentence pedagogical explanation.
- react prompt strongly instructs explicit tool calling and mandatory submit_move usage.

## 3. Testing and Quality Status (Source-Verified)

### 3.1 Current test outcome

Live run in this workspace:

- Command: python -m pytest -q
- Result: 182 passed

### 3.2 What is concretely tested

State tests (tests/state/test_state.py):

- TurnState annotation completeness against expected field set
- create_initial_turn_state defaults
- move_history copy safety

Parser tests (tests/validators/test_move_parser.py):

- strict UCI parsing
- regex fallback extraction
- case normalization
- parse/no-output handling
- null move rejection

Symbolic validator tests (tests/validators/test_symbolic_validator.py):

- all major taxonomy branches:
  INVALID_PIECE, ILLEGAL_DESTINATION, LEAVES_IN_CHECK, CASTLING_VIOLATION,
  EN_PASSANT_VIOLATION, PROMOTION_ERROR, PARSE_ERROR, NO_OUTPUT

Tool tests (tests/tools/test_chess_tools.py):

- JSON tool response contracts
- legal move list size at start position
- square-based move extraction
- attacked squares output behavior

Data tests (tests/data/test_puzzle_sampler.py):

- phase classifier behavior
- invalid-FEN row skipping
- phase-stratified count guarantees

Engine tests (tests/engine/test_stockfish_wrapper.py):

- invalid FEN failure path
- path resolution priority checks
- legal move return when engine is available

Graph tests (tests/graph/test_conditions.py):

- parse_and_validate behavior
- condition A acceptance/forfeit branches
- condition B no-retry behavior
- condition D retry success and retry exhaustion behavior
- config_for_condition retry map checks

Agent base tests (tests/agents/test_base.py):

- prompt loading success/failure
- board representation mode behavior
- feedback block formatting
- side-to-move utility behavior

Metrics tests (tests/metrics/):

- Pydantic model validation and JSON round-trip behavior (definitions)
- Collector lifecycle, wall-clock timing, and game-phase inference (collector)
- Aggregate metric formulas and edge cases across Exp 1 and Exp 2/3 (aggregator)
- Recurrence/clustering metrics and quartile/bin outputs (recurrence)

### 3.3 Current test coverage boundaries

From actual test files:

- There are no dedicated unit tests for condition C, E, or F graph-level transitions in tests/graph/.
- ReAct internals are exercised by smoke_test.py (script-level), not by deterministic unit tests.
- Critic/explainer/router/strategist/tactician role modules are not deeply unit-tested in isolation.

Interpretation:

- Core deterministic logic is well covered.
- LLM-dependent and tool-loop heavy paths are currently validated mostly through manual/smoke workflows.

## 4. Scripts Folder: What Is Implemented (Code-Level)

### 4.1 scripts/prompt_lab.py (backend)

This is a lightweight HTTP backend built on BaseHTTPRequestHandler.
Implemented API surface:

- GET /
  serves scripts/prompt_lab_ui.html
- GET /api/roles
  exposes role metadata and parameter schemas for dynamic UI rendering
- GET /api/presets
  exposes curated chess position presets
- GET /api/prompt/<role_id>
  returns the role template body
- POST /api/preview
  renders role prompt without model call
- POST /api/invoke
  invokes either standard role LLM call or full ReAct loop
- POST /api/save-prompt
  persists edited template text to src/prompts/<file>

Notable backend implementation details:

- ReAct path returns submitted_move, tool call trace, step count, token totals, and forfeited flag.
- Output validation is immediate: parse_uci_move + validate_move are run on returned move text.
- Tactician in prompt lab uses an inline template in this script (not loaded from src/prompts).

### 4.2 scripts/prompt_lab_ui.html (frontend)

Implemented UX behavior:

- Dynamic role loading and grouping from backend metadata
- Dynamic parameter form generation by role field schema
- Position preset application to fen/move_history fields
- Tabbed workflow:
  Template editor -> Rendered messages -> LLM Output/validation
- Template dirty-state indicator and save/reset controls
- Validation badges and metrics cards (tokens, latency, steps)
- Tool call transcript rendering for ReAct invocations
- Keyboard shortcuts:
  Ctrl/Cmd+Enter invoke, Ctrl/Cmd+Shift+Enter preview

### 4.3 scripts/smoke_test.py (operational test harness)

The script implements three explicit layers:

- Layer 1 (offline deterministic checks)
- Layer 2 (live role calls)
- Layer 3 (live condition calls, including bonus generation strategy checks)

Actual layer coverage highlights:

- Offline layer includes parser/validator/tools/state/config/base helpers and parse_and_validate integration.
- Role layer calls generator, strategist, tactician, critic, explainer, router, specialists, and react loop.
- Condition layer executes A-F and includes planner_actor/router_specialists checks on condition B.

Operational details:

- CLI supports --layer offline|roles|conditions|all and --offline shortcut.
- Colored pass/fail output and total timing are built in.
- Script exits non-zero if any checks fail.

## 5. What Is Explicitly Not Implemented Yet (From Actual Tree)

The following implementation areas are still missing in src/ and root structure:

- No experiment orchestration package (e.g., puzzle_manager/game_manager runtime modules).
- No analysis package or result-report generation scripts in repository root.
- No experiment YAML config directory or run-config files.
- No results/ output structure for persisted experiment runs in tracked tree.

What this means in practice:

- Turn-level logic and condition execution are implemented.
- End-to-end research campaign automation is partially implemented (metrics are complete; orchestration/analysis remain pending).

## 6. Consistency Notes Found During Source Analysis

### 6.1 Model naming drift in docs vs code

- Current code default is gemma-4-31b-it (src/config.py).
- Some planning/reference docs still mention Gemma 4 31B.

### 6.2 Packaging metadata path drift

- pyproject.toml uses readme = "implementation_plan.md".
- There is no implementation_plan.md at repository root.
- The plan file exists under docs/reference/implementation_plan.md.

### 6.3 Tactician template duplication risk

- Tactician prompt text is hardcoded in src/agents/tactician.py.
- Prompt Lab also keeps a separate inline tactician template in scripts/prompt_lab.py.
- This creates drift risk if one template is edited without the other.

### 6.4 Skip behavior is environment-dependent

- docs mention potential Stockfish skip condition.
- In this environment, full suite ran with 182 passed.

## 7. Practical Readiness Assessment

### 7.1 What is production-ready for internal experiments

- Deterministic parse + symbolic legality classification pipeline
- Condition-level execution for A-F
- Strategy swapping for B-E generation stage
- Prompt iteration tooling with live feedback and legality checks
- Smoke-testing workflow that exercises both offline and live paths

### 7.2 What is not yet ready for thesis-scale batch experimentation

- Automated large-run orchestrators
- Persistent metric aggregation and statistical computation modules
- Structured experiment configuration and run metadata management
- Analysis/report artifact generation pipeline

### 7.3 Maturity summary

- Core runtime and condition logic: implemented and test-backed
- Prompt/tool experimentation stack: implemented and usable
- Large-scale experiment operations and analytics: pending
