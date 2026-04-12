# State and Error Model

## Turn State Contract

`TurnState` is the shared typed payload used by all condition graphs for turn processing.

Field groups:

- Position: `board_fen`, `move_history`, `move_number`
- Input mode: `input_mode` (`fen` or `history`)
- Current turn: `proposed_move`, `is_valid`, `retry_count`, `max_retries`, `feedback_history`
- Messages: `messages`
- Turn metrics: `first_try_valid`, `error_types`, `tool_calls`, `total_attempts`, `llm_calls_this_turn`, `tokens_this_turn`, `prompt_token_count`
- Critic-specific: `critic_verdict`, `ground_truth_verdict`
- Generation strategy metadata: `generation_strategy`, `strategic_plan`, `routed_phase`
- Game-level: `game_id`, `condition`, `turn_results`, `game_status`

Initialization helper: `create_initial_turn_state(...)`.

### Generation Strategy Fields

| Field | Used By | Description |
|-------|---------|-------------|
| `generation_strategy` | All conditions | One of `generator_only`, `planner_actor`, `router_specialists` |
| `strategic_plan` | Planner-Actor | Natural-language plan from the Strategist agent |
| `routed_phase` | Router-Specialists | Phase classified by the Router (`opening`, `middlegame`, `endgame`) |

## Error Taxonomy

`ErrorType` values:

- `INVALID_PIECE`
- `ILLEGAL_DESTINATION`
- `LEAVES_IN_CHECK`
- `CASTLING_VIOLATION`
- `EN_PASSANT_VIOLATION`
- `PROMOTION_ERROR`
- `PARSE_ERROR`
- `NO_OUTPUT`

Taxonomy usage:

- Parser emits: `PARSE_ERROR`, `NO_OUTPUT`
- Symbolic validator emits move legality categories and may also emit parser-like errors when inputs are malformed.

## Classification Priority in Symbolic Validator

The validator checks categories in this order:

1. missing output
2. invalid FEN
3. invalid UCI / null move
4. invalid piece source / wrong side
5. promotion errors
6. castling violation
7. en passant violation
8. legal move
9. leaves king in check
10. illegal destination (fallback)
