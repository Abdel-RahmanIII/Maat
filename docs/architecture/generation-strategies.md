# Generation Strategies

Maat supports three generation strategies, each representing a different
decomposition of the move-generation task.  All strategies are implemented
as compiled LangGraph `StateGraph` subgraphs and are interchangeable via
the `build_generation_subgraph()` factory.

## 1. Generator Only (G)

**Single LLM baseline.**

- **Calls per attempt**: 1
- **Subgraph**: `generator_only.py`
- **Flow**: `generate → parse_validate → END`
- **Agent**: `generator.py` using `generator.yaml`

The simplest strategy — one LLM call produces a UCI move string directly.
Used as the baseline for all conditions.

## 2. Planner-Actor (PA)

**Strategy → Tactics decomposition.**

- **Calls per first attempt**: 2
- **Calls per retry**: 1 (Strategist cached)
- **Subgraph**: `planner_actor.py`
- **Flow**: `strategist → tactician → parse_validate → END`
- **Agents**: `strategist.py`, `tactician.py`

Decomposes generation into strategic planning (what should I do?) and
tactical execution (which UCI move implements the plan?).  The Strategist
produces a natural-language plan; the Tactician converts it into a move.

On retry, the Strategist is **skipped** (plan cached in `TurnState.strategic_plan`)
and only the Tactician re-runs with the original plan plus enforcement feedback.

## 3. Threat-Analyst (TA)

**Analysis → Execution decomposition.**

- **Calls per first attempt**: 2
- **Calls per retry**: 1 (Analyst cached)
- **Subgraph**: `threat_analyst.py`
- **Flow**: `threat_analyst → constrained_generator → parse_validate → END`
- **Agents**: `threat_analyst.py`, `constrained_generator.py`

Decomposes generation into constraint identification (what can't I do?)
and constrained move selection (what's best given constraints?).  The
Threat Analyst produces a structured report covering:

- **King Safety** — check detection and escape squares
- **Pinned Pieces** — pins to king with movement restrictions
- **Hanging Pieces** — undefended pieces on both sides
- **Castling** — available rights and conditions
- **En Passant** — target squares and eligible pawns
- **Movement Constraints** — pieces that cannot move this turn

The Constrained Generator receives this report and must respect every
constraint when selecting a move.  This directly targets the most common
error types: `LEAVES_IN_CHECK`, `INVALID_PIECE`, `ILLEGAL_DESTINATION`.

On retry, the Threat Analyst is **skipped** (report cached in
`TurnState.threat_report`) and only the Generator re-runs with the
original report plus enforcement feedback.

## Why Router-Specialist Was Replaced

The former Router-Specialist (RS) strategy classified the game phase
(opening/middlegame/endgame) and routed to a phase-themed specialist.
It was replaced because:

1. **No legality mechanism** — phase classification provides zero
   information about move legality.
2. **Router wastes an LLM call** — phase can be determined
   deterministically via move count + material heuristics.
3. **Specialists ≈ Generator** — each specialist is just a Generator
   with a phase-themed system prompt.
4. **Low-insight data risk** — likely to produce a null result, wasting
   LLM calls to confirm that phase-routing doesn't prevent illegal moves.

The TA strategy provides a **plausible legality mechanism** by explicitly
surfacing constraints that the Generator must respect, enabling clean
error attribution and genuine information enrichment.

## Orthogonality of PA and TA

PA and TA decompose generation along genuinely different axes:

| Axis | PA | TA |
|------|----|----|
| Question answered by Agent 1 | "What should I do?" | "What can't I do?" |
| Question answered by Agent 2 | "How do I do it?" | "What's best given constraints?" |
| Decomposition type | Strategy → Tactics | Analysis → Execution |
| Target error types | Strategic blunders | Legality violations |
