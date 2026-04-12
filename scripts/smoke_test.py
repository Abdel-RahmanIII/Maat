#!/usr/bin/env python3
"""Maat Smoke-Test Suite — test the setup and every role.

Usage:
    # Run everything (offline + live LLM):
    python scripts/smoke_test.py

    # Offline only (no API calls):
    python scripts/smoke_test.py --offline

    # Test a specific layer:
    python scripts/smoke_test.py --layer offline
    python scripts/smoke_test.py --layer roles
    python scripts/smoke_test.py --layer conditions

Layers
------
1. **offline**     — parser, symbolic validator, chess tools, state helpers
                     (no API key needed, fast, deterministic)
2. **roles**       — each individual agent/role against the live LLM
                     (generator, strategist, tactician, critic, explainer,
                      router, specialist, react)
3. **conditions**  — full condition A–F graphs end-to-end
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import textwrap
import time
import traceback
from pathlib import Path

# ── Force UTF-8 on Windows console ──────────────────────────────────────
if sys.platform == "win32":
    os.system("")  # enable ANSI escape processing
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ── Make sure `src` is importable ────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import chess  # noqa: E402

# ── Pretty printing helpers ──────────────────────────────────────────────

_GREEN = "\033[92m"
_RED = "\033[91m"
_YELLOW = "\033[93m"
_CYAN = "\033[96m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


def _header(title: str) -> None:
    print(f"\n{_BOLD}{_CYAN}{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}{_RESET}\n")


def _section(title: str) -> None:
    print(f"\n{_BOLD}{_YELLOW}-- {title} --{_RESET}")


def _pass(label: str, detail: str = "") -> None:
    suffix = f"  ->  {detail}" if detail else ""
    print(f"  {_GREEN}[PASS]{_RESET}  {label}{suffix}")


def _fail(label: str, detail: str = "") -> None:
    suffix = f"  ->  {detail}" if detail else ""
    print(f"  {_RED}[FAIL]{_RESET}  {label}{suffix}")


def _info(msg: str) -> None:
    print(f"  {_CYAN}[INFO]{_RESET}  {msg}")


# ── Test positions ───────────────────────────────────────────────────────

STARTING_FEN = chess.STARTING_FEN
# Italian Game after 1.e4 e5 2.Nf3 Nc6 3.Bc4
MID_FEN = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"
MID_HISTORY = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4"]
# Endgame: K+R vs K
END_FEN = "8/8/8/4k3/8/8/8/4K2R w - - 0 1"

# Promotion position
PROMO_FEN = "8/P7/8/8/8/8/8/4K2k w - - 0 1"


# ═══════════════════════════════════════════════════════════════════════════
# LAYER 1 — OFFLINE TESTS
# ═══════════════════════════════════════════════════════════════════════════


def test_offline() -> tuple[int, int]:
    """Run all offline (no-API) tests. Returns (passed, failed)."""

    _header("LAYER 1 — Offline Tests (no API calls)")
    passed = 0
    failed = 0

    # ── 1a. Move parser ──────────────────────────────────────────────────

    _section("Move Parser")

    from src.validators.move_parser import parse_uci_move

    tests_parser = [
        ("Clean UCI", "e2e4", True, "e2e4"),
        ("With whitespace", "  e2e4 \n", True, "e2e4"),
        ("Promotion", "a7a8q", True, "a7a8q"),
        ("Embedded in text", "I suggest playing e2e4 because it opens the center", True, "e2e4"),
        ("Garbage text", "hello world", False, None),
        ("Empty string", "", False, None),
        ("None input", None, False, None),
    ]

    for label, raw, expect_valid, expect_move in tests_parser:
        try:
            result = parse_uci_move(raw)
            if result["is_valid"] == expect_valid:
                if expect_move is None or result["move_uci"] == expect_move:
                    _pass(label, f"parsed={result['move_uci']}")
                    passed += 1
                else:
                    _fail(label, f"expected move={expect_move}, got={result['move_uci']}")
                    failed += 1
            else:
                _fail(label, f"expected valid={expect_valid}, got={result['is_valid']}")
                failed += 1
        except Exception as e:
            _fail(label, str(e))
            failed += 1

    # ── 1b. Symbolic validator ───────────────────────────────────────────

    _section("Symbolic Validator")

    from src.validators.symbolic import validate_move

    tests_validator = [
        ("Legal e2e4 from start", STARTING_FEN, "e2e4", True, None),
        ("Illegal e2e5 from start", STARTING_FEN, "e2e5", False, "ILLEGAL_DESTINATION"),
        ("No piece on e5", STARTING_FEN, "e5e6", False, "INVALID_PIECE"),
        ("Opponent piece", STARTING_FEN, "e7e6", False, "INVALID_PIECE"),
        ("Legal Nf3", STARTING_FEN, "g1f3", True, None),
        ("Promotion a7a8q", PROMO_FEN, "a7a8q", True, None),
        ("Missing promotion", PROMO_FEN, "a7a8", False, "PROMOTION_ERROR"),
        ("Null move", STARTING_FEN, "0000", False, "PARSE_ERROR"),
        ("Empty move", STARTING_FEN, "", False, "NO_OUTPUT"),
    ]

    for label, fen, move, expect_valid, expect_error in tests_validator:
        try:
            result = validate_move(fen, move)
            ok = result["valid"] == expect_valid
            if expect_error is not None:
                ok = ok and result["error_type"] == expect_error
            if ok:
                _pass(label, f"error_type={result.get('error_type')}")
                passed += 1
            else:
                _fail(label, f"expected valid={expect_valid} error={expect_error}, "
                             f"got valid={result['valid']} error={result.get('error_type')}")
                failed += 1
        except Exception as e:
            _fail(label, str(e))
            failed += 1

    # ── 1c. Chess tools (offline, no LLM) ────────────────────────────────

    _section("Chess Tools (offline)")

    from src.tools.chess_tools import (
        get_attackers,
        get_board_visual,
        get_defenders,
        get_game_phase,
        get_move_history_pgn,
        get_piece_at,
        get_position_after_moves,
        is_in_check,
        is_square_safe,
        validate_move as tool_validate,
    )

    try:
        result = json.loads(tool_validate.invoke({"fen": STARTING_FEN, "move_uci": "e2e4"}))
        assert result["legal"] is True
        _pass("tool: validate_move (legal)", f"legal={result['legal']}")
        passed += 1
    except Exception as e:
        _fail("tool: validate_move (legal)", str(e))
        failed += 1

    try:
        result = json.loads(tool_validate.invoke({"fen": STARTING_FEN, "move_uci": "e2e5"}))
        assert result["legal"] is False
        _pass("tool: validate_move (illegal)", f"reason={result['reason']}")
        passed += 1
    except Exception as e:
        _fail("tool: validate_move (illegal)", str(e))
        failed += 1

    try:
        result = json.loads(is_in_check.invoke({"fen": STARTING_FEN}))
        assert result["in_check"] is False
        _pass("tool: is_in_check", "starting position not in check")
        passed += 1
    except Exception as e:
        _fail("tool: is_in_check", str(e))
        failed += 1

    try:
        phase = get_game_phase.invoke({"move_history": ["e2e4"] * 25})
        assert phase == "middlegame"
        _pass("tool: get_game_phase", f"phase={phase}")
        passed += 1
    except Exception as e:
        _fail("tool: get_game_phase", str(e))
        failed += 1

    try:
        pgn = get_move_history_pgn.invoke({"move_history": ["e2e4", "e7e5", "g1f3"]})
        assert pgn == "1. e4 e5 2. Nf3"
        _pass("tool: get_move_history_pgn", pgn)
        passed += 1
    except Exception as e:
        _fail("tool: get_move_history_pgn", str(e))
        failed += 1

    try:
        board_visual = get_board_visual.invoke({"fen": STARTING_FEN})
        assert "r n b q k b n r" in board_visual
        _pass("tool: get_board_visual", "ASCII board rendered")
        passed += 1
    except Exception as e:
        _fail("tool: get_board_visual", str(e))
        failed += 1

    try:
        piece = get_piece_at.invoke({"fen": STARTING_FEN, "square": "e1"})
        assert piece == "wK"
        _pass("tool: get_piece_at", f"piece={piece}")
        passed += 1
    except Exception as e:
        _fail("tool: get_piece_at", str(e))
        failed += 1

    try:
        fen_attack = "4k3/8/8/8/8/8/4r3/4K3 w - - 0 1"
        attackers = json.loads(get_attackers.invoke({"fen": fen_attack, "square": "e1"}))
        assert attackers and attackers[0]["square"] == "e2"
        _pass("tool: get_attackers", f"count={len(attackers)}")
        passed += 1
    except Exception as e:
        _fail("tool: get_attackers", str(e))
        failed += 1

    try:
        fen_defenders = "8/8/8/8/8/5k2/4r3/K7 w - - 0 1"
        defenders = json.loads(get_defenders.invoke({"fen": fen_defenders, "square": "e2"}))
        assert defenders and defenders[0]["square"] == "f3"
        _pass("tool: get_defenders", f"count={len(defenders)}")
        passed += 1
    except Exception as e:
        _fail("tool: get_defenders", str(e))
        failed += 1

    try:
        fen_attack = "4k3/8/8/8/8/8/4r3/4K3 w - - 0 1"
        unsafe = json.loads(is_square_safe.invoke({"fen": fen_attack, "square": "e1", "color": "white"}))
        assert unsafe["safe"] is False
        _pass("tool: is_square_safe", f"threats={unsafe['threats']}")
        passed += 1
    except Exception as e:
        _fail("tool: is_square_safe", str(e))
        failed += 1

    try:
        resulting_fen = get_position_after_moves.invoke(
            {"fen": STARTING_FEN, "moves": ["e2e4", "e7e5"]}
        )
        assert isinstance(resulting_fen, str) and len(resulting_fen.split()) == 6
        _pass("tool: get_position_after_moves", resulting_fen)
        passed += 1
    except Exception as e:
        _fail("tool: get_position_after_moves", str(e))
        failed += 1

    # ── 1d. State helpers ────────────────────────────────────────────────

    _section("State & Config Helpers")

    from src.state import create_initial_turn_state
    from src.config import Condition, config_for_condition, GenerationStrategy

    try:
        state = create_initial_turn_state(
            board_fen=STARTING_FEN, game_id="smoke", condition="A"
        )
        assert state["board_fen"] == STARTING_FEN
        assert state["game_id"] == "smoke"
        assert state["condition"] == "A"
        assert state["retry_count"] == 0
        assert state["game_status"] == "ongoing"
        _pass("create_initial_turn_state", "all default fields OK")
        passed += 1
    except Exception as e:
        _fail("create_initial_turn_state", str(e))
        failed += 1

    for cond, expected_retries in [("A", 0), ("B", 0), ("C", 3), ("D", 3), ("E", 3), ("F", 0)]:
        try:
            cfg = config_for_condition(cond)
            assert cfg.max_retries == expected_retries
            _pass(f"config_for_condition({cond})", f"retries={cfg.max_retries}")
            passed += 1
        except Exception as e:
            _fail(f"config_for_condition({cond})", str(e))
            failed += 1

    # ── 1e. Base agent helpers ───────────────────────────────────────────

    _section("Base Agent Helpers")

    from src.agents.base import (
        build_board_representation,
        format_feedback_block,
        get_side_to_move,
        load_prompt,
    )

    try:
        assert get_side_to_move(STARTING_FEN) == "white"
        assert get_side_to_move(MID_FEN) == "black"
        _pass("get_side_to_move", "white/black correct")
        passed += 1
    except Exception as e:
        _fail("get_side_to_move", str(e))
        failed += 1

    try:
        repr_fen = build_board_representation(STARTING_FEN, "fen")
        assert "FEN:" in repr_fen
        assert "Board:" in repr_fen
        repr_hist = build_board_representation(STARTING_FEN, "history")
        assert "withheld" in repr_hist.lower()
        _pass("build_board_representation", "fen + history modes OK")
        passed += 1
    except Exception as e:
        _fail("build_board_representation", str(e))
        failed += 1

    try:
        fb = format_feedback_block(["Move e2e5 is illegal"])
        assert "Attempt 1" in fb
        empty = format_feedback_block([])
        assert empty == ""
        _pass("format_feedback_block", "feedback formatting OK")
        passed += 1
    except Exception as e:
        _fail("format_feedback_block", str(e))
        failed += 1

    prompts_to_check = [
        "generator.txt", "strategist.txt", "router.txt",
        "critic.txt", "explainer.txt", "react.txt",
        "opening_specialist.txt", "middlegame_specialist.txt", "endgame_specialist.txt",
    ]
    for p in prompts_to_check:
        try:
            text = load_prompt(p)
            assert len(text) > 10
            _pass(f"load_prompt({p})", f"len={len(text)}")
            passed += 1
        except Exception as e:
            _fail(f"load_prompt({p})", str(e))
            failed += 1

    # ── 1f. parse_and_validate integration ───────────────────────────────

    _section("parse_and_validate integration")

    from src.graph.base_graph import parse_and_validate

    try:
        pv = parse_and_validate("e2e4", STARTING_FEN)
        assert pv["is_valid"] is True
        assert pv["proposed_move"] == "e2e4"
        _pass("parse_and_validate (valid)", f"move={pv['proposed_move']}")
        passed += 1
    except Exception as e:
        _fail("parse_and_validate (valid)", str(e))
        failed += 1

    try:
        pv = parse_and_validate("e2e5", STARTING_FEN)
        assert pv["is_valid"] is False
        _pass("parse_and_validate (illegal)", f"error={pv['error_type']}")
        passed += 1
    except Exception as e:
        _fail("parse_and_validate (illegal)", str(e))
        failed += 1

    return passed, failed


# ═══════════════════════════════════════════════════════════════════════════
# LAYER 2 — INDIVIDUAL ROLE TESTS (LIVE LLM)
# ═══════════════════════════════════════════════════════════════════════════


def test_roles() -> tuple[int, int]:
    """Test each agent role individually against the live LLM."""

    _header("LAYER 2 — Individual Role Tests (live LLM calls)")
    passed = 0
    failed = 0

    # ── 2a. Generator ────────────────────────────────────────────────────

    _section("Generator (move generation)")

    from src.agents.generator import generate_move

    try:
        t0 = time.time()
        result = generate_move(fen=STARTING_FEN, move_history=[])
        elapsed = time.time() - t0
        raw = result["raw_output"]
        _pass("Generator", f"output='{raw}' | tokens={result['prompt_tokens']}+{result['completion_tokens']} | {elapsed:.1f}s")

        # Check if the output is parseable as a move
        from src.validators.move_parser import parse_uci_move
        parse_r = parse_uci_move(raw)
        if parse_r["is_valid"]:
            _pass("  ↳ Parseable", f"uci={parse_r['move_uci']}")
        else:
            _info(f"  ↳ Not cleanly parseable: {raw[:60]}")
        passed += 1
    except Exception as e:
        _fail("Generator", f"{e}")
        traceback.print_exc()
        failed += 1

    # ── 2b. Strategist (plan generation) ─────────────────────────────────

    _section("Strategist (plan generation)")

    from src.agents.strategist import create_plan

    try:
        t0 = time.time()
        result = create_plan(fen=MID_FEN, move_history=MID_HISTORY)
        elapsed = time.time() - t0
        plan = result["plan"]
        _pass("Strategist", f"plan_len={len(plan)} | {elapsed:.1f}s")
        _info(f"Plan: {textwrap.shorten(plan, 120)}")
        passed += 1
    except Exception as e:
        _fail("Strategist", str(e))
        traceback.print_exc()
        failed += 1

    # ── 2c. Tactician (execute plan) ─────────────────────────────────────

    _section("Tactician (execute plan)")

    from src.agents.tactician import execute_plan

    try:
        fake_plan = "Develop the dark-squared bishop to c5, eyeing f2. The Italian game setup. Consider Bc5."
        t0 = time.time()
        result = execute_plan(
            fen=MID_FEN,
            move_history=MID_HISTORY,
            strategic_plan=fake_plan,
        )
        elapsed = time.time() - t0
        raw = result["raw_output"]
        _pass("Tactician", f"output='{raw}' | {elapsed:.1f}s")
        passed += 1
    except Exception as e:
        _fail("Tactician", str(e))
        traceback.print_exc()
        failed += 1

    # ── 2d. Critic (move evaluation) ─────────────────────────────────────

    _section("Critic (LLM move evaluation)")

    from src.agents.critic import critique_move

    try:
        t0 = time.time()
        result = critique_move(fen=STARTING_FEN, proposed_move="e2e4")
        elapsed = time.time() - t0
        _pass("Critic (legal move)", f"verdict={result['valid']} | {elapsed:.1f}s")
        _info(f"Reasoning: {textwrap.shorten(result['reasoning'], 100)}")
        passed += 1
    except Exception as e:
        _fail("Critic (legal move)", str(e))
        traceback.print_exc()
        failed += 1

    try:
        t0 = time.time()
        result = critique_move(fen=STARTING_FEN, proposed_move="e2e5")
        elapsed = time.time() - t0
        _pass("Critic (illegal move)", f"verdict={result['valid']} | {elapsed:.1f}s")
        _info(f"Reasoning: {textwrap.shorten(result['reasoning'], 100)}")
        if result["valid"]:
            _info(f"  ⚠ Critic incorrectly approved an illegal move — this is expected LLM behavior to study!")
        passed += 1
    except Exception as e:
        _fail("Critic (illegal move)", str(e))
        traceback.print_exc()
        failed += 1

    # ── 2e. Explainer (error explanation) ────────────────────────────────

    _section("Explainer (error explanation)")

    from src.agents.explainer import explain_error

    try:
        t0 = time.time()
        result = explain_error(
            fen=STARTING_FEN,
            proposed_move="e2e5",
            error_type="ILLEGAL_DESTINATION",
            error_reason="Piece cannot move to the destination square.",
        )
        elapsed = time.time() - t0
        _pass("Explainer", f"explanation_len={len(result['explanation'])} | {elapsed:.1f}s")
        _info(f"Explanation: {textwrap.shorten(result['explanation'], 120)}")
        passed += 1
    except Exception as e:
        _fail("Explainer", str(e))
        traceback.print_exc()
        failed += 1

    # ── 2f. Router (phase classification) ────────────────────────────────

    _section("Router (phase classification)")

    from src.agents.router import classify_phase

    positions = [
        ("Starting position", STARTING_FEN, [], "opening"),
        ("Italian Game (move 3)", MID_FEN, MID_HISTORY, "opening"),
        ("K+R vs K endgame", END_FEN, [], "endgame"),
    ]

    for label, fen, hist, expected_phase in positions:
        try:
            t0 = time.time()
            result = classify_phase(fen=fen, move_history=hist)
            elapsed = time.time() - t0
            match = result["phase"] == expected_phase
            if match:
                _pass(f"Router ({label})", f"phase={result['phase']} | {elapsed:.1f}s")
            else:
                _info(f"Router ({label}): expected={expected_phase}, got={result['phase']} (LLM disagreement is possible)")
            passed += 1
        except Exception as e:
            _fail(f"Router ({label})", str(e))
            traceback.print_exc()
            failed += 1

    # ── 2g. Specialists (phase-specific generation) ──────────────────────

    _section("Specialists (phase-specific generation)")

    from src.agents.specialists import generate_specialist_move

    for phase, fen, hist in [
        ("opening", STARTING_FEN, []),
        ("middlegame", MID_FEN, MID_HISTORY),
        ("endgame", END_FEN, []),
    ]:
        try:
            t0 = time.time()
            result = generate_specialist_move(phase=phase, fen=fen, move_history=hist)
            elapsed = time.time() - t0
            _pass(f"Specialist ({phase})", f"output='{result['raw_output']}' | {elapsed:.1f}s")
            passed += 1
        except Exception as e:
            _fail(f"Specialist ({phase})", str(e))
            traceback.print_exc()
            failed += 1

    # ── 2h. ReAct Agent ──────────────────────────────────────────────────

    _section("ReAct Agent (tool-using)")

    from src.agents.react_agent import run_react_loop

    try:
        t0 = time.time()
        result = run_react_loop(
            fen=STARTING_FEN,
            move_history=[],
            max_steps=4,  # keep it short for smoke test
        )
        elapsed = time.time() - t0
        _pass(
            "ReAct Agent",
            f"move='{result['submitted_move']}' | "
            f"steps={result['steps_taken']} | "
            f"tools={len(result['tool_calls_log'])} | "
            f"forfeited={result['forfeited']} | "
            f"{elapsed:.1f}s"
        )
        if result["tool_calls_log"]:
            for tc in result["tool_calls_log"]:
                _info(f"  Tool call: {tc['tool']}({tc.get('args', {})})")
        passed += 1
    except Exception as e:
        _fail("ReAct Agent", str(e))
        traceback.print_exc()
        failed += 1

    return passed, failed


# ═══════════════════════════════════════════════════════════════════════════
# LAYER 3 — FULL CONDITION TESTS (LIVE LLM)
# ═══════════════════════════════════════════════════════════════════════════


def test_conditions() -> tuple[int, int]:
    """Test each condition (A–F) end-to-end with a real LLM call."""

    _header("LAYER 3 — Full Condition End-to-End Tests (live LLM)")
    passed = 0
    failed = 0

    test_fen = STARTING_FEN
    test_history: list[str] = []

    # ── Condition A ──────────────────────────────────────────────────────

    _section("Condition A — Single LLM (no LangGraph)")

    from src.graph.condition_a import run_condition_a

    try:
        t0 = time.time()
        state = run_condition_a(fen=test_fen, game_id="smoke-A")
        elapsed = time.time() - t0
        _pass(
            "Condition A",
            f"move={state['proposed_move']} | valid={state['is_valid']} | "
            f"status={state['game_status']} | attempts={state['total_attempts']} | "
            f"llm_calls={state['llm_calls_this_turn']} | {elapsed:.1f}s"
        )
        passed += 1
    except Exception as e:
        _fail("Condition A", str(e))
        traceback.print_exc()
        failed += 1

    # ── Condition B ──────────────────────────────────────────────────────

    _section("Condition B — LangGraph (no retry)")

    from src.graph.condition_b import run_condition_b

    try:
        t0 = time.time()
        state = run_condition_b(fen=test_fen, game_id="smoke-B")
        elapsed = time.time() - t0
        _pass(
            "Condition B",
            f"move={state['proposed_move']} | valid={state['is_valid']} | "
            f"status={state['game_status']} | {elapsed:.1f}s"
        )
        passed += 1
    except Exception as e:
        _fail("Condition B", str(e))
        traceback.print_exc()
        failed += 1

    # ── Condition C ──────────────────────────────────────────────────────

    _section("Condition C — MAS + LLM Critic")

    from src.graph.condition_c import run_condition_c

    try:
        t0 = time.time()
        state = run_condition_c(fen=test_fen, game_id="smoke-C")
        elapsed = time.time() - t0
        _pass(
            "Condition C",
            f"move={state['proposed_move']} | valid={state['is_valid']} | "
            f"critic_verdict={state.get('critic_verdict')} | "
            f"gt_verdict={state.get('ground_truth_verdict')} | "
            f"retries={state['retry_count']} | "
            f"llm_calls={state['llm_calls_this_turn']} | {elapsed:.1f}s"
        )
        passed += 1
    except Exception as e:
        _fail("Condition C", str(e))
        traceback.print_exc()
        failed += 1

    # ── Condition D ──────────────────────────────────────────────────────

    _section("Condition D — MAS + Symbolic Validator (terse feedback)")

    from src.graph.condition_d import run_condition_d

    try:
        t0 = time.time()
        state = run_condition_d(fen=test_fen, game_id="smoke-D")
        elapsed = time.time() - t0
        _pass(
            "Condition D",
            f"move={state['proposed_move']} | valid={state['is_valid']} | "
            f"retries={state['retry_count']} | attempts={state['total_attempts']} | "
            f"status={state['game_status']} | {elapsed:.1f}s"
        )
        if state["feedback_history"]:
            for fb in state["feedback_history"]:
                _info(f"  Feedback: {textwrap.shorten(fb, 100)}")
        passed += 1
    except Exception as e:
        _fail("Condition D", str(e))
        traceback.print_exc()
        failed += 1

    # ── Condition E ──────────────────────────────────────────────────────

    _section("Condition E — MAS + Symbolic + LLM Explainer")

    from src.graph.condition_e import run_condition_e

    try:
        t0 = time.time()
        state = run_condition_e(fen=test_fen, game_id="smoke-E")
        elapsed = time.time() - t0
        _pass(
            "Condition E",
            f"move={state['proposed_move']} | valid={state['is_valid']} | "
            f"retries={state['retry_count']} | attempts={state['total_attempts']} | "
            f"llm_calls={state['llm_calls_this_turn']} | {elapsed:.1f}s"
        )
        if state["feedback_history"]:
            for fb in state["feedback_history"]:
                _info(f"  Explainer feedback: {textwrap.shorten(fb, 120)}")
        passed += 1
    except Exception as e:
        _fail("Condition E", str(e))
        traceback.print_exc()
        failed += 1

    # ── Condition F ──────────────────────────────────────────────────────

    _section("Condition F — ReAct + Tool Calling")

    from src.graph.condition_f import run_condition_f

    try:
        t0 = time.time()
        state = run_condition_f(fen=test_fen, game_id="smoke-F", max_steps=4)
        elapsed = time.time() - t0
        _pass(
            "Condition F",
            f"move={state['proposed_move']} | valid={state['is_valid']} | "
            f"status={state['game_status']} | "
            f"tool_calls={len(state['tool_calls'])} | {elapsed:.1f}s"
        )
        if state["tool_calls"]:
            for tc in state["tool_calls"][:5]:  # show first 5
                _info(f"  Tool: {tc.get('tool', '?')}({tc.get('args', {})})")
        passed += 1
    except Exception as e:
        _fail("Condition F", str(e))
        traceback.print_exc()
        failed += 1

    # ── Bonus: test with generation strategies ───────────────────────────

    _section("Bonus — Generation Strategies (planner_actor on Condition B)")

    try:
        t0 = time.time()
        state = run_condition_b(
            fen=MID_FEN,
            move_history=MID_HISTORY,
            game_id="smoke-B-planner",
            generation_strategy="planner_actor",
        )
        elapsed = time.time() - t0
        _pass(
            "Cond B + planner_actor",
            f"move={state['proposed_move']} | plan_len={len(state.get('strategic_plan', ''))} | {elapsed:.1f}s"
        )
        if state.get("strategic_plan"):
            _info(f"  Plan: {textwrap.shorten(state['strategic_plan'], 120)}")
        passed += 1
    except Exception as e:
        _fail("Cond B + planner_actor", str(e))
        traceback.print_exc()
        failed += 1

    _section("Bonus — Generation Strategies (router_specialists on Condition B)")

    try:
        t0 = time.time()
        state = run_condition_b(
            fen=MID_FEN,
            move_history=MID_HISTORY,
            game_id="smoke-B-router",
            generation_strategy="router_specialists",
        )
        elapsed = time.time() - t0
        _pass(
            "Cond B + router_specialists",
            f"move={state['proposed_move']} | phase={state.get('routed_phase', '')} | {elapsed:.1f}s"
        )
        passed += 1
    except Exception as e:
        _fail("Cond B + router_specialists", str(e))
        traceback.print_exc()
        failed += 1

    return passed, failed


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(description="Maat Smoke Test Suite")
    parser.add_argument(
        "--layer",
        choices=["offline", "roles", "conditions", "all"],
        default="all",
        help="Which test layer to run (default: all)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Shorthand for --layer offline",
    )
    args = parser.parse_args()

    if args.offline:
        args.layer = "offline"

    total_passed = 0
    total_failed = 0
    start = time.time()

    if args.layer in ("offline", "all"):
        p, f = test_offline()
        total_passed += p
        total_failed += f

    if args.layer in ("roles", "all"):
        p, f = test_roles()
        total_passed += p
        total_failed += f

    if args.layer in ("conditions", "all"):
        p, f = test_conditions()
        total_passed += p
        total_failed += f

    elapsed = time.time() - start

    # ── Summary ──────────────────────────────────────────────────────────

    _header("SUMMARY")

    total = total_passed + total_failed
    color = _GREEN if total_failed == 0 else _RED
    print(f"  {color}{_BOLD}{total_passed}/{total} tests passed{_RESET}  ({total_failed} failed)")
    print(f"  Total time: {elapsed:.1f}s\n")

    if total_failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
