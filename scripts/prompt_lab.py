#!/usr/bin/env python3
"""Maat Prompt Lab — interactive prompt testing UI.

Run:
    python scripts/prompt_lab.py [--port 8111]

Then open http://localhost:8111 in your browser.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

# ── Make sure `src` is importable ────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import chess  # noqa: E402

from src.agents.base import (  # noqa: E402
    build_board_representation,
    format_feedback_block,
    get_side_to_move,
    load_prompt,
)
from src.config import ModelConfig  # noqa: E402
from src.engine.stockfish_wrapper import StockfishWrapper  # noqa: E402
from src.llm.llm_client import get_model  # noqa: E402
from src.validators.move_parser import parse_uci_move  # noqa: E402
from src.validators.symbolic import validate_move  # noqa: E402

# ── Prompt templates directory ───────────────────────────────────────────
PROMPTS_DIR = PROJECT_ROOT / "src" / "prompts"
UI_FILE = Path(__file__).resolve().parent / "prompt_lab_ui.html"
RATE_MOVE_ELO = int(os.getenv("PROMPT_LAB_RATE_MOVE_ELO", "2500"))
RATE_MOVE_TIME_LIMIT_S = float(os.getenv("PROMPT_LAB_RATE_MOVE_TIME_LIMIT_S", "0.12"))

# ── Tactician inline template (not in a file) ───────────────────────────
TACTICIAN_TEMPLATE = """\
You are a chess tactician. You are playing as {color}.

{board_representation}

Move history (UCI): {move_history}

A strategist has provided the following plan:
--- STRATEGIC PLAN ---
{strategic_plan}
--- END PLAN ---
{feedback_block}
Based on this plan, select the best concrete move.
Output exactly one move in UCI format (e.g., e2e4, g1f3).
Respond with ONLY the UCI move, no explanation."""

# ── Role definitions ─────────────────────────────────────────────────────

ROLES: dict[str, dict[str, Any]] = {
    "generator": {
        "name": "Generator",
        "description": "Generates a chess move from a position (Conditions A-E)",
        "prompt_file": "generator.txt",
        "system_message": "You are a chess-playing assistant.",
        "fields": [
            {"id": "fen", "label": "FEN", "type": "text", "default": chess.STARTING_FEN},
            {"id": "move_history", "label": "Move History (space-separated UCI)", "type": "text", "default": ""},
            {"id": "feedback_history", "label": "Feedback History (one per line)", "type": "textarea", "default": ""},
            {"id": "input_mode", "label": "Input Mode", "type": "select", "options": ["fen", "history"], "default": "fen"},
        ],
    },
    "strategist": {
        "name": "Strategist",
        "description": "Creates a strategic plan — no move, just NL analysis (Planner-Actor)",
        "prompt_file": "strategist.txt",
        "system_message": "You are a chess strategist.",
        "fields": [
            {"id": "fen", "label": "FEN", "type": "text", "default": chess.STARTING_FEN},
            {"id": "move_history", "label": "Move History (space-separated UCI)", "type": "text", "default": ""},
            {"id": "input_mode", "label": "Input Mode", "type": "select", "options": ["fen", "history"], "default": "fen"},
        ],
    },
    "tactician": {
        "name": "Tactician",
        "description": "Converts a strategic plan into a concrete UCI move (Planner-Actor)",
        "prompt_file": None,  # uses inline template
        "system_message": "You are a chess-playing assistant executing a strategic plan.",
        "fields": [
            {"id": "fen", "label": "FEN", "type": "text", "default": chess.STARTING_FEN},
            {"id": "move_history", "label": "Move History (space-separated UCI)", "type": "text", "default": ""},
            {"id": "strategic_plan", "label": "Strategic Plan (from Strategist)", "type": "textarea", "default": "Develop the knight to f3 to control the center and prepare for kingside castling."},
            {"id": "feedback_history", "label": "Feedback History (one per line)", "type": "textarea", "default": ""},
            {"id": "input_mode", "label": "Input Mode", "type": "select", "options": ["fen", "history"], "default": "fen"},
        ],
    },
    "critic": {
        "name": "Critic",
        "description": "Evaluates whether a proposed move is legal (Condition C)",
        "prompt_file": "critic.txt",
        "system_message": "You are a chess rules validation expert.",
        "fields": [
            {"id": "fen", "label": "FEN", "type": "text", "default": chess.STARTING_FEN},
            {"id": "proposed_move", "label": "Proposed Move (UCI)", "type": "text", "default": "e2e4"},
        ],
    },
    "explainer": {
        "name": "Explainer",
        "description": "Translates symbolic validation errors into pedagogical feedback (Condition E)",
        "prompt_file": "explainer.txt",
        "system_message": "You are a chess rules teacher.",
        "fields": [
            {"id": "fen", "label": "FEN", "type": "text", "default": chess.STARTING_FEN},
            {"id": "proposed_move", "label": "Proposed Move (UCI)", "type": "text", "default": "e2e5"},
            {"id": "error_type", "label": "Error Type", "type": "select",
             "options": ["ILLEGAL_DESTINATION", "INVALID_PIECE", "LEAVES_IN_CHECK",
                         "CASTLING_VIOLATION", "EN_PASSANT_VIOLATION", "PROMOTION_ERROR",
                         "PARSE_ERROR", "NO_OUTPUT"],
             "default": "ILLEGAL_DESTINATION"},
            {"id": "error_reason", "label": "Error Reason", "type": "text",
             "default": "Piece cannot move to the destination square."},
        ],
    },
    "router": {
        "name": "Router",
        "description": "Classifies game phase: opening / middlegame / endgame (Router-Specialists)",
        "prompt_file": "router.txt",
        "system_message": "You are a chess game-phase classifier.",
        "fields": [
            {"id": "fen", "label": "FEN", "type": "text", "default": chess.STARTING_FEN},
            {"id": "move_history", "label": "Move History (space-separated UCI)", "type": "text", "default": ""},
            {"id": "input_mode", "label": "Input Mode", "type": "select", "options": ["fen", "history"], "default": "fen"},
        ],
    },
    "opening_specialist": {
        "name": "Opening Specialist",
        "description": "Generates a move with opening-phase expertise (Router-Specialists)",
        "prompt_file": "opening_specialist.txt",
        "system_message": "You are a chess-playing assistant.",
        "fields": [
            {"id": "fen", "label": "FEN", "type": "text", "default": chess.STARTING_FEN},
            {"id": "move_history", "label": "Move History (space-separated UCI)", "type": "text", "default": ""},
            {"id": "feedback_history", "label": "Feedback History (one per line)", "type": "textarea", "default": ""},
            {"id": "input_mode", "label": "Input Mode", "type": "select", "options": ["fen", "history"], "default": "fen"},
        ],
    },
    "middlegame_specialist": {
        "name": "Middlegame Specialist",
        "description": "Generates a move with middlegame tactical focus (Router-Specialists)",
        "prompt_file": "middlegame_specialist.txt",
        "system_message": "You are a chess-playing assistant.",
        "fields": [
            {"id": "fen", "label": "FEN", "type": "text",
             "default": "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"},
            {"id": "move_history", "label": "Move History (space-separated UCI)", "type": "text",
             "default": "e2e4 e7e5 g1f3 b8c6 f1c4"},
            {"id": "feedback_history", "label": "Feedback History (one per line)", "type": "textarea", "default": ""},
            {"id": "input_mode", "label": "Input Mode", "type": "select", "options": ["fen", "history"], "default": "fen"},
        ],
    },
    "endgame_specialist": {
        "name": "Endgame Specialist",
        "description": "Generates a move with endgame technique focus (Router-Specialists)",
        "prompt_file": "endgame_specialist.txt",
        "system_message": "You are a chess-playing assistant.",
        "fields": [
            {"id": "fen", "label": "FEN", "type": "text", "default": "8/8/8/4k3/8/8/8/4K2R w - - 0 1"},
            {"id": "move_history", "label": "Move History (space-separated UCI)", "type": "text", "default": ""},
            {"id": "feedback_history", "label": "Feedback History (one per line)", "type": "textarea", "default": ""},
            {"id": "input_mode", "label": "Input Mode", "type": "select", "options": ["fen", "history"], "default": "fen"},
        ],
    },
    "react": {
        "name": "ReAct Agent",
        "description": "Autonomous tool-using chess player (Condition F) — system prompt only",
        "prompt_file": "react.txt",
        "system_message": "(This is the system prompt — ReAct uses tool calling)",
        "fields": [
            {"id": "fen", "label": "FEN", "type": "text", "default": chess.STARTING_FEN},
            {"id": "move_history", "label": "Move History (space-separated UCI)", "type": "text", "default": ""},
            {"id": "input_mode", "label": "Input Mode", "type": "select", "options": ["fen", "history"], "default": "fen"},
        ],
        "is_react": True,
    },
}

# ── Position presets ─────────────────────────────────────────────────────

PRESETS = [
    {"id": "starting", "name": "Starting Position",
     "fen": chess.STARTING_FEN, "history": ""},
    {"id": "italian", "name": "Italian Game (move 3)",
     "fen": "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
     "history": "e2e4 e7e5 g1f3 b8c6 f1c4"},
    {"id": "sicilian", "name": "Sicilian Najdorf",
     "fen": "rnbqkb1r/1p2pppp/p2p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6",
     "history": "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 a7a6"},
    {"id": "endgame_kr", "name": "K+R vs K (endgame)",
     "fen": "8/8/8/4k3/8/8/8/4K2R w - - 0 1", "history": ""},
    {"id": "endgame_kp", "name": "K+P vs K (endgame)",
     "fen": "8/8/8/4k3/8/4K3/4P3/8 w - - 0 1", "history": ""},
    {"id": "promotion", "name": "Pawn Promotion",
     "fen": "8/P7/8/8/8/8/8/4K2k w - - 0 1", "history": ""},
    {"id": "in_check", "name": "King in Check",
     "fen": "rnbqkbnr/ppppp1pp/8/8/4Pp1Q/8/PPPP1PPP/RNB1KBNR b KQkq - 1 3",
     "history": "e2e4 f7f5 d1h5"},
]


def rate_move(fen: str, move_uci: str) -> dict[str, Any]:
    """Evaluate a move played on the given FEN and return quality metadata."""
    with StockfishWrapper(elo=RATE_MOVE_ELO) as engine:
        return engine.analyze_move(fen=fen, move_uci=move_uci, time_limit=RATE_MOVE_TIME_LIMIT_S)


# ═══════════════════════════════════════════════════════════════════════════
# RENDERING
# ═══════════════════════════════════════════════════════════════════════════


def render_prompt(role_id: str, params: dict[str, Any], template_override: str | None = None) -> dict[str, Any]:
    """Render a prompt template with the given parameters.

    Returns a dict with `system_message`, `human_message`, and `template_raw`.
    """
    role = ROLES[role_id]
    fen = params.get("fen", chess.STARTING_FEN)
    move_history_str = params.get("move_history", "")
    move_history = move_history_str.split() if move_history_str.strip() else []
    input_mode = params.get("input_mode", "fen")

    # Load template
    if template_override:
        template = template_override
    elif role["prompt_file"] is None:
        template = TACTICIAN_TEMPLATE
    else:
        template = load_prompt(role["prompt_file"])

    template_raw = template

    # Build common variables
    fmt_vars: dict[str, str] = {}

    if role_id in ("generator", "strategist", "tactician",
                   "opening_specialist", "middlegame_specialist", "endgame_specialist",
                   "react"):
        fmt_vars["color"] = get_side_to_move(fen)
        fmt_vars["board_representation"] = build_board_representation(fen, input_mode, move_history)
        fmt_vars["move_history"] = " ".join(move_history) if move_history else "(none)"

    if role_id in ("generator", "tactician",
                   "opening_specialist", "middlegame_specialist", "endgame_specialist"):
        feedback_lines = [l for l in params.get("feedback_history", "").split("\n") if l.strip()]
        fmt_vars["feedback_block"] = format_feedback_block(feedback_lines)

    if role_id == "tactician":
        fmt_vars["strategic_plan"] = params.get("strategic_plan", "")

    if role_id == "router":
        fmt_vars["board_representation"] = build_board_representation(fen, input_mode, move_history)
        fmt_vars["move_history"] = " ".join(move_history) if move_history else "(none)"

    if role_id in ("critic", "explainer"):
        board = chess.Board(fen)
        fmt_vars["fen"] = fen
        fmt_vars["board_ascii"] = str(board)
        fmt_vars["proposed_move"] = params.get("proposed_move", "")

    if role_id == "explainer":
        fmt_vars["error_type"] = params.get("error_type", "")
        fmt_vars["error_reason"] = params.get("error_reason", "")

    # Render
    try:
        rendered = template.format(**fmt_vars)
    except KeyError as e:
        rendered = f"[Template rendering error: missing variable {e}]\n\nTemplate:\n{template}\n\nVariables:\n{json.dumps(fmt_vars, indent=2)}"

    return {
        "system_message": role["system_message"],
        "human_message": rendered,
        "template_raw": template_raw,
    }


def _format_response_content(content: Any) -> str:
    """Format structured model content (blocks) into readable text."""
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        thinking_parts: list[str] = []
        answer_parts: list[str] = []
        other_parts: list[str] = []

        for block in content:
            if isinstance(block, str):
                text = block.strip()
                if text:
                    answer_parts.append(text)
                continue

            if not isinstance(block, dict):
                text = str(block).strip()
                if text:
                    other_parts.append(text)
                continue

            block_type = str(block.get("type", "")).lower()
            if block_type == "thinking":
                thinking = block.get("thinking") or block.get("text") or ""
                if isinstance(thinking, str) and thinking.strip():
                    thinking_parts.append(thinking.strip())
                continue

            if block_type in ("text", "output_text"):
                text = block.get("text") or block.get("content") or ""
                if isinstance(text, str) and text.strip():
                    answer_parts.append(text.strip())
                continue

            fallback_text = block.get("text") or block.get("content") or block.get("thinking")
            if isinstance(fallback_text, str) and fallback_text.strip():
                other_parts.append(fallback_text.strip())
            else:
                dumped = json.dumps(block, ensure_ascii=False, default=str).strip()
                if dumped and dumped != "{}":
                    other_parts.append(dumped)

        sections: list[str] = []
        if thinking_parts:
            sections.append("Thinking:\n" + "\n\n".join(thinking_parts))
        if answer_parts:
            sections.append("Final Answer:\n" + "\n\n".join(answer_parts))
        if other_parts:
            sections.append("Other Output:\n" + "\n\n".join(other_parts))

        return "\n\n".join(sections).strip()

    if isinstance(content, dict):
        nested = content.get("content")
        if nested is not None:
            nested_text = _format_response_content(nested)
            if nested_text:
                return nested_text

        for key in ("text", "output_text", "thinking"):
            value = content.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        return json.dumps(content, ensure_ascii=False, default=str).strip()

    return str(content).strip()


def _extract_text_content(content: Any) -> str:
    """Extract plain text answer content for move parsing and validation."""
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        text_parts: list[str] = []
        for block in content:
            if isinstance(block, str) and block.strip():
                text_parts.append(block.strip())
                continue

            if not isinstance(block, dict):
                continue

            block_type = str(block.get("type", "")).lower()
            if block_type in ("text", "output_text"):
                text = block.get("text") or block.get("content")
                if isinstance(text, str) and text.strip():
                    text_parts.append(text.strip())

        if text_parts:
            return "\n".join(text_parts).strip()

        return _format_response_content(content)

    if isinstance(content, dict):
        nested = content.get("content")
        if nested is not None:
            nested_text = _extract_text_content(nested)
            if nested_text:
                return nested_text

        for key in ("text", "output_text"):
            value = content.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        return _format_response_content(content)

    return str(content).strip()


def invoke_llm(role_id: str, params: dict[str, Any], template_override: str | None = None) -> dict[str, Any]:
    """Invoke the LLM for the given role with the given parameters."""
    from langchain_core.messages import HumanMessage, SystemMessage

    role = ROLES[role_id]
    is_react = role.get("is_react", False)

    rendered = render_prompt(role_id, params, template_override)

    if is_react:
        # For ReAct, run the full loop
        from src.agents.react_agent import run_react_loop
        fen = params.get("fen", chess.STARTING_FEN)
        move_history_str = params.get("move_history", "")
        move_history = move_history_str.split() if move_history_str.strip() else []

        t0 = time.time()
        react_result = run_react_loop(
            fen=fen, move_history=move_history, max_steps=4,
        )
        elapsed = time.time() - t0

        # Build tool calls text
        tool_log = []
        for tc in react_result["tool_calls_log"]:
            tool_log.append(f"Step {tc.get('step', '?')}: {tc['tool']}({json.dumps(tc['args'])})")
            if "result" in tc:
                tool_log.append(f"  -> {tc['result'][:200]}")

        return {
            **rendered,
            "raw_output": react_result["submitted_move"] or "(no move submitted)",
            "tool_calls_text": "\n".join(tool_log),
            "prompt_tokens": react_result["total_prompt_tokens"],
            "completion_tokens": react_result["total_completion_tokens"],
            "elapsed_s": round(elapsed, 2),
            "steps_taken": react_result["steps_taken"],
            "forfeited": react_result["forfeited"],
            "validation": _validate_output(react_result["submitted_move"], params.get("fen", chess.STARTING_FEN)),
        }

    # Standard (non-ReAct) invocation
    model = get_model()
    messages = [
        SystemMessage(content=rendered["system_message"]),
        HumanMessage(content=rendered["human_message"]),
    ]

    t0 = time.time()
    response = model.invoke(messages)
    elapsed = time.time() - t0

    raw = _format_response_content(response.content)
    validation_text = _extract_text_content(response.content) or raw
    usage = response.usage_metadata or {}

    return {
        **rendered,
        "raw_output": raw,
        "prompt_tokens": usage.get("input_tokens", 0),
        "completion_tokens": usage.get("output_tokens", 0),
        "elapsed_s": round(elapsed, 2),
        "validation": _validate_output(validation_text, params.get("fen", chess.STARTING_FEN)),
    }


def _validate_output(raw_output: str, fen: str) -> dict[str, Any]:
    """Parse and validate the raw output against the board."""
    if not raw_output or raw_output.startswith("("):
        return {"parsed": False, "move": None, "valid": False, "error": "No move to validate"}

    parse_result = parse_uci_move(raw_output)
    if not parse_result["is_valid"]:
        return {
            "parsed": False,
            "move": None,
            "valid": False,
            "error": parse_result.get("reason", "Could not parse"),
            "used_fallback": parse_result["used_fallback"],
        }

    val_result = validate_move(fen, parse_result["move_uci"])
    return {
        "parsed": True,
        "move": parse_result["move_uci"],
        "valid": val_result["valid"],
        "error": val_result["reason"] if not val_result["valid"] else None,
        "error_type": val_result.get("error_type"),
        "used_fallback": parse_result["used_fallback"],
    }


# ═══════════════════════════════════════════════════════════════════════════
# HTTP SERVER
# ═══════════════════════════════════════════════════════════════════════════


class PromptLabHandler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        # Quieter logging
        sys.stderr.write(f"[PromptLab] {args[0]}\n")

    def _send_json(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        return json.loads(raw) if raw else {}

    # ── GET routes ────────────────────────────────────────────────────────

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"

        if path == "/":
            html = UI_FILE.read_text(encoding="utf-8")
            self._send_html(html)
            return

        if path == "/api/roles":
            # Strip fields to a simpler format for the UI
            roles_out = {}
            for rid, rdef in ROLES.items():
                roles_out[rid] = {
                    "name": rdef["name"],
                    "description": rdef["description"],
                    "system_message": rdef["system_message"],
                    "fields": rdef["fields"],
                    "is_react": rdef.get("is_react", False),
                }
            self._send_json(roles_out)
            return

        if path == "/api/presets":
            self._send_json(PRESETS)
            return

        if path.startswith("/api/prompt/"):
            role_id = path.split("/api/prompt/")[1]
            if role_id not in ROLES:
                self._send_json({"error": f"Unknown role: {role_id}"}, 404)
                return
            role = ROLES[role_id]
            if role["prompt_file"] is None:
                text = TACTICIAN_TEMPLATE
            else:
                text = load_prompt(role["prompt_file"])
            self._send_json({"template": text, "prompt_file": role["prompt_file"]})
            return

        self._send_json({"error": "Not found"}, 404)

    # ── POST routes ───────────────────────────────────────────────────────

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path == "/api/rate-move":
            body = self._read_body()
            fen = str(body.get("fen", "")).strip()
            move_uci = str(body.get("move_uci") or body.get("move") or "").strip().lower()
            source = str(body.get("source", "")).strip()

            if not fen:
                self._send_json({"error": "Missing required field: fen"}, 400)
                return

            if not move_uci:
                self._send_json({"error": "Missing required field: move_uci"}, 400)
                return

            try:
                result = rate_move(fen, move_uci)
                if source:
                    result["source"] = source
                self._send_json(result)
            except ValueError as e:
                self._send_json({"error": str(e)}, 400)
            except FileNotFoundError as e:
                self._send_json({"error": str(e)}, 503)
            except Exception as e:
                self._send_json({"error": str(e), "traceback": traceback.format_exc()}, 500)
            return

        if path == "/api/preview":
            body = self._read_body()
            role_id = body.get("role_id")
            params = body.get("params", {})
            template_override = body.get("template_override")
            if role_id not in ROLES:
                self._send_json({"error": f"Unknown role: {role_id}"}, 400)
                return
            try:
                result = render_prompt(role_id, params, template_override)
                self._send_json(result)
            except Exception as e:
                self._send_json({"error": str(e), "traceback": traceback.format_exc()}, 500)
            return

        if path == "/api/invoke":
            body = self._read_body()
            role_id = body.get("role_id")
            params = body.get("params", {})
            template_override = body.get("template_override")
            if role_id not in ROLES:
                self._send_json({"error": f"Unknown role: {role_id}"}, 400)
                return

            request_id = f"{int(time.time() * 1000)}-{threading.get_ident()}"
            sys.stderr.write(f"[PromptLab][invoke][{request_id}] started role={role_id}\n")
            invoke_t0 = time.time()
            try:
                result = invoke_llm(role_id, params, template_override)
                result["server_elapsed_s"] = round(time.time() - invoke_t0, 2)
                sys.stderr.write(
                    f"[PromptLab][invoke][{request_id}] completed role={role_id} "
                    f"server_elapsed={result['server_elapsed_s']}s "
                    f"model_elapsed={result.get('elapsed_s', 'n/a')}s\n"
                )
                self._send_json(result)
            except Exception as e:
                server_elapsed_s = round(time.time() - invoke_t0, 2)
                sys.stderr.write(
                    f"[PromptLab][invoke][{request_id}] failed role={role_id} "
                    f"server_elapsed={server_elapsed_s}s error={e}\n"
                )
                self._send_json(
                    {
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                        "server_elapsed_s": server_elapsed_s,
                    },
                    500,
                )
            return

        if path == "/api/save-prompt":
            body = self._read_body()
            role_id = body.get("role_id")
            new_template = body.get("template", "")
            if role_id not in ROLES:
                self._send_json({"error": f"Unknown role: {role_id}"}, 400)
                return
            role = ROLES[role_id]
            if role["prompt_file"] is None:
                self._send_json({"error": "Tactician uses an inline template — edit it in src/agents/tactician.py"}, 400)
                return
            prompt_path = PROMPTS_DIR / role["prompt_file"]
            prompt_path.write_text(new_template, encoding="utf-8")
            self._send_json({"saved": True, "path": str(prompt_path)})
            return

        self._send_json({"error": "Not found"}, 404)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Maat Prompt Lab")
    parser.add_argument("--port", type=int, default=8111)
    args = parser.parse_args()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), PromptLabHandler)
    print(f"\n  Maat Prompt Lab running at http://localhost:{args.port}")
    print(f"  Press Ctrl+C to stop.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Shutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
