"""LLM-based error explainer — used by Condition E."""

from __future__ import annotations

import json
from typing import Any, TypedDict

import chess
from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.base import get_side_to_move, load_agent_prompt
from src.llm.llm_client import get_model
from src.config import ModelConfig
from src.state import InputMode


class ExplainerResult(TypedDict):
    error_summary: str
    explanation: str
    feedback_text: str
    raw_output: str
    prompt_tokens: int
    completion_tokens: int


def explain_error(
    *,
    fen: str,
    attempted_move: str,
    error_type: str,
    engine_error_message: str,
    move_history: list[str] | None = None,
    input_mode: InputMode = "fen",
    model_config: ModelConfig | None = None,
) -> ExplainerResult:
    """Translate a symbolic validation error into pedagogical feedback.

    Returns a rich natural-language explanation plus token usage data.
    """

    board = chess.Board(fen)
    ascii_board = str(board)
    color = get_side_to_move(fen)
    history_str = " ".join(move_history or []) if move_history else "(none)"

    system_text = load_agent_prompt("explainer", input_mode, "system")
    user_template = load_agent_prompt("explainer", input_mode, "user")
    prompt_text = user_template.format(
        color=color,
        fen=fen,
        ascii_board=ascii_board,
        move_history=history_str,
        attempted_move=attempted_move,
        engine_error_message=engine_error_message,
        error_type=error_type,
    )

    model = get_model(model_config)
    messages = [
        SystemMessage(content=system_text),
        HumanMessage(content=prompt_text),
    ]

    response = model.invoke(messages)
    raw = response.content.strip() if isinstance(response.content, str) else str(response.content).strip()
    usage = response.usage_metadata or {}

    parsed = _parse_explainer_json(raw)
    error_summary = str(parsed.get("error_summary", "")).strip()
    explanation = str(parsed.get("explanation", "")).strip()

    feedback_text = explanation
    if error_summary and explanation:
        feedback_text = f"{error_summary} {explanation}"
    elif error_summary:
        feedback_text = error_summary
    elif not feedback_text:
        feedback_text = raw

    return {
        "error_summary": error_summary,
        "explanation": explanation or raw,
        "feedback_text": feedback_text,
        "raw_output": raw,
        "prompt_tokens": usage.get("input_tokens", 0),
        "completion_tokens": usage.get("output_tokens", 0),
    }


def _parse_explainer_json(raw: str) -> dict[str, Any]:
    """Best-effort parse of the explainer's JSON response."""

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    import re

    fenced_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, re.DOTALL)
    if fenced_match:
        try:
            parsed = json.loads(fenced_match.group(1))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    first = raw.find("{")
    last = raw.rfind("}")
    if first != -1 and last != -1 and last > first:
        try:
            parsed = json.loads(raw[first:last + 1])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    return {"error_summary": "", "explanation": raw}
