"""ReAct agent helpers — prompt builders and tool execution for Condition F.

The ReAct reasoning loop itself lives in the LangGraph graph at
``src/graph/condition_f.py``.  This module provides shared utilities:

- ``build_react_messages`` — constructs the initial ``[SystemMessage, HumanMessage]``
- ``execute_tool_calls`` — direct tool invocation returning ``ToolMessage`` objects
- ``extract_submit_from_text`` — fallback regex parser for text-embedded moves
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import chess
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from src.agents.base import (
    get_side_to_move,
    load_agent_prompt,
)
from src.state import InputMode


# ── Prompt builder ───────────────────────────────────────────────────────


def build_react_messages(
    *,
    fen: str,
    move_history: list[str],
    input_mode: InputMode = "fen",
    conversation_history: list[Any] | None = None,
) -> list[SystemMessage | HumanMessage]:
    """Build the initial message list for the ReAct agent.

    Returns a list starting with ``SystemMessage``, optionally followed
    by prior-turn conversation history, then the current ``HumanMessage``.
    """

    color = get_side_to_move(fen)
    board = chess.Board(fen)
    ascii_board = str(board)
    history_str = " ".join(move_history) if move_history else "(none)"

    system_template = load_agent_prompt("react", input_mode, "system")
    user_template = load_agent_prompt("react", input_mode, "user")

    system_prompt = system_template.format(
        color=color,
        fen=fen,
        ascii_board=ascii_board,
        move_history=history_str,
    )
    turn_prompt = user_template.format(
        color=color,
        fen=fen,
        ascii_board=ascii_board,
        move_history=history_str,
    )

    messages: list[Any] = [SystemMessage(content=system_prompt)]
    if conversation_history:
        messages.extend(conversation_history)
    messages.append(HumanMessage(content=turn_prompt))
    return messages



# ── Tool execution ──────────────────────────────────────────────────────


def execute_tool_calls(
    tool_calls: Sequence[Any],
    tool_map: dict[str, Any],
) -> list[ToolMessage]:
    """Execute tool calls directly and return ``ToolMessage`` objects.

    This avoids LangGraph's ``ToolNode`` to sidestep config-passing
    issues in LangGraph >=1.1 while keeping execution deterministic.
    """

    tool_messages: list[ToolMessage] = []
    for tc in tool_calls:
        tool_name = tc["name"]
        tool_args = tc["args"]
        tool_id = tc.get("id", "")

        tool_fn = tool_map.get(tool_name)
        if tool_fn is None:
            content = f"Error: Unknown tool '{tool_name}'"
        else:
            try:
                content = tool_fn.invoke(tool_args)
            except Exception as e:
                content = f"Error invoking {tool_name}: {e}"

        tool_messages.append(
            ToolMessage(content=content, tool_call_id=tool_id)
        )

    return tool_messages


# ── Fallback text parser ────────────────────────────────────────────────


def extract_submit_from_text(text: str) -> str:
    """Fallback: try to find a move the agent embedded in plain text.

    Looks for patterns like ``SUBMIT:e2e4`` or ``MOVE: e2e4``.
    """

    match = re.search(r"(?:SUBMIT:|MOVE:\s*)([a-h][1-8][a-h][1-8][qrbn]?)", text)
    if match:
        return match.group(1)
    return ""
