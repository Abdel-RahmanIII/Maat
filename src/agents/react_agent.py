"""ReAct agent — autonomous tool-using chess player (Condition F)."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.agents.base import (
    build_board_representation,
    get_side_to_move,
    load_prompt,
)
from src.config import ModelConfig
from src.llm.llm_client import get_model_with_tools
from src.state import InputMode
from src.tools.chess_tools import get_tools_for_input_mode


def _execute_tool_calls(
    tool_calls: Sequence[Any],
    tool_map: dict[str, Any],
) -> list[ToolMessage]:
    """Execute tool calls directly and return ToolMessages."""

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


def run_react_loop(
    *,
    fen: str,
    move_history: list[str],
    input_mode: InputMode = "fen",
    max_steps: int = 6,
    model_config: ModelConfig | None = None,
) -> dict[str, Any]:
    """Execute the ReAct reasoning loop.

    The agent iterates think → (optionally) call tools → observe results
    until it either calls ``submit_move`` or reaches *max_steps*.

    Returns
    -------
    dict
        ``submitted_move`` — the UCI string the agent submitted (or ``""``
        if it ran out of steps).
        ``tool_calls_log`` — list of ``{tool, args, result}`` dicts.
        ``steps_taken`` — number of think/act cycles.
        ``total_prompt_tokens`` / ``total_completion_tokens`` — aggregate
        token usage across all LLM calls in the loop.
        ``forfeited`` — ``True`` if the agent failed to submit.
    """

    color = get_side_to_move(fen)
    board_repr = build_board_representation(fen, input_mode, move_history)
    history_str = " ".join(move_history) if move_history else "(none)"

    template = load_prompt("react.txt")
    system_prompt = template.format(
        color=color,
        board_representation=board_repr,
        move_history=history_str,
    )

    selected_tools = get_tools_for_input_mode(input_mode)
    tool_map: dict[str, Any] = {tool.name: tool for tool in selected_tools}
    model = get_model_with_tools(selected_tools, model_config)

    if input_mode == "fen":
        turn_prompt = f"It is your turn as {color}. The current FEN is: {fen}"
    else:
        turn_prompt = (
            f"It is your turn as {color}. FEN is withheld in history mode; "
            "reason from move history and tool outputs."
        )

    messages: list[Any] = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=turn_prompt),
    ]

    tool_calls_log: list[dict[str, Any]] = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    submitted_move = ""
    steps_taken = 0

    for step in range(max_steps):
        steps_taken = step + 1
        response: AIMessage = model.invoke(messages)
        usage = response.usage_metadata or {}
        total_prompt_tokens += usage.get("input_tokens", 0)
        total_completion_tokens += usage.get("output_tokens", 0)

        messages.append(response)

        # If no tool calls, the agent is done thinking without submitting
        if not response.tool_calls:
            # Check if it embedded a submit in text (fallback)
            submitted_move = _extract_submit_from_text(
                response.content if isinstance(response.content, str) else str(response.content)
            )
            if submitted_move:
                break
            continue

        # Process tool calls
        for tc in response.tool_calls:
            tool_calls_log.append({
                "tool": tc["name"],
                "args": tc["args"],
                "step": step,
            })

        # Execute tools directly (avoids ToolNode config issues in LangGraph >=1.1)
        tool_messages = _execute_tool_calls(response.tool_calls, tool_map)
        messages.extend(tool_messages)

        # Update tool call log with results
        for i, msg in enumerate(tool_messages):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if i < len(response.tool_calls):
                tool_calls_log[-(len(response.tool_calls) - i)]["result"] = content

        # Check if submit_move was called
        for tc in response.tool_calls:
            if tc["name"] == "submit_move":
                submitted_move = tc["args"].get("uci_move", "")
                break

        if submitted_move:
            break

    return {
        "submitted_move": submitted_move,
        "tool_calls_log": tool_calls_log,
        "steps_taken": steps_taken,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "forfeited": not bool(submitted_move),
    }


def _extract_submit_from_text(text: str) -> str:
    """Fallback: try to find a move the agent mentioned but forgot to tool-call."""

    match = re.search(r"SUBMIT:([a-h][1-8][a-h][1-8][qrbn]?)", text)
    if match:
        return match.group(1)
    return ""

