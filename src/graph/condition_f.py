"""Condition F — ReAct + Tool Calling (LangGraph implementation).

The agent autonomously reasons, optionally invokes chess analysis tools,
and eventually calls ``submit_move``.  The submitted move is checked
against python-chess ground truth.  No retries after submission.

Graph topology::

    START → agent_reason → (route)
        ├─ has tool calls with submit_move  → ground_truth
        ├─ has tool calls (no submit)       → execute_tools → (route)
        │       ├─ steps < max  → agent_reason
        │       └─ steps >= max → forfeit
        ├─ text-fallback submit found       → ground_truth
        ├─ steps < max                      → agent_reason
        └─ steps >= max                     → forfeit

    ground_truth → accept  (valid)
                 → forfeit (invalid)
"""

from __future__ import annotations

import time
from typing import Any, cast

from langchain_core.messages import AIMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.agents.react_agent import (
    build_react_messages,
    execute_tool_calls,
    extract_submit_from_text,
)
from src.config import ModelConfig
from src.context import ConversationContext
from src.graph.base_graph import parse_and_validate, snapshot_turn_result
from src.llm.llm_client import invoke_llm_timed
from src.state import InputMode, TurnState, create_initial_turn_state
from src.tools.chess_tools import get_tools_for_input_mode


# ── Graph builder ─────────────────────────────────────────────────────────


def build_graph(
    model_config: ModelConfig | None = None,
    input_mode: InputMode = "fen",
    context: ConversationContext | None = None,
) -> CompiledStateGraph:
    """Build and return the compiled Condition F graph."""

    selected_tools = get_tools_for_input_mode(input_mode)
    tool_map: dict[str, Any] = {tool.name: tool for tool in selected_tools}

    # ── Nodes ─────────────────────────────────────────────────────────

    def _agent_reason_node(state: TurnState) -> dict[str, Any]:
        """Invoke the tool-bound LLM with the current message history."""

        messages = list(state["messages"])
        response, elapsed_ms = invoke_llm_timed(messages, model_config, tools=selected_tools)
        usage = response.usage_metadata or {}

        # Capture raw AI response text
        response_text = (
            response.content
            if isinstance(response.content, str)
            else str(response.content)
        )
        prev_raw = state.get("raw_llm_response", "")
        new_raw = (
            f"{prev_raw}\n---\n{response_text}" if prev_raw else response_text
        )

        # Append the AI response to messages
        messages.append(response)

        return {
            "messages": messages,
            "react_steps_taken": state["react_steps_taken"] + 1,
            "llm_calls_this_turn": state["llm_calls_this_turn"] + 1,
            "tokens_this_turn": (
                state["tokens_this_turn"]
                + usage.get("input_tokens", 0)
                + usage.get("output_tokens", 0)
            ),
            "prompt_token_count": (
                state["prompt_token_count"]
                + usage.get("input_tokens", 0)
            ),
            "wall_clock_ms": state["wall_clock_ms"] + elapsed_ms,
            "raw_llm_response": new_raw,
        }

    def _execute_tools_node(state: TurnState) -> dict[str, Any]:
        """Execute non-submit tool calls and append results to messages."""

        messages = list(state["messages"])
        # The last message should be the AIMessage with tool_calls
        last_msg = messages[-1]
        ai_tool_calls = getattr(last_msg, "tool_calls", []) or []

        # Execute tool calls
        tool_start = time.perf_counter()
        tool_messages = execute_tool_calls(ai_tool_calls, tool_map)
        tool_elapsed_ms = (time.perf_counter() - tool_start) * 1000
        messages.extend(tool_messages)

        # Build tool call log entries
        tool_calls_log = list(state["tool_calls"])
        for i, tc in enumerate(ai_tool_calls):
            entry: dict[str, Any] = {
                "tool": tc["name"],
                "args": tc["args"],
                "step": state["react_steps_taken"],
            }
            if i < len(tool_messages):
                content = tool_messages[i].content
                entry["result"] = (
                    content if isinstance(content, str) else str(content)
                )
            tool_calls_log.append(entry)

        return {
            "messages": messages,
            "tool_calls": tool_calls_log,
            "wall_clock_ms": state["wall_clock_ms"] + tool_elapsed_ms,
        }

    def _ground_truth_node(state: TurnState) -> dict[str, Any]:
        """Extract submitted move and validate against python-chess."""

        submitted = state.get("proposed_move", "")

        # If proposed_move wasn't set yet, try to find it from messages
        if not submitted:
            messages = state["messages"]
            for msg in reversed(messages):
                ai_tool_calls = getattr(msg, "tool_calls", None) or []
                for tc in ai_tool_calls:
                    if tc["name"] == "submit_move":
                        submitted = tc["args"].get("uci_move", "")
                        break
                if submitted:
                    break

        if not submitted:
            # Try text fallback from the last AI message
            for msg in reversed(state["messages"]):
                if isinstance(msg, AIMessage):
                    text = (
                        msg.content
                        if isinstance(msg.content, str)
                        else str(msg.content)
                    )
                    submitted = extract_submit_from_text(text)
                    if submitted:
                        break

        pv = parse_and_validate(submitted, state["board_fen"])

        error_types = list(state["error_types"])
        if not pv["is_valid"] and pv["error_type"]:
            error_types.append(pv["error_type"])

        return {
            "proposed_move": pv["proposed_move"],
            "is_valid": pv["is_valid"],
            "first_try_valid": pv["is_valid"],
            "total_attempts": 1,
            "error_types": error_types,
        }

    def _accept_node(state: TurnState) -> dict[str, Any]:
        return {
            "game_status": "ongoing",
            "turn_results": state["turn_results"]
            + [snapshot_turn_result(state)],
        }

    def _forfeit_node(state: TurnState) -> dict[str, Any]:
        # If no move was ever proposed, record NO_OUTPUT
        error_types = list(state["error_types"])
        if not state.get("proposed_move"):
            error_types.append("NO_OUTPUT")
        return {
            "game_status": "forfeit",
            "is_valid": False,
            "first_try_valid": False,
            "total_attempts": state.get("total_attempts", 0) or 1,
            "error_types": error_types,
            "turn_results": state["turn_results"]
            + [snapshot_turn_result(state)],
        }

    # ── Routing ───────────────────────────────────────────────────────

    def _route_after_reason(state: TurnState) -> str:
        """Decide next step after the agent produces a response."""

        messages = state["messages"]
        last_msg = messages[-1] if messages else None
        ai_tool_calls = getattr(last_msg, "tool_calls", None) or []

        # Check for submit_move in tool calls
        has_submit = any(tc["name"] == "submit_move" for tc in ai_tool_calls)

        if ai_tool_calls and has_submit:
            # Extract the submitted move for ground-truth
            for tc in ai_tool_calls:
                if tc["name"] == "submit_move":
                    # Store proposed_move for ground_truth_node
                    return "ground_truth_with_submit"
            return "ground_truth"

        if ai_tool_calls:
            # Has non-submit tool calls — execute them
            return "execute_tools"

        # No tool calls — check for text-fallback submit
        if last_msg is not None:
            text = (
                last_msg.content
                if isinstance(getattr(last_msg, "content", None), str)
                else str(getattr(last_msg, "content", ""))
            )
            if extract_submit_from_text(text):
                return "ground_truth"

        # No submit — check step budget
        if state["react_steps_taken"] >= state["max_react_steps"]:
            return "forfeit"

        return "agent_reason"

    def _route_after_tools(state: TurnState) -> str:
        """Decide next step after executing tools."""

        # Check if submit_move was among the executed tools
        tool_calls = state.get("tool_calls", [])
        if tool_calls:
            last_step = state["react_steps_taken"]
            for tc in tool_calls:
                if tc.get("step") == last_step and tc.get("tool") == "submit_move":
                    return "ground_truth"

        # Check step budget
        if state["react_steps_taken"] >= state["max_react_steps"]:
            return "forfeit"

        return "agent_reason"

    def _route_after_ground_truth(state: TurnState) -> str:
        if state["is_valid"]:
            return "accept"
        return "forfeit"

    # ── Intermediate node to extract submit_move before ground truth ──

    def _extract_submit_node(state: TurnState) -> dict[str, Any]:
        """Extract the submitted move from the last AI tool call."""

        messages = state["messages"]
        submitted = ""
        for msg in reversed(messages):
            ai_tool_calls = getattr(msg, "tool_calls", None) or []
            for tc in ai_tool_calls:
                if tc["name"] == "submit_move":
                    submitted = tc["args"].get("uci_move", "")
                    break
            if submitted:
                break

        # Also execute all tool calls (including submit_move) to keep
        # the message history consistent
        last_msg = messages[-1]
        ai_tool_calls = getattr(last_msg, "tool_calls", []) or []
        tool_start = time.perf_counter()
        tool_messages = execute_tool_calls(ai_tool_calls, tool_map)
        tool_elapsed_ms = (time.perf_counter() - tool_start) * 1000

        updated_messages = list(messages) + list(tool_messages)

        # Build tool call log entries
        tool_calls_log = list(state["tool_calls"])
        for i, tc in enumerate(ai_tool_calls):
            entry: dict[str, Any] = {
                "tool": tc["name"],
                "args": tc["args"],
                "step": state["react_steps_taken"],
            }
            if i < len(tool_messages):
                content = tool_messages[i].content
                entry["result"] = (
                    content if isinstance(content, str) else str(content)
                )
            tool_calls_log.append(entry)

        return {
            "messages": updated_messages,
            "proposed_move": submitted,
            "tool_calls": tool_calls_log,
            "wall_clock_ms": state["wall_clock_ms"] + tool_elapsed_ms,
        }

    # ── Assemble graph ────────────────────────────────────────────────

    graph = StateGraph(TurnState)

    graph.add_node("agent_reason", _agent_reason_node)
    graph.add_node("execute_tools", _execute_tools_node)
    graph.add_node("extract_submit", _extract_submit_node)
    graph.add_node("ground_truth", _ground_truth_node)
    graph.add_node("accept", _accept_node)
    graph.add_node("forfeit", _forfeit_node)

    graph.set_entry_point("agent_reason")

    graph.add_conditional_edges(
        "agent_reason",
        _route_after_reason,
        {
            "ground_truth_with_submit": "extract_submit",
            "ground_truth": "ground_truth",
            "execute_tools": "execute_tools",
            "agent_reason": "agent_reason",
            "forfeit": "forfeit",
        },
    )
    graph.add_edge("extract_submit", "ground_truth")
    graph.add_conditional_edges(
        "execute_tools",
        _route_after_tools,
        {
            "ground_truth": "ground_truth",
            "agent_reason": "agent_reason",
            "forfeit": "forfeit",
        },
    )
    graph.add_conditional_edges(
        "ground_truth",
        _route_after_ground_truth,
        {"accept": "accept", "forfeit": "forfeit"},
    )
    graph.add_edge("accept", END)
    graph.add_edge("forfeit", END)

    return graph.compile()


# ── Public entry point ────────────────────────────────────────────────────


def run_condition_f(
    *,
    fen: str,
    move_history: list[str] | None = None,
    move_number: int = 1,
    game_id: str = "",
    input_mode: InputMode = "fen",
    max_steps: int = 6,
    model_config: ModelConfig | None = None,
    context: ConversationContext | None = None,
) -> TurnState:
    """Execute one turn under Condition F.

    Builds and runs the ReAct LangGraph with think → tool → observe
    cycles, bounded by *max_steps*.
    """

    state = create_initial_turn_state(
        board_fen=fen,
        game_id=game_id,
        condition="F",
        input_mode=input_mode,
        move_history=list(move_history or []),
        move_number=move_number,
        max_retries=0,
    )
    state["generation_strategy"] = "generator_only"  # ReAct is its own paradigm
    state["max_react_steps"] = max_steps

    # Build initial messages for the agent
    history = (
        context.get_history(
            "react",
            tokens_used_so_far=state["prompt_token_count"],
            tokenizer_name=model_config.model_name if model_config else None,
        )
        if context
        else None
    )
    state["messages"] = build_react_messages(
        fen=fen,
        move_history=list(move_history or []),
        input_mode=input_mode,
        conversation_history=history,
    )

    compiled = build_graph(model_config, input_mode, context)
    result = cast(TurnState, compiled.invoke(state))

    # Persist the full unified message thread for next turn's context only if move is valid
    if context and result.get("messages") and result.get("is_valid"):
        context.add_turn_messages("react", list(result["messages"]))

    return result
