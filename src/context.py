"""ConversationContext — per-game conversation history for multi-turn consistency.

Created once per game (Exp 2/3) or once per puzzle (Exp 1).  Maintains
per-agent message histories so each agent type sees only its own prior
exchanges across turns.

Design decisions:
- Per-agent histories: each agent type sees only its own prior exchanges.
- Critic and Explainer are EXCLUDED (they evaluate single moves in isolation).
- Only the final successful exchange per turn is stored (retry failures excluded).
"""

from __future__ import annotations

from typing import Any

from langchain_core.load import dumpd, load


DEFAULT_HISTORY_TOKEN_LIMIT = 250000
DEFAULT_TOKENIZER_NAME = "google/gemma-4-31b-it"


def _resolve_tokenizer_name(tokenizer_name: str | None) -> str:
    """Return a Hugging Face tokenizer identifier for the requested model."""

    if not tokenizer_name:
        return DEFAULT_TOKENIZER_NAME
    if "/" not in tokenizer_name and tokenizer_name.startswith("gemma-4"):
        return f"google/{tokenizer_name}"
    return tokenizer_name


def _normalize_history_item(message: Any) -> Any:
    """Return a prompt-safe history item while preserving unknown payloads."""

    if isinstance(message, dict):
        try:
            return load(message)
        except Exception:
            return message
    return message


def _estimate_message_tokens(message: Any, tokenizer_name: str) -> int:
    """Estimate token usage for a stored history item using a local heuristic.

    The heuristic counts whitespace-separated words as tokens, which is
    sufficient for budgeting historical context.
    """

    content = getattr(message, "content", message)
    if isinstance(content, list):
        text = " ".join(str(part) for part in content)
    else:
        text = str(content)

    text = text.strip()
    if not text:
        return 0

    # Simple whitespace-based token estimate; keep at least 1 token.
    return max(1, len(text.split()))


class ConversationContext:
    """Per-game conversation history, keyed by agent_id.

    Agent IDs:
        ``"generator"``, ``"strategist"``, ``"tactician"``,
        ``"observer"``, ``"executor"``, ``"react"``
    """

    def __init__(self) -> None:
        self._histories: dict[str, list[Any]] = {}

    def get_history(
        self,
        agent_id: str,
        *,
        max_tokens: int | None = DEFAULT_HISTORY_TOKEN_LIMIT,
        tokens_used_so_far: int = 0,
        tokenizer_name: str | None = DEFAULT_TOKENIZER_NAME,
    ) -> list[Any]:
        """Return a prompt-safe, token-bounded copy of accumulated messages.

        Messages are returned from newest to oldest until the remaining token
        budget is exhausted. Pass ``max_tokens=None`` to return the full
        history. ``tokens_used_so_far`` is the number of tokens already spent
        in the current prompt assembly and is subtracted from the budget.
        """

        history = self._histories.get(agent_id, [])
        if max_tokens is None:
            return [_normalize_history_item(msg) for msg in history]

        remaining_budget = max(max_tokens - max(tokens_used_so_far, 0), 0)
        if remaining_budget <= 0:
            return []

        selected: list[Any] = []
        used_tokens = 0
        for msg in reversed(history):
            normalized = _normalize_history_item(msg)
            msg_tokens = _estimate_message_tokens(
                normalized,
                _resolve_tokenizer_name(tokenizer_name),
            )
            if msg_tokens > remaining_budget - used_tokens:
                break
            selected.append(normalized)
            used_tokens += msg_tokens

        selected.reverse()
        return selected

    def add_turn_messages(self, agent_id: str, messages: list[Any]) -> None:
        """Append this turn's messages for *agent_id*.

        Only the final successful exchange should be stored — retry
        failures within a turn must NOT be added.
        """
        self._histories.setdefault(agent_id, []).extend(messages)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the conversation histories to a dictionary.

        Uses LangChain's ``dumpd`` to properly serialize BaseMessage
        subclasses (HumanMessage, AIMessage, etc.) into JSON-safe dicts.
        """
        return {
            "histories": {
                agent_id: [dumpd(msg) for msg in msgs]
                for agent_id, msgs in self._histories.items()
            }
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConversationContext:
        """Deserialize from a dictionary.

        Uses LangChain's ``load`` to reconstruct BaseMessage objects
        from their serialized dict representations.
        """
        instance = cls()
        raw_histories = data.get("histories", {})
        instance._histories = {
            agent_id: [load(msg) for msg in msgs]
            for agent_id, msgs in raw_histories.items()
        }
        return instance
