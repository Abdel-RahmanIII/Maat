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


class ConversationContext:
    """Per-game conversation history, keyed by agent_id.

    Agent IDs:
        ``"generator"``, ``"strategist"``, ``"tactician"``,
        ``"threat_analyst"``, ``"constrained_generator"``, ``"react"``
    """

    def __init__(self) -> None:
        self._histories: dict[str, list[Any]] = {}

    def get_history(self, agent_id: str) -> list[Any]:
        """Return a *copy* of accumulated messages for *agent_id*."""
        return list(self._histories.get(agent_id, []))

    def add_turn_messages(self, agent_id: str, messages: list[Any]) -> None:
        """Append this turn's messages for *agent_id*.

        Only the final successful exchange should be stored — retry
        failures within a turn must NOT be added.
        """
        self._histories.setdefault(agent_id, []).extend(messages)
