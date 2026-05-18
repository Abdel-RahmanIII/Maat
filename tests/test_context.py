"""Tests for src.context — ConversationContext."""

from __future__ import annotations

from langchain_core.load import dumpd
from langchain_core.messages import HumanMessage

import src.context as context_module

from src.context import ConversationContext


class _FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if not text.strip():
            return []
        token_count = max(1, (len(text) + 7) // 8)
        return list(range(token_count))


def _patch_tokenizer(monkeypatch) -> None:
    monkeypatch.setattr(
        context_module,
        "_get_tokenizer",
        lambda tokenizer_name: _FakeTokenizer(),
    )

def test_add_get_history(monkeypatch):
    _patch_tokenizer(monkeypatch)
    ctx = ConversationContext()
    ctx.add_turn_messages("generator", [{"role": "user", "content": "hello"}])

    history = ctx.get_history("generator")
    assert len(history) == 1
    assert history[0]["content"] == "hello"

def test_serialization(monkeypatch):
    _patch_tokenizer(monkeypatch)
    ctx = ConversationContext()
    ctx.add_turn_messages("generator", [{"role": "user", "content": "hello"}])
    ctx.add_turn_messages("critic", [{"role": "system", "content": "eval"}])

    data = ctx.to_dict()
    assert "histories" in data
    assert len(data["histories"]["generator"]) == 1

    new_ctx = ConversationContext.from_dict(data)
    gen_hist = new_ctx.get_history("generator")
    critic_hist = new_ctx.get_history("critic")

    assert len(gen_hist) == 1
    assert gen_hist[0]["content"] == "hello"
    assert len(critic_hist) == 1
    assert critic_hist[0]["content"] == "eval"


def test_get_history_uses_token_budget_and_newest_messages_first(monkeypatch):
    _patch_tokenizer(monkeypatch)
    ctx = ConversationContext()
    for idx in range(14):
        content = f"{idx:02d}" + ("x" * 14)
        ctx.add_turn_messages(
            "generator",
            [dumpd(HumanMessage(content=content))],
        )

    history = ctx.get_history("generator", max_tokens=5)

    assert len(history) == 2
    assert isinstance(history[0], HumanMessage)
    assert history[0].content == "12" + ("x" * 14)
    assert history[-1].content == "13" + ("x" * 14)


def test_get_history_respects_tokens_used_so_far(monkeypatch):
    _patch_tokenizer(monkeypatch)
    ctx = ConversationContext()
    for idx in range(3):
        content = f"{idx:02d}" + ("x" * 14)
        ctx.add_turn_messages(
            "generator",
            [dumpd(HumanMessage(content=content))],
        )

    history = ctx.get_history("generator", max_tokens=5, tokens_used_so_far=3)

    assert len(history) == 1
    assert history[0].content == "02" + ("x" * 14)


def test_get_history_can_return_full_history_when_requested(monkeypatch):
    _patch_tokenizer(monkeypatch)
    ctx = ConversationContext()
    ctx.add_turn_messages("generator", [dumpd(HumanMessage(content="first"))])
    ctx.add_turn_messages("generator", [dumpd(HumanMessage(content="second"))])

    history = ctx.get_history("generator", max_tokens=None)

    assert len(history) == 2
    assert [msg.content for msg in history] == ["first", "second"]
