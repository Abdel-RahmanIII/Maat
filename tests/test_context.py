"""Tests for src.context — ConversationContext."""

from src.context import ConversationContext

def test_add_get_history():
    ctx = ConversationContext()
    ctx.add_turn_messages("generator", [{"role": "user", "content": "hello"}])
    
    history = ctx.get_history("generator")
    assert len(history) == 1
    assert history[0]["content"] == "hello"

def test_serialization():
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
