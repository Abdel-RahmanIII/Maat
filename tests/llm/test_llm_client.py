from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from src.config import ModelConfig
from src.llm.llm_client import LoggedModelRunnable, _build_model


def test_build_model_returns_queued_model():
    cfg = ModelConfig(api_key="test_key", model_name="gemma-test")
    model = _build_model(cfg)
    
    # Assert it returns a QueuedChatModel
    from src.runner.requests.queued_model import QueuedChatModel
    assert isinstance(model, QueuedChatModel)
    assert model.model == "gemma-test"
    # Even if config has no api key, it provides a fallback "dummy_key_for_queue"
    cfg_no_key = ModelConfig(api_key="", model_name="test")
    model2 = _build_model(cfg_no_key)
    assert model2.google_api_key.get_secret_value() == "dummy_key_for_queue"


def test_logged_model_runnable_no_retry():
    # Verify that LoggedModelRunnable doesn't swallow exceptions and retry anymore
    mock_runnable = MagicMock()
    mock_runnable.invoke.side_effect = Exception("API Failure")
    
    logged = LoggedModelRunnable(runnable=mock_runnable, model_name="test", has_tools=False)
    
    with pytest.raises(Exception, match="API Failure"):
        logged.invoke("Hello")
        
    # Should only be called exactly once (no retries)
    assert mock_runnable.invoke.call_count == 1


@pytest.mark.anyio
async def test_logged_model_runnable_async_no_retry():
    mock_runnable = AsyncMock()
    mock_runnable.ainvoke.side_effect = Exception("Async API Failure")
    
    logged = LoggedModelRunnable(runnable=mock_runnable, model_name="test", has_tools=False)
    
    with pytest.raises(Exception, match="Async API Failure"):
        await logged.ainvoke("Hello")
        
    assert mock_runnable.ainvoke.call_count == 1
