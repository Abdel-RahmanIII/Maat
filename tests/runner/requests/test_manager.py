"""Tests for the RequestsManager."""

import time
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.runner.requests.manager import APIConfig, RequestsManager


def test_api_config_limits():
    config = APIConfig(api_key="test_key", rpm_limit=2, rpd_limit=4)
    
    # Should accept 2 requests in the same minute
    assert config.can_accept() is True
    config.record_usage()
    
    assert config.can_accept() is True
    config.record_usage()
    
    # RPM limit reached
    assert config.can_accept() is False
    
    # Manually reset minute
    config._minute_start = time.time() - 61.0
    config._exhausted_until = 0.0
    
    # Can accept 2 more
    assert config.can_accept() is True
    config.record_usage()
    assert config.can_accept() is True
    config.record_usage()
    
    # Both RPM and RPD reached
    assert config.can_accept() is False
    config._minute_start = time.time() - 61.0
    
    # RPD limits cross-minute
    assert config.can_accept() is False


def test_api_config_exhausted_manual():
    config = APIConfig(api_key="test_key", rpm_limit=10, rpd_limit=100)
    config.mark_exhausted_minute()
    assert config.can_accept() is False
    
    config._exhausted_until = time.time() - 1.0
    assert config.can_accept() is True
    
    config.mark_exhausted_day()
    assert config.can_accept() is False


@patch("src.runner.requests.manager.ChatGoogleGenerativeAI")
def test_manager_routing_success(mock_chat_class):
    # Setup mock LLM response
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="Hello")
    mock_chat_class.return_value = mock_llm

    manager = RequestsManager(api_keys=["key1", "key2"])
    manager.start()
    
    try:
        messages = [HumanMessage(content="Hi")]
        future1 = manager.submit(messages, {"model_name": "gemma-4-31b-it"})
        future2 = manager.submit(messages, {"model_name": "gemma-4-31b-it"})
        
        # Wait for futures
        res1 = future1.result(timeout=2.0)
        res2 = future2.result(timeout=2.0)
        
        assert res1.content == "Hello"
        assert res2.content == "Hello"
        
        # Verify call counts and args
        assert mock_llm.invoke.call_count == 2
    finally:
        manager.stop()


@patch("src.runner.requests.manager.ChatGoogleGenerativeAI")
def test_manager_429_exhausts_key_and_requeues(mock_chat_class):
    from google.api_core.exceptions import ResourceExhausted
    
    mock_llm = MagicMock()
    # Fails first time, succeeds second time
    mock_llm.invoke.side_effect = [ResourceExhausted("Rate Limit Exceeded"), AIMessage(content="Success")]
    mock_chat_class.return_value = mock_llm

    manager = RequestsManager(api_keys=["key1", "key2"])
    manager.start()
    
    try:
        future = manager.submit([HumanMessage(content="Hi")], {})
        res = future.result(timeout=2.0)
        
        assert res.content == "Success"
        assert mock_llm.invoke.call_count == 2
        
        # Verify key1 got exhausted
        exhausted = sum(1 for c in manager._api_configs if not c.can_accept())
        assert exhausted == 1
    finally:
        manager.stop()


@patch("src.runner.requests.manager.ChatGoogleGenerativeAI")
def test_manager_500_requeues_and_fails_after_retries(mock_chat_class):
    from google.api_core.exceptions import InternalServerError
    
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = InternalServerError("Server down")
    mock_chat_class.return_value = mock_llm

    manager = RequestsManager(api_keys=["key1"])
    manager.start()
    
    try:
        future = manager.submit([HumanMessage(content="Hi")], {})
        
        with pytest.raises(InternalServerError):
            future.result(timeout=2.0)
            
        # 1 original try + 3 retries = 4 calls total
        assert mock_llm.invoke.call_count == 4
    finally:
        manager.stop()


@patch("src.runner.requests.manager.ChatGoogleGenerativeAI")
def test_manager_400_fails_immediately(mock_chat_class):
    from google.api_core.exceptions import InvalidArgument
    
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = InvalidArgument("Context too long")
    mock_chat_class.return_value = mock_llm

    manager = RequestsManager(api_keys=["key1"])
    manager.start()
    
    try:
        future = manager.submit([HumanMessage(content="Hi")], {})
        
        with pytest.raises(InvalidArgument):
            future.result(timeout=2.0)
            
        assert mock_llm.invoke.call_count == 1
    finally:
        manager.stop()


def test_manager_global_rpd_exhaustion():
    callback_called = False
    def on_rpd():
        nonlocal callback_called
        callback_called = True

    manager = RequestsManager(api_keys=["key1", "key2"], on_global_rpd_limit=on_rpd)
    # Manually exhaust all keys for the day
    for config in manager._api_configs:
        config.mark_exhausted_day()
        
    manager.start()
    
    try:
        future = manager.submit([HumanMessage(content="Hi")], {})
        
        with pytest.raises(Exception, match="All API keys reached RPD limit"):
            future.result(timeout=2.0)
            
        assert callback_called is True
    finally:
        manager.stop()
