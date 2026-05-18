"""Tests for the RequestsManager."""

import time
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from src.config import ModelConfig
from src.runner.requests.manager import APIConfig, RequestsManager


def test_api_config_limits():
    config = APIConfig(api_key="test_key", rpm_limit=2, rpd_limit=4)

    assert config.can_accept() is True
    config.record_usage()

    assert config.can_accept() is True
    config.record_usage()

    assert config.can_accept() is False

    config._minute_start = time.time() - 61.0
    config._exhausted_until = 0.0

    assert config.can_accept() is True
    config.record_usage()
    assert config.can_accept() is True
    config.record_usage()

    assert config.can_accept() is False
    config._minute_start = time.time() - 61.0

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
def test_manager_eager_pool_and_least_rpm_routing(mock_chat_class):
    first_model = MagicMock()
    first_model.invoke.return_value = AIMessage(content="First")
    second_model = MagicMock()
    second_model.invoke.return_value = AIMessage(content="Second")
    mock_chat_class.side_effect = [first_model, second_model]

    cfg = ModelConfig(api_key="key1", api_keys=["key1", "key2"])
    manager = RequestsManager(cfg, request_send_delay=0.0)

    # Models are built eagerly on manager init.
    assert mock_chat_class.call_count == 2
    for call in mock_chat_class.call_args_list:
        assert call.kwargs.get("request_timeout") == pytest.approx(300.0)
    manager.start()

    try:
        messages: list[BaseMessage] = [HumanMessage(content="Hi")]
        res1 = manager.submit(messages).result(timeout=2.0)
        res2 = manager.submit(messages).result(timeout=2.0)

        assert res1.content == "First"
        assert res2.content == "Second"
        assert mock_chat_class.call_count == 2
        assert first_model.invoke.call_count == 1
        assert second_model.invoke.call_count == 1
    finally:
        manager.stop()


@patch("src.runner.requests.manager.ChatGoogleGenerativeAI")
def test_manager_posts_delay_after_each_dispatch(mock_chat_class):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="Done")
    mock_chat_class.return_value = mock_llm

    cfg = ModelConfig(api_key="key1", api_keys=["key1"])
    manager = RequestsManager(cfg, request_send_delay=0.2)

    class FakeThread:
        def __init__(self, target, args=(), daemon=False):
            self._target = target
            self._args = args

        def start(self):
            self._target(*self._args)

    manager.start()

    try:
        with patch("src.runner.requests.manager.threading.Thread") as mock_thread_class:
            mock_thread_class.side_effect = lambda target, args=(), daemon=False: FakeThread(target, args, daemon)
            with patch.object(manager, "_manager_send_sleep") as mock_sleep:
                future = manager.submit([HumanMessage(content="Hi")])
                assert future.result(timeout=2.0).content == "Done"
                mock_sleep.assert_called_once()

        assert mock_llm.invoke.call_count == 1
        assert manager.request_send_delay == pytest.approx(0.2)
    finally:
        manager.stop()



@patch("src.runner.requests.manager.ChatGoogleGenerativeAI")
def test_manager_429_exhausts_key_and_requeues(mock_chat_class):
    from google.api_core.exceptions import ResourceExhausted

    first_model = MagicMock()
    first_model.invoke.side_effect = ResourceExhausted("Rate Limit Exceeded")
    second_model = MagicMock()
    second_model.invoke.return_value = AIMessage(content="Success")
    mock_chat_class.side_effect = [first_model, second_model]

    cfg = ModelConfig(api_key="key1", api_keys=["key1", "key2"])
    manager = RequestsManager(cfg, request_send_delay=0.0)
    manager.start()

    try:
        future = manager.submit([HumanMessage(content="Hi")])
        res = future.result(timeout=2.0)

        assert res.content == "Success"
        assert first_model.invoke.call_count == 1
        assert second_model.invoke.call_count == 1
        exhausted = sum(1 for c in manager._api_configs if not c.can_accept())
        assert exhausted == 1
    finally:
        manager.stop()


@patch("src.runner.requests.manager.ChatGoogleGenerativeAI")
def test_manager_500_requeues_and_fails_after_retries(mock_chat_class):
    from google.api_core.exceptions import InternalServerError

    from src.runner.requests.manager import RequestTerminatedError

    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = InternalServerError("Server down")
    mock_chat_class.return_value = mock_llm

    cfg = ModelConfig(api_key="key1", api_keys=["key1"])
    # Use near-zero backoff so the test finishes quickly
    manager = RequestsManager(
        cfg,
        request_send_delay=0.0,
        max_retries=3,
        backoff_base=0.01,
        backoff_max=0.05,
    )
    manager.start()

    try:
        future = manager.submit([HumanMessage(content="Hi")])

        with pytest.raises(RequestTerminatedError):
            future.result(timeout=5.0)

        # 1 initial + 3 retries = 4 total invocations
        assert mock_llm.invoke.call_count == 4
    finally:
        manager.stop()


@patch("src.runner.requests.manager.ChatGoogleGenerativeAI")
def test_manager_400_fails_immediately(mock_chat_class):
    from google.api_core.exceptions import InvalidArgument

    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = InvalidArgument("Context too long")
    mock_chat_class.return_value = mock_llm

    cfg = ModelConfig(api_key="key1", api_keys=["key1"])
    manager = RequestsManager(cfg, request_send_delay=0.0)
    manager.start()

    try:
        future = manager.submit([HumanMessage(content="Hi")])

        with pytest.raises(InvalidArgument):
            future.result(timeout=2.0)

        assert mock_llm.invoke.call_count == 1
    finally:
        manager.stop()


@patch("src.runner.requests.manager.ChatGoogleGenerativeAI")
def test_manager_timeout_is_treated_as_bad_request(mock_chat_class):
    from google.api_core.exceptions import DeadlineExceeded

    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = DeadlineExceeded("Request timed out")
    mock_chat_class.return_value = mock_llm

    cfg = ModelConfig(api_key="key1", api_keys=["key1"])
    manager = RequestsManager(
        cfg,
        request_send_delay=0.0,
        max_retries=10,
        backoff_base=0.01,
        backoff_max=0.05,
    )
    manager.start()

    try:
        future = manager.submit([HumanMessage(content="Hi")])

        with pytest.raises(DeadlineExceeded):
            future.result(timeout=2.0)

        # Should fail immediately (no retries).
        assert mock_llm.invoke.call_count == 1
    finally:
        manager.stop()


def test_manager_global_rpd_exhaustion():
    callback_called = False

    def on_rpd() -> None:
        nonlocal callback_called
        callback_called = True

    cfg = ModelConfig(api_key="key1", api_keys=["key1", "key2"])
    manager = RequestsManager(cfg, request_send_delay=0.0, on_global_rpd_limit=on_rpd)

    for config in manager._api_configs:
        config.mark_exhausted_day()

    manager.start()

    try:
        future = manager.submit([HumanMessage(content="Hi")])

        with pytest.raises(Exception, match="RequestsManager paused"):
            future.result(timeout=2.0)

        assert callback_called is True
    finally:
        manager.stop()