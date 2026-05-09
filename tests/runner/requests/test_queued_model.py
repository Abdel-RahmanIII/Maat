import asyncio
from concurrent.futures import Future
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from pydantic import SecretStr

from src.runner.requests.queued_model import QueuedChatModel
from src.runner.requests.manager import set_global_manager


@pytest.fixture(autouse=True)
def reset_global_manager():
    set_global_manager(None)
    yield
    set_global_manager(None)


def test_queued_chat_model_fallback():
    # If no global manager, it should fall back to ChatGoogleGenerativeAI's _generate
    model = QueuedChatModel(model="gemma-4-31b-it", google_api_key=SecretStr("fake"))
    
    from langchain_core.outputs import ChatResult, ChatGeneration
    with patch("langchain_google_genai.ChatGoogleGenerativeAI._generate") as mock_generate:
        mock_generate.return_value = ChatResult(generations=[ChatGeneration(message=AIMessage(content="Fallback"))])
        model.invoke("Hello")
        
        # Verify the fallback was called
        mock_generate.assert_called_once()


def test_queued_chat_model_routing():
    # If global manager exists, it should route to the manager
    mock_manager = MagicMock()
    mock_future = Future()
    mock_future.set_result(AIMessage(content="Hi there"))
    mock_manager.submit.return_value = mock_future
    
    set_global_manager(mock_manager)
    
    model = QueuedChatModel(model="gemma-4-31b-it", google_api_key=SecretStr("fake"), temperature=0.5)
    
    response = model.invoke([HumanMessage(content="Hello")])
    
    assert response.content == "Hi there"
    mock_manager.submit.assert_called_once()
    
    args, kwargs = mock_manager.submit.call_args
    assert len(args[0]) == 1
    assert args[0][0].content == "Hello"
    assert kwargs["model_kwargs"]["temperature"] == 0.5


@pytest.mark.anyio
async def test_queued_chat_model_async_routing():
    mock_manager = MagicMock()
    mock_future = Future()
    mock_future.set_result(AIMessage(content="Async Hi there"))
    mock_manager.submit.return_value = mock_future
    
    set_global_manager(mock_manager)
    
    model = QueuedChatModel(model="gemma-4-31b-it", google_api_key=SecretStr("fake"))
    
    response = await model.ainvoke([HumanMessage(content="Async Hello")])
    
    assert response.content == "Async Hi there"
    mock_manager.submit.assert_called_once()
