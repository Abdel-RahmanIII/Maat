from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import LLMCallMode, ModelConfig
from src.llm.llm_client import _build_direct_model, beutifyOutput, get_model, invoke_llm


def test_build_direct_model():
    """Test that direct model construction returns ChatGoogleGenerativeAI."""

    cfg = ModelConfig(
        api_key="test_key",
        model_name="gemma-test",
        call_mode=LLMCallMode.DIRECT,
    )

    model = _build_direct_model(cfg)

    assert isinstance(model, ChatGoogleGenerativeAI)
    assert model.model == "gemma-test"


def test_get_model_returns_direct_model():
    """Test that get_model returns a direct LangChain chat model."""

    cfg = ModelConfig(api_key="test_key", call_mode=LLMCallMode.DIRECT)

    model = get_model(cfg)

    assert isinstance(model, ChatGoogleGenerativeAI)


@patch("src.llm.llm_client.ChatGoogleGenerativeAI")
def test_invoke_llm_direct_mode(mock_chat_class):
    """Direct mode should invoke the Google chat model immediately."""

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="Hello")
    mock_chat_class.return_value = mock_llm

    cfg = ModelConfig(
        api_key="test_key",
        model_name="gemma-test",
        call_mode=LLMCallMode.DIRECT,
    )

    response = invoke_llm([HumanMessage(content="Hi")], cfg)

    assert response.content == "Hello"
    mock_chat_class.assert_called_once()
    mock_llm.invoke.assert_called_once()


@patch("src.llm.llm_client.ChatGoogleGenerativeAI")
def test_invoke_llm_direct_mode_cleans_structured_content(mock_chat_class):
    """Structured content blocks should be flattened into prompt-safe text."""

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(
        content=[
            {"type": "thinking", "thinking": ""},
            {
                "type": "text",
                "text": '{\n  "error_summary": "The white knight on b3 cannot move to a4.",\n  "explanation": "Knights move in an \'L\' shape."\n}',
            },
        ]
    )
    mock_chat_class.return_value = mock_llm

    cfg = ModelConfig(
        api_key="test_key",
        model_name="gemma-test",
        call_mode=LLMCallMode.DIRECT,
    )

    response = invoke_llm([HumanMessage(content="Hi")], cfg)

    assert response.content == '{\n  "error_summary": "The white knight on b3 cannot move to a4.",\n  "explanation": "Knights move in an \'L\' shape."\n}'


def test_beutify_output_extracts_text_blocks():
    """beutifyOutput should ignore thinking blocks and keep user-facing text."""

    content = [
        {"type": "thinking", "thinking": ""},
        {"type": "text", "text": "First line"},
        {"type": "text", "text": "Second line"},
    ]

    assert beutifyOutput(content) == "First line\nSecond line"


@patch("src.llm.llm_client.ChatGoogleGenerativeAI")
def test_invoke_llm_logs_prompt_before_direct_call(mock_chat_class, caplog):
    """Prompt text should be logged before a direct model invocation."""

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="Hello")
    mock_chat_class.return_value = mock_llm

    cfg = ModelConfig(
        api_key="test_key",
        model_name="gemma-test",
        call_mode=LLMCallMode.DIRECT,
    )

    with caplog.at_level("INFO", logger="src.llm.llm_client"):
        invoke_llm([HumanMessage(content="Hi")], cfg)

    assert any("[LLM] DIRECT PROMPT" in record.message for record in caplog.records)
    assert any("1. human: Hi" in record.message for record in caplog.records)


def test_invoke_llm_queue_mode_uses_manager():
    """Queue mode should submit the request to the active RequestsManager."""

    cfg = ModelConfig(
        api_key="test_key",
        api_keys=["test_key"],
        model_name="gemma-test",
        call_mode=LLMCallMode.QUEUE,
    )

    mock_manager = MagicMock()
    mock_manager.model_config = cfg
    mock_future = MagicMock()
    mock_future.result.return_value = AIMessage(content="Queued")
    mock_manager.submit.return_value = mock_future

    with patch("src.llm.llm_client.get_global_manager", return_value=mock_manager):
        response = invoke_llm([HumanMessage(content="Hi")], cfg)

    assert response.content == "Queued"
    mock_manager.submit.assert_called_once()