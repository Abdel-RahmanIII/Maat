# LLM Client Module Reference

## File

- `src/llm/llm_client.py`

## Purpose

Single source of truth for LLM model instantiation. All agents call these factory functions rather than constructing models directly.

## APIs

```python
get_model(cfg: ModelConfig | None = None) -> ChatGoogleGenerativeAI
get_model_with_tools(tools: Sequence[BaseTool], cfg: ModelConfig | None = None) -> ChatGoogleGenerativeAI
```

## Configuration

Uses `ModelConfig` from `src/config.py`:

| Field | Default | Description |
|-------|---------|-------------|
| `model_name` | `gemma-4-31b-it` | Google AI Studio model identifier |
| `temperature` | `0.0` | Greedy decoding for reproducibility |
| `max_output_tokens` | `1024` | Maximum response length |
| `api_key` | from `GOOGLE_API_KEY` env var | API authentication |

## Tool Binding

`get_model_with_tools(tools)` calls `model.bind_tools(tools)` to enable LangChain's function-calling protocol. Used by Condition F's ReAct agent.

## Error Handling

- Missing `GOOGLE_API_KEY` → raises `ValueError` with instructions.
