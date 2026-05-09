"""Queued chat model wrapper for LangChain."""

import asyncio
from typing import Any, List, Optional

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_google_genai import ChatGoogleGenerativeAI

from src.runner.requests.manager import get_global_manager


class QueuedChatModel(ChatGoogleGenerativeAI):
    """
    A drop-in replacement for ChatGoogleGenerativeAI that routes requests
    through the global RequestsManager queue instead of executing them immediately.
    If no global manager is active, it falls back to standard execution.
    """

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Route the request to the global manager queue, or fallback to standard behavior."""
        manager = get_global_manager()

        if manager is None:
            # Fallback for standalone executions outside the Orchestrator
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

        # Standard execution uses the manager
        invoke_kwargs = kwargs.copy()
        if stop is not None:
            invoke_kwargs["stop"] = stop

        # Extract model initialization kwargs from self, filtering out None values
        model_kwargs = {
            k: v for k, v in {
                "model_name": self.model,
                "temperature": self.temperature,
                "max_output_tokens": self.max_output_tokens,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "timeout": self.timeout,
                "max_retries": self.max_retries,
            }.items() if v is not None
        }

        # Submit to the queue and block for the result
        future = manager.submit(messages, model_kwargs=model_kwargs, invoke_kwargs=invoke_kwargs)
        response_message = future.result()

        # Build and return the ChatResult
        generation = ChatGeneration(message=response_message)
        return ChatResult(generations=[generation])

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Route the request asynchronously to the global manager queue."""
        manager = get_global_manager()

        if manager is None:
            return await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)

        invoke_kwargs = kwargs.copy()
        if stop is not None:
            invoke_kwargs["stop"] = stop

        model_kwargs = {
            k: v for k, v in {
                "model_name": self.model,
                "temperature": self.temperature,
                "max_output_tokens": self.max_output_tokens,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "timeout": self.timeout,
                "max_retries": self.max_retries,
            }.items() if v is not None
        }

        # Submit to the queue
        future = manager.submit(messages, model_kwargs=model_kwargs, invoke_kwargs=invoke_kwargs)
        
        # Asyncio wrapper to await the concurrent.futures.Future
        loop = asyncio.get_running_loop()
        response_message = await loop.run_in_executor(None, future.result)

        generation = ChatGeneration(message=response_message)
        return ChatResult(generations=[generation])
