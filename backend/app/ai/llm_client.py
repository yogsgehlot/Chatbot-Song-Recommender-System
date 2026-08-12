"""
LLM Provider Abstraction Layer.

Supports: Gemini, OpenAI, Anthropic, Ollama (local).
The active provider is determined by the LLM_PROVIDER environment variable.

Architecture principle:
- The LLM understands language and generates explanations.
- The LLM does NOT generate song titles, artists, or catalog data.
- All song recommendations come from the local catalog via the recommendation engine.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


# =============================================================================
# Abstract Interface
# =============================================================================


class BaseLLMProvider(ABC):
    """Abstract base class for all LLM providers."""

    @abstractmethod
    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        """Send a list of chat messages and return the text response."""
        ...

    @abstractmethod
    def structured_output(
        self, messages: list[dict[str, str]], schema: type[BaseModel]
    ) -> BaseModel:
        """
        Send messages and return a structured response parsed into the given schema.
        Raises ValueError if the model output cannot be parsed.
        """
        ...

    def build_system_message(self, content: str) -> dict[str, str]:
        return {"role": "system", "content": content}

    def build_user_message(self, content: str) -> dict[str, str]:
        return {"role": "user", "content": content}

    def build_assistant_message(self, content: str) -> dict[str, str]:
        return {"role": "assistant", "content": content}


# =============================================================================
# Gemini Provider
# =============================================================================


class GeminiProvider(BaseLLMProvider):
    """Google Gemini via langchain-google-genai."""

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash") -> None:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as e:
            raise ImportError(
                "langchain-google-genai is required for Gemini. "
                "Install it with: pip install langchain-google-genai"
            ) from e

        self._llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=0.7,
        )
        logger.info("Gemini provider initialized with model=%s", model)

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        lc_messages = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "user":
                lc_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))

        response = self._llm.invoke(lc_messages)
        return str(response.content)

    def structured_output(
        self, messages: list[dict[str, str]], schema: type[BaseModel]
    ) -> BaseModel:
        structured_llm = self._llm.with_structured_output(schema)
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        lc_messages = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "user":
                lc_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))

        result = structured_llm.invoke(lc_messages)
        if not isinstance(result, schema):
            raise ValueError(f"Expected {schema.__name__}, got {type(result).__name__}")
        return result


# =============================================================================
# OpenAI Provider
# =============================================================================


class OpenAIProvider(BaseLLMProvider):
    """OpenAI via langchain-openai."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as e:
            raise ImportError(
                "langchain-openai is required for OpenAI. "
                "Install it with: pip install langchain-openai"
            ) from e

        self._llm = ChatOpenAI(model=model, api_key=api_key, temperature=0.7)
        logger.info("OpenAI provider initialized with model=%s", model)

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        lc_messages = []
        for msg in messages:
            role, content = msg["role"], msg["content"]
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "user":
                lc_messages.append(HumanMessage(content=content))
            else:
                lc_messages.append(AIMessage(content=content))

        return str(self._llm.invoke(lc_messages).content)

    def structured_output(
        self, messages: list[dict[str, str]], schema: type[BaseModel]
    ) -> BaseModel:
        structured_llm = self._llm.with_structured_output(schema)
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        lc_messages = []
        for msg in messages:
            role, content = msg["role"], msg["content"]
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "user":
                lc_messages.append(HumanMessage(content=content))
            else:
                lc_messages.append(AIMessage(content=content))

        result = structured_llm.invoke(lc_messages)
        if not isinstance(result, schema):
            raise ValueError(f"Expected {schema.__name__}, got {type(result).__name__}")
        return result


# =============================================================================
# Anthropic Provider
# =============================================================================


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude via langchain-anthropic."""

    def __init__(self, api_key: str, model: str = "claude-3-5-haiku-20241022") -> None:
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as e:
            raise ImportError(
                "langchain-anthropic is required. "
                "Install it with: pip install langchain-anthropic"
            ) from e

        self._llm = ChatAnthropic(model=model, api_key=api_key, temperature=0.7)
        logger.info("Anthropic provider initialized with model=%s", model)

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        lc_messages = []
        for msg in messages:
            role, content = msg["role"], msg["content"]
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "user":
                lc_messages.append(HumanMessage(content=content))
            else:
                lc_messages.append(AIMessage(content=content))

        return str(self._llm.invoke(lc_messages).content)

    def structured_output(
        self, messages: list[dict[str, str]], schema: type[BaseModel]
    ) -> BaseModel:
        structured_llm = self._llm.with_structured_output(schema)
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        lc_messages = []
        for msg in messages:
            role, content = msg["role"], msg["content"]
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "user":
                lc_messages.append(HumanMessage(content=content))
            else:
                lc_messages.append(AIMessage(content=content))

        result = structured_llm.invoke(lc_messages)
        if not isinstance(result, schema):
            raise ValueError(f"Expected {schema.__name__}, got {type(result).__name__}")
        return result


# =============================================================================
# Ollama Provider (local)
# =============================================================================


class OllamaProvider(BaseLLMProvider):
    """Local Ollama models — no API key required."""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2") -> None:
        try:
            from langchain_community.chat_models import ChatOllama
        except ImportError as e:
            raise ImportError(
                "langchain-community is required for Ollama. "
                "Install it with: pip install langchain-community"
            ) from e

        self._llm = ChatOllama(base_url=base_url, model=model, temperature=0.7)
        logger.info("Ollama provider initialized with model=%s at %s", model, base_url)

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        lc_messages = []
        for msg in messages:
            role, content = msg["role"], msg["content"]
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "user":
                lc_messages.append(HumanMessage(content=content))
            else:
                lc_messages.append(AIMessage(content=content))

        return str(self._llm.invoke(lc_messages).content)

    def structured_output(
        self, messages: list[dict[str, str]], schema: type[BaseModel]
    ) -> BaseModel:
        structured_llm = self._llm.with_structured_output(schema)
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        lc_messages = []
        for msg in messages:
            role, content = msg["role"], msg["content"]
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "user":
                lc_messages.append(HumanMessage(content=content))
            else:
                lc_messages.append(AIMessage(content=content))

        result = structured_llm.invoke(lc_messages)
        if not isinstance(result, schema):
            raise ValueError(f"Expected {schema.__name__}, got {type(result).__name__}")
        return result


# =============================================================================
# Factory
# =============================================================================


def create_llm_provider(settings: Any | None = None) -> BaseLLMProvider:
    """
    Factory function that creates the appropriate LLM provider based on config.
    Falls back gracefully if the primary provider fails to initialize.
    """
    if settings is None:
        from app.config import get_settings
        settings = get_settings()

    provider_name = settings.llm_provider
    logger.info("Creating LLM provider: %s", provider_name)

    try:
        if provider_name == "gemini":
            if not settings.gemini_api_key:
                raise ValueError("GEMINI_API_KEY is not set in environment")
            return GeminiProvider(
                api_key=settings.gemini_api_key,
                model=settings.gemini_model,
            )
        elif provider_name == "openai":
            if not settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY is not set in environment")
            return OpenAIProvider(
                api_key=settings.openai_api_key,
                model=settings.openai_model,
            )
        elif provider_name == "anthropic":
            if not settings.anthropic_api_key:
                raise ValueError("ANTHROPIC_API_KEY is not set in environment")
            return AnthropicProvider(
                api_key=settings.anthropic_api_key,
                model=settings.anthropic_model,
            )
        elif provider_name == "ollama":
            return OllamaProvider(
                base_url=settings.ollama_base_url,
                model=settings.ollama_model,
            )
        else:
            raise ValueError(f"Unknown LLM provider: {provider_name!r}")
    except Exception as e:
        logger.error("Failed to initialize LLM provider %r: %s", provider_name, e)
        raise
