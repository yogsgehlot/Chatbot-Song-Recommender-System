"""
Dependency injection container.
Manages singleton instances of heavyweight objects (LLM client, recommendation engine).
Objects are initialized once at startup and reused across requests.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from app.ai.llm_client import BaseLLMProvider, create_llm_provider
from app.config import get_settings
from app.recommender.engine import RecommendationEngine
from app.services.chat_service import ChatService
from app.services.recommendation_service import RecommendationService

logger = logging.getLogger(__name__)

# Module-level singletons — set during app startup
_llm_provider: BaseLLMProvider | None = None
_recommendation_engine: RecommendationEngine | None = None
_chat_service: ChatService | None = None
_recommendation_service: RecommendationService | None = None


def initialize_dependencies() -> None:
    """
    Called once at application startup.
    Initializes the LLM provider and recommendation engine.
    """
    global _llm_provider, _recommendation_engine, _chat_service, _recommendation_service

    settings = get_settings()

    # Load recommendation engine
    logger.info("Loading recommendation engine from: %s", settings.dataset_path)
    try:
        _recommendation_engine = RecommendationEngine.load(settings.dataset_path)
        logger.info("Recommendation engine loaded: %d songs", _recommendation_engine.catalog_size)
    except FileNotFoundError:
        logger.error(
            "Dataset not found at %s. "
            "Run: python scripts/generate_sample_data.py",
            settings.dataset_path,
        )
        raise

    # Initialize LLM provider
    logger.info("Initializing LLM provider: %s", settings.llm_provider)
    try:
        _llm_provider = create_llm_provider(settings)
    except Exception as e:
        logger.error("Failed to initialize LLM provider: %s", e)
        raise

    # Build services
    _recommendation_service = RecommendationService(_recommendation_engine)
    _chat_service = ChatService(_llm_provider, _recommendation_engine)

    logger.info("All dependencies initialized successfully.")


def get_chat_service() -> ChatService:
    """FastAPI dependency for the chat service."""
    if _chat_service is None:
        raise RuntimeError("ChatService not initialized. Application startup incomplete.")
    return _chat_service


def get_recommendation_service() -> RecommendationService:
    """FastAPI dependency for the recommendation service."""
    if _recommendation_service is None:
        raise RuntimeError("RecommendationService not initialized.")
    return _recommendation_service


def get_llm_provider() -> BaseLLMProvider:
    """FastAPI dependency for the LLM provider."""
    if _llm_provider is None:
        raise RuntimeError("LLM provider not initialized.")
    return _llm_provider


def get_engine() -> RecommendationEngine:
    """FastAPI dependency for the recommendation engine."""
    if _recommendation_engine is None:
        raise RuntimeError("Recommendation engine not initialized.")
    return _recommendation_engine
