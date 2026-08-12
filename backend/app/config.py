"""
Application settings loaded from environment variables.
Uses pydantic-settings for type-safe configuration management.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All application configuration — sourced from .env or environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---- Application ----
    app_env: Literal["development", "production", "test"] = "development"
    app_debug: bool = False
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    secret_key: str = Field(default="dev-secret-change-me", min_length=16)

    # ---- Database ----
    database_url: str = "sqlite+aiosqlite:///./data/music_recommender.db"

    # ---- LLM Provider ----
    llm_provider: Literal["gemini", "openai", "anthropic", "ollama"] = "gemini"

    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"

    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    anthropic_api_key: str = ""
    anthropic_model: str = "claude-3-5-haiku-20241022"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"

    # ---- Recommendation Engine ----
    default_recommendation_count: int = 5
    max_recommendation_count: int = 20
    content_weight: float = 0.6
    mood_weight: float = 0.3
    popularity_weight: float = 0.1

    # ---- Dataset ----
    dataset_path: str = "./data/songs.csv"

    # ---- CORS ----
    cors_origins: str = "http://localhost:8000,http://127.0.0.1:8000"

    # ---- Logging ----
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    @field_validator("content_weight", "mood_weight", "popularity_weight")
    @classmethod
    def validate_weight(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("Weights must be between 0.0 and 1.0")
        return v

    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",")]

    @property
    def is_development(self) -> bool:
        return self.app_env == "development"

    @property
    def active_llm_api_key(self) -> str:
        """Return the API key for the currently configured provider."""
        return {
            "gemini": self.gemini_api_key,
            "openai": self.openai_api_key,
            "anthropic": self.anthropic_api_key,
            "ollama": "",  # Ollama runs locally, no key needed
        }.get(self.llm_provider, "")


@lru_cache
def get_settings() -> Settings:
    """Cached settings instance — read once at startup."""
    return Settings()
