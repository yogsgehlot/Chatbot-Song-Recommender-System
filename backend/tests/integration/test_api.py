"""
Integration tests for FastAPI REST API endpoints.
Tests health check, catalog search, single song, similar songs, feedback, and mock chat.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.ai.llm_client import BaseLLMProvider
from app.config import Settings, get_settings
from app.db.session import get_db
from app.dependencies import get_chat_service, get_engine
from app.main import app
from app.models.database import Base, Song
from app.models.schemas import MusicPreference
from app.recommender.engine import RecommendationEngine
from app.services.chat_service import ChatService
from app.services.recommendation_service import RecommendationService



# =============================================================================
# Mocks & Test Config
# =============================================================================


class MockLLMProvider(BaseLLMProvider):
    """Mock LLM that returns deterministic responses to avoid hitting external APIs."""

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        return "Here is a mock recommendation response!"

    def structured_output(
        self, messages: list[dict[str, str]], schema: type[Any]
    ) -> Any:
        # Detect if it's looking for IntentClassification or MusicPreference
        schema_name = schema.__name__
        if schema_name == "IntentClassification":
            from app.models.schemas import IntentClassification
            return IntentClassification(
                intent="recommendation",
                confidence=0.95,
                reasoning="User wants calm music.",
            )
        elif schema_name == "MusicPreference":
            return MusicPreference(
                mood="calm",
                language="english",
                genre=None,
                is_refinement=False,
            )
        elif schema_name == "SongExplanation":
            from app.models.schemas import SongExplanation
            return SongExplanation(
                reasons=["Matches your study mood", "Chill background tempo", "Low energy focus track"]
            )
        raise ValueError(f"Unknown mock schema: {schema_name}")


# =============================================================================
# Fixtures
# =============================================================================


@pytest_asyncio.fixture(name="test_db")
async def test_db_fixture() -> AsyncGenerator[AsyncSession, None]:
    """In-memory SQLite database session fixture."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with session_factory() as session:
        # Ingest basic songs to test DB
        songs = [
            Song(id="s1", title="Calm Song", artist="Chill Artist", genre="lo-fi", language="english", year=2022, energy=0.2, valence=0.5, tempo=80.0, popularity=80),
            Song(id="s2", title="Happy Song", artist="Pop Artist", genre="pop", language="english", year=2020, energy=0.8, valence=0.9, tempo=120.0, popularity=90),
            Song(id="s3", title="Hindi Love", artist="Bollywood Singer", genre="bollywood", language="hindi", year=2018, energy=0.3, valence=0.6, tempo=75.0, popularity=85),
        ]
        session.add_all(songs)
        await session.commit()

    async with session_factory() as session:
        yield session

    await engine.dispose()


@pytest_asyncio.fixture(name="client")
async def client_fixture(
    sample_dataset_path: Path, test_db: AsyncSession
) -> AsyncGenerator[AsyncClient, None]:
    """FastAPI AsyncClient fixture overriding database and recommendation engine dependencies."""
    # Load recommendation engine using test dataset
    engine = RecommendationEngine.load(sample_dataset_path)

    # Initialize services with mocked LLM
    mock_llm = MockLLMProvider()
    chat_service = ChatService(mock_llm, engine)
    rec_service = RecommendationService(engine)

    # Set dependency module-level variables directly so non-Depends calls (like in /health) work
    import app.dependencies as deps
    deps._recommendation_engine = engine
    deps._llm_provider = mock_llm
    deps._chat_service = chat_service
    deps._recommendation_service = rec_service

    async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
        yield test_db
        await test_db.commit()

    # Override dependencies for FastAPI routing
    from app.dependencies import get_recommendation_service
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_engine] = lambda: engine
    app.dependency_overrides[get_chat_service] = lambda: chat_service
    app.dependency_overrides[get_recommendation_service] = lambda: rec_service

    # Override configuration settings
    def mock_settings() -> Settings:
        s = get_settings()
        s.llm_provider = "gemini"
        s.gemini_api_key = "mock-key-for-testing"
        return s

    app.dependency_overrides[get_settings] = mock_settings

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    # Clean overrides and dependency globals
    app.dependency_overrides.clear()
    deps._recommendation_engine = None
    deps._llm_provider = None
    deps._chat_service = None
    deps._recommendation_service = None



# =============================================================================
# API Endpoint Tests
# =============================================================================


@pytest.mark.asyncio
async def test_health_endpoint(client: AsyncClient) -> None:
    """Verify health check endpoint returns success."""
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["database"] == "connected"
    assert data["catalog_size"] > 0


@pytest.mark.asyncio
async def test_get_songs_list(client: AsyncClient) -> None:
    """Verify song listing with search queries works."""
    response = await client.get("/api/v1/songs")
    assert response.status_code == 200
    data = response.json()
    assert "songs" in data
    assert "total" in data
    assert len(data["songs"]) > 0


@pytest.mark.asyncio
async def test_search_songs_query(client: AsyncClient) -> None:
    """Verify searching songs by keyword matches correctly."""
    response = await client.get("/api/v1/songs?query=Calm")
    assert response.status_code == 200
    data = response.json()
    assert len(data["songs"]) > 0
    assert any("calm" in s["title"].lower() for s in data["songs"])


@pytest.mark.asyncio
async def test_get_popular_discover(client: AsyncClient) -> None:
    """Verify the popular songs discover route works."""
    response = await client.get("/api/v1/songs/discover/popular?n=2")
    assert response.status_code == 200
    data = response.json()
    assert len(data) <= 2
    assert "song" in data[0]


@pytest.mark.asyncio
async def test_feedback_submission(client: AsyncClient) -> None:
    """Verify submitting song feedback saves successfully."""
    # First create a mock chat session to make sure session exists
    chat_req = {"message": "Hello!", "session_id": "test-session-123"}
    chat_resp = await client.post("/api/v1/chat", json=chat_req)
    assert chat_resp.status_code == 200
    session_id = chat_resp.json()["session_id"]

    # Submit feedback for song ID 's1'
    feedback_req = {
        "session_id": session_id,
        "song_id": "s1",
        "feedback_type": "like",
    }
    response = await client.post("/api/v1/feedback", json=feedback_req)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "recorded" in data["message"]


@pytest.mark.asyncio
async def test_chat_workflow(client: AsyncClient) -> None:
    """Verify conversational AI endpoint triggers LangGraph correctly."""
    chat_req = {
        "message": "Give me something calm for reading",
        "session_id": "test-session-456",
    }
    response = await client.post("/api/v1/chat", json=chat_req)
    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == "test-session-456"
    assert "recommendations" in data
    assert len(data["recommendations"]) > 0
    assert data["message"] == "Here is a mock recommendation response!"
    # Verify dynamic LLM-generated explanations are returned
    first_rec = data["recommendations"][0]
    assert "reasons" in first_rec
    assert "Matches your study mood" in first_rec["reasons"]


@pytest.mark.asyncio
async def test_feedback_refinement_workflow(client: AsyncClient) -> None:
    """Verify that liking a song affects recommendations in subsequent turns of the same session."""
    # 1. First chat message to start a session
    chat_req = {
        "message": "Give me something calm for reading",
        "session_id": "refinement-session-789",
    }
    response = await client.post("/api/v1/chat", json=chat_req)
    assert response.status_code == 200
    data = response.json()
    
    # Grab the recommended song ID (e.g. the first one)
    song_id = data["recommendations"][0]["song"]["id"]

    # 2. Like the recommended song
    feedback_req = {
        "session_id": "refinement-session-789",
        "song_id": song_id,
        "feedback_type": "like",
    }
    response = await client.post("/api/v1/feedback", json=feedback_req)
    assert response.status_code == 200
    
    # 3. Second chat request in the same session — should prioritize similar items
    chat_req2 = {
        "message": "Recommend me more songs",
        "session_id": "refinement-session-789",
    }
    response2 = await client.post("/api/v1/chat", json=chat_req2)
    assert response2.status_code == 200
    data2 = response2.json()
    
    # The subsequent recommendations should contain songs similar to the liked song.
    recs2 = data2["recommendations"]
    assert len(recs2) > 0
    # At least one recommendation should show it is similar to the user liked history
    assert any("Similar to songs you liked" in r["reasons"] for r in recs2)
