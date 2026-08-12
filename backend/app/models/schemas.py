"""
Pydantic v2 schemas for all API request/response models.
These are the contracts between the frontend and backend.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


# =============================================================================
# Music Domain Models
# =============================================================================


class SongResponse(BaseModel):
    """A single song returned from the catalog."""

    id: str
    title: str
    artist: str
    album: str | None = None
    genre: str | None = None
    language: str = "english"
    year: int | None = None
    energy: float | None = None         # 0.0 (calm) → 1.0 (intense)
    valence: float | None = None        # 0.0 (sad) → 1.0 (happy)
    tempo: float | None = None          # BPM
    danceability: float | None = None
    acousticness: float | None = None
    mood_tags: list[str] = Field(default_factory=list)
    popularity: int = 0
    preview_url: str | None = None
    cover_url: str | None = None


class RecommendationItem(BaseModel):
    """A recommended song with its score and explanation."""

    song: SongResponse
    score: float = Field(ge=0.0, le=1.0, description="Relevance score 0-1")
    reasons: list[str] = Field(
        default_factory=list,
        description="Human-readable reasons why this song was recommended",
        examples=[["Matches calm mood", "Acoustic style", "Similar artist"]],
    )
    rank: int = Field(ge=1, description="1-indexed position in recommendation list")


# =============================================================================
# Preference Extraction (LLM structured output)
# =============================================================================


class MusicPreference(BaseModel):
    """
    Structured music preferences extracted from a user's natural language message.
    Used as structured output from the LLM — validated before passing to the engine.
    """

    mood: str | None = Field(
        default=None,
        description="Emotional mood",
        examples=["calm", "energetic", "sad", "happy", "romantic", "motivational"],
    )
    genre: str | None = Field(
        default=None,
        examples=["pop", "rock", "hip-hop", "lo-fi", "classical", "jazz", "bollywood"],
    )
    language: str | None = Field(
        default=None,
        examples=["english", "hindi", "punjabi", "spanish"],
    )
    artist: str | None = Field(default=None, description="Specific artist preference")
    activity: str | None = Field(
        default=None,
        examples=["study", "workout", "sleep", "party", "travel", "meditation"],
    )
    energy_level: Literal["low", "medium", "high"] | None = None
    tempo: Literal["slow", "medium", "fast"] | None = None
    era: str | None = Field(default=None, examples=["90s", "2000s", "modern", "classic"])
    is_refinement: bool = Field(
        default=False,
        description="True if user is refining a previous request rather than starting fresh",
    )
    similar_to_song_id: str | None = Field(
        default=None,
        description="Song ID if user wants songs similar to a specific track",
    )

    def has_any_preference(self) -> bool:
        """True if at least one preference field is set."""
        return any([
            self.mood, self.genre, self.language, self.artist,
            self.activity, self.energy_level, self.tempo, self.era,
        ])

    def to_display_dict(self) -> dict[str, str]:
        """Return a clean dict of set preferences for display."""
        return {
            k: str(v)
            for k, v in self.model_dump().items()
            if v is not None and k not in ("is_refinement", "similar_to_song_id")
        }


class IntentClassification(BaseModel):
    """Classified intent of a user message."""

    intent: Literal["recommendation", "refinement", "feedback", "chitchat", "help"]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(description="Brief explanation of why this intent was chosen")


# =============================================================================
# Chat API
# =============================================================================


class ChatRequest(BaseModel):
    """User sends a message to the assistant."""

    message: str = Field(min_length=1, max_length=1000)
    session_id: str | None = Field(
        default=None,
        description="Existing session ID. If None, a new session is created.",
    )

    @field_validator("message")
    @classmethod
    def strip_message(cls, v: str) -> str:
        return v.strip()


class ChatResponse(BaseModel):
    """Assistant response including message and recommendations."""

    session_id: str
    message: str = Field(description="Natural language response from the AI assistant")
    intent: str = Field(description="Classified intent of the user's message")
    recommendations: list[RecommendationItem] = Field(default_factory=list)
    extracted_preferences: MusicPreference | None = None
    is_streaming: bool = False


# =============================================================================
# Songs API
# =============================================================================


class SongListResponse(BaseModel):
    """Paginated list of songs."""

    songs: list[SongResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


class SongSearchParams(BaseModel):
    """Query parameters for song search."""

    query: str | None = None
    genre: str | None = None
    mood: str | None = None
    language: str | None = None
    min_energy: float | None = Field(default=None, ge=0.0, le=1.0)
    max_energy: float | None = Field(default=None, ge=0.0, le=1.0)
    min_year: int | None = None
    max_year: int | None = None
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=20, ge=1, le=100)


# =============================================================================
# Feedback API
# =============================================================================


class FeedbackRequest(BaseModel):
    """User submits feedback on a recommendation."""

    session_id: str
    song_id: str
    feedback_type: Literal["like", "dislike", "skip"]


class FeedbackResponse(BaseModel):
    """Confirmation of feedback received."""

    success: bool
    message: str
    updated_preferences: bool = Field(
        default=False,
        description="True if feedback caused preference updates",
    )


# =============================================================================
# Session
# =============================================================================


class SessionInfo(BaseModel):
    """Information about an active session."""

    session_id: str
    message_count: int
    recommendations_count: int
    liked_songs: list[str] = Field(default_factory=list)
    disliked_songs: list[str] = Field(default_factory=list)
    accumulated_preferences: MusicPreference | None = None


# =============================================================================
# Health
# =============================================================================


class HealthResponse(BaseModel):
    """API health check response."""

    status: Literal["ok", "degraded", "error"]
    version: str
    catalog_size: int
    llm_provider: str
    database: Literal["connected", "error"]


# =============================================================================
# Error
# =============================================================================


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str
    detail: str | None = None
    request_id: str | None = None


class SongExplanation(BaseModel):
    """Structured song recommendation explanations generated by the LLM."""

    reasons: list[str] = Field(
        default_factory=list,
        description="2-3 short bullet-point reasons (4-8 words each) why this song was recommended."
    )
