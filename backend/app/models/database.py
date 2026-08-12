"""
SQLAlchemy 2.x ORM models — defines the database schema.
Uses the modern mapped_column / Mapped annotation style.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import JSON


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


def _new_uuid() -> str:
    return str(uuid.uuid4())


# =============================================================================
# Music Catalog
# =============================================================================


class Song(Base):
    """
    A song in the music catalog.
    Features align with the Million Song Dataset / Spotify audio features.
    """

    __tablename__ = "songs"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_new_uuid)
    title: Mapped[str] = mapped_column(String(500), nullable=False, index=True)
    artist: Mapped[str] = mapped_column(String(500), nullable=False, index=True)
    album: Mapped[str | None] = mapped_column(String(500))

    # Taxonomy
    genre: Mapped[str | None] = mapped_column(String(100), index=True)
    language: Mapped[str] = mapped_column(String(50), default="english", index=True)
    year: Mapped[int | None] = mapped_column(Integer, index=True)

    # Audio features (0.0 → 1.0 normalized unless noted)
    tempo: Mapped[float | None] = mapped_column(Float)          # BPM (raw)
    energy: Mapped[float | None] = mapped_column(Float)         # 0-1: intensity
    valence: Mapped[float | None] = mapped_column(Float)        # 0-1: positivity
    danceability: Mapped[float | None] = mapped_column(Float)
    acousticness: Mapped[float | None] = mapped_column(Float)
    instrumentalness: Mapped[float | None] = mapped_column(Float)
    loudness: Mapped[float | None] = mapped_column(Float)       # dB

    # Mood / tag metadata (stored as JSON list)
    mood_tags: Mapped[list[str] | None] = mapped_column(JSON, default=list)

    # Catalog metadata
    popularity: Mapped[int] = mapped_column(Integer, default=0, index=True)
    preview_url: Mapped[str | None] = mapped_column(String(1000))
    cover_url: Mapped[str | None] = mapped_column(String(1000))

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), nullable=False
    )

    # Relationships
    recommendations: Mapped[list["Recommendation"]] = relationship(
        back_populates="song"
    )
    feedback: Mapped[list["Feedback"]] = relationship(back_populates="song")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "artist": self.artist,
            "album": self.album,
            "genre": self.genre,
            "language": self.language,
            "year": self.year,
            "energy": self.energy,
            "valence": self.valence,
            "tempo": self.tempo,
            "danceability": self.danceability,
            "acousticness": self.acousticness,
            "mood_tags": self.mood_tags or [],
            "popularity": self.popularity,
            "preview_url": self.preview_url,
            "cover_url": self.cover_url,
        }

    def __repr__(self) -> str:
        return f"<Song id={self.id!r} title={self.title!r} artist={self.artist!r}>"


# =============================================================================
# Sessions
# =============================================================================


class Session(Base):
    """
    A chat session. Created on first message, identified by UUID.
    No authentication required — sessions are session-scoped.
    """

    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_new_uuid)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    last_active: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), onupdate=func.now()
    )

    # Accumulated preferences as JSON (serialized MusicPreference)
    accumulated_preferences: Mapped[dict | None] = mapped_column(JSON)

    # Relationships
    messages: Mapped[list["ConversationMessage"]] = relationship(
        back_populates="session", order_by="ConversationMessage.created_at"
    )
    recommendations: Mapped[list["Recommendation"]] = relationship(
        back_populates="session"
    )
    feedback: Mapped[list["Feedback"]] = relationship(back_populates="session")

    def __repr__(self) -> str:
        return f"<Session id={self.id!r}>"


# =============================================================================
# Conversation
# =============================================================================


class ConversationMessage(Base):
    """A single message in a conversation session."""

    __tablename__ = "conversation_messages"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_new_uuid)
    session_id: Mapped[str] = mapped_column(
        String, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    role: Mapped[str] = mapped_column(String(20), nullable=False)   # "user" | "assistant"
    content: Mapped[str] = mapped_column(Text, nullable=False)
    intent: Mapped[str | None] = mapped_column(String(50))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())

    # Relationship
    session: Mapped["Session"] = relationship(back_populates="messages")

    def __repr__(self) -> str:
        return f"<Message role={self.role!r} session={self.session_id!r}>"


# =============================================================================
# Recommendations
# =============================================================================


class Recommendation(Base):
    """A recommendation event — what was recommended in which session and why."""

    __tablename__ = "recommendations"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_new_uuid)
    session_id: Mapped[str] = mapped_column(
        String, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    song_id: Mapped[str] = mapped_column(
        String, ForeignKey("songs.id", ondelete="CASCADE"), nullable=False
    )
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    reasons: Mapped[list[str] | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())

    # Relationships
    session: Mapped["Session"] = relationship(back_populates="recommendations")
    song: Mapped["Song"] = relationship(back_populates="recommendations")

    def __repr__(self) -> str:
        return f"<Recommendation rank={self.rank} song={self.song_id!r}>"


# =============================================================================
# Feedback
# =============================================================================


class Feedback(Base):
    """User feedback on a recommendation — used for session personalization."""

    __tablename__ = "feedback"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_new_uuid)
    session_id: Mapped[str] = mapped_column(
        String, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    song_id: Mapped[str] = mapped_column(
        String, ForeignKey("songs.id", ondelete="CASCADE"), nullable=False
    )
    feedback_type: Mapped[str] = mapped_column(
        String(20), nullable=False  # "like" | "dislike" | "skip"
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())

    # Relationships
    session: Mapped["Session"] = relationship(back_populates="feedback")
    song: Mapped["Song"] = relationship(back_populates="feedback")

    def __repr__(self) -> str:
        return f"<Feedback type={self.feedback_type!r} song={self.song_id!r}>"
