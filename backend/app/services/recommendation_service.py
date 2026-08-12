"""
Recommendation service — exposes the engine to the API layer.
Handles song search, retrieval, and catalog statistics.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from app.models.schemas import (
    RecommendationItem,
    SongListResponse,
    SongResponse,
)
from app.recommender.engine import RecommendationEngine

logger = logging.getLogger(__name__)


class RecommendationService:
    """High-level service wrapping the recommendation engine for API use."""

    def __init__(self, engine: RecommendationEngine) -> None:
        self._engine = engine

    @property
    def catalog_size(self) -> int:
        return self._engine.catalog_size

    def get_songs(
        self,
        query: str | None = None,
        genre: str | None = None,
        language: str | None = None,
        min_energy: float | None = None,
        max_energy: float | None = None,
        min_year: int | None = None,
        max_year: int | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> SongListResponse:
        """Search and filter the song catalog with pagination."""
        songs_raw, total = self._engine.search_songs(
            query=query,
            genre=genre,
            language=language,
            min_energy=min_energy,
            max_energy=max_energy,
            min_year=min_year,
            max_year=max_year,
            page=page,
            page_size=page_size,
        )
        songs = [self._to_song_response(s) for s in songs_raw]
        return SongListResponse(
            songs=songs,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=math.ceil(total / page_size) if page_size > 0 else 0,
        )

    def get_song_by_id(self, song_id: str) -> SongResponse | None:
        """Retrieve a single song by ID."""
        raw = self._engine.get_song_by_id(song_id)
        if raw is None:
            return None
        return self._to_song_response(raw)

    def get_similar(
        self,
        song_id: str,
        n: int = 5,
        exclude_ids: list[str] | None = None,
    ) -> list[RecommendationItem]:
        """Get songs similar to a given song."""
        results = self._engine.find_similar(song_id, n=n, exclude_ids=exclude_ids)
        return [
            RecommendationItem(
                song=self._to_song_response(r),
                score=round(r.get("score", 0.0), 3),
                reasons=r.get("reasons", ["Musically similar"]),
                rank=r.get("rank", i + 1),
            )
            for i, r in enumerate(results)
        ]

    def get_popular(self, n: int = 5, genre: str | None = None) -> list[RecommendationItem]:
        """Get popular songs — used for cold-start discovery."""
        results = self._engine.get_popular(n=n, genre=genre)
        return [
            RecommendationItem(
                song=self._to_song_response(r),
                score=round(r.get("score", 0.0), 3),
                reasons=["Popular in catalog", "Trending"],
                rank=i + 1,
            )
            for i, r in enumerate(results)
        ]

    @staticmethod
    def _to_song_response(raw: dict[str, Any]) -> SongResponse:
        """Convert a raw dict from the engine to a SongResponse schema."""
        import pandas as pd

        def clean(val: Any) -> str | None:
            if val is None or (isinstance(val, float) and pd.isna(val)):
                return None
            s = str(val).strip()
            return s if s else None

        mood_tags = raw.get("mood_tags") or []
        if isinstance(mood_tags, str):
            mood_tags = [t.strip() for t in mood_tags.split(",") if t.strip()]

        return SongResponse(
            id=str(raw.get("id", "")),
            title=str(raw.get("title", "")),
            artist=str(raw.get("artist", "")),
            album=clean(raw.get("album")),
            genre=clean(raw.get("genre")),
            language=str(raw.get("language", "english")),
            year=int(raw["year"]) if raw.get("year") and not pd.isna(raw["year"]) else None,
            energy=float(raw["energy"]) if raw.get("energy") is not None and not pd.isna(raw["energy"]) else None,
            valence=float(raw["valence"]) if raw.get("valence") is not None and not pd.isna(raw["valence"]) else None,
            tempo=float(raw["tempo"]) if raw.get("tempo") is not None and not pd.isna(raw["tempo"]) else None,
            danceability=float(raw["danceability"]) if raw.get("danceability") is not None and not pd.isna(raw["danceability"]) else None,
            acousticness=float(raw["acousticness"]) if raw.get("acousticness") is not None and not pd.isna(raw["acousticness"]) else None,
            mood_tags=mood_tags,
            popularity=int(raw.get("popularity", 0)) if not pd.isna(raw.get("popularity")) else 0,
            preview_url=clean(raw.get("preview_url")),
            cover_url=clean(raw.get("cover_url")),
        )

