"""Songs API — GET /api/v1/songs, GET /api/v1/songs/{id}, GET /api/v1/songs/{id}/similar"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.dependencies import get_recommendation_service
from app.models.schemas import (
    ErrorResponse,
    RecommendationItem,
    SongListResponse,
    SongResponse,
)
from app.services.recommendation_service import RecommendationService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/songs", tags=["Songs"])


@router.get(
    "",
    response_model=SongListResponse,
    summary="Browse and search the song catalog",
)
async def get_songs(
    query: str | None = Query(default=None, description="Search by title, artist, or genre"),
    genre: str | None = Query(default=None, description="Filter by genre (e.g., pop, rock, lo-fi)"),
    language: str | None = Query(default=None, description="Filter by language (e.g., english, hindi)"),
    min_energy: float | None = Query(default=None, ge=0.0, le=1.0, description="Minimum energy level"),
    max_energy: float | None = Query(default=None, ge=0.0, le=1.0, description="Maximum energy level"),
    min_year: int | None = Query(default=None, description="Minimum release year"),
    max_year: int | None = Query(default=None, description="Maximum release year"),
    page: int = Query(default=1, ge=1, description="Page number"),
    page_size: int = Query(default=20, ge=1, le=100, description="Items per page"),
    rec_service: RecommendationService = Depends(get_recommendation_service),
) -> SongListResponse:
    """Search and filter the music catalog with pagination."""
    return rec_service.get_songs(
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


@router.get(
    "/{song_id}",
    response_model=SongResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Get a song by ID",
)
async def get_song(
    song_id: str,
    rec_service: RecommendationService = Depends(get_recommendation_service),
) -> SongResponse:
    """Retrieve a single song's full details by its ID."""
    song = rec_service.get_song_by_id(song_id)
    if song is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Song with id={song_id!r} not found",
        )
    return song


@router.get(
    "/{song_id}/similar",
    response_model=list[RecommendationItem],
    responses={404: {"model": ErrorResponse}},
    summary="Get songs similar to a given song",
)
async def get_similar_songs(
    song_id: str,
    n: int = Query(default=5, ge=1, le=20, description="Number of similar songs to return"),
    rec_service: RecommendationService = Depends(get_recommendation_service),
) -> list[RecommendationItem]:
    """Find musically similar songs using feature-based cosine similarity."""
    if rec_service.get_song_by_id(song_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Song with id={song_id!r} not found",
        )
    return rec_service.get_similar(song_id, n=n)


@router.get(
    "/discover/popular",
    response_model=list[RecommendationItem],
    summary="Discover popular songs (cold-start)",
)
async def get_popular_songs(
    n: int = Query(default=5, ge=1, le=20),
    genre: str | None = Query(default=None),
    rec_service: RecommendationService = Depends(get_recommendation_service),
) -> list[RecommendationItem]:
    """Get popular songs — useful for cold-start or discovery page."""
    return rec_service.get_popular(n=n, genre=genre)
