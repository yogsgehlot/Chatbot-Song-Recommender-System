"""Feedback API — POST /api/v1/feedback"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.models.database import Feedback, Session
from app.models.schemas import ErrorResponse, FeedbackRequest, FeedbackResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/feedback", tags=["Feedback"])


@router.post(
    "",
    response_model=FeedbackResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Session not found"},
        400: {"model": ErrorResponse},
    },
    summary="Submit feedback on a song recommendation",
    description="Like, dislike, or skip a recommended song. Feedback influences future recommendations in the same session.",
)
async def submit_feedback(
    request: FeedbackRequest,
    db: AsyncSession = Depends(get_db),
) -> FeedbackResponse:
    """Record user feedback for a song in a session."""
    # Validate session exists
    result = await db.execute(
        select(Session).where(Session.id == request.session_id)
    )
    session = result.scalar_one_or_none()
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {request.session_id!r} not found. Start a chat first.",
        )

    # Check if feedback already exists for this song in this session
    existing = await db.execute(
        select(Feedback).where(
            Feedback.session_id == request.session_id,
            Feedback.song_id == request.song_id,
        )
    )
    existing_feedback = existing.scalar_one_or_none()

    if existing_feedback:
        # Update existing feedback
        existing_feedback.feedback_type = request.feedback_type
        logger.info(
            "Updated feedback: session=%s, song=%s, type=%s",
            request.session_id,
            request.song_id,
            request.feedback_type,
        )
    else:
        # Create new feedback
        feedback = Feedback(
            session_id=request.session_id,
            song_id=request.song_id,
            feedback_type=request.feedback_type,
        )
        db.add(feedback)
        logger.info(
            "New feedback: session=%s, song=%s, type=%s",
            request.session_id,
            request.song_id,
            request.feedback_type,
        )

    return FeedbackResponse(
        success=True,
        message=f"Feedback recorded: {request.feedback_type}",
        updated_preferences=request.feedback_type in ("like", "dislike"),
    )
