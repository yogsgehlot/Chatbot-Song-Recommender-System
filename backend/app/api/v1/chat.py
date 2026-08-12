"""Chat API — POST /api/v1/chat"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.dependencies import get_chat_service
from app.models.schemas import ChatRequest, ChatResponse, ErrorResponse
from app.services.chat_service import ChatService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post(
    "",
    response_model=ChatResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        503: {"model": ErrorResponse, "description": "AI service unavailable"},
    },
    summary="Send a message to the AI music assistant",
    description=(
        "Send a natural language message. The AI will classify your intent, "
        "extract music preferences, retrieve songs from the catalog, and return "
        "both a conversational response and personalized recommendations."
    ),
)
async def chat(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
    chat_service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    """Handle a chat message and return AI response with recommendations."""
    logger.info(
        "Chat request: session=%s, message=%r",
        request.session_id or "new",
        request.message[:50],
    )

    try:
        response = await chat_service.handle_message(request, db)
        logger.info(
            "Chat response: session=%s, intent=%s, recommendations=%d",
            response.session_id,
            response.intent,
            len(response.recommendations),
        )
        return response
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error("Unexpected error in chat endpoint: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The AI service encountered an error. Please try again.",
        )
