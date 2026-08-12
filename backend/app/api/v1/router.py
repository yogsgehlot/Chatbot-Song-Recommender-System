"""API v1 router — aggregates all route modules."""

from fastapi import APIRouter

from app.api.v1 import chat, feedback, songs

router = APIRouter(prefix="/api/v1")

router.include_router(chat.router)
router.include_router(songs.router)
router.include_router(feedback.router)
