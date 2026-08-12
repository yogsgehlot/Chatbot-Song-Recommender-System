"""
Chat service — orchestrates the conversation workflow.

Manages session state in SQLite and delegates to the LangGraph workflow.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai import prompts as prompts_module
from app.ai.workflow import ConversationState, build_conversation_graph
from app.models.database import ConversationMessage, Feedback, Recommendation, Session
from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    MusicPreference,
    RecommendationItem,
    SongResponse,
)

logger = logging.getLogger(__name__)


class ChatService:
    """Handles the full chat request lifecycle."""

    def __init__(self, llm_client: object, engine: object) -> None:
        self._llm_client = llm_client
        self._engine = engine
        self._graph = build_conversation_graph(
            llm_client=llm_client,
            engine=engine,
            prompts_module=prompts_module,
        )

    async def handle_message(
        self, request: ChatRequest, db: AsyncSession
    ) -> ChatResponse:
        """
        Process a chat message:
        1. Load or create session
        2. Restore conversation state
        3. Run LangGraph workflow
        4. Persist results
        5. Return structured response
        """
        # --- Step 1: Session management ---
        session_id = request.session_id or str(uuid.uuid4())
        session = await self._get_or_create_session(session_id, db)

        # --- Step 2: Load history ---
        messages = await self._load_messages(session_id, db)
        liked_ids, disliked_ids = await self._load_feedback(session_id, db)
        recommended_ids = await self._load_recommended_ids(session_id, db)

        # --- Step 3: Build initial state ---
        accumulated_prefs = session.accumulated_preferences or {}
        initial_state: ConversationState = {
            "session_id": session_id,
            "current_message": request.message,
            "messages": messages,
            "intent": "",
            "current_preferences": {},
            "accumulated_preferences": accumulated_prefs,
            "recommended_song_ids": recommended_ids,
            "liked_song_ids": liked_ids,
            "disliked_song_ids": disliked_ids,
            "raw_candidates": [],
            "final_recommendations": [],
            "response_text": "",
            "error": None,
        }

        # --- Step 4: Run LangGraph workflow ---
        try:
            result = await self._graph.ainvoke(initial_state)
        except Exception as e:
            logger.error("LangGraph workflow error: %s", e)
            return ChatResponse(
                session_id=session_id,
                message="I'm having trouble processing your request right now. Please try again!",
                intent="error",
                recommendations=[],
            )

        # --- Step 5: Persist results ---
        await self._save_message(session_id, "user", request.message, result["intent"], db)
        await self._save_message(session_id, "assistant", result["response_text"], None, db)

        if result["final_recommendations"]:
            await self._save_recommendations(
                session_id, result["final_recommendations"], db
            )

        # Update session accumulated preferences
        session.accumulated_preferences = result.get("accumulated_preferences", {})
        session.last_active = datetime.now(timezone.utc)
        await db.flush()

        # --- Step 6: Build response ---
        recommendations = self._build_recommendation_items(result["final_recommendations"])
        extracted_prefs = None
        if result.get("current_preferences"):
            try:
                extracted_prefs = MusicPreference(**result["current_preferences"])
            except Exception:
                pass

        return ChatResponse(
            session_id=session_id,
            message=result["response_text"],
            intent=result.get("intent", "recommendation"),
            recommendations=recommendations,
            extracted_preferences=extracted_prefs,
        )

    # ------------------------------------------------------------------
    # Session helpers
    # ------------------------------------------------------------------

    async def _get_or_create_session(
        self, session_id: str, db: AsyncSession
    ) -> Session:
        result = await db.execute(
            select(Session).where(Session.id == session_id)
        )
        session = result.scalar_one_or_none()
        if session is None:
            session = Session(id=session_id)
            db.add(session)
            await db.flush()
            logger.info("Created new session: %s", session_id)
        return session

    async def _load_messages(
        self, session_id: str, db: AsyncSession
    ) -> list[dict[str, str]]:
        result = await db.execute(
            select(ConversationMessage)
            .where(ConversationMessage.session_id == session_id)
            .order_by(ConversationMessage.created_at)
        )
        msgs = result.scalars().all()
        return [{"role": m.role, "content": m.content} for m in msgs[-20:]]  # Last 20

    async def _load_feedback(
        self, session_id: str, db: AsyncSession
    ) -> tuple[list[str], list[str]]:
        result = await db.execute(
            select(Feedback).where(Feedback.session_id == session_id)
        )
        feedbacks = result.scalars().all()
        liked = [f.song_id for f in feedbacks if f.feedback_type == "like"]
        disliked = [f.song_id for f in feedbacks if f.feedback_type == "dislike"]
        return liked, disliked

    async def _load_recommended_ids(
        self, session_id: str, db: AsyncSession
    ) -> list[str]:
        result = await db.execute(
            select(Recommendation.song_id)
            .where(Recommendation.session_id == session_id)
        )
        return [row[0] for row in result.all()]

    async def _save_message(
        self,
        session_id: str,
        role: str,
        content: str,
        intent: str | None,
        db: AsyncSession,
    ) -> None:
        msg = ConversationMessage(
            session_id=session_id,
            role=role,
            content=content,
            intent=intent,
        )
        db.add(msg)

    async def _save_recommendations(
        self,
        session_id: str,
        recommendations: list[dict],
        db: AsyncSession,
    ) -> None:
        for rec in recommendations:
            db_rec = Recommendation(
                session_id=session_id,
                song_id=rec["id"],
                rank=rec.get("rank", 1),
                score=rec.get("score", 0.0),
                reasons=rec.get("reasons", []),
            )
            db.add(db_rec)

    @staticmethod
    def _build_recommendation_items(
        recommendations: list[dict],
    ) -> list[RecommendationItem]:
        items = []
        for rec in recommendations:
            try:
                song = SongResponse(
                    id=rec["id"],
                    title=rec["title"],
                    artist=rec["artist"],
                    album=rec.get("album"),
                    genre=rec.get("genre"),
                    language=rec.get("language", "english"),
                    year=rec.get("year"),
                    energy=rec.get("energy"),
                    valence=rec.get("valence"),
                    tempo=rec.get("tempo"),
                    danceability=rec.get("danceability"),
                    acousticness=rec.get("acousticness"),
                    mood_tags=rec.get("mood_tags", []),
                    popularity=rec.get("popularity", 0),
                    preview_url=rec.get("preview_url"),
                    cover_url=rec.get("cover_url"),
                )
                item = RecommendationItem(
                    song=song,
                    score=round(rec.get("score", 0.0), 3),
                    reasons=rec.get("reasons", []),
                    rank=rec.get("rank", len(items) + 1),
                )
                items.append(item)
            except Exception as e:
                logger.warning("Failed to build recommendation item: %s", e)
        return items
