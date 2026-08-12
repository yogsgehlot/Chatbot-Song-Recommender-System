"""
LangGraph conversational workflow for the music recommendation assistant.

State machine with nodes:
  classify_intent → extract_preferences → retrieve_candidates →
  rank_results → generate_explanation → generate_response

Conditional routing:
  - recommendation/refinement → full recommendation pipeline
  - chitchat/help → direct LLM response
  - feedback → record and respond
"""

from __future__ import annotations

import logging
from typing import Any, Annotated
from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph

logger = logging.getLogger(__name__)


# =============================================================================
# State
# =============================================================================


class ConversationState(TypedDict):
    """The full state passed between LangGraph nodes."""

    # Input
    session_id: str
    current_message: str

    # Conversation history — list of {"role": str, "content": str}
    messages: list[dict[str, str]]

    # Classification
    intent: str  # "recommendation" | "refinement" | "feedback" | "chitchat" | "help"

    # Preferences extracted from current message
    current_preferences: dict[str, Any]

    # Accumulated preferences across the session (merged)
    accumulated_preferences: dict[str, Any]

    # Songs already recommended this session (IDs)
    recommended_song_ids: list[str]

    # Session feedback
    liked_song_ids: list[str]
    disliked_song_ids: list[str]

    # Recommendation pipeline results
    raw_candidates: list[dict[str, Any]]
    final_recommendations: list[dict[str, Any]]

    # Output
    response_text: str
    error: str | None


# =============================================================================
# Node Functions
# =============================================================================


def node_classify_intent(state: ConversationState, llm_client: Any, prompts: Any) -> dict:
    """Classify the user's intent using the LLM."""
    from app.models.schemas import IntentClassification

    try:
        messages = prompts.build_intent_classification_prompt(
            user_message=state["current_message"],
            conversation_history=state["messages"],
        )
        classification: IntentClassification = llm_client.structured_output(
            messages, IntentClassification
        )
        intent = classification.intent
        logger.debug("Intent classified as %r (confidence=%.2f)", intent, classification.confidence)
    except Exception as e:
        logger.warning("Intent classification failed: %s — defaulting to 'recommendation'", e)
        intent = "recommendation"

    return {"intent": intent}


def node_extract_preferences(
    state: ConversationState, llm_client: Any, prompts: Any
) -> dict:
    """Extract structured preferences from the user message using the LLM."""
    from app.models.schemas import MusicPreference

    prev_songs = [
        f"{r['title']} by {r['artist']}"
        for r in state.get("final_recommendations", [])
        if r.get("title")
    ]

    try:
        messages = prompts.build_preference_extraction_prompt(
            user_message=state["current_message"],
            conversation_history=state["messages"],
            previous_recommendations=prev_songs,
        )
        prefs: MusicPreference = llm_client.structured_output(messages, MusicPreference)
        current_prefs = {k: v for k, v in prefs.model_dump().items() if v is not None}
    except Exception as e:
        logger.warning("Preference extraction failed: %s — using empty preferences", e)
        current_prefs = {}

    # Merge with accumulated preferences
    is_refinement = current_prefs.pop("is_refinement", False)
    accumulated = dict(state.get("accumulated_preferences", {}))

    if is_refinement or state["intent"] == "refinement":
        # Only update fields that the user explicitly mentioned
        accumulated.update({k: v for k, v in current_prefs.items() if v})
    else:
        # Fresh request — start from new preferences but keep language if not specified
        language = current_prefs.get("language") or accumulated.get("language")
        accumulated = current_prefs
        if language:
            accumulated["language"] = language

    logger.debug("Merged preferences: %s", accumulated)
    return {
        "current_preferences": current_prefs,
        "accumulated_preferences": accumulated,
    }


def node_retrieve_candidates(
    state: ConversationState, engine: Any
) -> dict:
    """Retrieve candidate songs from the recommendation engine."""
    prefs = state.get("accumulated_preferences", {})
    exclude_ids = list(state.get("disliked_song_ids", []))
    liked_ids = list(state.get("liked_song_ids", []))

    # For refinements, also exclude what we've already shown
    if state.get("intent") == "refinement":
        exclude_ids.extend(state.get("recommended_song_ids", []))

    try:
        candidates = engine.recommend(
            preferences=prefs,
            n=10,  # Get extra candidates for ranking diversity
            exclude_ids=exclude_ids,
            liked_ids=liked_ids,
        )
    except Exception as e:
        logger.error("Recommendation engine error: %s", e)
        # Fallback: return popular songs
        candidates = engine.get_popular(n=5)

    logger.debug("Retrieved %d candidates", len(candidates))
    return {"raw_candidates": candidates}


def node_rank_results(state: ConversationState) -> dict:
    """Select final top-N from candidates (already ranked by engine)."""
    candidates = state.get("raw_candidates", [])
    final = candidates[:5]  # Take top 5
    return {"final_recommendations": final}


def node_generate_explanations(
    state: ConversationState, llm_client: Any, prompts: Any
) -> dict:
    """Generate dynamic, LLM-enriched explanations for each recommended song."""
    from app.models.schemas import SongExplanation

    recommendations = state.get("final_recommendations", [])
    prefs = state.get("accumulated_preferences", {})

    if not recommendations:
        return {}

    enriched_recs = []
    for rank, song in enumerate(recommendations, start=1):
        try:
            # Build messages for the LLM
            messages = prompts.build_explanation_prompt(
                song=song,
                preferences=prefs,
                rank=rank,
            )
            # Request structured output from LLM
            explanation_obj: SongExplanation = llm_client.structured_output(
                messages, SongExplanation
            )
            
            # Create a new dict with the generated reasons
            new_song = dict(song)
            if explanation_obj.reasons:
                existing_reasons = new_song.get("reasons") or []
                # Keep liked reason at the top if present, combine with LLM reasons up to 3 total
                liked_reasons = [r for r in existing_reasons if "liked" in r]
                new_reasons = liked_reasons + [r for r in explanation_obj.reasons if r not in liked_reasons]
                new_song["reasons"] = new_reasons[:3]
                
            enriched_recs.append(new_song)
        except Exception as e:
            logger.warning("Failed to generate dynamic explanation for song %s: %s", song.get("title"), e)
            enriched_recs.append(song)  # Fallback to existing reasons

    return {"final_recommendations": enriched_recs}


def node_generate_response(
    state: ConversationState, llm_client: Any, prompts: Any
) -> dict:
    """Generate the natural language response using the LLM."""
    intent = state.get("intent", "recommendation")
    recommendations = state.get("final_recommendations", [])
    prefs = state.get("accumulated_preferences", {})

    # Build a summary of recommended songs for the LLM
    song_dicts = [
        {"title": r["title"], "artist": r["artist"], "genre": r.get("genre")}
        for r in recommendations
    ]

    if intent in ("recommendation", "refinement"):
        # Update tracked recommended IDs
        new_ids = [r["id"] for r in recommendations if r.get("id")]
        updated_ids = list(set(state.get("recommended_song_ids", [])) | set(new_ids))
    else:
        updated_ids = state.get("recommended_song_ids", [])

    system_prompt = (
        prompts.CHITCHAT_SYSTEM_PROMPT
        if intent in ("chitchat", "help")
        else prompts.MUSIC_ASSISTANT_SYSTEM_PROMPT
    )

    try:
        messages = prompts.build_response_generation_prompt(
            user_message=state["current_message"],
            intent=intent,
            recommendations=song_dicts if recommendations else None,
            preferences=prefs,
            conversation_history=state.get("messages", []),
        )
        response_text = llm_client.chat(messages)
    except Exception as e:
        logger.error("Response generation failed: %s", e)
        if recommendations:
            response_text = (
                f"I found {len(recommendations)} songs that match your preferences! "
                f"Here are my recommendations based on what you're looking for."
            )
        else:
            response_text = (
                "I'm here to help you discover music! "
                "Tell me what kind of songs you're in the mood for."
            )

    return {
        "response_text": response_text,
        "recommended_song_ids": updated_ids,
    }


# =============================================================================
# Routing
# =============================================================================


def route_by_intent(state: ConversationState) -> str:
    """Conditional edge — route based on classified intent."""
    intent = state.get("intent", "recommendation")
    if intent in ("recommendation", "refinement"):
        return "extract_preferences"
    elif intent in ("chitchat", "help", "feedback"):
        return "generate_response"
    else:
        return "extract_preferences"


# =============================================================================
# Graph Builder
# =============================================================================


def build_conversation_graph(llm_client: Any, engine: Any, prompts_module: Any) -> Any:
    """
    Build and compile the LangGraph conversation workflow.

    Returns a compiled graph that can be invoked with a ConversationState.
    """
    import functools

    # Bind dependencies to node functions
    classify = functools.partial(node_classify_intent, llm_client=llm_client, prompts=prompts_module)
    extract = functools.partial(node_extract_preferences, llm_client=llm_client, prompts=prompts_module)
    retrieve = functools.partial(node_retrieve_candidates, engine=engine)
    explain = functools.partial(node_generate_explanations, llm_client=llm_client, prompts=prompts_module)
    respond = functools.partial(node_generate_response, llm_client=llm_client, prompts=prompts_module)

    # Build graph
    builder = StateGraph(ConversationState)

    # Add nodes
    builder.add_node("classify_intent", classify)
    builder.add_node("extract_preferences", extract)
    builder.add_node("retrieve_candidates", retrieve)
    builder.add_node("rank_results", node_rank_results)
    builder.add_node("generate_explanations", explain)
    builder.add_node("generate_response", respond)

    # Edges
    builder.add_edge(START, "classify_intent")
    builder.add_conditional_edges("classify_intent", route_by_intent)
    builder.add_edge("extract_preferences", "retrieve_candidates")
    builder.add_edge("retrieve_candidates", "rank_results")
    builder.add_edge("rank_results", "generate_explanations")
    builder.add_edge("generate_explanations", "generate_response")
    builder.add_edge("generate_response", END)

    return builder.compile()
