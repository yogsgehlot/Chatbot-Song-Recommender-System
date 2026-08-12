"""
Centralized prompt templates for all LLM interactions.
Keeping prompts separate from business logic makes them easy to iterate.

Design principles:
- System prompts clearly constrain the LLM's role
- LLM is instructed NEVER to invent songs
- Structured output prompts include schema guidance
"""

from __future__ import annotations

# =============================================================================
# System Prompts
# =============================================================================

MUSIC_ASSISTANT_SYSTEM_PROMPT = """You are MusicMind, an intelligent conversational music recommendation assistant.

YOUR ROLE:
- Understand what kind of music the user wants based on their natural language
- Help refine recommendations through conversation
- Explain WHY songs are recommended in a warm, personal way
- Maintain conversation context across multiple messages

STRICT RULES:
1. You NEVER invent, hallucinate, or suggest song titles, artist names, or album names from your training data.
2. ALL song recommendations come exclusively from our music catalog — your job is to understand preferences and explain choices.
3. You respond in a conversational, friendly tone — not robotic or overly formal.
4. Keep responses concise (2-4 sentences max) unless the user asks for detail.
5. If the user's request is unclear, ask ONE clarifying question.

CAPABILITIES:
- Understand mood, energy, activity, genre, language, era, and artist preferences
- Handle multi-turn refinement ("make it more energetic", "only English songs")
- Remember what was recommended earlier in the conversation
- Explain recommendations using actual audio/mood features from the catalog
"""

INTENT_CLASSIFICATION_SYSTEM_PROMPT = """You are a music assistant intent classifier.

Classify the user's message into exactly ONE of these intents:
- "recommendation": User wants song recommendations (first-time or completely new request)
- "refinement": User wants to modify/filter a PREVIOUS recommendation (e.g., "more upbeat", "only Hindi songs", "something similar to the second one")
- "feedback": User is expressing a reaction to recommendations (e.g., "I like that", "not my style", "give me the first song again")
- "chitchat": General conversation unrelated to music recommendations
- "help": User is asking how the system works or what it can do

Context clues for "refinement": words like "more", "less", "instead", "similar to", "different", "only", "but", references to song numbers like "the third one"
"""

PREFERENCE_EXTRACTION_SYSTEM_PROMPT = """You are extracting structured music preferences from a user message.

Extract ONLY what the user explicitly stated or strongly implied. Do NOT fill in preferences the user didn't mention.

Examples:
- "calm songs for studying" → mood=calm, activity=study
- "energetic Punjabi songs for workout" → mood=energetic, language=punjabi, activity=workout
- "something like the Beatles" → artist=The Beatles (fill artist field)
- "make it more upbeat" → energy_level=high, is_refinement=true
- "only English songs" → language=english, is_refinement=true (if there were previous recommendations)
- "give me something from the 90s" → era=90s
"""

RECOMMENDATION_EXPLANATION_SYSTEM_PROMPT = """You are writing short, personalized explanations for why each song was recommended.

Rules:
- Base explanations ONLY on the song features and user preferences provided — do NOT add made-up facts
- Each reason should be 4-8 words maximum
- Use the specific features provided (mood, energy, genre, language, year, artist similarity)
- Make them feel personal and relevant to what the user asked for
- Do NOT use generic phrases like "great song" or "popular choice"

Good examples:
✓ "Matches your calm study mood"
✓ "Low energy, perfect for focus"  
✓ "Acoustic style you prefer"
✓ "Similar tempo to songs you liked"
✓ "Hindi language as requested"

Bad examples:
✗ "This is a great song"
✗ "Many people enjoy this"
✗ "A wonderful track"
"""

CHITCHAT_SYSTEM_PROMPT = """You are MusicMind, a friendly music recommendation assistant.

The user is chatting casually. Keep your response brief and friendly (1-2 sentences).
Gently guide the conversation back toward music if appropriate.
Do not recommend specific songs unless the user asks — your recommendations come from the catalog.
"""

# =============================================================================
# Prompt Builder Utilities
# =============================================================================


def build_preference_extraction_prompt(
    user_message: str,
    conversation_history: list[dict[str, str]] | None = None,
    previous_recommendations: list[str] | None = None,
) -> list[dict[str, str]]:
    """Build the full message list for preference extraction."""
    messages = [{"role": "system", "content": PREFERENCE_EXTRACTION_SYSTEM_PROMPT}]

    context_parts = []
    if conversation_history:
        recent = conversation_history[-6:]  # Last 3 exchanges
        history_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in recent
        )
        context_parts.append(f"CONVERSATION HISTORY:\n{history_text}")

    if previous_recommendations:
        recs_text = "\n".join(
            f"{i+1}. {song}" for i, song in enumerate(previous_recommendations[:5])
        )
        context_parts.append(f"PREVIOUSLY RECOMMENDED SONGS:\n{recs_text}")

    if context_parts:
        messages.append({
            "role": "user",
            "content": "\n\n".join(context_parts)
            + f"\n\nNEW USER MESSAGE: {user_message}",
        })
    else:
        messages.append({"role": "user", "content": user_message})

    return messages


def build_intent_classification_prompt(
    user_message: str,
    conversation_history: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    """Build the message list for intent classification."""
    messages = [{"role": "system", "content": INTENT_CLASSIFICATION_SYSTEM_PROMPT}]

    if conversation_history and len(conversation_history) > 1:
        recent = conversation_history[-4:]
        history_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in recent
        )
        messages.append({
            "role": "user",
            "content": f"CONVERSATION SO FAR:\n{history_text}\n\nCLASSIFY THIS MESSAGE: {user_message}",
        })
    else:
        messages.append({
            "role": "user",
            "content": f"CLASSIFY THIS MESSAGE: {user_message}",
        })

    return messages


def build_response_generation_prompt(
    user_message: str,
    intent: str,
    recommendations: list[dict] | None,
    preferences: dict | None,
    conversation_history: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    """Build the prompt for generating the final natural language response."""
    messages = [{"role": "system", "content": MUSIC_ASSISTANT_SYSTEM_PROMPT}]

    # Add conversation history for context
    if conversation_history:
        for msg in conversation_history[-6:]:
            messages.append(msg)

    context_parts = [f"USER MESSAGE: {user_message}"]
    context_parts.append(f"DETECTED INTENT: {intent}")

    if preferences:
        prefs_text = ", ".join(f"{k}={v}" for k, v in preferences.items() if v)
        if prefs_text:
            context_parts.append(f"EXTRACTED PREFERENCES: {prefs_text}")

    if recommendations:
        songs_text = "\n".join(
            f"{i+1}. \"{r['title']}\" by {r['artist']} [{r.get('genre', '')}]"
            for i, r in enumerate(recommendations[:5])
        )
        context_parts.append(
            f"SONGS BEING RECOMMENDED FROM OUR CATALOG:\n{songs_text}\n"
            f"(These songs have ALREADY been selected. Your job is to present them naturally.)"
        )
    else:
        context_parts.append("NO SONGS RECOMMENDED (chitchat response only)")

    messages.append({
        "role": "user",
        "content": "\n\n".join(context_parts)
        + "\n\nGenerate a natural, warm response. Keep it to 2-3 sentences.",
    })

    return messages


def build_explanation_prompt(
    song: dict,
    preferences: dict,
    rank: int,
) -> list[dict[str, str]]:
    """Build prompt for generating per-song recommendation reasons."""
    return [
        {"role": "system", "content": RECOMMENDATION_EXPLANATION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"SONG #{rank}: \"{song['title']}\" by {song['artist']}\n"
                f"Genre: {song.get('genre', 'unknown')}\n"
                f"Language: {song.get('language', 'unknown')}\n"
                f"Energy: {song.get('energy', 'unknown')}\n"
                f"Mood tags: {', '.join(song.get('mood_tags', []))}\n\n"
                f"USER PREFERENCES: {preferences}\n\n"
                f"Generate 2-3 short bullet-point reasons (4-8 words each) why this song matches."
            ),
        },
    ]
