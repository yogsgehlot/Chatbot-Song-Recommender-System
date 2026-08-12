"""
Content-based music recommendation engine.

Algorithm:
1. Load the music catalog (CSV → DataFrame)
2. Build hybrid feature vectors per song:
   - Numerical: energy, valence, tempo (normalized), danceability, acousticness
   - Categorical: genre (one-hot), language (one-hot), mood_tags (multi-hot)
3. Given user preferences (MusicPreference), score each song:
   - Mood match: valence/energy alignment with stated mood
   - Feature similarity: cosine similarity to preference vector
   - Popularity boost: log-normalized popularity
4. Apply hard filters (language, genre) then rank by score
5. Apply diversity: no more than 2 songs per artist in top-N

This engine is fully offline and deterministic — no LLM calls.
"""

from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


# =============================================================================
# Mood → Feature Mapping
# Maps natural language moods to approximate audio feature targets
# =============================================================================

MOOD_FEATURES: dict[str, dict[str, float]] = {
    "calm":         {"energy": 0.25, "valence": 0.50, "tempo_norm": 0.30},
    "relaxing":     {"energy": 0.20, "valence": 0.55, "tempo_norm": 0.25},
    "peaceful":     {"energy": 0.15, "valence": 0.60, "tempo_norm": 0.20},
    "happy":        {"energy": 0.65, "valence": 0.85, "tempo_norm": 0.60},
    "upbeat":       {"energy": 0.70, "valence": 0.80, "tempo_norm": 0.65},
    "energetic":    {"energy": 0.85, "valence": 0.70, "tempo_norm": 0.80},
    "intense":      {"energy": 0.90, "valence": 0.50, "tempo_norm": 0.85},
    "workout":      {"energy": 0.88, "valence": 0.65, "tempo_norm": 0.82},
    "sad":          {"energy": 0.30, "valence": 0.15, "tempo_norm": 0.30},
    "melancholic":  {"energy": 0.35, "valence": 0.20, "tempo_norm": 0.35},
    "romantic":     {"energy": 0.40, "valence": 0.70, "tempo_norm": 0.40},
    "love":         {"energy": 0.45, "valence": 0.75, "tempo_norm": 0.42},
    "motivational": {"energy": 0.75, "valence": 0.80, "tempo_norm": 0.70},
    "focus":        {"energy": 0.30, "valence": 0.45, "tempo_norm": 0.30},
    "study":        {"energy": 0.25, "valence": 0.50, "tempo_norm": 0.28},
    "party":        {"energy": 0.90, "valence": 0.85, "tempo_norm": 0.88},
    "dance":        {"energy": 0.85, "valence": 0.80, "tempo_norm": 0.85},
    "chill":        {"energy": 0.30, "valence": 0.60, "tempo_norm": 0.32},
}

ENERGY_LEVEL_MAP: dict[str, tuple[float, float]] = {
    "low":    (0.0, 0.40),
    "medium": (0.30, 0.70),
    "high":   (0.60, 1.0),
}

TEMPO_MAP: dict[str, tuple[float, float]] = {
    "slow":   (0.0, 0.40),
    "medium": (0.30, 0.65),
    "fast":   (0.55, 1.0),
}


# =============================================================================
# Recommendation Engine
# =============================================================================


class RecommendationEngine:
    """
    Content-based music recommendation engine.

    Load once at application startup via the `load()` class method,
    then use `recommend()` for each user query.
    """

    def __init__(self, songs_df: pd.DataFrame) -> None:
        self._df = songs_df.copy()
        self._scaler = MinMaxScaler()
        self._tfidf: TfidfVectorizer | None = None
        self._tfidf_matrix: np.ndarray | None = None
        self._num_features: np.ndarray | None = None
        self._build_index()
        from app.recommender.external_provider import ExternalMusicProvider
        self.external_provider = ExternalMusicProvider()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, dataset_path: str | Path) -> "RecommendationEngine":
        """
        Load the song catalog from a CSV file and build the recommendation index.
        Expected columns: id, title, artist, genre, language, year, energy,
                          valence, tempo, danceability, acousticness, popularity, mood_tags
        """
        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {path}. "
                "Run: python scripts/generate_sample_data.py"
            )

        df = pd.read_csv(path, dtype={"id": str})
        df = cls._validate_and_clean(df)
        logger.info("Loaded %d songs from %s", len(df), path)
        return cls(df)

    @staticmethod
    def _validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure required columns exist and clean data."""
        required = ["id", "title", "artist"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Dataset missing required columns: {missing}")

        # Fill optional columns with sensible defaults
        df["genre"] = df.get("genre", pd.Series(["unknown"] * len(df))).fillna("unknown")
        df["language"] = df.get("language", pd.Series(["english"] * len(df))).fillna("english").str.lower()
        df["energy"] = pd.to_numeric(df.get("energy", 0.5), errors="coerce").fillna(0.5)
        df["valence"] = pd.to_numeric(df.get("valence", 0.5), errors="coerce").fillna(0.5)
        df["tempo"] = pd.to_numeric(df.get("tempo", 120.0), errors="coerce").fillna(120.0)
        df["danceability"] = pd.to_numeric(df.get("danceability", 0.5), errors="coerce").fillna(0.5)
        df["acousticness"] = pd.to_numeric(df.get("acousticness", 0.5), errors="coerce").fillna(0.5)
        df["popularity"] = pd.to_numeric(df.get("popularity", 0), errors="coerce").fillna(0).astype(int)
        df["year"] = pd.to_numeric(df.get("year", 2000), errors="coerce").fillna(2000).astype(int)
        df["mood_tags"] = df.get("mood_tags", pd.Series([""] * len(df))).fillna("")

        # Remove duplicates
        df = df.drop_duplicates(subset=["title", "artist"]).reset_index(drop=True)
        return df

    def _build_index(self) -> None:
        """Pre-compute feature matrices for fast similarity lookups."""
        df = self._df

        # 1. Numerical feature matrix
        num_cols = ["energy", "valence", "danceability", "acousticness"]
        num_matrix = df[num_cols].values.astype(float)

        # Normalize tempo to [0, 1]
        tempo_norm = (df["tempo"].values - df["tempo"].min()) / (
            df["tempo"].max() - df["tempo"].min() + 1e-8
        )
        self._num_features = np.column_stack([num_matrix, tempo_norm])

        # 2. TF-IDF over text tags (genre + mood_tags)
        text_corpus = (
            df["genre"].str.lower()
            + " "
            + df["mood_tags"].astype(str).str.lower()
            + " "
            + df["artist"].str.lower()
        )
        self._tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
        self._tfidf_matrix = self._tfidf.fit_transform(text_corpus).toarray()

        logger.info(
            "Built recommendation index: %d songs, %d numerical features, %d text features",
            len(df),
            self._num_features.shape[1],
            self._tfidf_matrix.shape[1],
        )

    @property
    def catalog_size(self) -> int:
        return len(self._df)

    @property
    def songs_dataframe(self) -> pd.DataFrame:
        return self._df

    # ------------------------------------------------------------------
    # Core Recommendation
    # ------------------------------------------------------------------

    def recommend(
        self,
        preferences: dict[str, Any],
        n: int = 5,
        exclude_ids: list[str] | None = None,
        liked_ids: list[str] | None = None,
        weights: dict[str, float] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Generate ranked recommendations from user preferences.

        Args:
            preferences: Dict from MusicPreference.model_dump()
            n: Number of recommendations to return
            exclude_ids: Song IDs to exclude (previously recommended / disliked)
            liked_ids: Song IDs the user liked in this session
            weights: Override default scoring weights

        Returns:
            List of song dicts with score and reasons fields
        """
        # Try fetching real tracks from Spotify/iTunes first (skip during tests)
        import sys
        is_testing = "pytest" in sys.modules or os.getenv("APP_ENV") == "test"
        if not is_testing:
            try:
                ext_recs = self.external_provider.recommend(preferences, n)
                if ext_recs:
                    return ext_recs
            except Exception as e:
                logger.warning("External provider failed: %s - falling back to local catalog", e)

        if weights is None:
            if liked_ids:
                weights = {"content": 0.5, "mood": 0.25, "popularity": 0.05, "feedback": 0.20}
            else:
                weights = {"content": 0.6, "mood": 0.3, "popularity": 0.1, "feedback": 0.0}
        elif "feedback" not in weights:
            weights["feedback"] = 0.0

        df = self._df.copy()

        # Step 1: Hard filters
        df = self._apply_hard_filters(df, preferences)
        if df.empty:
            logger.warning("No songs after filtering — returning popular songs as fallback")
            df = self._df.copy()

        # Step 2: Exclude disliked/already-seen
        if exclude_ids:
            df = df[~df["id"].isin(exclude_ids)]

        if df.empty:
            return []

        indices = df.index.tolist()

        # Step 3: Compute scores
        mood_scores = self._score_mood(df, preferences)
        content_scores = self._score_content(indices, preferences)
        popularity_scores = self._score_popularity(df)
        feedback_scores = self._score_feedback(df, liked_ids)

        # Step 4: Weighted combination
        final_scores = (
            weights["mood"] * mood_scores
            + weights["content"] * content_scores
            + weights["popularity"] * popularity_scores
            + weights["feedback"] * feedback_scores
        )

        # Add random exploration jitter (up to 5% score variance)
        # to ensure fresh, diverse recommendations on identical queries
        final_scores += np.random.uniform(0.0, 0.05, size=len(final_scores))

        df = df.copy()
        df["_score"] = final_scores

        # Step 5: Diversity — max 2 songs per artist in top results
        df_sorted = df.sort_values("_score", ascending=False)
        selected = self._apply_diversity(df_sorted, n=n)

        return self._format_results(selected, preferences, liked_ids=liked_ids)

    def find_similar(
        self,
        song_id: str,
        n: int = 5,
        exclude_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Find songs most similar to a given song by ID."""
        if song_id not in self._df["id"].values:
            logger.warning("Song ID %s not found in catalog", song_id)
            return []

        idx = self._df[self._df["id"] == song_id].index[0]

        # Compute cosine similarity of the target song's full feature vector
        # against all other songs
        target_num = self._num_features[idx].reshape(1, -1)
        target_tfidf = self._tfidf_matrix[idx].reshape(1, -1)

        num_sim = cosine_similarity(target_num, self._num_features)[0]
        tfidf_sim = cosine_similarity(target_tfidf, self._tfidf_matrix)[0]
        similarity = 0.5 * num_sim + 0.5 * tfidf_sim

        df = self._df.copy()
        df["_score"] = similarity
        df = df[df["id"] != song_id]

        if exclude_ids:
            df = df[~df["id"].isin(exclude_ids)]

        df_sorted = df.sort_values("_score", ascending=False)
        selected = self._apply_diversity(df_sorted, n=n)

        return self._format_results(selected, preferences={})

    def get_popular(self, n: int = 5, genre: str | None = None) -> list[dict[str, Any]]:
        """Return the most popular songs — used for cold-start (new sessions)."""
        df = self._df.copy()
        if genre:
            genre_filtered = df[df["genre"].str.lower() == genre.lower()]
            if not genre_filtered.empty:
                df = genre_filtered
        df = df.sort_values("popularity", ascending=False).head(n * 3)
        selected = self._apply_diversity(df, n=n)
        return self._format_results(selected, preferences={})

    def get_song_by_id(self, song_id: str) -> dict[str, Any] | None:
        """Retrieve a single song by ID."""
        row = self._df[self._df["id"] == song_id]
        if row.empty:
            return None
        return row.iloc[0].to_dict()

    def search_songs(
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
    ) -> tuple[list[dict[str, Any]], int]:
        """
        Search/filter the catalog. Returns (songs_list, total_count).
        """
        df = self._df.copy()

        if query:
            q = query.lower()
            mask = (
                df["title"].str.lower().str.contains(q, na=False)
                | df["artist"].str.lower().str.contains(q, na=False)
                | df["genre"].str.lower().str.contains(q, na=False)
            )
            df = df[mask]

        if genre:
            df = df[df["genre"].str.lower() == genre.lower()]
        if language:
            df = df[df["language"].str.lower() == language.lower()]
        if min_energy is not None:
            df = df[df["energy"] >= min_energy]
        if max_energy is not None:
            df = df[df["energy"] <= max_energy]
        if min_year is not None:
            df = df[df["year"] >= min_year]
        if max_year is not None:
            df = df[df["year"] <= max_year]

        total = len(df)
        start = (page - 1) * page_size
        end = start + page_size
        page_df = df.iloc[start:end]

        return [row.to_dict() for _, row in page_df.iterrows()], total

    # ------------------------------------------------------------------
    # Scoring Methods
    # ------------------------------------------------------------------

    def _score_mood(self, df: pd.DataFrame, preferences: dict) -> np.ndarray:
        """Score songs by how well they match the stated mood."""
        mood = (preferences.get("mood") or "").lower()
        if not mood:
            return np.ones(len(df))

        targets = MOOD_FEATURES.get(mood, {"energy": 0.5, "valence": 0.5, "tempo_norm": 0.5})

        # Normalized tempo
        tempo_norm = (df["tempo"] - df["tempo"].min()) / (
            df["tempo"].max() - df["tempo"].min() + 1e-8
        )

        energy_score = 1.0 - np.abs(df["energy"].values - targets.get("energy", 0.5))
        valence_score = 1.0 - np.abs(df["valence"].values - targets.get("valence", 0.5))
        tempo_score = 1.0 - np.abs(tempo_norm.values - targets.get("tempo_norm", 0.5))

        return (energy_score + valence_score + tempo_score) / 3.0

    def _score_content(self, original_indices: list[int], preferences: dict) -> np.ndarray:
        """Score songs by TF-IDF similarity to preference text."""
        if self._tfidf is None:
            return np.zeros(len(original_indices))

        pref_text = " ".join(filter(None, [
            preferences.get("genre") or "",
            preferences.get("mood") or "",
            preferences.get("artist") or "",
            preferences.get("activity") or "",
        ])).strip()

        if not pref_text:
            return np.zeros(len(original_indices))

        pref_vec = self._tfidf.transform([pref_text]).toarray()
        tfidf_subset = self._tfidf_matrix[original_indices]
        similarities = cosine_similarity(pref_vec, tfidf_subset)[0]
        return similarities

    def _score_popularity(self, df: pd.DataFrame) -> np.ndarray:
        """Log-normalize popularity scores."""
        pop = df["popularity"].values.astype(float)
        if pop.max() == 0:
            return np.zeros(len(df))
        return np.log1p(pop) / np.log1p(pop.max())

    def _score_feedback(self, df: pd.DataFrame, liked_ids: list[str] | None) -> np.ndarray:
        """Score songs based on their similarity to songs the user liked in this session."""
        if not liked_ids:
            return np.zeros(len(df))

        # Find valid liked indices in the catalog
        liked_indices = self._df[self._df["id"].isin(liked_ids)].index.tolist()
        if not liked_indices:
            return np.zeros(len(df))

        # Build average feature vector of liked songs
        liked_features = self._num_features[liked_indices]
        profile_vector = np.mean(liked_features, axis=0).reshape(1, -1)

        # Compute cosine similarity between the profile vector and candidate features
        candidate_indices = df.index.tolist()
        candidate_features = self._num_features[candidate_indices]
        similarities = cosine_similarity(profile_vector, candidate_features)[0]
        return similarities

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------

    def _apply_hard_filters(
        self, df: pd.DataFrame, preferences: dict
    ) -> pd.DataFrame:
        """Apply hard constraints — language, genre, year, energy/tempo ranges."""
        language = (preferences.get("language") or "").lower()
        genre = (preferences.get("genre") or "").lower()
        energy_level = (preferences.get("energy_level") or "").lower()
        tempo = (preferences.get("tempo") or "").lower()
        era = preferences.get("era") or ""

        if language:
            lang_df = df[df["language"].str.lower() == language]
            if not lang_df.empty:
                df = lang_df
            else:
                logger.debug("No songs found for language=%s, removing filter", language)

        if genre:
            genre_df = df[df["genre"].str.lower().str.contains(genre, na=False)]
            if not genre_df.empty:
                df = genre_df
            else:
                logger.debug("No songs found for genre=%s, removing filter", genre)

        if energy_level and energy_level in ENERGY_LEVEL_MAP:
            lo, hi = ENERGY_LEVEL_MAP[energy_level]
            e_df = df[(df["energy"] >= lo) & (df["energy"] <= hi)]
            if not e_df.empty:
                df = e_df

        if tempo and tempo in TEMPO_MAP:
            lo, hi = TEMPO_MAP[tempo]
            tempo_norm = (df["tempo"] - df["tempo"].min()) / (
                df["tempo"].max() - df["tempo"].min() + 1e-8
            )
            t_df = df[(tempo_norm >= lo) & (tempo_norm <= hi)]
            if not t_df.empty:
                df = t_df

        if era:
            decade = self._parse_decade(era)
            if decade:
                decade_df = df[
                    (df["year"] >= decade) & (df["year"] < decade + 10)
                ]
                if not decade_df.empty:
                    df = decade_df

        return df

    @staticmethod
    def _parse_decade(era: str) -> int | None:
        """Parse era string like '90s', '2000s', '80s' into start year."""
        match = re.search(r"(\d{2,4})s?", era)
        if not match:
            return None
        year = int(match.group(1))
        if year < 100:
            year = 1900 + year if year > 20 else 2000 + year
        return year

    # ------------------------------------------------------------------
    # Diversity & Formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_diversity(df: pd.DataFrame, n: int, max_per_artist: int = 2) -> pd.DataFrame:
        """Select top-n songs with max `max_per_artist` songs per artist."""
        selected = []
        artist_counts: dict[str, int] = {}

        for _, row in df.iterrows():
            artist = str(row["artist"]).lower()
            count = artist_counts.get(artist, 0)
            if count < max_per_artist:
                selected.append(row)
                artist_counts[artist] = count + 1
            if len(selected) >= n:
                break

        return pd.DataFrame(selected)

    def _format_results(
        self,
        df: pd.DataFrame,
        preferences: dict,
        liked_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Convert result DataFrame rows to dicts with reasons."""
        results = []
        for rank, (_, row) in enumerate(df.iterrows(), start=1):
            reasons = _build_basic_reasons(row, preferences)

            # Check if this song is highly similar to user's liked songs
            if liked_ids and row.get("id"):
                idx = self._df[self._df["id"] == row["id"]].index[0]
                liked_indices = self._df[self._df["id"].isin(liked_ids)].index.tolist()
                if liked_indices:
                    liked_features = self._num_features[liked_indices]
                    profile_vector = np.mean(liked_features, axis=0).reshape(1, -1)
                    song_vector = self._num_features[idx].reshape(1, -1)
                    sim = cosine_similarity(profile_vector, song_vector)[0][0]
                    if sim > 0.85:
                        reasons.insert(0, "Similar to songs you liked")

            song_dict = {
                "id": str(row.get("id", "")),
                "title": str(row.get("title", "")),
                "artist": str(row.get("artist", "")),
                "album": str(row.get("album", "")) if row.get("album") else None,
                "genre": str(row.get("genre", "")) if row.get("genre") else None,
                "language": str(row.get("language", "english")),
                "year": int(row.get("year", 0)) if row.get("year") else None,
                "energy": float(row.get("energy", 0.5)),
                "valence": float(row.get("valence", 0.5)),
                "tempo": float(row.get("tempo", 120.0)),
                "danceability": float(row.get("danceability", 0.5)),
                "acousticness": float(row.get("acousticness", 0.5)),
                "mood_tags": str(row.get("mood_tags", "")).split(",") if row.get("mood_tags") else [],
                "popularity": int(row.get("popularity", 0)),
                "preview_url": str(row.get("preview_url", "")) or None,
                "cover_url": str(row.get("cover_url", "")) or None,
                "score": float(row.get("_score", 0.0)),
                "rank": rank,
                # Reasons will be enriched by the AI layer
                "reasons": reasons[:3],
            }
            results.append(song_dict)

        return results


# =============================================================================
# Helpers
# =============================================================================


def _build_basic_reasons(row: Any, preferences: dict) -> list[str]:
    """
    Generate simple deterministic reasons before the LLM enriches them.
    These are factual, feature-based reasons.
    """
    reasons = []

    mood = (preferences.get("mood") or "").lower()
    if mood and mood in MOOD_FEATURES:
        reasons.append(f"Matches {mood} mood")

    if preferences.get("language"):
        reasons.append(f"{str(preferences['language']).capitalize()} language")

    if preferences.get("genre"):
        reasons.append(f"{str(preferences['genre']).capitalize()} genre")

    energy = float(row.get("energy", 0.5))
    if energy < 0.35:
        reasons.append("Low energy — calm and relaxing")
    elif energy > 0.75:
        reasons.append("High energy track")

    acousticness = float(row.get("acousticness", 0.5))
    if acousticness > 0.7:
        reasons.append("Acoustic style")

    return reasons[:3]  # Cap at 3 reasons
