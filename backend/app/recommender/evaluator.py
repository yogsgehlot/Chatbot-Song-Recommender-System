"""
Recommendation System Evaluator.

Computes academic metrics to measure the quality of recommendations:
1. Catalog Coverage: The % of unique songs recommended across a set of test profiles.
2. Musical Diversity: Mean pairwise cosine distance of recommended song feature vectors.
3. Personalization: How unique recommendations are to each profile (1 - average overlap).
4. Popularity/Novelty: Average popularity score of recommended tracks.
"""

from __future__ import annotations

import itertools
import logging
from typing import Any

import numpy as np
from sklearn.metrics.pairwise import cosine_distances

from app.recommender.engine import RecommendationEngine

logger = logging.getLogger(__name__)


# Standard test profiles representing different user personas
TEST_PROFILES: list[dict[str, Any]] = [
    {"mood": "calm", "language": "english"},
    {"mood": "happy", "language": "english"},
    {"mood": "sad", "language": "english"},
    {"mood": "energetic", "language": "english"},
    {"mood": "workout", "energy_level": "high"},
    {"mood": "romantic", "genre": "pop"},
    {"mood": "study", "language": "instrumental"},
    {"mood": "party", "energy_level": "high"},
    {"language": "hindi", "mood": "romantic"},
    {"language": "hindi", "mood": "sad"},
    {"language": "punjabi", "mood": "upbeat"},
    {"genre": "lo-fi", "mood": "chill"},
    {"genre": "jazz", "mood": "relaxing"},
    {"genre": "classical", "mood": "peaceful"},
    {"genre": "rock", "mood": "intense"},
    {"genre": "hip-hop", "mood": "workout"},
]


class RecommendationEvaluator:
    """Evaluates a recommendation engine using standard offline metrics."""

    def __init__(self, engine: RecommendationEngine) -> None:
        self.engine = engine

    def run_evaluation(self, n: int = 5) -> dict[str, Any]:
        """
        Run recommendations for all test profiles and compute aggregate metrics.
        """
        results_per_profile = []
        all_recommended_ids = set()
        all_recommendations = []

        for profile in TEST_PROFILES:
            recs = self.engine.recommend(profile, n=n)
            results_per_profile.append(recs)
            all_recommended_ids.update(r["id"] for r in recs)
            all_recommendations.append(recs)

        # 1. Catalog Coverage
        coverage = len(all_recommended_ids) / self.engine.catalog_size

        # 2. Average Popularity
        popularity_scores = [
            r["popularity"]
            for recs in results_per_profile
            for r in recs
        ]
        avg_popularity = float(np.mean(popularity_scores)) if popularity_scores else 0.0

        # 3. Musical Diversity (Mean Pairwise Distance)
        diversity_scores = []
        for recs in results_per_profile:
            if len(recs) > 1:
                # Extract numerical vectors for the recommended songs
                song_vectors = []
                for r in recs:
                    song_id = r["id"]
                    idx = self.engine._df[self.engine._df["id"] == song_id].index[0]
                    song_vectors.append(self.engine._num_features[idx])

                # Compute pairwise cosine distances
                distances = cosine_distances(song_vectors)
                # Mean of the upper triangle of the distance matrix (excluding diagonal)
                triu_indices = np.triu_indices_from(distances, k=1)
                mean_dist = float(np.mean(distances[triu_indices]))
                diversity_scores.append(mean_dist)

        avg_diversity = float(np.mean(diversity_scores)) if diversity_scores else 0.0

        # 4. Personalization (1 - Jaccard overlap between different users' recommendations)
        overlap_scores = []
        for recs_a, recs_b in itertools.combinations(results_per_profile, 2):
            ids_a = {r["id"] for r in recs_a}
            ids_b = {r["id"] for r in recs_b}
            if not ids_a or not ids_b:
                continue
            jaccard = len(ids_a & ids_b) / len(ids_a | ids_b)
            overlap_scores.append(jaccard)

        personalization = 1.0 - (float(np.mean(overlap_scores)) if overlap_scores else 0.0)

        return {
            "catalog_size": self.engine.catalog_size,
            "test_profiles_count": len(TEST_PROFILES),
            "catalog_coverage": round(coverage, 4),
            "mean_diversity": round(avg_diversity, 4),
            "personalization": round(personalization, 4),
            "average_popularity": round(avg_popularity, 2),
        }
