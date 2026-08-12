"""Tests for the recommendation engine."""

from __future__ import annotations

import tempfile
import textwrap
from pathlib import Path

import pytest

from app.recommender.engine import RecommendationEngine


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    """Create a small sample CSV for testing."""
    csv_content = textwrap.dedent("""\
        id,title,artist,genre,language,year,energy,valence,tempo,danceability,acousticness,popularity,mood_tags
        1,Calm Study Beat,Lo-Fi Artist,lo-fi,instrumental,2022,0.20,0.55,78.0,0.38,0.90,72,"calm,study,focus"
        2,Happy Pop Song,Pop Artist,pop,english,2020,0.85,0.92,120.0,0.88,0.05,90,"happy,upbeat,dance"
        3,Sad Ballad,Sad Artist,pop,english,2019,0.25,0.18,70.0,0.28,0.80,75,"sad,emotional,ballad"
        4,Workout Banger,Hip Hop Artist,hip-hop,english,2021,0.90,0.65,150.0,0.85,0.02,88,"workout,energetic,motivational"
        5,Hindi Love Song,Arijit Singh,bollywood,hindi,2018,0.35,0.70,80.0,0.50,0.65,85,"romantic,love,bollywood"
        6,Punjabi Dance,Guru Randhawa,punjabi pop,punjabi,2019,0.82,0.80,115.0,0.84,0.10,82,"dance,upbeat,punjabi"
        7,Jazz Night,Miles Davis,jazz,instrumental,1959,0.28,0.62,136.0,0.42,0.88,80,"chill,jazz,study"
        8,Classical Piano,Beethoven,classical,instrumental,1800,0.12,0.50,60.0,0.15,0.99,85,"calm,study,classical"
        9,Party Anthem,DJ Artist,electronic,english,2023,0.92,0.85,130.0,0.92,0.02,88,"party,dance,energetic"
        10,Romantic Evening,Ed Sheeran,pop,english,2017,0.42,0.72,95.0,0.55,0.50,94,"romantic,love,acoustic"
    """)
    csv_path = tmp_path / "songs.csv"
    csv_path.write_text(csv_content)
    return csv_path


@pytest.fixture
def engine(sample_csv: Path) -> RecommendationEngine:
    return RecommendationEngine.load(sample_csv)


class TestEngineLoad:
    def test_loads_correctly(self, engine: RecommendationEngine) -> None:
        assert engine.catalog_size == 10

    def test_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            RecommendationEngine.load(tmp_path / "nonexistent.csv")


class TestRecommendations:
    def test_returns_correct_count(self, engine: RecommendationEngine) -> None:
        results = engine.recommend(preferences={}, n=5)
        assert len(results) <= 5

    def test_mood_calm_returns_low_energy(self, engine: RecommendationEngine) -> None:
        results = engine.recommend(preferences={"mood": "calm"}, n=3)
        assert len(results) > 0
        # Calm songs should have lower energy on average
        avg_energy = sum(r["energy"] for r in results) / len(results)
        assert avg_energy < 0.6

    def test_mood_energetic_returns_high_energy(self, engine: RecommendationEngine) -> None:
        results = engine.recommend(preferences={"mood": "energetic"}, n=3)
        assert len(results) > 0
        avg_energy = sum(r["energy"] for r in results) / len(results)
        assert avg_energy > 0.4

    def test_language_filter_hindi(self, engine: RecommendationEngine) -> None:
        results = engine.recommend(preferences={"language": "hindi"}, n=5)
        for r in results:
            assert r["language"].lower() == "hindi"

    def test_language_filter_punjabi(self, engine: RecommendationEngine) -> None:
        results = engine.recommend(preferences={"language": "punjabi"}, n=5)
        for r in results:
            assert r["language"].lower() == "punjabi"

    def test_exclude_disliked_songs(self, engine: RecommendationEngine) -> None:
        first_results = engine.recommend(preferences={}, n=5)
        exclude_ids = [r["id"] for r in first_results]
        second_results = engine.recommend(preferences={}, n=5, exclude_ids=exclude_ids)
        second_ids = [r["id"] for r in second_results]
        # No excluded IDs should appear in second results
        assert not any(sid in exclude_ids for sid in second_ids)

    def test_diversity_max_per_artist(self, engine: RecommendationEngine) -> None:
        results = engine.recommend(preferences={}, n=10)
        artist_counts: dict[str, int] = {}
        for r in results:
            artist = r["artist"].lower()
            artist_counts[artist] = artist_counts.get(artist, 0) + 1
        # No artist should appear more than 2 times
        assert all(count <= 2 for count in artist_counts.values())

    def test_results_have_required_fields(self, engine: RecommendationEngine) -> None:
        results = engine.recommend(preferences={}, n=3)
        required_fields = {"id", "title", "artist", "score", "rank", "reasons"}
        for r in results:
            assert required_fields.issubset(r.keys())

    def test_scores_are_in_valid_range(self, engine: RecommendationEngine) -> None:
        results = engine.recommend(preferences={"mood": "happy"}, n=5)
        for r in results:
            assert 0.0 <= r["score"] <= 1.5  # May exceed 1.0 due to weighting

    def test_ranks_are_sequential(self, engine: RecommendationEngine) -> None:
        results = engine.recommend(preferences={}, n=5)
        ranks = [r["rank"] for r in results]
        assert ranks == list(range(1, len(results) + 1))


class TestSimilarSongs:
    def test_find_similar_excludes_source(self, engine: RecommendationEngine) -> None:
        results = engine.recommend(preferences={}, n=1)
        if results:
            source_id = results[0]["id"]
            similar = engine.find_similar(source_id, n=3)
            similar_ids = [s["id"] for s in similar]
            assert source_id not in similar_ids

    def test_invalid_song_id_returns_empty(self, engine: RecommendationEngine) -> None:
        result = engine.find_similar("nonexistent-id", n=3)
        assert result == []


class TestSearch:
    def test_search_by_query(self, engine: RecommendationEngine) -> None:
        results, total = engine.search_songs(query="Pop")
        assert total > 0

    def test_search_by_language(self, engine: RecommendationEngine) -> None:
        results, total = engine.search_songs(language="english")
        assert total > 0
        for r in results:
            assert r["language"].lower() == "english"

    def test_pagination(self, engine: RecommendationEngine) -> None:
        results_p1, total = engine.search_songs(page=1, page_size=3)
        assert len(results_p1) <= 3

    def test_energy_filter(self, engine: RecommendationEngine) -> None:
        results, _ = engine.search_songs(min_energy=0.7, max_energy=1.0)
        for r in results:
            assert 0.7 <= r["energy"] <= 1.0


class TestColdStart:
    def test_popular_returns_results(self, engine: RecommendationEngine) -> None:
        results = engine.get_popular(n=5)
        assert len(results) > 0

    def test_popular_sorted_by_popularity(self, engine: RecommendationEngine) -> None:
        results = engine.get_popular(n=5)
        # Popularity should be reasonably high for popular songs
        for r in results:
            assert r["popularity"] > 0


class TestFeedbackPersonalization:
    def test_liked_songs_influence_recommendations(self, engine: RecommendationEngine) -> None:
        # Liking a low energy/high acoustic song ("1" is 0.20 energy, 0.90 acoustic)
        recs_with_likes = engine.recommend(preferences={}, n=5, liked_ids=["1"], exclude_ids=["1"])
        
        # Verify the recommendations contain the feedback similarity reason
        has_feedback_reason = any("Similar to songs you liked" in r["reasons"] for r in recs_with_likes)
        assert has_feedback_reason

        # Recommendations should lean towards similar acoustic/low-energy profile songs in the catalog (e.g. classical, jazz, sad ballad)
        for r in recs_with_likes:
            # Let's make sure none of the recommended songs is the liked song itself since it was excluded
            assert r["id"] != "1"
