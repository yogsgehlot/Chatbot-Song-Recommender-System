"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import pytest_asyncio


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def sample_dataset_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Shared sample dataset for all tests."""
    import csv
    import uuid

    tmp = tmp_path_factory.mktemp("data")
    csv_path = tmp / "songs.csv"

    songs = [
        {"id": str(uuid.uuid4()), "title": "Calm Study Beat", "artist": "Lo-Fi Artist", "genre": "lo-fi",
         "language": "instrumental", "year": 2022, "energy": 0.20, "valence": 0.55, "tempo": 78.0,
         "danceability": 0.38, "acousticness": 0.90, "popularity": 72, "mood_tags": "calm,study,focus",
         "album": "", "preview_url": "", "cover_url": ""},
        {"id": str(uuid.uuid4()), "title": "Happy Pop Song", "artist": "Pop Artist", "genre": "pop",
         "language": "english", "year": 2020, "energy": 0.85, "valence": 0.92, "tempo": 120.0,
         "danceability": 0.88, "acousticness": 0.05, "popularity": 90, "mood_tags": "happy,upbeat,dance",
         "album": "", "preview_url": "", "cover_url": ""},
        {"id": str(uuid.uuid4()), "title": "Hindi Love Song", "artist": "Arijit Singh", "genre": "bollywood",
         "language": "hindi", "year": 2018, "energy": 0.35, "valence": 0.70, "tempo": 80.0,
         "danceability": 0.50, "acousticness": 0.65, "popularity": 85, "mood_tags": "romantic,love,bollywood",
         "album": "", "preview_url": "", "cover_url": ""},
        {"id": str(uuid.uuid4()), "title": "Workout Banger", "artist": "Hip Hop Artist", "genre": "hip-hop",
         "language": "english", "year": 2021, "energy": 0.90, "valence": 0.65, "tempo": 150.0,
         "danceability": 0.85, "acousticness": 0.02, "popularity": 88, "mood_tags": "workout,energetic,motivational",
         "album": "", "preview_url": "", "cover_url": ""},
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["id", "title", "artist", "album", "genre", "language", "year",
                      "energy", "valence", "tempo", "danceability", "acousticness",
                      "popularity", "mood_tags", "preview_url", "cover_url"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(songs)

    return csv_path
