"""
Database seeding utility.
Populates the database songs catalog from songs.csv if it's empty.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import Song

logger = logging.getLogger(__name__)


async def seed_database_from_csv(db: AsyncSession, csv_path: str | Path) -> None:
    """Seed the database songs table from the CSV file if it's empty."""
    path = Path(csv_path)
    if not path.exists():
        logger.warning("Catalog CSV not found at %s. Skipping database seeding.", path)
        return

    # Check if we already have songs in the database
    result = await db.execute(select(Song).limit(1))
    if result.scalar_one_or_none() is not None:
        logger.debug("Database songs table is already seeded.")
        return

    logger.info("Database songs table is empty. Seeding from %s...", path)
    songs_to_insert = []

    try:
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Handle pandas nan values / empty strings
                def clean_str(val: str | None) -> str | None:
                    if not val:
                        return None
                    v = str(val).strip()
                    return v if v and v.lower() != "nan" else None

                def clean_float(val: str | None) -> float | None:
                    if not val:
                        return None
                    try:
                        f_val = float(val)
                        import math
                        return f_val if not math.isnan(f_val) else None
                    except ValueError:
                        return None

                def clean_int(val: str | None) -> int | None:
                    if not val:
                        return None
                    try:
                        return int(float(val))
                    except ValueError:
                        return None

                mood_tags = []
                if row.get("mood_tags"):
                    mood_tags = [t.strip() for t in row["mood_tags"].split(",") if t.strip()]

                song = Song(
                    id=row["id"],
                    title=row["title"],
                    artist=row["artist"],
                    album=clean_str(row.get("album")),
                    genre=clean_str(row.get("genre")),
                    language=row.get("language", "english").lower(),
                    year=clean_int(row.get("year")),
                    energy=clean_float(row.get("energy")),
                    valence=clean_float(row.get("valence")),
                    tempo=clean_float(row.get("tempo")),
                    danceability=clean_float(row.get("danceability")),
                    acousticness=clean_float(row.get("acousticness")),
                    instrumentalness=clean_float(row.get("instrumentalness")),
                    loudness=clean_float(row.get("loudness")),
                    mood_tags=mood_tags,
                    popularity=clean_int(row.get("popularity")) or 0,
                    preview_url=clean_str(row.get("preview_url")),
                    cover_url=clean_str(row.get("cover_url")),
                )
                songs_to_insert.append(song)

        if songs_to_insert:
            db.add_all(songs_to_insert)
            await db.commit()
            logger.info("Successfully seeded %d songs into the database.", len(songs_to_insert))
    except Exception as e:
        await db.rollback()
        logger.error("Failed to seed database from CSV: %s", e, exc_info=True)
