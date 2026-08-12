"""
Generate a curated sample music dataset for development and testing.

This creates a realistic 200-song CSV with proper audio features,
matching the schema expected by the recommendation engine.

Usage:
    cd backend
    python scripts/generate_sample_data.py

Output:
    backend/data/songs.csv  (200 songs across multiple genres/languages/moods)
"""

from __future__ import annotations

import csv
import os
import random
import uuid
from pathlib import Path


# Seed for reproducibility
random.seed(42)

OUTPUT_PATH = Path(__file__).parent.parent / "data" / "songs.csv"


# ---------------------------------------------------------------------------
# Song catalog data
# ---------------------------------------------------------------------------

CATALOG: list[dict] = [
    # --- Lo-Fi / Study ---
    {"title": "Midnight Lofi", "artist": "Chillhop Essentials", "genre": "lo-fi", "language": "instrumental", "year": 2022, "energy": 0.22, "valence": 0.55, "tempo": 78.0, "danceability": 0.42, "acousticness": 0.88, "popularity": 72, "mood_tags": "calm,study,focus,chill"},
    {"title": "Coffee Study Beats", "artist": "Lofi Girl", "genre": "lo-fi", "language": "instrumental", "year": 2021, "energy": 0.18, "valence": 0.60, "tempo": 82.0, "danceability": 0.38, "acousticness": 0.91, "popularity": 85, "mood_tags": "study,calm,focus,chill"},
    {"title": "Late Night Studying", "artist": "ChilledCow", "genre": "lo-fi", "language": "instrumental", "year": 2020, "energy": 0.20, "valence": 0.50, "tempo": 75.0, "danceability": 0.35, "acousticness": 0.90, "popularity": 78, "mood_tags": "study,focus,late-night,calm"},
    {"title": "Rainy Day Focus", "artist": "Ambition Beats", "genre": "lo-fi", "language": "instrumental", "year": 2023, "energy": 0.16, "valence": 0.45, "tempo": 70.0, "danceability": 0.30, "acousticness": 0.93, "popularity": 65, "mood_tags": "calm,focus,study,relaxing"},
    {"title": "Bookstore Vibes", "artist": "Lo-Fi Hip Hop Radio", "genre": "lo-fi", "language": "instrumental", "year": 2022, "energy": 0.19, "valence": 0.58, "tempo": 80.0, "danceability": 0.40, "acousticness": 0.89, "popularity": 70, "mood_tags": "study,calm,chill,focus"},

    # --- Calm / Acoustic ---
    {"title": "The Night Will Always Win", "artist": "Manchester Orchestra", "genre": "indie", "language": "english", "year": 2014, "energy": 0.28, "valence": 0.32, "tempo": 88.0, "danceability": 0.38, "acousticness": 0.72, "popularity": 58, "mood_tags": "calm,melancholic,acoustic,indie"},
    {"title": "Skinny Love", "artist": "Bon Iver", "genre": "indie folk", "language": "english", "year": 2007, "energy": 0.24, "valence": 0.28, "tempo": 95.0, "danceability": 0.32, "acousticness": 0.85, "popularity": 74, "mood_tags": "sad,melancholic,acoustic,folk"},
    {"title": "Holocene", "artist": "Bon Iver", "genre": "indie folk", "language": "english", "year": 2011, "energy": 0.22, "valence": 0.35, "tempo": 90.0, "danceability": 0.35, "acousticness": 0.88, "popularity": 81, "mood_tags": "calm,melancholic,peaceful,acoustic"},
    {"title": "Re: Stacks", "artist": "Bon Iver", "genre": "indie folk", "language": "english", "year": 2007, "energy": 0.20, "valence": 0.30, "tempo": 78.0, "danceability": 0.30, "acousticness": 0.90, "popularity": 72, "mood_tags": "calm,melancholic,acoustic"},
    {"title": "Be Still", "artist": "The Killers", "genre": "rock", "language": "english", "year": 2012, "energy": 0.30, "valence": 0.40, "tempo": 84.0, "danceability": 0.35, "acousticness": 0.68, "popularity": 65, "mood_tags": "calm,reflective,acoustic"},

    # --- Happy / Upbeat ---
    {"title": "Happy", "artist": "Pharrell Williams", "genre": "pop", "language": "english", "year": 2013, "energy": 0.80, "valence": 0.96, "tempo": 160.0, "danceability": 0.82, "acousticness": 0.10, "popularity": 92, "mood_tags": "happy,upbeat,dance,feel-good"},
    {"title": "Can't Stop the Feeling!", "artist": "Justin Timberlake", "genre": "pop", "language": "english", "year": 2016, "energy": 0.84, "valence": 0.95, "tempo": 113.0, "danceability": 0.85, "acousticness": 0.05, "popularity": 90, "mood_tags": "happy,dance,upbeat,feel-good"},
    {"title": "Good as Hell", "artist": "Lizzo", "genre": "pop", "language": "english", "year": 2016, "energy": 0.75, "valence": 0.92, "tempo": 96.0, "danceability": 0.80, "acousticness": 0.15, "popularity": 88, "mood_tags": "happy,empowering,upbeat,feel-good"},
    {"title": "Shake It Off", "artist": "Taylor Swift", "genre": "pop", "language": "english", "year": 2014, "energy": 0.80, "valence": 0.90, "tempo": 160.0, "danceability": 0.88, "acousticness": 0.08, "popularity": 91, "mood_tags": "happy,upbeat,dance,fun"},
    {"title": "Uptown Funk", "artist": "Mark Ronson ft. Bruno Mars", "genre": "funk pop", "language": "english", "year": 2014, "energy": 0.88, "valence": 0.88, "tempo": 115.0, "danceability": 0.90, "acousticness": 0.05, "popularity": 95, "mood_tags": "happy,dance,upbeat,party"},
    {"title": "Levitating", "artist": "Dua Lipa", "genre": "pop", "language": "english", "year": 2020, "energy": 0.82, "valence": 0.88, "tempo": 103.0, "danceability": 0.86, "acousticness": 0.08, "popularity": 93, "mood_tags": "happy,upbeat,dance,feel-good"},
    {"title": "Blinding Lights", "artist": "The Weeknd", "genre": "synth-pop", "language": "english", "year": 2019, "energy": 0.83, "valence": 0.78, "tempo": 171.0, "danceability": 0.85, "acousticness": 0.03, "popularity": 98, "mood_tags": "upbeat,energetic,synth,retro"},

    # --- Sad / Emotional ---
    {"title": "Someone Like You", "artist": "Adele", "genre": "pop", "language": "english", "year": 2011, "energy": 0.28, "valence": 0.18, "tempo": 67.0, "danceability": 0.25, "acousticness": 0.88, "popularity": 95, "mood_tags": "sad,heartbreak,emotional,ballad"},
    {"title": "The Night We Met", "artist": "Lord Huron", "genre": "indie folk", "language": "english", "year": 2014, "energy": 0.26, "valence": 0.22, "tempo": 90.0, "danceability": 0.28, "acousticness": 0.80, "popularity": 82, "mood_tags": "sad,melancholic,nostalgic,romantic"},
    {"title": "Fix You", "artist": "Coldplay", "genre": "alternative rock", "language": "english", "year": 2005, "energy": 0.38, "valence": 0.35, "tempo": 69.0, "danceability": 0.30, "acousticness": 0.70, "popularity": 89, "mood_tags": "sad,emotional,hopeful,comforting"},
    {"title": "Let Her Go", "artist": "Passenger", "genre": "folk pop", "language": "english", "year": 2012, "energy": 0.32, "valence": 0.25, "tempo": 75.0, "danceability": 0.42, "acousticness": 0.82, "popularity": 85, "mood_tags": "sad,heartbreak,acoustic,folk"},
    {"title": "Skinny Love", "artist": "Birdy", "genre": "indie pop", "language": "english", "year": 2011, "energy": 0.22, "valence": 0.20, "tempo": 90.0, "danceability": 0.28, "acousticness": 0.88, "popularity": 78, "mood_tags": "sad,emotional,acoustic,ballad"},
    {"title": "Hurt", "artist": "Johnny Cash", "genre": "country", "language": "english", "year": 2002, "energy": 0.25, "valence": 0.12, "tempo": 80.0, "danceability": 0.22, "acousticness": 0.85, "popularity": 88, "mood_tags": "sad,deep,melancholic,emotional"},
    {"title": "All I Want", "artist": "Kodaline", "genre": "indie rock", "language": "english", "year": 2012, "energy": 0.30, "valence": 0.22, "tempo": 85.0, "danceability": 0.28, "acousticness": 0.75, "popularity": 80, "mood_tags": "sad,romantic,heartbreak,emotional"},

    # --- Romantic ---
    {"title": "Perfect", "artist": "Ed Sheeran", "genre": "pop", "language": "english", "year": 2017, "energy": 0.42, "valence": 0.72, "tempo": 95.0, "danceability": 0.55, "acousticness": 0.50, "popularity": 94, "mood_tags": "romantic,love,ballad,acoustic"},
    {"title": "All of Me", "artist": "John Legend", "genre": "pop", "language": "english", "year": 2013, "energy": 0.38, "valence": 0.68, "tempo": 120.0, "danceability": 0.50, "acousticness": 0.62, "popularity": 92, "mood_tags": "romantic,love,ballad,emotional"},
    {"title": "Thinking Out Loud", "artist": "Ed Sheeran", "genre": "pop", "language": "english", "year": 2014, "energy": 0.45, "valence": 0.75, "tempo": 79.0, "danceability": 0.60, "acousticness": 0.45, "popularity": 91, "mood_tags": "romantic,love,dance,feel-good"},
    {"title": "La Vie en Rose", "artist": "Édith Piaf", "genre": "chanson", "language": "french", "year": 1946, "energy": 0.28, "valence": 0.70, "tempo": 75.0, "danceability": 0.38, "acousticness": 0.80, "popularity": 85, "mood_tags": "romantic,classic,love,nostalgic"},
    {"title": "Make You Feel My Love", "artist": "Adele", "genre": "pop", "language": "english", "year": 2008, "energy": 0.30, "valence": 0.62, "tempo": 76.0, "danceability": 0.40, "acousticness": 0.78, "popularity": 86, "mood_tags": "romantic,love,acoustic,emotional"},
    {"title": "At Last", "artist": "Etta James", "genre": "soul", "language": "english", "year": 1960, "energy": 0.32, "valence": 0.78, "tempo": 68.0, "danceability": 0.38, "acousticness": 0.82, "popularity": 87, "mood_tags": "romantic,classic,soul,love"},

    # --- Workout / Energetic ---
    {"title": "Eye of the Tiger", "artist": "Survivor", "genre": "rock", "language": "english", "year": 1982, "energy": 0.92, "valence": 0.78, "tempo": 109.0, "danceability": 0.72, "acousticness": 0.02, "popularity": 90, "mood_tags": "workout,motivational,energetic,rock"},
    {"title": "Lose Yourself", "artist": "Eminem", "genre": "hip-hop", "language": "english", "year": 2002, "energy": 0.90, "valence": 0.55, "tempo": 171.0, "danceability": 0.75, "acousticness": 0.05, "popularity": 92, "mood_tags": "workout,motivational,intense,hip-hop"},
    {"title": "Till I Collapse", "artist": "Eminem", "genre": "hip-hop", "language": "english", "year": 2002, "energy": 0.93, "valence": 0.48, "tempo": 171.0, "danceability": 0.70, "acousticness": 0.03, "popularity": 90, "mood_tags": "workout,intense,motivational,hip-hop"},
    {"title": "Stronger", "artist": "Kanye West", "genre": "hip-hop", "language": "english", "year": 2007, "energy": 0.85, "valence": 0.62, "tempo": 104.0, "danceability": 0.82, "acousticness": 0.02, "popularity": 88, "mood_tags": "workout,motivational,energetic,hip-hop"},
    {"title": "Run The World (Girls)", "artist": "Beyoncé", "genre": "pop", "language": "english", "year": 2011, "energy": 0.90, "valence": 0.82, "tempo": 130.0, "danceability": 0.88, "acousticness": 0.03, "popularity": 87, "mood_tags": "workout,empowering,dance,energetic"},
    {"title": "Power", "artist": "Kanye West", "genre": "hip-hop", "language": "english", "year": 2010, "energy": 0.88, "valence": 0.58, "tempo": 89.0, "danceability": 0.70, "acousticness": 0.05, "popularity": 86, "mood_tags": "workout,powerful,motivational,hip-hop"},
    {"title": "Sabotage", "artist": "Beastie Boys", "genre": "rock", "language": "english", "year": 1994, "energy": 0.96, "valence": 0.65, "tempo": 142.0, "danceability": 0.65, "acousticness": 0.01, "popularity": 84, "mood_tags": "workout,intense,energetic,rock"},
    {"title": "Thunder", "artist": "Imagine Dragons", "genre": "pop rock", "language": "english", "year": 2017, "energy": 0.82, "valence": 0.74, "tempo": 168.0, "danceability": 0.72, "acousticness": 0.03, "popularity": 88, "mood_tags": "workout,energetic,motivational,pop"},

    # --- Hindi Bollywood ---
    {"title": "Tum Hi Ho", "artist": "Arijit Singh", "genre": "bollywood", "language": "hindi", "year": 2013, "energy": 0.30, "valence": 0.48, "tempo": 72.0, "danceability": 0.42, "acousticness": 0.72, "popularity": 88, "mood_tags": "romantic,sad,emotional,bollywood"},
    {"title": "Dil Diyan Gallan", "artist": "Atif Aslam", "genre": "bollywood", "language": "hindi", "year": 2017, "energy": 0.35, "valence": 0.65, "tempo": 78.0, "danceability": 0.50, "acousticness": 0.65, "popularity": 82, "mood_tags": "romantic,love,bollywood"},
    {"title": "Ae Dil Hai Mushkil", "artist": "Arijit Singh", "genre": "bollywood", "language": "hindi", "year": 2016, "energy": 0.32, "valence": 0.40, "tempo": 80.0, "danceability": 0.45, "acousticness": 0.68, "popularity": 85, "mood_tags": "sad,romantic,heartbreak,bollywood"},
    {"title": "Channa Mereya", "artist": "Arijit Singh", "genre": "bollywood", "language": "hindi", "year": 2016, "energy": 0.38, "valence": 0.35, "tempo": 84.0, "danceability": 0.48, "acousticness": 0.70, "popularity": 90, "mood_tags": "sad,heartbreak,emotional,bollywood"},
    {"title": "Tera Ban Jaunga", "artist": "Akhil Sachdeva", "genre": "bollywood", "language": "hindi", "year": 2019, "energy": 0.28, "valence": 0.60, "tempo": 75.0, "danceability": 0.40, "acousticness": 0.78, "popularity": 80, "mood_tags": "romantic,love,calm,bollywood"},
    {"title": "Kesariya", "artist": "Arijit Singh", "genre": "bollywood", "language": "hindi", "year": 2022, "energy": 0.40, "valence": 0.72, "tempo": 88.0, "danceability": 0.55, "acousticness": 0.62, "popularity": 92, "mood_tags": "romantic,love,happy,bollywood"},
    {"title": "Raataan Lambiyan", "artist": "Jubin Nautiyal", "genre": "bollywood", "language": "hindi", "year": 2021, "energy": 0.35, "valence": 0.68, "tempo": 82.0, "danceability": 0.50, "acousticness": 0.65, "popularity": 88, "mood_tags": "romantic,love,calm,bollywood"},
    {"title": "Butta Bomma", "artist": "Armaan Malik", "genre": "bollywood", "language": "telugu", "year": 2020, "energy": 0.72, "valence": 0.85, "tempo": 108.0, "danceability": 0.78, "acousticness": 0.20, "popularity": 82, "mood_tags": "happy,love,upbeat,bollywood"},
    {"title": "Srivalli", "artist": "Sid Sriram", "genre": "bollywood", "language": "telugu", "year": 2021, "energy": 0.65, "valence": 0.70, "tempo": 98.0, "danceability": 0.68, "acousticness": 0.28, "popularity": 86, "mood_tags": "romantic,energetic,love,bollywood"},
    {"title": "Manike", "artist": "Yohani", "genre": "pop", "language": "sinhala", "year": 2021, "energy": 0.68, "valence": 0.80, "tempo": 105.0, "danceability": 0.75, "acousticness": 0.22, "popularity": 78, "mood_tags": "happy,upbeat,dance"},

    # --- Punjabi ---
    {"title": "Lahore", "artist": "Guru Randhawa", "genre": "punjabi pop", "language": "punjabi", "year": 2017, "energy": 0.78, "valence": 0.82, "tempo": 114.0, "danceability": 0.82, "acousticness": 0.12, "popularity": 84, "mood_tags": "happy,dance,upbeat,punjabi"},
    {"title": "High Rated Gabru", "artist": "Guru Randhawa", "genre": "punjabi pop", "language": "punjabi", "year": 2018, "energy": 0.82, "valence": 0.80, "tempo": 118.0, "danceability": 0.85, "acousticness": 0.08, "popularity": 82, "mood_tags": "upbeat,dance,party,punjabi"},
    {"title": "Lal Ghaghra", "artist": "Guru Randhawa", "genre": "punjabi", "language": "punjabi", "year": 2020, "energy": 0.80, "valence": 0.78, "tempo": 112.0, "danceability": 0.82, "acousticness": 0.10, "popularity": 80, "mood_tags": "dance,upbeat,happy,punjabi"},
    {"title": "Kala Chashma", "artist": "Baar Baar Dekho", "genre": "punjabi", "language": "punjabi", "year": 2016, "energy": 0.85, "valence": 0.85, "tempo": 120.0, "danceability": 0.88, "acousticness": 0.06, "popularity": 88, "mood_tags": "party,dance,upbeat,punjabi"},
    {"title": "Suit Suit", "artist": "Guru Randhawa", "genre": "punjabi pop", "language": "punjabi", "year": 2016, "energy": 0.80, "valence": 0.82, "tempo": 116.0, "danceability": 0.84, "acousticness": 0.10, "popularity": 82, "mood_tags": "dance,upbeat,happy,punjabi"},

    # --- Hip-Hop / Rap ---
    {"title": "HUMBLE.", "artist": "Kendrick Lamar", "genre": "hip-hop", "language": "english", "year": 2017, "energy": 0.88, "valence": 0.42, "tempo": 150.0, "danceability": 0.92, "acousticness": 0.02, "popularity": 94, "mood_tags": "hip-hop,intense,motivational,rap"},
    {"title": "God's Plan", "artist": "Drake", "genre": "hip-hop", "language": "english", "year": 2018, "energy": 0.68, "valence": 0.47, "tempo": 77.0, "danceability": 0.75, "acousticness": 0.02, "popularity": 96, "mood_tags": "hip-hop,chill,rap,feel-good"},
    {"title": "Rockstar", "artist": "Post Malone", "genre": "hip-hop", "language": "english", "year": 2017, "energy": 0.62, "valence": 0.32, "tempo": 160.0, "danceability": 0.78, "acousticness": 0.05, "popularity": 92, "mood_tags": "hip-hop,trap,chill,rap"},
    {"title": "Old Town Road", "artist": "Lil Nas X", "genre": "country rap", "language": "english", "year": 2019, "energy": 0.72, "valence": 0.60, "tempo": 136.0, "danceability": 0.82, "acousticness": 0.18, "popularity": 95, "mood_tags": "fun,upbeat,hip-hop,country"},
    {"title": "In My Feelings", "artist": "Drake", "genre": "hip-hop", "language": "english", "year": 2018, "energy": 0.75, "valence": 0.65, "tempo": 91.0, "danceability": 0.80, "acousticness": 0.05, "popularity": 90, "mood_tags": "hip-hop,dance,feel-good,rap"},
    {"title": "Sicko Mode", "artist": "Travis Scott", "genre": "trap", "language": "english", "year": 2018, "energy": 0.82, "valence": 0.48, "tempo": 155.0, "danceability": 0.78, "acousticness": 0.02, "popularity": 92, "mood_tags": "hip-hop,intense,trap,workout"},
    {"title": "Mask Off", "artist": "Future", "genre": "trap", "language": "english", "year": 2017, "energy": 0.72, "valence": 0.35, "tempo": 85.0, "danceability": 0.80, "acousticness": 0.02, "popularity": 86, "mood_tags": "trap,chill,hip-hop,dark"},

    # --- Classical / Instrumental ---
    {"title": "Clair de Lune", "artist": "Claude Debussy", "genre": "classical", "language": "instrumental", "year": 1905, "energy": 0.12, "valence": 0.55, "tempo": 56.0, "danceability": 0.18, "acousticness": 0.98, "popularity": 85, "mood_tags": "calm,peaceful,classical,study,sleep"},
    {"title": "Moonlight Sonata", "artist": "Ludwig van Beethoven", "genre": "classical", "language": "instrumental", "year": 1801, "energy": 0.18, "valence": 0.32, "tempo": 60.0, "danceability": 0.15, "acousticness": 0.99, "popularity": 88, "mood_tags": "melancholic,calm,classical,study"},
    {"title": "Canon in D", "artist": "Johann Pachelbel", "genre": "classical", "language": "instrumental", "year": 1680, "energy": 0.14, "valence": 0.65, "tempo": 58.0, "danceability": 0.20, "acousticness": 0.99, "popularity": 86, "mood_tags": "calm,peaceful,classical,wedding,study"},
    {"title": "Experience", "artist": "Ludovico Einaudi", "genre": "classical", "language": "instrumental", "year": 2013, "energy": 0.22, "valence": 0.50, "tempo": 72.0, "danceability": 0.25, "acousticness": 0.95, "popularity": 80, "mood_tags": "calm,peaceful,study,emotional,focus"},
    {"title": "Comptine d'un autre été", "artist": "Yann Tiersen", "genre": "classical", "language": "instrumental", "year": 2001, "energy": 0.20, "valence": 0.58, "tempo": 76.0, "danceability": 0.28, "acousticness": 0.96, "popularity": 82, "mood_tags": "calm,nostalgic,study,peaceful"},

    # --- Jazz ---
    {"title": "So What", "artist": "Miles Davis", "genre": "jazz", "language": "instrumental", "year": 1959, "energy": 0.28, "valence": 0.60, "tempo": 136.0, "danceability": 0.42, "acousticness": 0.88, "popularity": 82, "mood_tags": "chill,jazz,study,cool,classic"},
    {"title": "Take Five", "artist": "Dave Brubeck Quartet", "genre": "jazz", "language": "instrumental", "year": 1959, "energy": 0.25, "valence": 0.62, "tempo": 174.0, "danceability": 0.40, "acousticness": 0.90, "popularity": 84, "mood_tags": "chill,jazz,study,classic"},
    {"title": "Fly Me to the Moon", "artist": "Frank Sinatra", "genre": "jazz", "language": "english", "year": 1964, "energy": 0.32, "valence": 0.80, "tempo": 138.0, "danceability": 0.55, "acousticness": 0.82, "popularity": 90, "mood_tags": "romantic,chill,jazz,classic,feel-good"},
    {"title": "L-O-V-E", "artist": "Nat King Cole", "genre": "jazz", "language": "english", "year": 1965, "energy": 0.35, "valence": 0.88, "tempo": 128.0, "danceability": 0.58, "acousticness": 0.78, "popularity": 85, "mood_tags": "romantic,happy,jazz,classic,feel-good"},

    # --- Party / Dance ---
    {"title": "Don't Start Now", "artist": "Dua Lipa", "genre": "disco pop", "language": "english", "year": 2019, "energy": 0.85, "valence": 0.82, "tempo": 124.0, "danceability": 0.90, "acousticness": 0.02, "popularity": 94, "mood_tags": "dance,party,upbeat,feel-good"},
    {"title": "Physical", "artist": "Dua Lipa", "genre": "disco pop", "language": "english", "year": 2020, "energy": 0.88, "valence": 0.78, "tempo": 126.0, "danceability": 0.88, "acousticness": 0.03, "popularity": 92, "mood_tags": "dance,workout,party,upbeat"},
    {"title": "As It Was", "artist": "Harry Styles", "genre": "pop", "language": "english", "year": 2022, "energy": 0.72, "valence": 0.58, "tempo": 174.0, "danceability": 0.80, "acousticness": 0.08, "popularity": 96, "mood_tags": "upbeat,feel-good,dance,pop"},
    {"title": "Watermelon Sugar", "artist": "Harry Styles", "genre": "pop", "language": "english", "year": 2019, "energy": 0.78, "valence": 0.86, "tempo": 95.0, "danceability": 0.82, "acousticness": 0.12, "popularity": 92, "mood_tags": "happy,summer,feel-good,dance"},
    {"title": "drivers license", "artist": "Olivia Rodrigo", "genre": "pop", "language": "english", "year": 2021, "energy": 0.45, "valence": 0.22, "tempo": 72.0, "danceability": 0.38, "acousticness": 0.55, "popularity": 96, "mood_tags": "sad,heartbreak,emotional,ballad"},
    {"title": "good 4 u", "artist": "Olivia Rodrigo", "genre": "pop punk", "language": "english", "year": 2021, "energy": 0.88, "valence": 0.68, "tempo": 166.0, "danceability": 0.78, "acousticness": 0.05, "popularity": 94, "mood_tags": "energetic,angry,empowering,pop"},

    # --- Indie / Alternative ---
    {"title": "Mr. Brightside", "artist": "The Killers", "genre": "indie rock", "language": "english", "year": 2003, "energy": 0.82, "valence": 0.52, "tempo": 148.0, "danceability": 0.62, "acousticness": 0.05, "popularity": 90, "mood_tags": "energetic,indie,anthemic,rock"},
    {"title": "Take Me to Church", "artist": "Hozier", "genre": "alternative", "language": "english", "year": 2013, "energy": 0.72, "valence": 0.32, "tempo": 129.0, "danceability": 0.52, "acousticness": 0.28, "popularity": 88, "mood_tags": "intense,dark,emotional,indie"},
    {"title": "Somebody That I Used to Know", "artist": "Gotye", "genre": "indie pop", "language": "english", "year": 2011, "energy": 0.58, "valence": 0.28, "tempo": 129.0, "danceability": 0.68, "acousticness": 0.22, "popularity": 90, "mood_tags": "sad,heartbreak,indie,pop"},
    {"title": "Seven Nation Army", "artist": "The White Stripes", "genre": "rock", "language": "english", "year": 2003, "energy": 0.78, "valence": 0.40, "tempo": 124.0, "danceability": 0.55, "acousticness": 0.12, "popularity": 88, "mood_tags": "energetic,rock,intense,anthemic"},
    {"title": "Pumped Up Kicks", "artist": "Foster the People", "genre": "indie pop", "language": "english", "year": 2010, "energy": 0.72, "valence": 0.58, "tempo": 128.0, "danceability": 0.80, "acousticness": 0.10, "popularity": 85, "mood_tags": "indie,chill,dance,pop"},
    {"title": "Electric Feel", "artist": "MGMT", "genre": "psychedelic pop", "language": "english", "year": 2007, "energy": 0.68, "valence": 0.75, "tempo": 104.0, "danceability": 0.78, "acousticness": 0.05, "popularity": 84, "mood_tags": "dance,upbeat,indie,chill"},

    # --- R&B / Soul ---
    {"title": "Redbone", "artist": "Childish Gambino", "genre": "r&b", "language": "english", "year": 2016, "energy": 0.48, "valence": 0.55, "tempo": 96.0, "danceability": 0.72, "acousticness": 0.20, "popularity": 87, "mood_tags": "chill,r&b,soulful,groove"},
    {"title": "No Air", "artist": "Jordin Sparks & Chris Brown", "genre": "r&b", "language": "english", "year": 2007, "energy": 0.55, "valence": 0.40, "tempo": 84.0, "danceability": 0.58, "acousticness": 0.18, "popularity": 80, "mood_tags": "sad,r&b,emotional,ballad"},
    {"title": "Adorn", "artist": "Miguel", "genre": "r&b", "language": "english", "year": 2012, "energy": 0.52, "valence": 0.68, "tempo": 88.0, "danceability": 0.72, "acousticness": 0.25, "popularity": 78, "mood_tags": "romantic,r&b,chill,groove"},
    {"title": "Say My Name", "artist": "Destiny's Child", "genre": "r&b", "language": "english", "year": 1999, "energy": 0.75, "valence": 0.62, "tempo": 90.0, "danceability": 0.80, "acousticness": 0.05, "popularity": 86, "mood_tags": "r&b,dance,90s,upbeat"},

    # --- Sleep / Ambient ---
    {"title": "Weightless", "artist": "Marconi Union", "genre": "ambient", "language": "instrumental", "year": 2011, "energy": 0.08, "valence": 0.45, "tempo": 60.0, "danceability": 0.12, "acousticness": 0.95, "popularity": 75, "mood_tags": "sleep,calm,ambient,meditation,relaxing"},
    {"title": "Sleep", "artist": "Max Richter", "genre": "ambient classical", "language": "instrumental", "year": 2015, "energy": 0.10, "valence": 0.42, "tempo": 55.0, "danceability": 0.10, "acousticness": 0.97, "popularity": 72, "mood_tags": "sleep,calm,ambient,peaceful"},
    {"title": "Gymnopédie No.1", "artist": "Erik Satie", "genre": "classical", "language": "instrumental", "year": 1888, "energy": 0.10, "valence": 0.50, "tempo": 52.0, "danceability": 0.15, "acousticness": 0.99, "popularity": 84, "mood_tags": "calm,peaceful,sleep,study,classical"},
    {"title": "Pure Shores", "artist": "All Saints", "genre": "pop", "language": "english", "year": 2000, "energy": 0.42, "valence": 0.68, "tempo": 125.0, "danceability": 0.65, "acousticness": 0.22, "popularity": 76, "mood_tags": "calm,relaxing,chill,90s"},

    # --- 90s Nostalgia ---
    {"title": "Smells Like Teen Spirit", "artist": "Nirvana", "genre": "grunge", "language": "english", "year": 1991, "energy": 0.92, "valence": 0.42, "tempo": 117.0, "danceability": 0.48, "acousticness": 0.02, "popularity": 92, "mood_tags": "90s,rock,intense,anthemic"},
    {"title": "Wonderwall", "artist": "Oasis", "genre": "britpop", "language": "english", "year": 1995, "energy": 0.52, "valence": 0.65, "tempo": 87.0, "danceability": 0.55, "acousticness": 0.42, "popularity": 90, "mood_tags": "90s,nostalgic,indie,classic"},
    {"title": "No Scrubs", "artist": "TLC", "genre": "r&b", "language": "english", "year": 1999, "energy": 0.72, "valence": 0.70, "tempo": 92.0, "danceability": 0.78, "acousticness": 0.08, "popularity": 85, "mood_tags": "90s,r&b,upbeat,feel-good"},
    {"title": "Waterfalls", "artist": "TLC", "genre": "r&b", "language": "english", "year": 1994, "energy": 0.52, "valence": 0.48, "tempo": 88.0, "danceability": 0.62, "acousticness": 0.12, "popularity": 84, "mood_tags": "90s,r&b,emotional,classic"},
    {"title": "Baby One More Time", "artist": "Britney Spears", "genre": "pop", "language": "english", "year": 1998, "energy": 0.75, "valence": 0.68, "tempo": 96.0, "danceability": 0.80, "acousticness": 0.08, "popularity": 88, "mood_tags": "90s,pop,upbeat,nostalgic"},

    # --- Spanish / Latin ---
    {"title": "Despacito", "artist": "Luis Fonsi & Daddy Yankee", "genre": "reggaeton", "language": "spanish", "year": 2017, "energy": 0.80, "valence": 0.82, "tempo": 89.0, "danceability": 0.88, "acousticness": 0.08, "popularity": 98, "mood_tags": "dance,upbeat,latin,summer"},
    {"title": "Hips Don't Lie", "artist": "Shakira", "genre": "pop", "language": "spanish", "year": 2005, "energy": 0.82, "valence": 0.88, "tempo": 100.0, "danceability": 0.90, "acousticness": 0.05, "popularity": 90, "mood_tags": "dance,upbeat,latin,feel-good"},
    {"title": "Con Calma", "artist": "Daddy Yankee", "genre": "reggaeton", "language": "spanish", "year": 2019, "energy": 0.85, "valence": 0.85, "tempo": 96.0, "danceability": 0.88, "acousticness": 0.04, "popularity": 88, "mood_tags": "dance,party,upbeat,latin"},

    # --- Motivational / Inspirational ---
    {"title": "Hall of Fame", "artist": "The Script", "genre": "pop", "language": "english", "year": 2012, "energy": 0.78, "valence": 0.82, "tempo": 85.0, "danceability": 0.68, "acousticness": 0.10, "popularity": 85, "mood_tags": "motivational,inspirational,uplifting,anthemic"},
    {"title": "We Are the Champions", "artist": "Queen", "genre": "rock", "language": "english", "year": 1977, "energy": 0.70, "valence": 0.85, "tempo": 60.0, "danceability": 0.42, "acousticness": 0.15, "popularity": 92, "mood_tags": "motivational,classic,anthemic,uplifting"},
    {"title": "Roar", "artist": "Katy Perry", "genre": "pop", "language": "english", "year": 2013, "energy": 0.80, "valence": 0.88, "tempo": 90.0, "danceability": 0.75, "acousticness": 0.05, "popularity": 90, "mood_tags": "motivational,empowering,upbeat,anthemic"},
    {"title": "Brave", "artist": "Sara Bareilles", "genre": "pop", "language": "english", "year": 2013, "energy": 0.72, "valence": 0.90, "tempo": 110.0, "danceability": 0.72, "acousticness": 0.08, "popularity": 82, "mood_tags": "motivational,empowering,happy,uplifting"},
    {"title": "Fight Song", "artist": "Rachel Platten", "genre": "pop", "language": "english", "year": 2015, "energy": 0.75, "valence": 0.85, "tempo": 98.0, "danceability": 0.70, "acousticness": 0.10, "popularity": 84, "mood_tags": "motivational,empowering,upbeat,inspirational"},
    {"title": "The Climb", "artist": "Miley Cyrus", "genre": "pop", "language": "english", "year": 2009, "energy": 0.58, "valence": 0.78, "tempo": 76.0, "danceability": 0.45, "acousticness": 0.35, "popularity": 82, "mood_tags": "motivational,inspirational,uplifting,ballad"},
    {"title": "Rise Up", "artist": "Andra Day", "genre": "soul", "language": "english", "year": 2015, "energy": 0.55, "valence": 0.72, "tempo": 68.0, "danceability": 0.40, "acousticness": 0.45, "popularity": 80, "mood_tags": "motivational,inspirational,soulful,uplifting"},
]


def generate_additional_songs(count: int = 100) -> list[dict]:
    """Generate additional synthetic songs to fill the dataset."""
    genres = ["pop", "rock", "indie", "hip-hop", "r&b", "jazz", "classical", "lo-fi", "bollywood", "punjabi"]
    languages = ["english", "hindi", "punjabi", "spanish", "french", "instrumental"]
    mood_combos = [
        "calm,study,focus", "happy,upbeat,dance", "sad,emotional,heartbreak",
        "romantic,love,ballad", "energetic,workout,motivational", "chill,relaxing,ambient",
        "party,dance,upbeat", "nostalgic,classic,indie",
    ]
    artists = [
        "The Resonance", "Echo Valley", "Midnight Souls", "Aurora Beats",
        "Crystal Waves", "Neon Dreams", "Velvet Underground Cover", "Deep Space",
        "Urban Flow", "Mountain Echo", "City Lights", "Forest Rain",
    ]

    songs = []
    for i in range(count):
        genre = random.choice(genres)
        lang = random.choice(languages) if genre != "classical" else "instrumental"
        energy = round(random.uniform(0.1, 0.95), 2)
        songs.append({
            "title": f"Track {i + 1:03d}",
            "artist": random.choice(artists),
            "genre": genre,
            "language": lang,
            "year": random.randint(1995, 2024),
            "energy": energy,
            "valence": round(random.uniform(0.1, 0.95), 2),
            "tempo": round(random.uniform(60, 180), 1),
            "danceability": round(random.uniform(0.15, 0.95), 2),
            "acousticness": round(random.uniform(0.02, 0.95), 2),
            "popularity": random.randint(40, 85),
            "mood_tags": random.choice(mood_combos),
        })
    return songs


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    all_songs = list(CATALOG)
    all_songs.extend(generate_additional_songs(100))

    # Assign UUIDs
    fieldnames = [
        "id", "title", "artist", "album", "genre", "language", "year",
        "energy", "valence", "tempo", "danceability", "acousticness",
        "popularity", "mood_tags", "preview_url", "cover_url",
    ]

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for song in all_songs:
            row = {
                "id": str(uuid.uuid4()),
                "album": "",
                "preview_url": "",
                "cover_url": "",
                **song,
            }
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    print(f"[OK] Generated {len(all_songs)} songs -> {OUTPUT_PATH}")
    print(f"   Curated songs: {len(CATALOG)}")
    print(f"   Synthetic filler: {len(all_songs) - len(CATALOG)}")
    print("\nNext step: cd backend && python -m uvicorn app.main:app --reload")


if __name__ == "__main__":
    main()
