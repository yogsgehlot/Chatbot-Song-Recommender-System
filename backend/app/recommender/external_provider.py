import os
import logging
import base64
import requests
from typing import Any

logger = logging.getLogger(__name__)

class ExternalMusicProvider:
    """Unified provider to fetch real-world music recommendations from Spotify and iTunes APIs."""

    def __init__(self) -> None:
        self.spotify_id = os.getenv("SPOTIFY_CLIENT_ID")
        self.spotify_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
        self._spotify_token = None

    @property
    def is_spotify_configured(self) -> bool:
        return bool(self.spotify_id and self.spotify_secret)

    def _auth_spotify(self) -> bool:
        if not self.is_spotify_configured:
            return False
        try:
            auth_url = "https://accounts.spotify.com/api/token"
            auth_header = base64.b64encode(f"{self.spotify_id}:{self.spotify_secret}".encode("utf-8")).decode("utf-8")
            headers = {
                "Authorization": f"Basic {auth_header}",
                "Content-Type": "application/x-www-form-urlencoded"
            }
            data = {"grant_type": "client_credentials"}
            res = requests.post(auth_url, headers=headers, data=data, timeout=6)
            if res.status_code == 200:
                self._spotify_token = res.json().get("access_token")
                return True
            return False
        except Exception as e:
            logger.warning("Spotify Auth exception: %s", e)
            return False

    def recommend(self, preferences: dict[str, Any], n: int = 5) -> list[dict[str, Any]] | None:
        """Get recommendations from Spotify (if configured) or iTunes (as keyless alternative)."""
        # Try Spotify first
        if self.is_spotify_configured:
            recs = self._recommend_spotify(preferences, n)
            if recs:
                logger.info("Fetched %d recommendations from Spotify API", len(recs))
                return recs

        # Fallback to iTunes Search API (completely public, no credentials needed)
        recs = self._recommend_itunes(preferences, n)
        if recs:
            logger.info("Fetched %d recommendations from iTunes Search API", len(recs))
            return recs

        return None

    def _recommend_spotify(self, preferences: dict[str, Any], n: int) -> list[dict[str, Any]] | None:
        if not self._spotify_token and not self._auth_spotify():
            return None

        try:
            genre = (preferences.get("genre") or "pop").lower().strip()
            genre_map = {
                "hip hop": "hip-hop", "hiphop": "hip-hop", "rap": "hip-hop",
                "punjabi": "indian", "hindi": "indian", "bollywood": "indian",
                "lofi": "chill", "lo-fi": "chill", "electronic": "electronic",
                "edm": "edm", "rock": "rock", "pop": "pop", "jazz": "jazz", "classical": "classical"
            }
            spotify_genre = genre_map.get(genre, genre if genre in ["pop", "rock", "hip-hop", "jazz", "classical", "chill", "dance", "electronic", "indian"] else "pop")

            params: dict[str, Any] = {
                "seed_genres": spotify_genre,
                "limit": n
            }

            # Map parameters
            energy = preferences.get("energy_level")
            if energy == "high": params["target_energy"] = 0.8
            elif energy == "low": params["target_energy"] = 0.3
            elif energy == "medium": params["target_energy"] = 0.6

            tempo = preferences.get("tempo")
            if tempo == "fast": params["target_tempo"] = 135
            elif tempo == "slow": params["target_tempo"] = 80
            elif tempo == "medium": params["target_tempo"] = 110

            mood = (preferences.get("mood") or "").lower()
            if "happy" in mood or "energetic" in mood: params["target_valence"] = 0.8
            elif "sad" in mood or "depressed" in mood: params["target_valence"] = 0.2
            elif "calm" in mood or "relaxed" in mood:
                params["target_valence"] = 0.5
                params["target_acousticness"] = 0.6

            headers = {
                "Authorization": f"Bearer {self._spotify_token}",
                "Content-Type": "application/json"
            }
            res = requests.get("https://api.spotify.com/v1/recommendations", headers=headers, params=params, timeout=6)
            
            if res.status_code == 401:
                if self._auth_spotify():
                    headers["Authorization"] = f"Bearer {self._spotify_token}"
                    res = requests.get("https://api.spotify.com/v1/recommendations", headers=headers, params=params, timeout=6)

            if res.status_code == 200:
                tracks = res.json().get("tracks", [])
                return [self._format_spotify_track(t) for t in tracks]
            return None
        except Exception as e:
            logger.warning("Spotify recommendation request failed: %s", e)
            return None

    def _recommend_itunes(self, preferences: dict[str, Any], n: int) -> list[dict[str, Any]] | None:
        """Fetch search hits from iTunes Search API as a high-quality keyless music source."""
        try:
            genre = (preferences.get("genre") or "").strip()
            lang = (preferences.get("language") or "").strip()
            
            # Construct a rich search term
            terms = []
            if genre:
                terms.append(genre)
            if lang:
                terms.append(lang)
            
            search_term = " ".join(terms) if terms else "top hits"
            
            url = "https://itunes.apple.com/search"
            params = {
                "term": search_term,
                "media": "music",
                "entity": "song",
                "limit": n
            }
            
            res = requests.get(url, params=params, timeout=6)
            if res.status_code == 200:
                results = res.json().get("results", [])
                return [self._format_itunes_track(r) for r in results]
            return None
        except Exception as e:
            logger.warning("iTunes Search API recommendation query failed: %s", e)
            return None

    def _format_spotify_track(self, track: dict[str, Any]) -> dict[str, Any]:
        album = track.get("album", {})
        images = album.get("images", [])
        cover_url = images[0].get("url") if images else None
        preview_url = track.get("preview_url")
        artists = [a.get("name") for a in track.get("artists", [])]
        artist_name = ", ".join(artists) if artists else "Unknown Artist"

        return {
            "id": f"sp_{track.get('id', '')}",
            "title": track.get("name", "Unknown Title"),
            "artist": artist_name,
            "album": album.get("name", "Unknown Album"),
            "genre": "pop",
            "language": "english",
            "year": int(album.get("release_date", "2026")[:4]) if album.get("release_date") else 2026,
            "energy": 0.7,
            "valence": 0.6,
            "tempo": 120.0,
            "danceability": 0.7,
            "acousticness": 0.2,
            "popularity": track.get("popularity", 50),
            "mood_tags": ["spotify-recs"],
            "preview_url": preview_url,
            "cover_url": cover_url,
            "reasons": ["Trending match from Spotify API"]
        }

    def _format_itunes_track(self, track: dict[str, Any]) -> dict[str, Any]:
        cover_url = track.get("artworkUrl100")
        if cover_url:
            cover_url = cover_url.replace("100x100bb.jpg", "400x400bb.jpg")

        year = 2026
        release_date = track.get("releaseDate")
        if release_date:
            try:
                year = int(release_date[:4])
            except ValueError:
                pass

        return {
            "id": f"it_{track.get('trackId', '')}",
            "title": track.get("trackName", "Unknown Title"),
            "artist": track.get("artistName", "Unknown Artist"),
            "album": track.get("collectionName", "Unknown Album"),
            "genre": track.get("primaryGenreName", "pop").lower(),
            "language": "english",
            "year": year,
            "energy": 0.6,
            "valence": 0.5,
            "tempo": 115.0,
            "danceability": 0.6,
            "acousticness": 0.3,
            "popularity": 80,
            "mood_tags": ["itunes-hits"],
            "preview_url": track.get("previewUrl"),
            "cover_url": cover_url,
            "reasons": [f"Popular {track.get('primaryGenreName', 'music')} hit on iTunes Platform"]
        }
