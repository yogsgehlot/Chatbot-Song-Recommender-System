# MusicMind — Conversational AI Music Recommender System

[![Python Version](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-009688.svg)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Stateful--AI-orange.svg)](https://github.com/langchain-ai/langgraph)
[![Spotify API](https://img.shields.io/badge/Spotify--API-Integrated-1ED760.svg)](https://developer.spotify.com/documentation/web-api/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED.svg)](https://www.docker.com/)

An end-to-end Agentic AI Music Recommendation & Chatbot system that uses FastAPI, LangGraph stateful conversation workflows, Spotify/iTunes API integrations, and a hybrid content-based recommendation engine.

### Tags / Topics for GitHub
`#recommender-system` `#fastapi` `#langgraph` `#spotify-api` `#machine-learning` `#chatbot` `#vanilla-js` `#python-313` `#ai-music-assistant` `#agentic-ai` `#content-based-filtering` `#docker` `#sqlite3` `#glassmorphism` `#web-audio-api`

---

# Project Overview

This project evolves from a basic university song recommender into a production-grade conversational agent. It simulates a real-world cognitive music assistant capable of:

* Stateful multi-turn conversational chat
* Music preference extraction & refinement using LLMs
* External API catalog ingestion (Spotify & iTunes Search integration)
* Free keyless fallback for instant real-world track retrieval
* Hybrid content-based scoring (TF-IDF + Cosine Similarity)
* Dynamic UX analysis (Valence, Danceability, Energy metrics)
* Session-based user feedback loops (Like / Dislike personalization)
* Production-ready Docker containerization

---

# Tech Stack

## Agentic AI / NLP

* **LangGraph:** Stateful multi-turn conversation orchestrator
* **Gemini AI / OpenAI / Anthropic:** Structured preference extraction
* **Pydantic v2:** Response schemas and settings validation

## Recommender & Data Processing

* **Scikit-learn:** Cosine similarity & MinMaxScaler
* **Pandas & NumPy:** Ingest, clean, and score datasets
* **External APIs:** Spotify Web API (spotipy) & iTunes Search API

## Backend / API

* **FastAPI:** High-performance async REST API
* **SQLAlchemy 2.x:** Async database interactions (SQLite)
* **Uvicorn:** ASGI web server

## Frontend

* **Vanilla HTML5 & CSS3:** Sleek "Classic Champagne & Gold" glassmorphic UI
* **Vanilla JavaScript:** Real-time web audio, metrics aggregation, state management

## Deployment / Testing

* **Docker & Docker Compose:** Containerization
* **Pytest:** Integration and unit testing suite

---

# Dataset

The system supports two catalog modes:

1. **Curated In-Memory Catalog:**
   - Source: `backend/data/songs.csv`
   - Generated using a custom dataset builder containing 204 highly diverse pop, lofi, hip-hop, classical, rock, Bollywood, and Punjabi songs.
2. **Global Real-World API Catalog:**
   - Instantly fetches tracks from the **Spotify Web API** or **iTunes API** (over 100M songs, real artwork, and 30-second preview streams).

---

# Problem Statement

Standard recommendation algorithms suffer from cold-start problems and rigid keyword search. Traditional chat interfaces do not capture nuanced user preferences or explain *why* songs were matched.

This project solves this by:
* Leveraging stateful agents to extract implicit mood, tempo, and language preferences.
* Dynamically mapping user conversational intent to acoustic metrics (Valence, Energy, Acousticness).
* Combining real-time user session likes/dislikes directly into the scoring formula.
* Generating rich LLM explanations for each recommended song.

---

# Project Structure

```bash
Chatbot-Song-Recommender-System/
│
├── backend/
│   ├── app/
│   │   ├── ai/                 # Intent classification, prompts, LangGraph workflow
│   │   ├── api/v1/             # Endpoints (chat, songs, feedback)
│   │   ├── core/               # Configuration settings
│   │   ├── db/                 # Database session management
│   │   ├── models/             # SQLAlchemy ORM and Pydantic schemas
│   │   ├── recommender/        # Cosine ranking, exploration jitter, Spotify/iTunes API
│   │   │   ├── engine.py
│   │   │   └── external_provider.py
│   │   ├── services/           # Business logic orchestrators
│   │   └── main.py             # FastAPI App root
│   │
│   ├── data/
│   │   └── songs.csv           # Seed song catalog
│   │
│   ├── scripts/
│   │   └── generate_sample_data.py
│   │
│   ├── tests/                  # Pytest unit and integration test suite
│   │   ├── integration/
│   │   └── unit/
│   │
│   ├── Dockerfile
│   ├── pyproject.toml
│   └── requirements.txt
│
├── docs/
│   └── project_report.md       # Full university report compilation
│
├── frontend/
│   ├── static/
│   │   ├── css/
│   │   │   └── styles.css      # "Champagne & Gold" theme rules
│   │   └── js/
│   │       └── app.js          # Chat and metrics visualization logic
│   └── index.html              # Main workspace dashboard
│
├── docker-compose.yml
├── README.md
└── .gitignore
```

---

# Features

* **LangGraph Cognitive Flow:** Intent classification, extraction, retrieve candidates, rank, explain, and respond.
* **Spotify / iTunes Dual Ingestion:** Real-time API query that resolves track details, covers, and audio preview links.
* **Dynamic Taste Profile Dashboard:** Client-side aggregation and animation of the recommendations' acoustic metadata.
* **Interactive Gold Filters:** Country (India/International), language, and genre filtering dropdowns.
* **Onboarding Quick-Pills:** Instant clickable prompts for first-time visitors.
* **Hermetic Fallbacks:** Fallback to local files if offline; bypass external APIs during tests.
* **Exploration Jitter:** Stochastic ranking diversity prevents deterministic recommendation loops.

---

# Recommender System Pipeline

```text
       User Query
           │
           ▼
    LangGraph Agent
           │
  (Extract Preferences)
           │
           ├──► Spotify API (If configured) ───┐
           │                                    ├─► Ingest Real Tracks
           ├──► iTunes API (Keyless Fallback) ──┘
           │
           ▼
  Local TF-IDF Catalog (Backup)
           │
           ▼
  Scoring Engine (Mood + Popularity + Feedback)
           │
           ▼
  Exploration Jitter (Diversity Variance)
           │
           ▼
  LangGraph Explanation Generator (LLM reasons)
           │
           ▼
  Visual Taste Dashboard & Gold Cards Grid
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/yogsgehlot/Chatbot-Song-Recommender-System
cd Chatbot-Song-Recommender-System
```

---

## Create Virtual Environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux/Mac

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

# Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

---

# Generate Curated Data

```bash
python scripts/generate_sample_data.py
```

---

# Configure Environment Variables

Create a `.env` file in the `backend/` directory:

```bash
cp .env.example .env
```

Open `.env` and fill in your keys:

```env
# LLM Configuration
GEMINI_API_KEY=your-gemini-api-key-here

# Spotify Credentials (Optional - Falls back to keyless iTunes API if empty)
SPOTIFY_CLIENT_ID=your-spotify-client-id
SPOTIFY_CLIENT_SECRET=your-spotify-client-secret
```

---

# Run FastAPI Server

```bash
uvicorn app.main:app --reload --port 8000
```

---

# Run Tests

```bash
pytest tests/ -v
```

---

# Swagger API Documentation

Open:
```text
http://127.0.0.1:8000/docs
```

---

# API Endpoints

## POST `/api/v1/chat`
Handles conversational messages, extracts preferences, and returns matched recommendations.

### Sample Request
```json
{
  "message": "Give me some high energy hip hop tracks",
  "session_id": "session-1234"
}
```

### Sample Response
```json
{
  "session_id": "session-1234",
  "message": "Here are 5 high-energy hip-hop tracks to get you moving!",
  "intent": "recommendation",
  "recommendations": [
    {
      "song": {
        "id": "it_1440854444",
        "title": "HUMBLE.",
        "artist": "Kendrick Lamar",
        "album": "DAMN.",
        "genre": "hip-hop",
        "language": "english",
        "year": 2017,
        "energy": 0.81,
        "valence": 0.42,
        "tempo": 150.0,
        "danceability": 0.90,
        "acousticness": 0.01,
        "preview_url": "https://audio-ssl.itunes.apple.com/.../preview.mp3",
        "cover_url": "https://is1-ssl.mzstatic.com/image/thumb/...",
        "popularity": 85
      },
      "score": 0.942,
      "reasons": [
        "Matches high energy requests",
        "Classic Kendrick Lamar hip-hop beat"
      ],
      "rank": 1
    }
  ]
}
```

---

## POST `/api/v1/feedback`
Submits session feedback to bias future recommendations.

### Sample Request
```json
{
  "song_id": "it_1440854444",
  "session_id": "session-1234",
  "feedback_type": "like"
}
```

---

# Docker Setup

## Build & Run with Docker Compose

```bash
docker-compose up --build
```

---

# Future Improvements

* **Vector Search Database:** Ingesting FAISS/Milvus for hybrid embedding semantic song search.
* **Multi-User Profiles:** Persistent database schemas with OAuth2 login.
* **Audio-Wave Visualizations:** Real-time canvas representation of played music.
* **Auto-Generated Playlists:** Exporting curated recommended songs directly to users' Spotify accounts.

---

# Resume Highlights

* **Designed Stateful AI Agent:** Built multi-turn music assistant utilizing LangGraph state machines, reducing conversational failure rates by handling refinements seamlessly.
* **Engineered Dual API Ingestion System:** Constructed an external provider layer merging Spotify API with an iTunes Search API fallback to deliver real-world songs, cover art, and preview audios.
* **Implemented Custom Hybrid Recommender:** Formulated a content-based indexing and scoring pipeline combining TF-IDF, Cosine Similarity, and dynamic session-based liked-song weights.
* **Constructed High-End UI/UX Dashboard:** Designed a glassmorphic dashboard showcasing real-time animated acoustic taste parameters (Valence, Energy, etc.).

---

# Key Learnings

* Stateful conversational workflows (LangGraph).
* Audio feature mapping and recommendation models.
* Dynamic client-side API metric aggregation.
* Fallback and credential management practices.
* Hermetic API mock testing practices in python (`pytest`).

---

# Author

Yogesh Gehlot  
AI/ML Engineer | Backend Developer

## Connect With Me

- GitHub: [yogsgehlot](https://github.com/yogsgehlot)
- LinkedIn: [Yogesh Gehlot](https://linkedin.com/in/yogsgehlot)
- X: [@yogsgehlot](https://x.com/yogsgehlot)

---

# License

This project is for educational and portfolio purposes.
