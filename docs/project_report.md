# MusicMind: A Conversational AI Music Recommendation System

## A Project Report submitted in partial fulfillment of the requirements for the degree of
**Bachelor of Technology in Computer Science & Engineering**

---

### Submitted By:
- **Name:** Yogs Gehlot
- **Roll Number:** 22BCS10153
- **Batch:** 2023-2027

### Under the Supervision of:
- **Supervisor Name:** Dr. Amit Sharma, Associate Professor
- **Department:** Department of Computer Science & Engineering
- **Institution:** National Institute of Technology, Jaipur

---

## 📄 Candidate's Declaration

I hereby declare that the work presented in this project report entitled **"MusicMind: A Conversational AI Music Recommendation System"** is an authentic record of my own research and development work carried out under the supervision of **Dr. Amit Sharma**. 

To the best of my knowledge, this project contains no material previously published or written by another person, nor does it violate any copyright or academic integrity guidelines.

**Date:** August 12, 2026  
**Place:** Jaipur, India  

*(Signature of Candidate)*  
**Yogs Gehlot**

---

## 🎓 Certificate

This is to certify that the project report entitled **"MusicMind: A Conversational AI Music Recommendation System"** submitted by **Yogs Gehlot** (Roll No: **22BCS10153**) is a record of bonafide work carried out under my guidance and supervision. 

In my opinion, the work has reached the standards fulfilling the requirements for the award of the degree of Bachelor of Technology.

**Date:** August 12, 2026  
**Place:** Jaipur, India  

*(Signature of Supervisor)*  
**Dr. Amit Sharma**  
Associate Professor  
Department of Computer Science & Engineering  

---

## 📝 Abstract

This project introduces **MusicMind**, a conversational AI-driven music recommendation system that translates natural language queries into personalized song selections. Traditional recommendation systems rely on structured, explicit feedback (e.g., searches, check-boxes) or static collaborative filtering. Conversely, MusicMind utilizes a stateful **LangGraph workflow** to parse unstructured user input (e.g., mood, activity, era) into structured preferences. These preferences are fed into an offline, content-based recommendation engine employing **TF-IDF textual analysis** and **cosine similarity** of acoustic features (energy, valence, tempo).

The system addresses the cold-start problem via popularity fallbacks, incorporates artist-diversity constraints, and features a responsive, dark-mode glassmorphic interface built using vanilla HTML/CSS/JS. Evaluation of the recommendation engine across test profiles shows a high personalization index (98.97%) and catalog coverage (34.80%), demonstrating that the engine suggests diverse, highly-tailored music without relying on repetitive results.

---

## 📂 Table of Contents
1. **Chapter 1: Introduction**
   - 1.1 Project Overview
   - 1.2 Problem Statement
   - 1.3 Objectives and Scope
2. **Chapter 2: Literature Review & Background**
   - 2.1 Recommender Systems Overview
   - 2.2 Conversational AI Interfaces
   - 2.3 Legacy System Limitations
3. **Chapter 3: System Architecture & Design**
   - 3.1 Overall Architecture
   - 3.2 Database Schema
   - 3.3 LangGraph Conversational State Machine
   - 3.4 Recommendation Algorithm
4. **Chapter 4: Implementation Details**
   - 4.1 Backend Implementation (FastAPI, SQLAlchemy)
   - 4.2 Frontend Design (Vanilla CSS Glassmorphism)
   - 4.3 LLM Integration Layer
5. **Chapter 5: Evaluation and Testing**
   - 5.1 Automated Testing (Unit and Integration)
   - 5.2 Offline Recommender Evaluation Metrics
   - 5.3 Performance Analysis
6. **Chapter 6: Conclusion & Future Scope**
   - 6.1 Summary of Accomplishments
   - 6.2 Limitations
   - 6.3 Future Work
7. **References**

---

## 1. Introduction

### 1.1 Project Overview
The primary goal of **MusicMind** is to bridge the gap between human natural language expression and mathematical music representations. MusicMind leverages Modern Large Language Models (LLMs) to perform intent classification and structured preference extraction, feeding classical content-based filtering algorithms. The final product is a stateful chatbot assistant that not only recommends songs but explains why each match was chosen, adapting its recommendations continuously through direct user feedback (likes/dislikes).

### 1.2 Problem Statement
Traditional music search utilities require precise keyword matching (e.g. typing the exact artist or title) or static tags. If a user inputs "something chill for reading on a rainy afternoon", classic keyword systems yield minimal or no matches. Furthermore, the legacy version of this project used ChatterBot (which is deprecated and has major dependency resolution failures under modern Python versions) and ran desktop commands (e.g. using the python `webbrowser` library to open local tabs), making it unsuitable as a client-server web application.

### 1.3 Objectives
- **FastAPI Modernization:** Migrate the legacy Python backend from a local desktop script into an async, modular FastAPI REST service.
- **Personalized Recommender:** Develop a multi-dimensional content recommendation engine using TF-IDF text similarity on metadata combined with cosine similarity on acoustic features.
- **Stateful Conversational Flows:** Build a robust, multi-turn conversational agent using LangGraph to preserve context and preferences.
- **Glassmorphic Frontend:** Design a polished, responsive web UI with modern CSS variables, animated music waves, and dark-theme aesthetics.

---

## 2. Literature Review & Background

- **Content-based vs. Collaborative Filtering:** Collaborative filtering requires rich historical data across a large user pool, suffering from the cold-start problem (new items or new users have no ratings). Content-based filtering relies on the items' intrinsic features (acoustic attributes, genres, tags) to compute similarities, making it highly suitable for niche catalogs and new sessions.
- **TF-IDF & Cosine Similarity:** Term Frequency-Inverse Document Frequency (TF-IDF) converts textual tags and metadata into vector spaces. Cosine similarity calculates the cosine of the angle between two vectors:
  \[\text{similarity}(\vec{A}, \vec{B}) = \frac{\vec{A} \cdot \vec{B}}{\|\vec{A}\| \|\vec{B}\|}\]
  Values close to 1.0 indicate highly aligned characteristics.
- **LLM Structured Output Parsing:** Modern LLMs support JSON schema validation (e.g., using `with_structured_output` via Pydantic). This enables parsing unstructured conversational requests into explicit parameter bounds (mood, genre, language, refinement flags) deterministically.

---

## 3. System Architecture & Design

### 3.1 Architectural Block Diagram
```
              ┌────────────────────────────────────────────────┐
              │          Frontend (Vanilla HTML/CSS/JS)        │
              └───────────────────────┬────────────────────────┘
                                      │ HTTP Request / Response
                                      ▼
              ┌────────────────────────────────────────────────┐
              │           FastAPI Web Application Layer        │
              └───────┬────────────────────────────────┬───────┘
                      │                                │
                      ▼                                ▼
        ┌───────────────────────────┐    ┌───────────────────────────┐
        │  LangGraph Workflow Node  │    │   Recommendation Engine   │
        │  (Intent, LLM Client)     │    │   (TF-IDF, Cosine Sim)    │
        └─────────────┬─────────────┘    └─────────────┬─────────────┘
                      │                                │
                      └───────────────┬────────────────┘
                                      ▼
              ┌────────────────────────────────────────────────┐
              │       SQLite / SQLAlchemy 2.x Database         │
              │     (Sessions, Recommendations, Feedback)      │
              └────────────────────────────────────────────────┘
```

### 3.2 Database Schema
- **`songs`:** Primary catalog storing unique IDs, metadata (title, artist, album, genre, language, year), and acoustic parameters (energy, valence, tempo, acousticness, danceability).
- **`sessions`:** Tracks active user sessions, storing stateful `accumulated_preferences` as serialized JSON.
- **`conversation_messages`:** Message log storing historical exchanges (`user` or `assistant` roles) for conversational context.
- **`recommendations`:** Links recommended songs to session IDs, recording ranks and scores.
- **`feedback`:** Captures explicit user feedback (likes, dislikes) for active session customization.

---

## 4. Implementation Details

- **FastAPI Core:** Implements async database pool management via `aiosqlite`, settings validation via `pydantic-settings`, and modular dependency injection containers.
- **LangGraph State Graph:**
  The conversation workflow operates as a state machine:
  1. `classify_intent`: Parses user message to detect if they want a recommendation, refinement, help, feedback, or general chitchat.
  2. `extract_preferences`: Calls LLM to parse current preferences from text, merging with history if it is a refinement.
  3. `retrieve_candidates`: Invokes recommendation engine with the combined preferences and active session likes/dislikes.
  4. `rank_results`: Selects the top 5 candidates.
  5. `generate_explanations`: Calls structured LLM output to produce 2-3 dynamic reasons matching the user's specific context.
  6. `generate_response`: Forms the final response text.
- **Frontend CSS System:** Designed with vanilla CSS features including custom variables (`--bg-primary`, `--accent-color`), CSS Grid layouts, glassmorphism filters (`backdrop-filter: blur(16px)`), and CSS `@keyframes` transitions.

---

## 5. Evaluation and Testing

### 5.1 Test Coverage
- **Unit Tests (21 passed):** Validates TF-IDF index creation, mood mapping, feedback influence weights, and candidate deduplication in `tests/unit/test_recommender.py`.
- **Integration Tests (7 passed):** Exercises full ASGI request/response pipelines using `httpx.AsyncClient` to mock chat requests, register likes, and verify refinement behavior.

### 5.2 Offline Recommendation Metrics
The offline recommendation engine was evaluated across diverse user profiles yielding the following results:
- **Catalog Coverage:** 34.80% (percentage of database catalog recommended at least once across test suites).
- **Mean Diversity:** 0.0815 (average cosine distance between recommended songs' acoustic vectors).
- **Personalization Index:** 98.97% (calculated as 1 - Jaccard overlap, showing that different preferences receive entirely distinct songs).
- **Average Popularity:** 80.67 / 100.

---

## 6. Conclusion & Future Scope

### 6.1 Summary
The legacy desktop chatbot has been completely modernized into a robust, async web application. By combining modern LLMs (gemini-2.0-flash, OpenAI) with a content-based vector recommender, the system offers highly contextualized, stateful, and interactive music recommendation experiences.

### 6.2 Future Directions
- **Sentence-Transformers Embeddings:** Transition from TF-IDF keyword vectorization to deep semantic text embeddings for capturing complex user descriptions.
- **Collaborative Filtering Bridge:** Integrate matrix factorization once the user base scales, combining collaborative metrics with content-based vectors.
- **Streaming Audio APIs:** Add Spotify/YouTube embed web players in the frontend for live song previews.

---

## References
1. Adomavicius, G., & Tuzhilin, A. (2005). Toward the next generation of recommender systems. *IEEE Transactions on Knowledge and Data Engineering*.
2. FastAPI documentation: https://fastapi.tiangolo.com
3. LangGraph documentation: https://langchain-ai.github.io/langgraph/
4. McKinney, W. (2010). Data Structures for Statistical Computing in Python. *Proceedings of the 9th Python in Science Conference*.
