# [PROJECT TITLE]
## Chatbot & Song Recommender System (MusicMind)

**A Project Report submitted in partial fulfillment of the requirements for the degree of**
**Bachelor of Technology / Master of Computer Applications**

*In*
**Computer Science & Engineering / Artificial Intelligence / Information Technology**

---

### Submitted By:
- **Name:** [Your Name]
- **Roll Number:** [Your Roll Number]
- **Batch:** [Your Batch, e.g., 2023-2027]

### Under the Supervision of:
- **Supervisor Name:** [Advisor's Name & Designation]
- **Department:** Department of Computer Science & Engineering
- **Institution:** [University Name / College Name]

---

## 📄 Candidate's Declaration

I hereby declare that the work presented in this project report entitled **"MusicMind: A Conversational AI Music Recommendation System"** is an authentic record of my own research and development work carried out under the supervision of **[Supervisor Name]**. 

To the best of my knowledge, this project contains no material previously published or written by another person, nor does it violate any copyright or academic integrity guidelines.

**Date:** [Submission Date]  
**Place:** [City, Country]  

*(Signature of Candidate)*  
**[Your Name]**

---

## 🎓 Certificate

This is to certify that the project report entitled **"MusicMind: A Conversational AI Music Recommendation System"** submitted by **[Your Name]** (Roll No: **[Your Roll Number]**) is a record of bonafide work carried out under my guidance and supervision. 

In my opinion, the work has reached the standards fulfilling the requirements for the award of the degree of Bachelor of Technology / MCA.

**Date:** [Date]  
**Place:** [Place]  

*(Signature of Supervisor)*  
**[Supervisor Name]**  
[Designation]  
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
Describe the goal of the project here. Explain how MusicMind leverages modern LLMs and classic content-based filtering to create an interactive conversational agent.

### 1.2 Problem Statement
Explain why traditional recommendation engines fail to capture contextual mood or multi-turn conversational adjustments ("make it faster", "only English tracks"). Detail the bugs and architectural failures of the original Flask/ChatterBot setup (e.g., ChatterBot dependency issues on Python 3.10+, server-side `webbrowser` execution).

### 1.3 Objectives
- Build a service-oriented FastAPI backend.
- Design a deterministic content-based filtering algorithm using vector spaces.
- Construct a stateful dialogue manager using LangGraph.
- Deliver a responsive frontend with glassmorphism styling.

---

## 2. Literature Review & Background

Discuss:
- Content-based filtering vs Collaborative filtering.
- TF-IDF Vectorization and Cosine Similarity in information retrieval.
- Large Language Models (LLMs) and structured schema parsing (with-structured-output).

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
Explain the tables:
- `songs`: Title, artist, album, genre, language, year, energy, valence, tempo, acousticness, danceability.
- `sessions`: Session ID, accumulated preferences.
- `conversation_messages`: Message log.
- `recommendations`: Recorded recommendation outcomes.
- `feedback`: User interactions (like, dislike, skip).

---

## 4. Implementation Details

- **FastAPI Core:** Type-safe settings, async database pool management, and dependency injection.
- **LangGraph State Graph:**
  - `classify_intent` -> `extract_preferences` -> `retrieve_candidates` -> `rank_results` -> `generate_response`
- **Frontend CSS System:** Describe the CSS variables, layout grid/flexbox rules, cover art styling, and `@keyframes` animations.

---

## 5. Evaluation and Testing

### 5.1 Test Coverage
- **Unit Tests (20 passed):** Testing individual engine functionality (mood filtering, diversity limit, similar song retrieval).
- **Integration Tests (6 passed):** Full ASGITransport request cycles testing `/health`, `/songs`, `/feedback`, and mock `/chat`.

### 5.2 Offline Recommendation Metrics
Present the evaluation metrics computed across the dataset:
- **Catalog Coverage:** 34.80% (percentage of songs recommended at least once across test profiles).
- **Mean Diversity:** 0.0815 (average cosine distance of features among recommended tracks).
- **Personalization Index:** 98.97% (1 - Jaccard overlap between profile recommendations).
- **Average Popularity:** 80.67 / 100.

---

## 6. Conclusion & Future Scope

### 6.1 Summary
Summarize the successful modernization of the Flask codebase into a modular, production-ready system.

### 6.2 Future Directions
- **Vector Embeddings:** Transition from TF-IDF + Cosine to semantic embeddings (e.g., sentence-transformers) for advanced keyword match.
- **Streaming UI:** Implementing Server-Sent Events (SSE) for real-time text completion.
- **Collaborative Filtering:** Storing user session records to transition to hybrid content-collaborative methods.

---

## References
1. Adomavicius, G., & Tuzhilin, A. (2005). Toward the next generation of recommender systems. *IEEE Transactions on Knowledge and Data Engineering*.
2. FastAPI documentation: https://fastapi.tiangolo.com
3. LangGraph documentation: https://langchain-ai.github.io/langgraph/
