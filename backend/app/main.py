"""
FastAPI application entry point.

Startup sequence:
1. Load settings from .env
2. Initialize database tables
3. Load recommendation engine (CSV → memory)
4. Initialize LLM provider
5. Mount API routes
6. Serve
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.v1.router import router as api_v1_router
from app.config import get_settings
from app.db.session import close_db, init_db
from app.dependencies import initialize_dependencies
from app.models.schemas import ErrorResponse, HealthResponse

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Application Lifecycle
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown."""
    logger.info("Starting Conversational AI Music Recommender...")

    # Initialize database
    await init_db()

    # Seed database from catalog CSV if empty
    from app.db.session import get_session_factory
    from app.db.seed import seed_database_from_csv
    session_factory = get_session_factory()
    async with session_factory() as db_session:
        try:
            await seed_database_from_csv(db_session, settings.dataset_path)
        except Exception as err:
            logger.error("Database seeding failed: %s", err)

    # Load engine + LLM (may raise if dataset/key missing)
    try:
        initialize_dependencies()
    except FileNotFoundError as e:
        logger.critical(
            "STARTUP FAILED: %s\n"
            "Fix: Run 'python scripts/generate_sample_data.py' in the backend/ directory.",
            e,
        )
        raise
    except Exception as e:
        logger.critical("STARTUP FAILED: %s", e)
        raise

    logger.info("Application started successfully. Listening on %s:%d", settings.app_host, settings.app_port)
    yield  # Application is running


    # Shutdown
    logger.info("Shutting down...")
    await close_db()
    logger.info("Shutdown complete.")


# ---------------------------------------------------------------------------
# App Instance
# ---------------------------------------------------------------------------


app = FastAPI(
    title="MusicMind — Conversational AI Music Recommender",
    description=(
        "An AI-powered music recommendation system that understands natural language. "
        "Tell it your mood, activity, or preference and get personalized song recommendations."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Global Exception Handler
# ---------------------------------------------------------------------------


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Unhandled exception on %s %s: %s", request.method, request.url, exc)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

# Mount API routes
app.include_router(api_v1_router)


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """Health check endpoint — verifies all components are operational."""
    from app.dependencies import get_engine

    try:
        engine = get_engine()
        catalog_size = engine.catalog_size
        db_status = "connected"
    except Exception:
        catalog_size = 0
        db_status = "error"

    return HealthResponse(
        status="ok" if db_status == "connected" else "degraded",
        version="1.0.0",
        catalog_size=catalog_size,
        llm_provider=settings.llm_provider,
        database=db_status,
    )


# ---------------------------------------------------------------------------
# Static files — serve the frontend
# ---------------------------------------------------------------------------

try:
    app.mount("/static", StaticFiles(directory="../frontend/static"), name="static")
except RuntimeError:
    logger.warning("Frontend static directory not found — skipping static file mounting.")

# Serve index.html at root if frontend exists
try:
    from pathlib import Path
    from fastapi.responses import FileResponse

    FRONTEND_INDEX = Path("../frontend/index.html")

    @app.get("/", include_in_schema=False)
    async def serve_frontend() -> FileResponse:
        if FRONTEND_INDEX.exists():
            return FileResponse(str(FRONTEND_INDEX))
        return JSONResponse({"message": "MusicMind API is running. See /docs for API documentation."})

except Exception:
    pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.is_development,
        log_level=settings.log_level.lower(),
    )
