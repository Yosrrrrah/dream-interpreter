"""
Oniromancer — Main Application Entry Point
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.api import router
from dotenv import load_dotenv
import os, pathlib

load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY is not set in .env")

app = FastAPI(
    title="Oniromancer — Dream Intelligence Platform",
    version="5.0",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")

# Serve frontend files
FRONTEND = pathlib.Path(__file__).parent.parent / "frontend"
if FRONTEND.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND)), name="static")

    @app.get("/")
    async def landing(): return FileResponse(str(FRONTEND / "landing.html"))

    @app.get("/app")
    async def main_app(): return FileResponse(str(FRONTEND / "index.html"))

    @app.get("/dashboard")
    async def dashboard(): return FileResponse(str(FRONTEND / "dashboard.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
