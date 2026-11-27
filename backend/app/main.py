import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.api import chat, videos
from app.core.config import get_settings
from app.core.db import init_db
from app.core.storage import LocalStorage, get_storage
from app.tools.registry import TOOLS, load_all

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    load_all()
    logging.getLogger("app").info("tools registered: %s", ", ".join(TOOLS))
    yield


app = FastAPI(title="Video Analysis Platform", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])
app.include_router(videos.router)
app.include_router(chat.router)


@app.get("/api/health")
def health():
    s = get_settings()
    return {"ok": True, "task_mode": s.task_mode, "storage": s.storage_backend, "vlm_model": s.vlm_model,
            "agent_model": s.agent_model_name, "tools": {n: t.tier for n, t in TOOLS.items()}}


@app.get("/media/{key:path}")
def media(key: str):
    """Serves stored files for the local storage backend (supports HTTP range requests for seeking)."""
    storage = get_storage()
    if not isinstance(storage, LocalStorage):
        raise HTTPException(404)
    try:
        path = storage.local_path(key)
    except ValueError:
        raise HTTPException(404)
    if path is None or not path.is_file():
        raise HTTPException(404)
    return FileResponse(path)
