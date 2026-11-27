"""Tool registry: every analysis capability the agent can call, with its cost tier.

Tiers: memory (reads the video memory) < vlm (one VLM call) < gpu / plugin (vision models, cached in tool_runs).
"""

import hashlib
import json
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select

from app.core.db import session_scope
from app.core.storage import url_or_none
from app.models import ToolRun


@dataclass
class ToolOutput:
    summary: str                      # one line for the UI step row
    data: dict                        # what the agent model sees (kept compact)
    tool_run_id: str | None = None
    overlay_url: str | None = None
    image_url: str | None = None      # a still picture to show in the chat (e.g. segment_scene)
    clip_url: str | None = None       # a video to show in the chat (e.g. make_highlight_clip)


@dataclass
class Tool:
    name: str
    tier: str
    description: str
    parameters: dict
    fn: Callable[..., ToolOutput]
    label: Callable[[dict], str] = field(default=lambda args: "")

    def openai_schema(self) -> dict:
        return {"type": "function", "function": {
            "name": self.name,
            "description": f"[{self.tier.upper()} tier] {self.description}",
            "parameters": self.parameters,
        }}


TOOLS: dict[str, Tool] = {}
# GPU tool implementations: name -> fn(video_id, params, run_id) -> (result dict, overlay_key | None)
GPU_IMPLS: dict[str, Callable[[uuid.UUID, dict, uuid.UUID], tuple[dict, str | None]]] = {}


def register(tool: Tool) -> Tool:
    TOOLS[tool.name] = tool
    return tool


def params_hash(tool_name: str, params: dict) -> str:
    return hashlib.sha256(f"{tool_name}:{json.dumps(params, sort_keys=True, default=str)}".encode()).hexdigest()


def find_cached_run(video_id, tool_name: str, params: dict) -> ToolRun | None:
    with session_scope() as s:
        return s.scalar(select(ToolRun).where(
            ToolRun.video_id == video_id, ToolRun.tool == tool_name,
            ToolRun.params_hash == params_hash(tool_name, params), ToolRun.status == "done",
        ).order_by(ToolRun.created_at.desc()).limit(1))


def execute_gpu_tool(tool_name: str, video_id: str, params: dict) -> dict:
    """Run (or fetch from cache) a GPU-tier tool. Runs in the worker (celery) or in-process (inline)."""
    vid = uuid.UUID(str(video_id))
    cached = find_cached_run(vid, tool_name, params)
    if cached:
        return {"tool_run_id": str(cached.id), "result": cached.result, "overlay_key": cached.overlay_key,
                "cached": True}
    tier = TOOLS[tool_name].tier if tool_name in TOOLS else "gpu"
    with session_scope() as s:
        run = ToolRun(video_id=vid, tool=tool_name, tier=tier, params=params,
                      params_hash=params_hash(tool_name, params), status="running")
        s.add(run)
        s.flush()
        run_id = run.id
    started = time.time()
    try:
        result, overlay_key = GPU_IMPLS[tool_name](vid, params, run_id)
    except Exception as e:
        with session_scope() as s:
            r = s.get(ToolRun, run_id)
            r.status, r.result = "failed", {"error": str(e)}
        raise
    with session_scope() as s:
        r = s.get(ToolRun, run_id)
        r.status, r.result, r.overlay_key = "done", result, overlay_key
        r.gpu_seconds = round(time.time() - started, 2)
    return {"tool_run_id": str(run_id), "result": result, "overlay_key": overlay_key, "cached": False}


def call_gpu(tool_name: str, video_id, params: dict) -> tuple[dict, str, str | None, bool]:
    """Agent-side helper: dispatch to the GPU queue and wait. Returns (result, run_id, overlay_url, cached)."""
    from app.core.tasks import run_gpu_tool

    out = run_gpu_tool(tool_name, str(video_id), params)
    return out["result"], out["tool_run_id"], url_or_none(out.get("overlay_key")), out.get("cached", False)


def load_all() -> None:
    """Import tool modules so they register themselves."""
    from app.tools import clips, memory, motion, scene, selection, vision, vlm  # noqa: F401
    from app.tools.plugins import checkout  # noqa: F401
