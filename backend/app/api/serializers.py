from app.core.storage import url_or_none
from app.models import Job, ToolRun, Video


def video_progress(jobs: list[Job]) -> float:
    if not jobs:
        return 0.0
    return round(sum(1.0 if j.status in ("done", "skipped") else (j.progress or 0) for j in jobs) / len(jobs), 3)


def video_dict(v: Video) -> dict:
    jobs = list(v.jobs or [])
    return {
        "id": str(v.id), "title": v.title, "filename": v.filename, "status": v.status,
        "progress": 1.0 if v.status == "ready" else video_progress(jobs),
        "duration_s": v.duration_s, "fps": v.fps, "width": v.width, "height": v.height, "has_audio": v.has_audio,
        "thumbnail_url": url_or_none(v.thumbnail_key),
        "playback_url": url_or_none(v.proxy_key) if v.status == "ready" else None,
        "error": v.error, "created_at": v.created_at.isoformat(),
    }


def step_dict(j: Job) -> dict:
    return {"step": j.step, "status": j.status, "progress": j.progress or 0.0, "detail": j.detail, "error": j.error}


def tool_run_dict(r: ToolRun) -> dict:
    return {"id": str(r.id), "tool": r.tool, "tier": r.tier, "params": r.params, "status": r.status,
            "result": r.result, "overlay_url": url_or_none(r.overlay_key), "created_at": r.created_at.isoformat()}


def sse(event: str, data) -> str:
    import json

    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"
