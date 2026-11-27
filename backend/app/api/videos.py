import asyncio
import json
import re
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agent.analysis import get_or_build_report
from app.api.serializers import sse, step_dict, tool_run_dict, video_dict
from app.core.db import SessionLocal, get_db
from app.core.storage import LocalStorage, get_storage, url_or_none, video_prefix
from app.core.tasks import enqueue_ingest
from app.ingest.pipeline import create_jobs
from app.models import Clip, Conversation, Job, Shot, ToolRun, TranscriptSegment, Video

router = APIRouter(prefix="/api")


def _get_video(db: Session, video_id: uuid.UUID) -> Video:
    v = db.get(Video, video_id)
    if v is None:
        raise HTTPException(404, "video not found")
    return v


class CreateVideo(BaseModel):
    filename: str
    content_type: str = "video/mp4"
    size: int | None = None


@router.post("/videos")
def create_video(body: CreateVideo, db: Session = Depends(get_db)):
    vid = uuid.uuid4()
    ext = Path(body.filename).suffix.lower() or ".mp4"
    safe_ext = ext if re.fullmatch(r"\.[a-z0-9]{1,5}", ext) else ".mp4"
    key = f"{video_prefix(vid)}/source{safe_ext}"
    title = Path(body.filename).stem.replace("_", " ").replace("-", " ").strip() or "Untitled video"
    v = Video(id=vid, title=title[:300], filename=body.filename[:300], storage_key=key, status="uploading")
    db.add(v)
    db.commit()
    url, headers = get_storage().upload_target(key, body.content_type, str(vid))
    return {"video": video_dict(v), "upload_url": url, "upload_method": "PUT", "upload_headers": headers}


@router.put("/videos/{video_id}/upload")
async def upload_local(video_id: uuid.UUID, request: Request):
    """Upload target for the local storage backend (S3 uses a pre-signed URL instead)."""
    storage = get_storage()
    if not isinstance(storage, LocalStorage):
        raise HTTPException(400, "uploads go to the pre-signed S3 URL")
    with SessionLocal() as db:
        v = _get_video(db, video_id)
        key = v.storage_key
    dest = storage._path(key)
    dest.parent.mkdir(parents=True, exist_ok=True)
    size = 0
    with open(dest, "wb") as f:
        async for chunk in request.stream():
            f.write(chunk)
            size += len(chunk)
    if size == 0:
        raise HTTPException(400, "empty upload")
    return {"ok": True, "size": size}


@router.post("/videos/{video_id}/ingest")
def ingest(video_id: uuid.UUID, db: Session = Depends(get_db)):
    v = _get_video(db, video_id)
    if v.status in ("queued", "processing"):
        raise HTTPException(409, "already ingesting")
    v.status, v.error = "queued", None
    create_jobs(db, v.id)
    db.commit()
    db.refresh(v)
    enqueue_ingest(str(v.id))
    return video_dict(v)


@router.get("/videos")
def list_videos(db: Session = Depends(get_db)):
    return [video_dict(v) for v in db.scalars(select(Video).order_by(Video.created_at.desc()))]


@router.get("/videos/{video_id}")
def get_video(video_id: uuid.UUID, db: Session = Depends(get_db)):
    return video_dict(_get_video(db, video_id))


@router.delete("/videos/{video_id}", status_code=204)
def delete_video(video_id: uuid.UUID, db: Session = Depends(get_db)):
    v = _get_video(db, video_id)
    db.delete(v)
    db.commit()
    get_storage().delete_prefix(video_prefix(video_id))
    return Response(status_code=204)


def _steps(db: Session, video_id) -> list[dict]:
    return [step_dict(j) for j in db.scalars(select(Job).where(Job.video_id == video_id).order_by(Job.order))]


@router.get("/videos/{video_id}/steps")
def get_steps(video_id: uuid.UUID, db: Session = Depends(get_db)):
    _get_video(db, video_id)
    return _steps(db, video_id)


@router.get("/videos/{video_id}/events")
async def video_events(video_id: uuid.UUID, request: Request):
    def snapshot():
        with SessionLocal() as db:
            v = db.get(Video, video_id)
            if v is None:
                return None
            return {"video": video_dict(v), "steps": _steps(db, video_id)}

    if await run_in_threadpool(snapshot) is None:
        raise HTTPException(404, "video not found")

    async def stream():
        last = None
        while not await request.is_disconnected():
            snap = await run_in_threadpool(snapshot)
            if snap is None:
                return
            encoded = json.dumps(snap, default=str)
            if encoded != last:
                last = encoded
                yield sse("progress", snap)
            if snap["video"]["status"] in ("ready", "failed"):
                yield sse("done", {"video": snap["video"]})
                return
            yield ": keep-alive\n\n"
            await asyncio.sleep(0.75)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.get("/videos/{video_id}/timeline")
def timeline(video_id: uuid.UUID, db: Session = Depends(get_db)):
    v = _get_video(db, video_id)
    if v.status != "ready":
        raise HTTPException(409, "video is not ready")
    shots = db.scalars(select(Shot).where(Shot.video_id == video_id).order_by(Shot.index)).all()
    clips = db.scalars(select(Clip).where(Clip.video_id == video_id).order_by(Clip.index)).all()
    segs = db.scalars(select(TranscriptSegment).where(TranscriptSegment.video_id == video_id)
                      .order_by(TranscriptSegment.t_start)).all()
    return {
        "duration_s": v.duration_s or 0,
        "shots": [{"id": str(s.id), "index": s.index, "t_start": s.t_start, "t_end": s.t_end} for s in shots],
        "clips": [{"id": str(c.id), "shot_id": str(c.shot_id), "t_start": c.t_start, "t_end": c.t_end,
                   "caption": c.caption, "objects": c.objects or [], "actions": c.actions or [],
                   "on_screen_text": c.on_screen_text or "", "keyframe_url": url_or_none(c.keyframe_key)}
                  for c in clips],
        "transcript": [{"t_start": x.t_start, "t_end": x.t_end, "text": x.text} for x in segs],
    }


class AnalysisRequest(BaseModel):
    refresh: bool = False


@router.post("/videos/{video_id}/analysis")
def analysis(video_id: uuid.UUID, body: AnalysisRequest | None = None, db: Session = Depends(get_db)):
    v = _get_video(db, video_id)
    if v.status != "ready":
        raise HTTPException(409, "video is not ready")
    try:
        return get_or_build_report(video_id, refresh=bool(body and body.refresh))
    except Exception as e:
        raise HTTPException(502, f"analysis failed: {e}")


@router.get("/videos/{video_id}/tool-runs")
def list_tool_runs(video_id: uuid.UUID, db: Session = Depends(get_db)):
    _get_video(db, video_id)
    runs = db.scalars(select(ToolRun).where(ToolRun.video_id == video_id).order_by(ToolRun.created_at.desc()))
    return [tool_run_dict(r) for r in runs]


@router.get("/tool-runs/{run_id}")
def get_tool_run(run_id: uuid.UUID, db: Session = Depends(get_db)):
    r = db.get(ToolRun, run_id)
    if r is None:
        raise HTTPException(404, "tool run not found")
    return tool_run_dict(r)


class SelectBody(BaseModel):
    t: float
    x: float  # 0..1 of the frame width
    y: float  # 0..1 of the frame height


@router.post("/videos/{video_id}/select")
async def select_object_at(video_id: uuid.UUID, body: SelectBody, db: Session = Depends(get_db)):
    """Click-to-ask: outline the object under a click on the frame at t (SAM). Cached per (t, x, y)."""
    v = _get_video(db, video_id)
    if v.status != "ready":
        raise HTTPException(409, "video is not ready")
    db.commit()  # end this request's transaction before the slow work (it would hold locks on videos)
    from app.tools.selection import select_object

    try:
        return await run_in_threadpool(select_object, video_id, body.t, body.x, body.y)
    except Exception as e:
        raise HTTPException(500, f"could not select an object there: {e}") from e


@router.post("/videos/{video_id}/conversations")
def create_conversation(video_id: uuid.UUID, db: Session = Depends(get_db)):
    _get_video(db, video_id)
    c = Conversation(video_id=video_id)
    db.add(c)
    db.commit()
    return {"id": str(c.id), "video_id": str(video_id), "created_at": c.created_at.isoformat()}


@router.get("/videos/{video_id}/conversations")
def list_conversations(video_id: uuid.UUID, db: Session = Depends(get_db)):
    _get_video(db, video_id)
    convs = db.scalars(select(Conversation).where(Conversation.video_id == video_id)
                       .order_by(Conversation.created_at.desc()))
    return [{"id": str(c.id), "video_id": str(c.video_id), "created_at": c.created_at.isoformat()} for c in convs]
