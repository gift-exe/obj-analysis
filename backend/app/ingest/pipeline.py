"""Always-on perception layer: turns an uploaded video into the searchable video memory.

Steps: probe -> proxy -> shots -> keyframes -> captions -> transcript -> embeddings -> summaries.
Each step updates its row in `jobs`, which the dashboard streams as progress.
"""

import logging
import math
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import cv2
from sqlalchemy import delete, select

from app.core.config import get_settings
from app.core.db import session_scope
from app.core.media import FrameReader, fmt_t, local_video_path, make_proxy, probe, to_jpeg_bytes
from app.core.storage import get_storage, video_prefix
from app.models import Clip, Job, Shot, Summary, ToolRun, TranscriptSegment, Video

log = logging.getLogger(__name__)

STEPS = ["probe", "proxy", "shots", "keyframes", "captions", "transcript", "embeddings", "summaries"]


class SkipStep(Exception):
    pass


def _now():
    return datetime.now(timezone.utc)


def create_jobs(session, video_id) -> None:
    session.execute(delete(Job).where(Job.video_id == video_id))
    for i, step in enumerate(STEPS):
        session.add(Job(video_id=video_id, step=step, order=i, status="pending"))


class StepTracker:
    def __init__(self, video_id, step: str):
        self.video_id, self.step = video_id, step

    def _update(self, **fields):
        with session_scope() as s:
            job = s.scalar(select(Job).where(Job.video_id == self.video_id, Job.step == self.step))
            for k, v in fields.items():
                setattr(job, k, v)

    def start(self):
        self._update(status="running", progress=0.0, started_at=_now(), error=None)

    def progress(self, frac: float, detail: str | None = None):
        self._update(progress=max(0.0, min(1.0, frac)), detail=detail)

    def done(self, detail: str | None = None):
        fields = dict(status="done", progress=1.0, finished_at=_now())
        if detail is not None:
            fields["detail"] = detail
        self._update(**fields)

    def skipped(self, detail: str):
        self._update(status="skipped", progress=1.0, detail=detail, finished_at=_now())

    def failed(self, error: str):
        self._update(status="failed", error=error[-2000:], finished_at=_now())


def run_ingest(video_id: str) -> None:
    vid = uuid.UUID(str(video_id))
    with session_scope() as s:
        video = s.get(Video, vid)
        if video is None:
            return
        video.status, video.error, video.analysis = "processing", None, None
        # Re-ingest: clear derived memory
        for model in (Clip, Shot, TranscriptSegment, Summary, ToolRun):
            s.execute(delete(model).where(model.video_id == vid))
        create_jobs(s, vid)

    ctx: dict = {"video_id": vid}
    for step in STEPS:
        tracker = StepTracker(vid, step)
        tracker.start()
        try:
            detail = STEP_FUNCS[step](ctx, tracker)
            tracker.done(detail)
        except SkipStep as e:
            tracker.skipped(str(e))
        except Exception as e:
            log.exception("ingest step %s failed", step)
            tracker.failed(f"{e}\n{traceback.format_exc(limit=3)}")
            with session_scope() as s:
                video = s.get(Video, vid)
                video.status, video.error = "failed", f"{step}: {e}"
            return

    with session_scope() as s:
        s.get(Video, vid).status = "ready"


# --- steps -----------------------------------------------------------------------------------


def step_probe(ctx, tr: StepTracker):
    with session_scope() as s:
        video = s.get(Video, ctx["video_id"])
        src = local_video_path(video.storage_key, video.id)
        p = probe(src)
        video.duration_s, video.fps, video.width, video.height, video.has_audio = (
            p.duration_s, p.fps, p.width, p.height, p.has_audio)
    ctx.update(src=src, probe=p)
    return f"{p.width}x{p.height}, {p.fps:.0f} fps, {fmt_t(p.duration_s)}, {'audio' if p.has_audio else 'no audio'}"


def step_proxy(ctx, tr):
    s_ = get_settings()
    vid = ctx["video_id"]
    work = s_.work_dir / str(vid)
    out = work / "proxy.mp4"
    make_proxy(ctx["src"], out, ctx["probe"].has_audio)
    key = f"{video_prefix(vid)}/proxy.mp4"
    get_storage().put_file(key, out, "video/mp4")
    with session_scope() as s:
        s.get(Video, vid).proxy_key = key
    ctx["proxy"] = out
    return "720p H.264 proxy written"


def step_shots(ctx, tr):
    from scenedetect import AdaptiveDetector, detect

    s_ = get_settings()
    duration = ctx["probe"].duration_s
    try:
        scenes = [(a.get_seconds(), b.get_seconds()) for a, b in detect(str(ctx["proxy"]), AdaptiveDetector())]
    except Exception:
        log.exception("scene detection failed; using one shot")
        scenes = []
    if not scenes:
        scenes = [(0.0, duration)]
    # Merge very short shots into their predecessor (flashes, detector noise).
    merged: list[list[float]] = []
    for a, b in scenes:
        if merged and (b - a) < 1.0:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    merged[-1][1] = max(merged[-1][1], duration)

    clips_meta = []
    with session_scope() as s:
        clip_index = 0
        for i, (a, b) in enumerate(merged):
            shot = Shot(video_id=ctx["video_id"], index=i, t_start=a, t_end=b)
            s.add(shot)
            s.flush()
            n = max(1, math.ceil((b - a) / s_.clip_max_seconds - 1e-6))
            step = (b - a) / n
            for j in range(n):
                clip = Clip(video_id=ctx["video_id"], shot_id=shot.id, index=clip_index,
                            t_start=round(a + j * step, 3), t_end=round(a + (j + 1) * step, 3))
                s.add(clip)
                s.flush()
                clips_meta.append((clip.id, clip.t_start, clip.t_end))
                clip_index += 1
    ctx["clips"] = clips_meta
    return f"{len(merged)} shots, {len(clips_meta)} clips"


def step_keyframes(ctx, tr):
    storage = get_storage()
    vid = ctx["video_id"]
    work = get_settings().work_dir / str(vid) / "keyframes"
    work.mkdir(parents=True, exist_ok=True)
    keys = {}
    with FrameReader(ctx["proxy"]) as reader:
        for n, (clip_id, t0, t1) in enumerate(ctx["clips"]):
            frame = reader.at((t0 + t1) / 2)
            if frame is None:
                continue
            path = work / f"{clip_id}.jpg"
            path.write_bytes(to_jpeg_bytes(frame))
            key = f"{video_prefix(vid)}/keyframes/{clip_id}.jpg"
            storage.put_file(key, path, "image/jpeg")
            keys[clip_id] = key
            tr.progress((n + 1) / len(ctx["clips"]), f"{n + 1} / {len(ctx['clips'])} keyframes")
        thumb = reader.at(min(1.0, reader.duration * 0.1))
    if thumb is not None:
        path = work / "thumbnail.jpg"
        path.write_bytes(to_jpeg_bytes(thumb))
        storage.put_file(f"{video_prefix(vid)}/thumbnail.jpg", path, "image/jpeg")
    with session_scope() as s:
        for clip_id, key in keys.items():
            s.get(Clip, clip_id).keyframe_key = key
        if thumb is not None:
            s.get(Video, vid).thumbnail_key = f"{video_prefix(vid)}/thumbnail.jpg"
    return f"{len(keys)} keyframes"


CAPTION_PROMPT = """You are indexing a video so it can be searched later.
The frames above come from one clip of the video, from {t0:.1f}s to {t1:.1f}s, in order.
Describe what is visible and what happens. Be concrete and factual; do not guess beyond what is visible.
Return only a JSON object with these keys:
- "description": 2-4 sentences: the setting, who/what is present, the actions, and what changes over time. Mention the time (in seconds) of notable moments.
- "objects": list of short lowercase nouns for visible people, vehicles, animals and objects (item types, e.g. "apple", "bottle", "truck", "person in blue jacket").
- "actions": list of short verb phrases (e.g. "scans an item", "car turns left").
- "on_screen_text": any readable text (signs, labels, captions), or "".
"""


def step_captions(ctx, tr):
    s_ = get_settings()
    from app.llm import chat_json, frames_content

    clips = ctx["clips"]
    with FrameReader(ctx["proxy"]) as reader:
        batches = [(cid, t0, t1, reader.sample(t0, t1, s_.caption_fps, s_.caption_max_frames))
                   for cid, t0, t1 in clips]

    done = 0

    def caption(item):
        cid, t0, t1, frames = item
        content = frames_content(frames, CAPTION_PROMPT.format(t0=t0, t1=t1))
        try:
            data = chat_json([{"role": "user", "content": content}], max_tokens=700)
        except Exception as e:  # one bad clip shouldn't fail the whole video
            log.warning("caption failed for clip %s: %s", cid, e)
            data = {"description": "", "objects": [], "actions": [], "on_screen_text": ""}
        return cid, data

    failures = 0
    with ThreadPoolExecutor(max_workers=s_.caption_concurrency) as pool:
        for cid, data in pool.map(caption, batches):
            done += 1
            desc = str(data.get("description") or "").strip()
            failures += 0 if desc else 1
            with session_scope() as s:
                clip = s.get(Clip, cid)
                clip.caption = desc
                clip.objects = [str(o).lower().strip() for o in (data.get("objects") or [])][:40]
                clip.actions = [str(a).strip() for a in (data.get("actions") or [])][:20]
                clip.on_screen_text = str(data.get("on_screen_text") or "")
            tr.progress(done / len(clips), f"{done} / {len(clips)} clips captioned")
    if failures == len(clips):
        raise RuntimeError("the VLM returned no captions; check VLM_BASE_URL / VLM_MODEL")
    return f"{len(clips)} clips captioned" + (f" ({failures} empty)" if failures else "")


def step_transcript(ctx, tr):
    if not ctx["probe"].has_audio:
        raise SkipStep("no audio track")
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise SkipStep("faster-whisper not installed (install the 'ml' extra)")
    from app.embeddings import pick_device

    device = "cuda" if pick_device() == "cuda" else "cpu"
    model = WhisperModel(get_settings().whisper_model, device=device,
                         compute_type="float16" if device == "cuda" else "int8")
    segments, _info = model.transcribe(str(ctx["proxy"]), vad_filter=True)
    n = 0
    with session_scope() as s:
        for seg in segments:
            if seg.text.strip():
                s.add(TranscriptSegment(video_id=ctx["video_id"], t_start=seg.start, t_end=seg.end,
                                        text=seg.text.strip()))
                n += 1
    return f"{n} speech segments"


def clip_search_text(clip: Clip) -> str:
    parts = [clip.caption]
    if clip.objects:
        parts.append("Objects: " + ", ".join(clip.objects))
    if clip.actions:
        parts.append("Actions: " + ", ".join(clip.actions))
    if clip.on_screen_text:
        parts.append("Text: " + clip.on_screen_text)
    return "\n".join(p for p in parts if p)


def step_embeddings(ctx, tr):
    try:
        from app import embeddings as emb
    except ImportError:
        raise SkipStep("ML extras not installed; search falls back to keywords")
    vid = ctx["video_id"]
    storage = get_storage()
    with session_scope() as s:
        clips = s.scalars(select(Clip).where(Clip.video_id == vid).order_by(Clip.index)).all()
        segs = s.scalars(select(TranscriptSegment).where(TranscriptSegment.video_id == vid)).all()
        try:
            text_vecs = emb.embed_texts([clip_search_text(c) for c in clips])
        except ImportError:
            raise SkipStep("ML extras not installed; search falls back to keywords")
        for c, v in zip(clips, text_vecs):
            c.caption_emb = v.tolist()
        tr.progress(0.4, "caption vectors")
        if segs:
            for seg, v in zip(segs, emb.embed_texts([x.text for x in segs])):
                seg.text_emb = v.tolist()
        tr.progress(0.5, "transcript vectors")
        images, owners = [], []
        for c in clips:
            if not c.keyframe_key:
                continue
            path = local_video_path(c.keyframe_key, vid)
            img = cv2.imread(str(path))
            if img is not None:
                images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                owners.append(c)
        for c, v in zip(owners, emb.embed_images(images)):
            c.image_emb = v.tolist()
    return f"{len(clips)} clip vectors, {len(images)} image vectors, {len(segs)} speech vectors"


SCENE_PROMPT = """These are timestamped descriptions of consecutive clips from one scene of a video.
Write a 2-3 sentence summary of the scene: what happens, in order, with approximate times. Plain prose, no lists.

{captions}"""

VIDEO_PROMPT = """These are timestamped summaries of the scenes in a video ({duration}).
Write one paragraph (4-6 sentences) summarising the whole video: the setting, the main people or objects,
and what happens from start to end, with approximate times. Plain prose.

{scenes}"""


def step_summaries(ctx, tr):
    from app.llm import chat_text

    vid = ctx["video_id"]
    with session_scope() as s:
        video = s.get(Video, vid)
        shots = s.scalars(select(Shot).where(Shot.video_id == vid).order_by(Shot.index)).all()
        clips = s.scalars(select(Clip).where(Clip.video_id == vid).order_by(Clip.index)).all()
        by_shot: dict = {}
        for c in clips:
            by_shot.setdefault(c.shot_id, []).append(c)
        scene_texts = []
        for i, shot in enumerate(shots):
            members = [c for c in by_shot.get(shot.id, []) if c.caption]
            if not members:
                continue
            if len(members) == 1:
                text = members[0].caption
            else:
                caps = "\n".join(f"[{fmt_t(c.t_start)}-{fmt_t(c.t_end)}] {c.caption}" for c in members)
                text = chat_text([{"role": "user", "content": SCENE_PROMPT.format(captions=caps)}], agent=True)
            s.add(Summary(video_id=vid, level="scene", t_start=shot.t_start, t_end=shot.t_end, text=text))
            scene_texts.append(f"[{fmt_t(shot.t_start)}-{fmt_t(shot.t_end)}] {text}")
            tr.progress((i + 1) / (len(shots) + 1), f"{i + 1} / {len(shots)} scenes summarised")
        if scene_texts:
            overview = chat_text([{"role": "user", "content": VIDEO_PROMPT.format(
                duration=fmt_t(video.duration_s or 0), scenes="\n".join(scene_texts))}], agent=True, max_tokens=600)
            s.add(Summary(video_id=vid, level="video", t_start=0, t_end=video.duration_s or 0, text=overview))
    try:  # the camera viewpoint picks the detector later (aerial footage needs a different model)
        from app.tools.vision import get_viewpoint

        view = get_viewpoint(vid)
    except Exception:  # optional extra: never fail ingestion on it
        view = "unknown"
    return f"{len(scene_texts)} scene summaries + video summary (viewpoint: {view})"


STEP_FUNCS = {
    "probe": step_probe,
    "proxy": step_proxy,
    "shots": step_shots,
    "keyframes": step_keyframes,
    "captions": step_captions,
    "transcript": step_transcript,
    "embeddings": step_embeddings,
    "summaries": step_summaries,
}
