"""Memory-tier tools: read the video memory built at ingestion. Fast and free."""

import logging
import re
import uuid

from sqlalchemy import Text, cast, func, select

from app.core.db import session_scope
from app.core.media import fmt_t
from app.core.storage import url_or_none
from app.models import Clip, Shot, Summary, TranscriptSegment, Video
from app.tools.registry import Tool, ToolOutput, register

log = logging.getLogger(__name__)


def _clip_dict(c: Clip, caption_chars: int = 400) -> dict:
    d = {"t_start": round(c.t_start, 1), "t_end": round(c.t_end, 1), "caption": c.caption[:caption_chars]}
    if c.objects:
        d["objects"] = c.objects[:15]
    if c.on_screen_text:
        d["on_screen_text"] = c.on_screen_text[:150]
    return d


def nearest_keyframe_url(video_id, t: float) -> str | None:
    with session_scope() as s:
        clip = s.scalar(select(Clip).where(Clip.video_id == video_id, Clip.keyframe_key.is_not(None))
                        .order_by(func.abs((Clip.t_start + Clip.t_end) / 2 - t)).limit(1))
        return url_or_none(clip.keyframe_key) if clip else None


# --- get_overview ---------------------------------------------------------------------------

def get_overview(video_id: uuid.UUID) -> ToolOutput:
    with session_scope() as s:
        v = s.get(Video, video_id)
        sums = s.scalars(select(Summary).where(Summary.video_id == video_id).order_by(Summary.t_start)).all()
        n_clips = s.scalar(select(func.count()).select_from(Clip).where(Clip.video_id == video_id))
        n_shots = s.scalar(select(func.count()).select_from(Shot).where(Shot.video_id == video_id))
    video_sum = next((x.text for x in sums if x.level == "video"), "")
    scenes = [{"t_start": round(x.t_start, 1), "t_end": round(x.t_end, 1), "summary": x.text}
              for x in sums if x.level == "scene"]
    data = {
        "video": {"title": v.title, "duration_s": round(v.duration_s or 0, 1), "resolution": f"{v.width}x{v.height}",
                  "has_audio": v.has_audio, "shots": n_shots, "clips": n_clips},
        "summary": video_sum,
        "scenes": scenes,
    }
    return ToolOutput(f"Read video summary and {len(scenes)} scene summaries", data)


register(Tool(
    name="get_overview", tier="memory",
    description="Video metadata, the whole-video summary and the per-scene summaries. Start here for "
                "general questions or to get oriented.",
    parameters={"type": "object", "properties": {}},
    fn=lambda video_id, **_: get_overview(video_id),
    label=lambda a: "Reading the video summary",
))


# --- search_moments -------------------------------------------------------------------------

def _or_tsquery(q: str) -> str:
    words = [w for w in re.findall(r"[a-zA-Z0-9]+", q.lower()) if len(w) > 2]
    return " | ".join(words)


def search_moments(video_id: uuid.UUID, query: str, k: int = 5) -> ToolOutput:
    k = max(1, min(int(k or 5), 10))
    ranked: list[list[uuid.UUID]] = []
    seg_hits: list[tuple[float, float, str]] = []
    with session_scope() as s:
        base = select(Clip).where(Clip.video_id == video_id)
        try:
            from app.embeddings import embed_image_query, embed_query_text

            qv = embed_query_text(query).tolist()
            ranked.append([c.id for c in s.scalars(base.where(Clip.caption_emb.is_not(None))
                                                   .order_by(Clip.caption_emb.cosine_distance(qv)).limit(20))])
            iv = embed_image_query(query).tolist()
            ranked.append([c.id for c in s.scalars(base.where(Clip.image_emb.is_not(None))
                                                   .order_by(Clip.image_emb.cosine_distance(iv)).limit(20))])
            segs = s.scalars(select(TranscriptSegment).where(TranscriptSegment.video_id == video_id,
                                                             TranscriptSegment.text_emb.is_not(None))
                             .order_by(TranscriptSegment.text_emb.cosine_distance(qv)).limit(5)).all()
            seg_hits = [(x.t_start, x.t_end, x.text) for x in segs]
        except Exception as e:  # models missing or failed to load: degrade to keyword search
            log.warning("vector search unavailable (%s); keyword search only", e)
        tsq = _or_tsquery(query)
        if tsq:
            doc = func.to_tsvector("english", Clip.caption + " " + Clip.on_screen_text + " "
                                   + cast(Clip.objects, Text))
            rank = func.ts_rank(doc, func.to_tsquery("english", tsq))
            ranked.append([c.id for c in s.scalars(base.where(rank > 0).order_by(rank.desc()).limit(20))])
        all_clips = {c.id: c for c in s.scalars(base)}

    # Reciprocal rank fusion; transcript hits boost the clips they overlap.
    scores: dict[uuid.UUID, float] = {}
    for lst in ranked:
        for r, cid in enumerate(lst):
            scores[cid] = scores.get(cid, 0) + 1 / (60 + r)
    for r, (a, b, _t) in enumerate(seg_hits):
        for cid, c in all_clips.items():
            if c.t_start < b and c.t_end > a:
                scores[cid] = scores.get(cid, 0) + 1 / (60 + r)
    top = sorted(scores.items(), key=lambda x: -x[1])[:k]
    results = []
    for cid, score in top:
        d = _clip_dict(all_clips[cid])
        d["score"] = round(score * 100, 2)
        results.append(d)
    data = {"query": query, "results": results, "total_clips_in_video": len(all_clips)}
    if seg_hits:
        data["speech"] = [{"t_start": round(a, 1), "t_end": round(b, 1), "text": t} for a, b, t in seg_hits[:3]]
    best = f"; best {fmt_t(results[0]['t_start'])}-{fmt_t(results[0]['t_end'])}" if results else ""
    return ToolOutput(f"{len(results)} candidate moments{best}", data)


register(Tool(
    name="search_moments", tier="memory",
    description="Find the moments most likely to show something, by searching clip captions, keyframe "
                "images and speech. Returns ranked time ranges with captions. Call it with several "
                "phrasings for event questions.",
    parameters={"type": "object", "properties": {
        "query": {"type": "string", "description": "What to look for, in plain words."},
        "k": {"type": "integer", "description": "Number of results (1-10).", "default": 5},
    }, "required": ["query"]},
    fn=lambda video_id, query, k=5, **_: search_moments(video_id, query, k),
    label=lambda a: f"Searching for '{a.get('query', '')}'",
))


# --- get_timeline ---------------------------------------------------------------------------

def get_timeline(video_id: uuid.UUID, t0: float = 0, t1: float | None = None) -> ToolOutput:
    with session_scope() as s:
        q = select(Clip).where(Clip.video_id == video_id, Clip.t_end > (t0 or 0))
        if t1 is not None:
            q = q.where(Clip.t_start < t1)
        clips = s.scalars(q.order_by(Clip.index)).all()
        sq = select(TranscriptSegment).where(TranscriptSegment.video_id == video_id,
                                             TranscriptSegment.t_end > (t0 or 0))
        if t1 is not None:
            sq = sq.where(TranscriptSegment.t_start < t1)
        segs = s.scalars(sq.order_by(TranscriptSegment.t_start)).all()
    data = {"clips": [_clip_dict(c, 600) for c in clips]}
    if segs:
        data["speech"] = [{"t_start": round(x.t_start, 1), "t_end": round(x.t_end, 1), "text": x.text}
                          for x in segs]
    rng = f"{fmt_t(t0 or 0)}-{fmt_t(t1)}" if t1 is not None else "the whole video"
    return ToolOutput(f"Read {len(clips)} clip captions for {rng}", data)


register(Tool(
    name="get_timeline", tier="memory",
    description="Clip-by-clip captions (and speech) for a time range, in order. Use it to read what "
                "happens around a moment, or the whole video when it is short.",
    parameters={"type": "object", "properties": {
        "t0": {"type": "number", "description": "Start time in seconds.", "default": 0},
        "t1": {"type": "number", "description": "End time in seconds (omit for the end of the video)."},
    }},
    fn=lambda video_id, t0=0, t1=None, **_: get_timeline(video_id, t0, t1),
    label=lambda a: "Reading the timeline" + (f" {fmt_t(a.get('t0', 0))}-{fmt_t(a['t1'])}" if a.get("t1") else ""),
))
