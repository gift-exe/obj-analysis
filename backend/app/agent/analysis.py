"""Path 1: general analysis report, built from memory tools only and cached per video."""

import uuid
from datetime import datetime, timezone

from app.core.db import session_scope
from app.core.media import fmt_t
from app.llm import chat_json
from app.models import Video
from app.tools.memory import get_overview, get_timeline

REPORT_PROMPT = """You are writing an analysis report for a video, based on the indexed descriptions below.
You cannot see the video; use only these descriptions and never invent events or times.

VIDEO OVERVIEW
{overview}

CLIP-BY-CLIP TIMELINE
{timeline}

Return only a JSON object:
{{
  "narration": "4-8 sentences narrating what happens, in order, in plain language, mentioning times as m:ss",
  "key_moments": [{{"t_start": seconds, "t_end": seconds, "title": "3-6 words", "description": "one sentence"}}],
  "entities": [{{"name": "short name, e.g. 'shopper in NASA t-shirt'", "kind": "person|object|vehicle|place|other",
                 "first_seen": seconds, "last_seen": seconds}}],
  "suggested_questions": ["4 specific follow-up questions a viewer might ask about this video"]
}}
Give 5-10 key moments (fewer for very short videos) and up to 8 entities."""

KINDS = {"person", "object", "vehicle", "place", "other"}


def _num(x, default=0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def normalise_report(data: dict, duration: float) -> dict:
    clamp = lambda t: round(min(max(_num(t), 0.0), duration or _num(t)), 2)  # noqa: E731
    moments = []
    for m in data.get("key_moments") or []:
        a, b = clamp(m.get("t_start")), clamp(m.get("t_end", m.get("t_start")))
        moments.append({"t_start": min(a, b), "t_end": max(a, b), "title": str(m.get("title") or "")[:80],
                        "description": str(m.get("description") or "")})
    entities = []
    for e in data.get("entities") or []:
        kind = str(e.get("kind") or "other").lower()
        entities.append({"name": str(e.get("name") or "")[:80], "kind": kind if kind in KINDS else "other",
                         "first_seen": clamp(e.get("first_seen")), "last_seen": clamp(e.get("last_seen"))})
    return {
        "narration": str(data.get("narration") or ""),
        "key_moments": sorted(moments, key=lambda m: m["t_start"])[:12],
        "entities": entities[:10],
        "suggested_questions": [str(q) for q in (data.get("suggested_questions") or [])][:6],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def build_report(video_id: uuid.UUID) -> tuple[dict, list]:
    """Returns (report, tool outputs used) so the chat path can show the steps."""
    ov = get_overview(video_id)
    tl = get_timeline(video_id)
    overview = ov.data.get("summary", "") + "\n" + "\n".join(
        f"[{fmt_t(sc['t_start'])}-{fmt_t(sc['t_end'])}] {sc['summary']}" for sc in ov.data.get("scenes", []))
    clips = tl.data["clips"][:150]
    timeline = "\n".join(
        f"[{c['t_start']:.1f}s-{c['t_end']:.1f}s] {c['caption'][:350]}"
        + (f" Objects: {', '.join(c.get('objects', [])[:10])}." if c.get("objects") else "")
        for c in clips)
    with session_scope() as s:
        duration = s.get(Video, video_id).duration_s or 0
    data = chat_json([{"role": "user", "content": REPORT_PROMPT.format(overview=overview, timeline=timeline)}],
                     agent=True, max_tokens=2000)
    return normalise_report(data, duration), [("get_overview", {}, ov), ("get_timeline", {}, tl)]


def get_or_build_report(video_id: uuid.UUID, refresh: bool = False) -> dict:
    with session_scope() as s:
        cached = s.get(Video, video_id).analysis
    if cached and not refresh:
        return cached
    report, _ = build_report(video_id)
    with session_scope() as s:
        s.get(Video, video_id).analysis = report
    return report
