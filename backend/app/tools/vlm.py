"""VLM-tier tools: send specific frames to the vision-language model with a focused question."""

import uuid

from app.core.db import session_scope
from app.core.media import FrameReader, fmt_t, local_video_path
from app.llm import chat_json, frames_content
from app.models import Video
from app.tools.registry import Tool, ToolOutput, register

WATCH_PROMPT = """You are inspecting a short part of a video ({t0:.1f}s to {t1:.1f}s) to answer a question.
Question: {question}

Use only what is clearly visible in these frames. Do not infer from context or typical scenes: if the thing the
question depends on (an object, a light's colour, a label, an action) cannot actually be made out at this
resolution and viewpoint, set "visible" to false and say it cannot be determined.
Return only a JSON object:
{{"visible": true or false (can the thing the question depends on actually be seen clearly?),
  "answer": "direct answer in 1-3 sentences, or what cannot be determined and why",
  "observations": [{{"t": seconds, "note": "what is visible at that time that matters for the question"}}],
  "confidence": number between 0 and 1}}"""


def _video_paths(video_id):
    with session_scope() as s:
        v = s.get(Video, video_id)
        return (local_video_path(v.proxy_key, v.id) if v.proxy_key else None,
                local_video_path(v.storage_key, v.id), v.duration_s or 0)


def watch_clip(video_id: uuid.UUID, t0: float, t1: float, question: str, fps: float = 2.0) -> ToolOutput:
    proxy, _src, duration = _video_paths(video_id)
    t0, t1 = float(t0), float(t1)
    if t1 - t0 < 2.0:  # too short to see motion: widen to 2s around the requested moment
        mid = (t0 + t1) / 2
        t0, t1 = mid - 1.0, mid + 1.0
    t0, t1 = max(0.0, t0), min(t1, duration or t1)
    fps = max(2.0, min(float(fps or 2.0), 6.0))
    with FrameReader(proxy) as r:
        frames = r.sample(t0, t1, fps, max_frames=16)
    data = chat_json([{"role": "user", "content": frames_content(
        frames, WATCH_PROMPT.format(t0=t0, t1=t1, question=question))}], max_tokens=800)
    data = {"t_start": round(t0, 1), "t_end": round(t1, 1), "frames_watched": len(frames), **data}
    return ToolOutput(f"Watched {fmt_t(t0)}-{fmt_t(t1)} ({len(frames)} frames): {str(data.get('answer', ''))[:120]}",
                      data)


register(Tool(
    name="watch_clip", tier="vlm",
    description="Watch a time range closely (up to 16 frames) and answer a focused question about it. "
                "Use it on candidate moments from search_moments to confirm or reject an event.",
    parameters={"type": "object", "properties": {
        "t0": {"type": "number", "description": "Start time in seconds."},
        "t1": {"type": "number", "description": "End time in seconds (keep ranges under ~10s)."},
        "question": {"type": "string", "description": "What to check in this range."},
        "fps": {"type": "number", "description": "Frames per second to sample (0.5-6).", "default": 2},
    }, "required": ["t0", "t1", "question"]},
    fn=lambda video_id, t0, t1, question, fps=2.0, **_: watch_clip(video_id, t0, t1, question, fps),
    label=lambda a: f"Watching {fmt_t(a.get('t0', 0))}-{fmt_t(a.get('t1', 0))} closely",
))


FRAME_PROMPT = """This is a single full-resolution frame at t={t:.1f}s.
Question: {question}
Use only what is clearly visible. Do not infer from context or typical scenes: if the thing the question asks
about cannot actually be made out (too small, hidden, wrong viewpoint), set "visible" to false and say so.
Return only a JSON object:
{{"visible": true or false, "answer": "direct answer in 1-3 sentences, or what cannot be determined and why",
  "details": "relevant fine details (colours, text, positions)", "confidence": number between 0 and 1}}"""


def inspect_frame(video_id: uuid.UUID, t: float, question: str) -> ToolOutput:
    _proxy, src, _d = _video_paths(video_id)
    with FrameReader(src) as r:
        frame = r.at(float(t))
    if frame is None:
        return ToolOutput(f"No frame at {fmt_t(t)}", {"error": "no frame at that time"})
    data = chat_json([{"role": "user", "content": frames_content(
        [(float(t), frame)], FRAME_PROMPT.format(t=float(t), question=question), max_side=1280)}], max_tokens=600)
    data = {"t": round(float(t), 1), **data}
    return ToolOutput(f"Inspected frame at {fmt_t(t)}: {str(data.get('answer', ''))[:120]}", data)


register(Tool(
    name="inspect_frame", tier="vlm",
    description="Look at one frame at full resolution, for fine detail (small text, colours, a traffic "
                "light, a label).",
    parameters={"type": "object", "properties": {
        "t": {"type": "number", "description": "Time in seconds."},
        "question": {"type": "string"},
    }, "required": ["t", "question"]},
    fn=lambda video_id, t, question, **_: inspect_frame(video_id, t, question),
    label=lambda a: f"Inspecting the frame at {fmt_t(a.get('t', 0))}",
))
