"""make_highlight_clip: cut the moments that matter into one short video ("every time a bus passes").

Moments come from (a) explicit time ranges taken from earlier tool results (checkout trips, zone events,
evidence), (b) the tracks of an object type (each tracked object's time on screen), or (c) a moment search.
Each piece gets a badge with its original time so the reel can be checked against the full video.
"""

import logging
import shutil
import uuid

from PIL import Image

from app.core.config import get_settings
from app.core.media import fmt_t, probe, run
from app.core.storage import get_storage, url_or_none, video_prefix
from app.tools.pictures import label_box
from app.tools.registry import GPU_IMPLS, Tool, ToolOutput, call_gpu, execute_gpu_tool, register
from app.tools.vision import MIN_TRACK_S, _clean, _proxy_path, _range, _singular, _track_params

log = logging.getLogger(__name__)
VERSION = 3
MAX_SEGMENTS = 12
MAX_TOTAL_S = 90.0


def merge_segments(segs: list[tuple[float, float]], pad: float, duration: float,
                   min_len: float = 1.0) -> list[tuple[float, float]]:
    """Pad, clamp to the video, and merge overlapping or touching ranges."""
    out: list[list[float]] = []
    for a, b in sorted((min(a, b), max(a, b)) for a, b in segs):
        a, b = max(0.0, a - pad), min(duration, b + pad)
        if b - a < min_len:  # very short moments: widen around their middle
            m = (a + b) / 2
            a, b = max(0.0, m - min_len / 2), min(duration, m + min_len / 2)
        if out and a <= out[-1][1] + 0.25:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(round(a, 2), round(b, 2)) for a, b in out]


def _parse_segments(segments) -> list[tuple[float, float]]:
    out = []
    for s in segments or []:
        try:
            if isinstance(s, dict):
                a = s.get("t_start", s.get("start", s.get("t")))
                b = s.get("t_end", s.get("end", a))
            else:
                a, b = (s[0], s[1]) if len(s) > 1 else (s[0], s[0])
            out.append((float(a), float(b)))
        except (TypeError, ValueError, IndexError, KeyError):
            continue
    return out


def _badge(text: str, w: int, path) -> None:
    scale = max(1.0, w / 640)
    img = Image.new("RGBA", (int(360 * scale), int(48 * scale)), (0, 0, 0, 0))
    label_box(img, (int(10 * scale), int(24 * scale)), text, (20, 20, 22, 215), scale, anchor="lm")
    img.save(path)


def _impl_clip(video_id, params, run_id):
    proxy = _proxy_path(video_id)
    info = probe(proxy)
    work = get_settings().work_dir / str(video_id) / "runs" / str(run_id)
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    parts = []
    for i, (a, b) in enumerate(params["segments"]):
        badge = work / f"badge_{i}.png"
        _badge(f"{fmt_t(a)} - {fmt_t(b)}", info.width, badge)  # default font has no en dash
        part = work / f"part_{i:02d}.mp4"
        cmd = ["ffmpeg", "-y", "-v", "error", "-ss", f"{a:.3f}", "-to", f"{b:.3f}", "-i", str(proxy),
               "-i", str(badge), "-filter_complex", "[0:v][1:v]overlay=12:12,format=yuv420p[v]", "-map", "[v]",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-r", f"{info.fps or 25:.3f}"]
        cmd += (["-map", "0:a?", "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2"] if info.has_audio
                else ["-an"])
        run(cmd + [str(part)])
        parts.append(part)
    listing = work / "parts.txt"
    listing.write_text("".join(f"file '{p.name}'\n" for p in parts))
    out = work / "highlights.mp4"
    run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy",
         "-movflags", "+faststart", str(out)])
    key = f"{video_prefix(video_id)}/runs/{run_id}/highlights.mp4"
    get_storage().put_file(key, out, "video/mp4")
    for p in parts:
        p.unlink(missing_ok=True)
    total = round(sum(b - a for a, b in params["segments"]), 1)
    return {**params, "clip_key": key, "total_s": total}, None


GPU_IMPLS["make_highlight_clip"] = _impl_clip


ARRIVAL_S = 6.0
ALWAYS_VISIBLE = 0.7  # on screen for more than 70% of the range: parked / static, not an "every time X" moment


def _track_segments(video_id, label, t0, t1) -> tuple[list[tuple[float, float]], int, int]:
    """-> (time on screen of each tracked object, objects used, objects left out as always visible)."""
    tp = _track_params(video_id, [label], t0, t1, 5.0, True)
    tr = execute_gpu_tool("segment_and_track", str(video_id), tp)
    min_frames = max(2, round(MIN_TRACK_S * 5.0))
    solid = [t for t in tr["result"]["tracks"] if t["frames"] >= min_frames]
    span = max(1e-6, tp["t1"] - tp["t0"])
    moments = [t for t in solid if (t["t_end"] - t["t_start"]) / span <= ALWAYS_VISIBLE]
    if not moments:  # everything is always visible: the whole range is the moment
        moments = solid
    # each object's arrival (first few seconds on screen), not its whole time on screen: in a busy scene the
    # full spans overlap into one long block
    return ([(t["t_start"], min(t["t_end"], t["t_start"] + ARRIVAL_S)) for t in moments], len(moments),
            len(solid) - len(moments))


def _checkout_segments(video_id, label) -> list[tuple[float, float]] | None:
    """If this video already has a checkout count (cached; never start the slow labelling pass from here),
    the trips of the matching item type. 'apple being scanned' matches 'apple'; 'items' / 'scan' match all."""
    from app.tools.plugins import checkout
    from app.tools.registry import find_cached_run

    vid = uuid.UUID(str(video_id))
    if not find_cached_run(vid, "checkout_scan_count", {"version": checkout.VERSION,
                                                         "fps": get_settings().checkout_fps}):
        return None
    trips = checkout.checkout_scan_count(vid).data["trips"]
    words = _words(label)
    if words & {"item", "scan", "product", "everything", "all", "grocery"}:
        return [(t["t_start"], t["t_end"]) for t in trips]

    def matches(item: str) -> bool:  # whole words only ('scanned' must not match 'can'); head word is enough
        iw = _words(item)
        return bool(iw) and (iw <= words or _singular(item.split()[0].lower()) in words)

    hits = [(t["t_start"], t["t_end"]) for t in trips if matches(t["item_type"])]
    return hits or None


def _words(text: str) -> set[str]:
    return {_singular(w) for w in str(text).lower().replace("_", " ").split()}


def make_highlight_clip(video_id, segments=None, label=None, query=None, t0=0, t1=None, pad=1.0) -> ToolOutput:
    lo, hi, duration = _range(video_id, t0, t1)
    pad = float(max(0.0, min(float(pad if pad is not None else 1.0), 5.0)))
    source, found = "", 0
    raw = _parse_segments(segments)
    if raw:
        source = "given time ranges"
    elif label and (trips := _checkout_segments(video_id, label)):
        raw, source = trips, f"checkout trips matching '{_clean(label)}' ({len(trips)} trips)"
    elif label:
        raw, found, static = _track_segments(video_id, label, lo, hi)
        source = f"tracks of {_clean(label)} ({found} objects"
        source += f"; {static} on screen almost the whole time left out, e.g. parked)" if static else ")"
    elif query:
        from app.tools.memory import search_moments

        res = search_moments(uuid.UUID(str(video_id)), query, 5).data["results"]
        raw = [(r["t_start"], r["t_end"]) for r in res if r.get("score", 0) > 0]
        source = f"moment search for '{query}'"
    if not raw:
        if label:
            raise ValueError(f"no {_clean(label)} was tracked in this video, so there are no moments to cut "
                             "(the detector may have missed it; check with search_moments or inspect_frame)")
        raise ValueError("no moments to cut: give segments, a label, or a query")
    merged = merge_segments([(a, b) for a, b in raw if b >= lo and a <= hi], pad, duration)
    kept, total = [], 0.0
    for a, b in merged:
        if len(kept) >= MAX_SEGMENTS or total + (b - a) > MAX_TOTAL_S:
            break
        kept.append((a, b))
        total += b - a
    if not kept:
        raise ValueError("the moments fall outside the requested time range")
    params = {"segments": kept, "version": VERSION}
    result, run_id, _, cached = call_gpu("make_highlight_clip", video_id, params)
    listed = ", ".join(f"{fmt_t(a)}–{fmt_t(b)}" for a, b in kept)
    summary = f"Highlight clip: {len(kept)} moment{'s' if len(kept) != 1 else ''}, {result['total_s']} s ({listed})"
    data = {"segments": [{"t_start": a, "t_end": b} for a, b in kept], "segments_as_text": listed,
            "total_s": result["total_s"],
            "source": source, "clip": "shown to the user in the chat as a video",
            "note": f"Each moment is padded by {pad} s and overlapping moments are merged."}
    if total > 0.7 * (hi - lo):
        data["coverage_note"] = (f"The moments cover {round(100 * total / max(hi - lo, 1e-6))}% of the video, "
                                 "because these objects are on screen most of the time; say so to the user.")
    if len(kept) < len(merged):
        data["left_out"] = (f"{len(merged) - len(kept)} more moments not included (limit {MAX_SEGMENTS} moments / "
                            f"{int(MAX_TOTAL_S)} s)")
    return ToolOutput(("(cached) " if cached else "") + summary, data, run_id, None,
                      clip_url=url_or_none(result["clip_key"]))


register(Tool(
    name="make_highlight_clip", tier="gpu",
    description="Cut moments of the video into ONE short highlight video shown to the user, each piece stamped "
                "with its original time. Use when the user asks for a clip / reel / montage / 'show me every "
                "time X'. Give exactly one source: segments = time ranges from earlier tool results "
                "([[start, end], ...] or objects with t_start/t_end, e.g. checkout trips, zone events, "
                "evidence); or label = an object type, cutting each tracked object's time on screen (e.g. 'bus'); "
                "or query = words to search for (least precise). Max 12 moments / 90 s.",
    parameters={"type": "object", "properties": {
        "segments": {"type": "array", "items": {}},
        "label": {"type": "string"}, "query": {"type": "string"},
        "t0": {"type": "number", "default": 0}, "t1": {"type": "number"},
        "pad": {"type": "number", "default": 1.0, "description": "Seconds added before and after each moment"},
    }},
    fn=lambda video_id, segments=None, label=None, query=None, t0=0, t1=None, pad=1.0, **_:
        make_highlight_clip(video_id, segments, label, query, t0, t1, pad),
    label=lambda a: "Making a highlight clip" + (f" of {_clean(a['label'])}" if a.get("label") else
                                                 f" for '{a['query']}'" if a.get("query") else ""),
))

