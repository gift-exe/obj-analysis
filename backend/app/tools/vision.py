"""GPU-tier tools: open-vocabulary detection, tracking (+ optional SAM masks), counting and zones.

These only run when the agent asks for them. Results are cached in tool_runs, so a second question
about the same objects reuses the tracks instead of recomputing them.
"""

import logging
import shutil
import threading
import uuid
from collections import Counter, defaultdict
from functools import lru_cache

import cv2
import numpy as np
from sqlalchemy import select

from app.core.config import get_settings
from app.core.db import session_scope
from app.core.media import FrameReader, encode_frames_to_mp4, fmt_t, local_video_path
from app.core.storage import get_storage, video_prefix
from app.llm import chat_json, frames_content
from app.models import Detection, Track, Video
from app.tools import overlay
from app.tools.registry import GPU_IMPLS, Tool, ToolOutput, call_gpu, execute_gpu_tool, register

log = logging.getLogger(__name__)
_model_lock = threading.Lock()


def _device():
    from app.embeddings import pick_device

    d = pick_device()
    return 0 if d == "cuda" else d


def _detector(labels: list[str]):
    """A fresh YOLO-World model per run, so tracker state never leaks between runs."""
    from ultralytics import YOLOWorld

    model = YOLOWorld(get_settings().detector_model)
    model.set_classes(labels)
    return model


@lru_cache
def _segmenter():
    from ultralytics import SAM

    return SAM(get_settings().segmenter_model)


# Generic words an open-vocabulary detector handles badly, expanded into concrete classes.
GENERIC_LABELS = {
    "vehicle": ["car", "truck", "bus", "motorcycle"], "automobile": ["car"], "motor vehicle": ["car", "truck", "bus"],
    "traffic": ["car", "truck", "bus", "motorcycle"], "people": ["person"], "pedestrian": ["person"],
    "human": ["person"], "man": ["person"], "woman": ["person"], "child": ["person"], "shopper": ["person"],
    "motorbike": ["motorcycle"], "scooter": ["motorcycle"], "bike": ["bicycle", "motorcycle"],
    "cyclist": ["bicycle", "person"], "lorry": ["truck"], "van": ["car"], "taxi": ["car"], "cab": ["car"],
}
_IRREGULAR = {"people": "people", "men": "man", "women": "woman", "children": "child", "buses": "bus",
              "glasses": "glass", "boxes": "box"}


def _singular(word: str) -> str:
    if word in _IRREGULAR:
        return _IRREGULAR[word]
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith(("ches", "shes", "sses", "xes")):
        return word[:-2]
    if word.endswith("s") and not word.endswith(("ss", "us", "is")) and len(word) > 3:
        return word[:-1]
    return word


def _clean(label) -> str:
    words = str(label).strip().lower().replace("_", " ").split()
    if words:
        words[-1] = _singular(words[-1])
    return " ".join(words)


def plan_labels(labels) -> tuple[list[str], dict[str, list[str]]]:
    """Requested words -> (concrete detector labels, groups for generic words, e.g. vehicle -> [car, ...])."""
    if isinstance(labels, str):
        labels = labels.split(",")
    out: list[str] = []
    groups: dict[str, list[str]] = {}
    for raw in labels:
        lb = _clean(raw)
        if not lb:
            continue
        members = GENERIC_LABELS.get(lb, [lb])
        if members != [lb] and len(members) > 1:
            groups[lb] = members
        for m in members:
            if m not in out:
                out.append(m)
    if not out:
        raise ValueError("at least one label is required")
    return sorted(out)[:12], groups


def _norm_labels(labels) -> list[str]:
    return plan_labels(labels)[0]


# --- viewpoint ---------------------------------------------------------------------------------

VIEW_PROMPT = """Look at these frames from one video. Classify the camera viewpoint.
"aerial": a drone/satellite-style view looking (nearly) straight down from high above, so cars and people look
small and are seen from the top (e.g. a road intersection filmed from a drone).
"elevated": a high-angle camera looking down at a scene at an angle (CCTV, a building, an overhead shop camera).
"ground": eye level or close up.
Return only JSON: {"viewpoint": "aerial" | "elevated" | "ground"}"""


def get_viewpoint(video_id) -> str:
    """Camera viewpoint, asked once from 3 frames and cached on the video. Chooses the detector."""
    with session_scope() as s:
        v = s.get(Video, video_id)
        if (v.meta or {}).get("viewpoint"):
            return v.meta["viewpoint"]
        duration = v.duration_s or 0
    proxy = _proxy_path(video_id)
    with FrameReader(proxy) as r:
        frames = [(t, f) for t in (duration * 0.1, duration * 0.5, duration * 0.9) if (f := r.at(t)) is not None]
    try:
        view = str(chat_json([{"role": "user", "content": frames_content(frames, VIEW_PROMPT)}],
                             max_tokens=60).get("viewpoint", "")).lower()
    except Exception as e:
        log.warning("viewpoint check failed: %s", e)
        return "unknown"  # not cached: try again next time
    view = view if view in {"aerial", "elevated", "ground"} else "unknown"
    with session_scope() as s:
        v = s.get(Video, video_id)
        v.meta = {**(v.meta or {}), "viewpoint": view}
    return view


# --- detector ----------------------------------------------------------------------------------

# Requested label -> class of the aerial (DOTA) model. Labels not listed fall back to YOLO-World.
AERIAL_CLASSES = {"car": "small vehicle", "suv": "small vehicle", "pickup truck": "small vehicle",
                  "truck": "large vehicle", "bus": "large vehicle", "trailer": "large vehicle",
                  "plane": "plane", "airplane": "plane", "aircraft": "plane", "ship": "ship", "boat": "ship",
                  "helicopter": "helicopter", "storage tank": "storage tank", "swimming pool": "swimming pool"}


@lru_cache
def _aerial_names() -> dict[str, int]:
    from ultralytics import YOLO

    return {v: k for k, v in YOLO(get_settings().aerial_detector_model).names.items()}


class Detector:
    """Runs the right model(s) for the requested labels and viewpoint, returning plain detections.

    Ground/elevated footage: YOLO-World (open vocabulary). Aerial footage: the DOTA-trained oriented-box
    model for vehicles/planes/ships, plus YOLO-World for anything it does not know. Fresh models per run,
    so tracker state never leaks between runs."""

    def __init__(self, labels: list[str], aerial: bool):
        self.obb = None
        self.obb_label: dict[int, str] = {}  # aerial class id -> output label
        world_labels = list(labels)
        if aerial:
            by_cls: dict[str, list[str]] = defaultdict(list)
            for lb in labels:
                if lb in AERIAL_CLASSES:
                    by_cls[AERIAL_CLASSES[lb]].append(lb)
            if by_cls:
                from ultralytics import YOLO

                self.obb = YOLO(get_settings().aerial_detector_model)
                names = _aerial_names()
                # "truck" and "bus" share the aerial "large vehicle" class -> reported as "bus/truck"
                self.obb_label = {names[c]: "/".join(sorted(lbs)) for c, lbs in by_cls.items()}
                world_labels = [lb for lb in labels if lb not in AERIAL_CLASSES]
        self.world = _detector(world_labels) if world_labels else None
        self.labels = sorted(self.obb_label.values()) + world_labels  # labels as they appear in results

    def __call__(self, frame, track: bool, conf: float, device) -> list[dict]:
        dets: list[dict] = []
        h, w = frame.shape[:2]
        if self.obb is not None:
            kw = dict(conf=max(conf, 0.25), imgsz=1024, verbose=False, device=device, classes=list(self.obb_label))
            res = (self.obb.track(frame, persist=True, tracker="bytetrack.yaml", **kw) if track
                   else self.obb.predict(frame, **kw))[0]
            o = res.obb
            if o is not None and len(o):
                ids = o.id.int().tolist() if (track and o.id is not None) else [None] * len(o)
                for bb, c, sc, tid in zip(o.xyxy.tolist(), o.cls.int().tolist(), o.conf.tolist(), ids):
                    if track and tid is None:
                        continue
                    box = [min(max(bb[0] / w, 0), 1), min(max(bb[1] / h, 0), 1),
                           min(max(bb[2] / w, 0), 1), min(max(bb[3] / h, 0), 1)]
                    dets.append({"bbox": [round(x, 4) for x in box], "label": self.obb_label[c],
                                 "score": round(sc, 3), "track": tid})
        if self.world is not None:
            kw = dict(conf=conf, verbose=False, device=device)
            res = (self.world.track(frame, persist=True, tracker="bytetrack.yaml", **kw) if track
                   else self.world.predict(frame, **kw))[0]
            boxes = res.boxes
            if boxes is not None and len(boxes):
                ids = boxes.id.int().tolist() if (track and boxes.id is not None) else [None] * len(boxes)
                for bb, c, sc, tid in zip(boxes.xyxyn.tolist(), boxes.cls.int().tolist(), boxes.conf.tolist(), ids):
                    if track and tid is None:
                        continue
                    if tid is not None and self.obb is not None:
                        tid += 100000  # keep track ids from the two models apart
                    dets.append({"bbox": [round(x, 4) for x in bb], "label": res.names[c],
                                 "score": round(sc, 3), "track": tid})
        return dets


def _groups_for(output_labels: list[str], groups: dict[str, list[str]]) -> dict[str, list[str]]:
    """Map generic groups onto output labels (e.g. vehicle -> ['car', 'bus/truck', 'motorcycle'])."""
    out = {}
    for g, members in groups.items():
        ms = [lb for lb in output_labels if set(lb.split("/")) & set(members)]
        if ms:
            out[g] = ms
    return out


def _range(video_id, t0, t1) -> tuple[float, float, float]:
    with session_scope() as s:
        duration = s.get(Video, video_id).duration_s or 0
    t0 = max(0.0, float(t0 or 0))
    t1 = duration if t1 is None else min(float(t1), duration)
    return round(t0, 2), round(t1, 2), duration


def _proxy_path(video_id):
    with session_scope() as s:
        v = s.get(Video, video_id)
        return local_video_path(v.proxy_key, v.id)


def run_vision(video_id: uuid.UUID, run_id: uuid.UUID, labels: list[str], t0: float, t1: float, fps: float,
               track: bool, masks: bool = False, conf: float = 0.2,
               aerial: bool = False) -> tuple[list[dict], str, list[str]]:
    """Analyse frames on a fixed grid (k / fps) and render an overlay covering the whole video, so the
    overlay stays time-aligned with the original.
    Returns (per-frame detections, overlay storage key, labels as they appear in the detections)."""
    duration = _range(video_id, 0, None)[2]
    proxy = _proxy_path(video_id)
    work = get_settings().work_dir / str(video_id) / "runs" / str(run_id)
    frames_dir = work / "frames"
    shutil.rmtree(frames_dir, ignore_errors=True)
    frames_dir.mkdir(parents=True)
    device = _device()
    detector = Detector(labels, aerial)
    per_frame: list[dict] = []
    n_grid = int(duration * fps) + 1
    last_frame = None
    with _model_lock, FrameReader(proxy) as reader:
        for k in range(n_grid):
            t = k / fps
            frame = reader.at(t)
            if frame is None:
                # keep the image sequence gap-free (ffmpeg stops at a gap) and time-aligned
                if last_frame is not None:
                    cv2.imwrite(str(frames_dir / f"{k:06d}.jpg"), last_frame)
                continue
            last_frame = frame
            dets: list[dict] = []
            if t0 == t1:  # single-frame request: analyse the grid frame nearest to t0
                in_range = t0 - 0.5 / fps < t <= t0 + 0.5 / fps
            else:
                in_range = t0 <= t <= t1
            if in_range:
                dets = detector(frame, track, conf, device)
                if dets and masks:
                    try:
                        h, w = frame.shape[:2]
                        px = [[d["bbox"][0] * w, d["bbox"][1] * h, d["bbox"][2] * w, d["bbox"][3] * h]
                              for d in dets]
                        m = _segmenter()(frame, bboxes=px, verbose=False, device=device)[0].masks
                        if m is not None:
                            for d, mk in zip(dets, m.data.cpu().numpy().astype(bool)):
                                d["mask"] = cv2.resize(mk.astype(np.uint8), (w, h)) > 0
                    except Exception as e:  # masks are a visual extra; never fail the run on them
                        log.warning("segmentation failed: %s", e)
                per_frame.append({"t": round(t, 3), "frame_idx": k, "dets": dets})
            cv2.imwrite(str(frames_dir / f"{k:06d}.jpg"), overlay.draw(frame, dets) if dets else frame)
    out = work / "overlay.mp4"
    encode_frames_to_mp4(frames_dir, fps, out)
    key = f"{video_prefix(video_id)}/runs/{run_id}/overlay.mp4"
    get_storage().put_file(key, out, "video/mp4")
    shutil.rmtree(frames_dir, ignore_errors=True)
    for f in per_frame:  # masks are not JSON
        for d in f["dets"]:
            d.pop("mask", None)
    return per_frame, key, detector.labels


def _label_stats(per_frame: list[dict], labels: list[str], groups: dict[str, list[str]] | None = None) -> dict:
    """Per-label (and per generic group) frame statistics. max_visible_at_once is the most seen in ONE frame."""
    stats = {}
    sets = {lb: {lb} for lb in labels}
    for g, members in (groups or {}).items():
        sets[g] = set(members)
    for name, members in sets.items():
        counts = [sum(1 for d in f["dets"] if d["label"] in members) for f in per_frame]
        if not counts:
            continue
        i = int(np.argmax(counts))
        stats[name] = {
            "frames_with_label": sum(1 for c in counts if c), "frames_analysed": len(counts),
            "max_visible_at_once": counts[i], "max_at_t": per_frame[i]["t"] if counts[i] else None,
            "p80_per_frame": float(np.percentile(counts, 80)), "mean_per_frame": round(float(np.mean(counts)), 2),
            "first_seen": next((f["t"] for f, c in zip(per_frame, counts) if c), None),
        }
        if len(counts) == 1:
            stats[name]["in_frame_at_t"] = {"t": per_frame[0]["t"], "count": counts[0]}
    return stats


MISS_WARNING = ("The detector found none. Detectors can miss objects (small, unusual viewpoint, unusual "
                "appearance). Do not answer 0 from this alone: check the captions or look with "
                "inspect_frame/watch_clip.")


def _vision_params(video_id, labels, t0, t1, fps) -> dict:
    concrete, groups = plan_labels(labels)
    t0, t1, _ = _range(video_id, t0, t1)
    return {"labels": concrete, "groups": groups, "t0": t0, "t1": t1, "fps": fps,
            "aerial": get_viewpoint(video_id) == "aerial", "detector": 2}


# --- detect_objects --------------------------------------------------------------------------

def _impl_detect(video_id, params, run_id):
    per_frame, key, out_labels = run_vision(video_id, run_id, params["labels"], params["t0"], params["t1"],
                                            params["fps"], track=False, aerial=params.get("aerial", False))
    groups = _groups_for(out_labels, params.get("groups") or {})
    return {**params, "per_label": _label_stats(per_frame, out_labels, groups)}, key


GPU_IMPLS["detect_objects"] = _impl_detect


def detect_objects(video_id, labels, t0=0, t1=None, fps=2.0) -> ToolOutput:
    params = _vision_params(video_id, labels, t0, t1, float(max(0.5, min(float(fps or 2), 5))))
    result, run_id, overlay_url, cached = call_gpu("detect_objects", video_id, params)
    single = result["t0"] == result["t1"]
    parts = []
    for lb, st in result["per_label"].items():
        if single:
            parts.append(f"{lb}: {st['max_visible_at_once']} in the frame at {fmt_t(result['t0'])}")
        else:
            parts.append(f"{lb}: in {st['frames_with_label']}/{st['frames_analysed']} frames, "
                         f"max {st['max_visible_at_once']} at once")
    concrete = {lb: st for lb, st in result["per_label"].items() if lb not in (result.get("groups") or {})}
    if len(concrete) > 1 and not (result.get("groups") or {}):
        total = sum(st["max_visible_at_once"] for st in concrete.values()) if single else None
        if single:
            parts.append(f"all requested types together: {total} in the frame")
            result = {**result, "total_in_frame_all_types": total}
    data = {**result, "detector_used": "aerial (DOTA)" if result.get("aerial") else "open-vocabulary (YOLO-World)",
            "note": "max_visible_at_once is the most seen in a single frame, NOT the number of distinct objects "
                    "in the video. For distinct objects use count_objects."}
    if all(st["max_visible_at_once"] == 0 for st in result["per_label"].values()):
        data["warning"] = MISS_WARNING
    return ToolOutput(("(cached) " if cached else "") + "; ".join(parts), data, run_id, overlay_url)


register(Tool(
    name="detect_objects", tier="gpu",
    description="Detect objects on sampled frames. Use for 'how many X at time t / in the first frame' "
                "(set t0 = t1 = that time) or 'when does X first appear'. Give concrete object words "
                "(['car','bus','person']); generic words like 'vehicle' are expanded automatically. Returns, per "
                "type, the frames it appears in and the most visible in one frame. It does NOT count distinct "
                "objects over time: use count_objects for 'how many X in the video'.",
    parameters={"type": "object", "properties": {
        "labels": {"type": "array", "items": {"type": "string"}},
        "t0": {"type": "number", "default": 0}, "t1": {"type": "number"},
        "fps": {"type": "number", "default": 2},
    }, "required": ["labels"]},
    fn=lambda video_id, labels, t0=0, t1=None, fps=2.0, **_: detect_objects(video_id, labels, t0, t1, fps),
    label=lambda a: f"Detecting {', '.join(_safe_labels(a))}",
))


def _safe_labels(a) -> list[str]:
    try:
        return _norm_labels(a.get("labels") or a.get("label") or [])
    except ValueError:
        return []


# --- segment_and_track -----------------------------------------------------------------------

def _impl_track(video_id, params, run_id):
    per_frame, key, out_labels = run_vision(video_id, run_id, params["labels"], params["t0"], params["t1"],
                                            params["fps"], track=True, masks=params.get("masks", False),
                                            aerial=params.get("aerial", False))
    by_track: dict[int, list] = defaultdict(list)
    for f in per_frame:
        for d in f["dets"]:
            by_track[d["track"]].append((f, d))
    tracks = []
    with session_scope() as s:
        for tid, items in sorted(by_track.items()):
            label = Counter(d["label"] for _, d in items).most_common(1)[0][0]
            tr = Track(tool_run_id=run_id, track_no=tid, label=label, t_start=items[0][0]["t"],
                       t_end=items[-1][0]["t"], mean_score=float(np.mean([d["score"] for _, d in items])))
            s.add(tr)
            s.flush()
            for f, d in items:
                s.add(Detection(track_id=tr.id, frame_idx=f["frame_idx"], t=f["t"], bbox=d["bbox"], score=d["score"]))
            tracks.append({"track": tid, "label": label, "t_start": tr.t_start, "t_end": tr.t_end,
                           "frames": len(items), "mean_score": round(tr.mean_score, 2)})
    groups = _groups_for(out_labels, params.get("groups") or {})
    return {**params, "tracks": tracks, "output_groups": groups,
            "per_label": _label_stats(per_frame, out_labels, groups)}, key


GPU_IMPLS["segment_and_track"] = _impl_track


def _track_params(video_id, labels, t0, t1, fps, masks) -> dict:
    params = _vision_params(video_id, labels, t0, t1, float(max(1, min(float(fps or 5), 10))))
    # masks are a visual extra; skip them for aerial footage (tiny objects, many of them, slow)
    params["masks"] = bool(masks) and get_settings().enable_masks and not params["aerial"]
    return params


MIN_TRACK_S = 1.0  # shorter tracks are mostly fragments or false detections


def _summarise_tracks(result: dict) -> dict:
    min_frames = max(2, round(MIN_TRACK_S * float(result.get("fps") or 5)))
    solid = [t for t in result["tracks"] if t["frames"] >= min_frames]
    by_label = Counter(t["label"] for t in solid)
    unique = dict(by_label)
    for g, members in (result.get("output_groups") or {}).items():
        unique[g] = sum(by_label.get(m, 0) for m in members)
    return {"unique_objects": unique, "short_tracks_ignored": len(result["tracks"]) - len(solid),
            "max_visible_at_once": {lb: st["max_visible_at_once"] for lb, st in result["per_label"].items()},
            "tracks": solid[:60]}


def segment_and_track(video_id, labels, t0=0, t1=None, fps=5.0, masks=True) -> ToolOutput:
    params = _track_params(video_id, labels, t0, t1, fps, masks)
    result, run_id, overlay_url, cached = call_gpu("segment_and_track", video_id, params)
    summ = _summarise_tracks(result)
    text = ", ".join(f"{n} {lb}" for lb, n in summ["unique_objects"].items()) or "no objects tracked"
    data = {"t0": params["t0"], "t1": params["t1"], **summ}
    if not summ["unique_objects"] or not any(summ["unique_objects"].values()):
        data["warning"] = MISS_WARNING
    return ToolOutput(("(cached) " if cached else "") + f"Tracked {text} distinct", data, run_id, overlay_url)


register(Tool(
    name="segment_and_track", tier="gpu",
    description="Detect, segment and track objects of the given types over a time range, giving each "
                "physical object a stable track ID with its time span. Produces an overlay video. Expensive: "
                "use only for counting, motion, duration or position questions.",
    parameters={"type": "object", "properties": {
        "labels": {"type": "array", "items": {"type": "string"}},
        "t0": {"type": "number", "default": 0}, "t1": {"type": "number"},
        "fps": {"type": "number", "default": 5},
    }, "required": ["labels"]},
    fn=lambda video_id, labels, t0=0, t1=None, fps=5.0, **_: segment_and_track(video_id, labels, t0, t1, fps),
    label=lambda a: f"Segmenting and tracking {', '.join(_safe_labels(a))}",
))


# --- count_objects ---------------------------------------------------------------------------

def count_objects(video_id, label, t0=0, t1=None, mode="unique") -> ToolOutput:
    lo, hi, _ = _range(video_id, t0, t1)
    if hi - lo < MIN_TRACK_S:  # a single moment: nothing to track, count what is in the frame
        out = detect_objects(video_id, [label], lo, lo)
        st = out.data["per_label"].get(_clean(label)) or {}
        n = st.get("max_visible_at_once", 0)
        data = {"label": _clean(label), "t": lo, "count_in_frame": n,
                "by_type": {lb: v["max_visible_at_once"] for lb, v in out.data["per_label"].items()
                            if lb != _clean(label)},
                "note": "Time range shorter than 1 s, so this is the count in the single frame at t."}
        if n == 0:
            data["warning"] = MISS_WARNING
        return ToolOutput(f"{n} {_clean(label)} in the frame at {fmt_t(lo)}", data, out.tool_run_id, out.overlay_url)
    params = _track_params(video_id, [label], t0, t1, 5.0, True)
    result, run_id, overlay_url, cached = call_gpu("segment_and_track", video_id, params)
    summ = _summarise_tracks(result)
    requested = _clean(label)
    names = list((result.get("output_groups") or {}).get(requested) or summ["unique_objects"].keys())
    by_type = {lb: summ["unique_objects"].get(lb, 0) for lb in names if lb != requested}
    unique = summ["unique_objects"].get(requested, sum(by_type.values()))
    concurrent = summ["max_visible_at_once"].get(requested, max(summ["max_visible_at_once"].values(), default=0))
    data = {"label": requested, "t0": params["t0"], "t1": params["t1"], "unique_objects": unique,
            "unique_by_type": by_type, "max_visible_at_once": concurrent, "mode_requested": mode,
            "detector_used": "aerial (DOTA)" if params["aerial"] else "open-vocabulary (YOLO-World)",
            "note": "unique = distinct tracked objects visible for at least 1 second; tracks can split "
                    "when objects are hidden, so treat as approximate and verify with watch_clip if it matters.",
            "tracks": summ["tracks"][:40]}
    if unique == 0:
        data["warning"] = MISS_WARNING
    by = f" ({', '.join(f'{n} {lb}' for lb, n in by_type.items())})" if len(by_type) > 1 else ""
    return ToolOutput(f"{unique} distinct {requested}{by}, max {concurrent} at once", data, run_id, overlay_url)


register(Tool(
    name="count_objects", tier="gpu",
    description="Count objects of one type over a time range by tracking them: the number of DISTINCT objects "
                "(use for 'how many X are in / appear in / pass through the video') and the most visible at "
                "once. Generic words like 'vehicle' are expanded (car, truck, bus, motorcycle). Reuses cached "
                "tracks when available.",
    parameters={"type": "object", "properties": {
        "label": {"type": "string"},
        "t0": {"type": "number", "default": 0}, "t1": {"type": "number"},
        "mode": {"type": "string", "enum": ["unique", "max_concurrent"], "default": "unique"},
    }, "required": ["label"]},
    fn=lambda video_id, label, t0=0, t1=None, mode="unique", **_: count_objects(video_id, label, t0, t1, mode),
    label=lambda a: f"Counting {a.get('label', 'objects')}",
))


# --- zone_events -----------------------------------------------------------------------------

ZONE_PRESETS = {
    "line:bottom": [[0.0, 0.85], [1.0, 0.85]], "line:middle": [[0.0, 0.5], [1.0, 0.5]],
    "line:top": [[0.0, 0.15], [1.0, 0.15]], "line:left": [[0.15, 0.0], [0.15, 1.0]],
    "line:right": [[0.85, 0.0], [0.85, 1.0]], "line:center_vertical": [[0.5, 0.0], [0.5, 1.0]],
    "region:center": [[0.3, 0.3], [0.7, 0.3], [0.7, 0.7], [0.3, 0.7]],
    "region:left": [[0.0, 0.0], [0.33, 0.0], [0.33, 1.0], [0.0, 1.0]],
    "region:right": [[0.67, 0.0], [1.0, 0.0], [1.0, 1.0], [0.67, 1.0]],
    "region:top": [[0.0, 0.0], [1.0, 0.0], [1.0, 0.33], [0.0, 0.33]],
    "region:bottom": [[0.0, 0.67], [1.0, 0.67], [1.0, 1.0], [0.0, 1.0]],
}


class Stabiliser:
    """Maps points from a reference frame into later frames, compensating camera motion (drones, pans)."""

    def __init__(self, ref: np.ndarray):
        self.orb = cv2.ORB_create(1500)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.ref_kp, self.ref_des = self.orb.detectAndCompute(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), None)

    def homography(self, frame: np.ndarray) -> np.ndarray:
        kp, des = self.orb.detectAndCompute(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None)
        if des is None or self.ref_des is None or len(kp) < 12:
            return np.eye(3)
        matches = sorted(self.matcher.match(self.ref_des, des), key=lambda m: m.distance)[:400]
        if len(matches) < 12:
            return np.eye(3)
        src = np.float32([self.ref_kp[m.queryIdx].pt for m in matches])
        dst = np.float32([kp[m.trainIdx].pt for m in matches])
        H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
        return H if H is not None else np.eye(3)


def _side(p1, p2, c) -> float:
    return float(np.sign((p2[0] - p1[0]) * (c[1] - p1[1]) - (p2[1] - p1[1]) * (c[0] - p1[0])))


def _impl_zone(video_id, params, run_id):
    track_params = {k: params[k] for k in ("labels", "groups", "t0", "t1", "fps", "aerial", "detector", "masks")}
    tr = execute_gpu_tool("segment_and_track", str(video_id), track_params)
    track_run = uuid.UUID(tr["tool_run_id"])
    with session_scope() as s:
        tracks = s.scalars(select(Track).where(Track.tool_run_id == track_run)).all()
        dets = defaultdict(list)
        for t in tracks:
            for d in s.scalars(select(Detection).where(Detection.track_id == t.id).order_by(Detection.t)):
                dets[(t.track_no, t.label)].append((d.t, d.bbox))
    zone_n = np.array(params["zone_points"], dtype=np.float32)
    proxy = _proxy_path(video_id)
    times = sorted({t for v in dets.values() for t, _ in v})
    zone_at: dict[float, np.ndarray] = {}
    with FrameReader(proxy) as reader:
        ref = reader.at(params["t0"])
        h, w = ref.shape[:2]
        zone_ref = zone_n * [w, h]
        stab = Stabiliser(ref)
        for t in times:
            frame = reader.at(t)
            H = stab.homography(frame) if frame is not None else np.eye(3)
            zone_at[t] = cv2.perspectiveTransform(zone_ref.reshape(-1, 1, 2), H).reshape(-1, 2)
    events = []
    is_line = len(zone_n) == 2
    for (tid, label), seq in dets.items():
        prev = None
        inside_since = None
        for t, bb in seq:
            c = ((bb[0] + bb[2]) / 2 * w, (bb[1] + bb[3]) / 2 * h)
            z = zone_at[t]
            if is_line:
                side = _side(z[0], z[1], c)
                if prev is not None and side != 0 and prev != 0 and side != prev:
                    events.append({"type": "line_crossing", "track": tid, "label": label, "t": t,
                                   "direction": "positive" if side > 0 else "negative"})
                prev = side if side != 0 else prev
            else:
                inside = cv2.pointPolygonTest(z.astype(np.float32).reshape(-1, 1, 2), c, False) >= 0
                if inside and inside_since is None:
                    inside_since = t
                    events.append({"type": "enter", "track": tid, "label": label, "t": t})
                elif not inside and inside_since is not None:
                    events.append({"type": "exit", "track": tid, "label": label, "t": t,
                                   "dwell_s": round(t - inside_since, 2)})
                    inside_since = None
        if not is_line and inside_since is not None:
            events.append({"type": "still_inside", "track": tid, "label": label, "t": seq[-1][0],
                           "dwell_s": round(seq[-1][0] - inside_since, 2)})
    events.sort(key=lambda e: e["t"])
    counts = Counter((e["type"], e["label"]) for e in events)
    return {**params, "tracks_run_id": str(track_run), "events": events[:200],
            "counts": [{"type": k[0], "label": k[1], "count": n} for k, n in counts.items()],
            "camera_compensated": True}, tr.get("overlay_key")


GPU_IMPLS["zone_events"] = _impl_zone


def zone_events(video_id, labels, zone="line:bottom", t0=0, t1=None) -> ToolOutput:
    base = _track_params(video_id, labels, t0, t1, 5.0, True)
    if isinstance(zone, str):
        if zone not in ZONE_PRESETS:
            raise ValueError(f"unknown zone preset {zone}; use one of {list(ZONE_PRESETS)} or a list of [x,y] points")
        points, zone_name = ZONE_PRESETS[zone], zone
    else:
        points, zone_name = [[float(x), float(y)] for x, y in zone], "custom"
    params = {**base, "zone": zone_name, "zone_points": points}
    result, run_id, overlay_url, cached = call_gpu("zone_events", video_id, params)
    parts = [f"{c['count']} {c['label']} {c['type'].replace('_', ' ')}" for c in result["counts"]]
    data = {k: result[k] for k in ("zone", "t0", "t1", "counts", "camera_compensated")}
    data["events"] = result["events"][:60]
    return ToolOutput(("(cached) " if cached else "") + ("; ".join(parts) or "no zone events"), data, run_id,
                      overlay_url)


register(Tool(
    name="zone_events", tier="gpu",
    description="Track objects and report when they cross a line or enter/leave a region, with camera-motion "
                "compensation. zone is a preset ('line:bottom', 'line:middle', 'line:top', 'line:left', "
                "'line:right', 'line:center_vertical', 'region:center', 'region:left', 'region:right', "
                "'region:top', 'region:bottom') or a list of normalised [x,y] points (2 points = line, 3+ = "
                "polygon) on the frame at t0.",
    parameters={"type": "object", "properties": {
        "labels": {"type": "array", "items": {"type": "string"}},
        "zone": {"description": "Preset name or list of [x,y] points in 0..1",
                 "anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "array", "items": {"type": "number"}}}]},
        "t0": {"type": "number", "default": 0}, "t1": {"type": "number"},
    }, "required": ["labels", "zone"]},
    fn=lambda video_id, labels, zone="line:bottom", t0=0, t1=None, **_: zone_events(video_id, labels, zone, t0, t1),
    label=lambda a: f"Checking {', '.join(_safe_labels(a))} against zone {a.get('zone') if isinstance(a.get('zone'), str) else 'custom'}",
))

__all__ = ["run_vision", "fmt_t"]
