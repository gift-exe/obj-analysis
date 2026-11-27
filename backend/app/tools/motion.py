"""Movement tools built on (cached) object tracks.

movement_map  -> a picture of where objects went: trails coloured by heading, or a heat map of time spent.
junction_flow -> turning movements at a junction: which road each vehicle came from and left by.

Both reuse the tracks from segment_and_track / count_objects (same cache), and map every position into ONE
reference frame with the camera-motion stabiliser, so trails stay put even when a drone drifts.
"""

import logging
import math
import uuid
from collections import Counter, defaultdict

import cv2
import numpy as np
from sqlalchemy import select

from app.core.db import session_scope
from app.core.media import FrameReader, fmt_t
from app.core.storage import url_or_none
from app.llm import chat_json, frames_content
from app.models import Detection, Track, Video
from app.tools.pictures import label_box, save_picture, ui_scale, with_legend
from app.tools.registry import GPU_IMPLS, Tool, ToolOutput, call_gpu, execute_gpu_tool, register
from app.tools.vision import MIN_TRACK_S, Stabiliser, _proxy_path, _safe_labels, _track_params

log = logging.getLogger(__name__)
VERSION = 6
HOMOGRAPHY_STEP_S = 0.5      # camera motion is smooth; one homography per half second is plenty
STATIONARY_FRAC = 0.05       # moved less than 5% of the frame diagonal overall -> stationary / parked

HEADINGS = {"heading right": (3, 155, 229), "heading left": (255, 112, 67), "heading up": (102, 187, 106),
            "heading down": (186, 104, 200)}
STATIONARY_COLOUR = (240, 240, 240)
GRID_NAMES = [["top-left", "top-centre", "top-right"], ["middle-left", "centre", "middle-right"],
              ["bottom-left", "bottom-centre", "bottom-right"]]


# --- shared: tracks -> stabilised paths -------------------------------------------------------

def _load_paths(video_id, track_run_id: uuid.UUID, t_ref: float, fps: float, aerial: bool):
    """-> (reference frame BGR, list of paths). A path is {track, label, pts: N x 3 array of (t, x, y)} with
    x, y in reference-frame pixels. Ground-level footage uses the bottom-centre of the box (where the object
    touches the ground); aerial footage the box centre."""
    min_frames = max(2, round(MIN_TRACK_S * fps))
    raw = []
    with session_scope() as s:
        for tr in s.scalars(select(Track).where(Track.tool_run_id == track_run_id)):
            dets = s.scalars(select(Detection).where(Detection.track_id == tr.id).order_by(Detection.t)).all()
            if len(dets) >= min_frames:
                raw.append((tr.track_no, tr.label, [(d.t, d.bbox) for d in dets]))
    with FrameReader(_proxy_path(video_id)) as reader:
        ref = reader.at(t_ref)
        h, w = ref.shape[:2]
        stab = Stabiliser(ref)
        inv: dict[float, np.ndarray] = {}
        for t in sorted({round(t / HOMOGRAPHY_STEP_S) * HOMOGRAPHY_STEP_S for _, _, seq in raw for t, _ in seq}):
            frame = reader.at(t)
            H = stab.homography(frame) if frame is not None else np.eye(3)
            try:
                inv[t] = np.linalg.inv(H)
            except np.linalg.LinAlgError:
                inv[t] = np.eye(3)
    paths = []
    for tid, label, seq in raw:
        pts = []
        for t, bb in seq:
            x = (bb[0] + bb[2]) / 2 * w
            y = ((bb[1] + bb[3]) / 2 if aerial else bb[3]) * h
            Hi = inv.get(round(t / HOMOGRAPHY_STEP_S) * HOMOGRAPHY_STEP_S, np.eye(3))
            p = Hi @ np.array([x, y, 1.0])
            pts.append((t, p[0] / p[2], p[1] / p[2]))
        paths.append({"track": tid, "label": label, "pts": np.array(pts)})
    return ref, paths


def _smooth(xy: np.ndarray, k: int = 5) -> np.ndarray:
    if len(xy) < k:
        return xy
    pad = np.pad(xy, ((k // 2, k // 2), (0, 0)), mode="edge")
    kernel = np.ones(k) / k
    return np.stack([np.convolve(pad[:, i], kernel, mode="valid") for i in range(2)], axis=1)


def _is_stationary(xy: np.ndarray, diag: float) -> bool:
    return float(np.max(np.linalg.norm(xy - xy[0], axis=1))) < STATIONARY_FRAC * diag


def _heading(xy: np.ndarray) -> str:
    dx, dy = xy[-1] - xy[0]
    if abs(dx) >= abs(dy):
        return "heading right" if dx > 0 else "heading left"
    return "heading down" if dy > 0 else "heading up"


def _track_run(video_id, params) -> uuid.UUID:
    keys = ("labels", "groups", "t0", "t1", "fps", "aerial", "detector", "masks")
    tr = execute_gpu_tool("segment_and_track", str(video_id), {k: params[k] for k in keys})
    return uuid.UUID(tr["tool_run_id"])


def _dim(frame_bgr: np.ndarray, f: float = 0.5) -> np.ndarray:
    return (cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) * f).astype(np.uint8)


def _arrow_end(xy: np.ndarray, out: np.ndarray, colour, thick: int) -> None:
    """Arrowhead at the end of a trail, pointing along the last stretch of movement."""
    end = xy[-1]
    back = next((p for p in xy[::-1] if np.linalg.norm(p - end) > 3 * thick), None)
    if back is not None:
        cv2.arrowedLine(out, tuple(int(v) for v in back), tuple(int(v) for v in end), colour, thick,
                        cv2.LINE_AA, tipLength=min(0.9, 6 * thick / max(1.0, float(np.linalg.norm(end - back)))))


# --- movement_map -----------------------------------------------------------------------------

def _impl_movement(video_id, params, run_id):
    track_run = _track_run(video_id, params)
    ref, paths = _load_paths(video_id, track_run, params["t0"], params["fps"], params["aerial"])
    h, w = ref.shape[:2]
    diag = math.hypot(w, h)
    scale = ui_scale(w)
    thick = max(2, int(2 * scale))
    out = _dim(ref, 0.5 if params["style"] == "trails" else 0.6)

    moving = defaultdict(Counter)   # label -> heading -> n
    stationary = Counter()
    grid = np.zeros((3, 3))
    for p in paths:
        xy = p["pts"][:, 1:]
        for x, y in xy:
            grid[min(2, max(0, int(3 * y / h))), min(2, max(0, int(3 * x / w)))] += 1
        if _is_stationary(xy, diag):
            stationary[p["label"]] += 1
        else:
            moving[p["label"]][_heading(xy)] += 1

    if params["style"] == "heatmap":
        acc = np.zeros((h, w), np.float32)
        r = max(2, int(0.006 * w))
        for p in paths:
            for x, y in p["pts"][:, 1:]:
                if 0 <= x < w and 0 <= y < h:
                    cv2.circle(acc, (int(x), int(y)), r, 1.0, -1)
        acc = cv2.GaussianBlur(acc, (0, 0), sigmaX=0.015 * w)
        top = float(np.percentile(acc[acc > 0], 99)) if (acc > 0).any() else 1.0
        norm = np.clip(acc / max(top, 1e-6), 0, 1)
        heat = cv2.cvtColor(cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
        alpha = np.clip(norm * 1.6, 0, 0.75)[..., None]
        out = (out * (1 - alpha) + heat * alpha).astype(np.uint8)
        lo = tuple(int(v) for v in cv2.cvtColor(cv2.applyColorMap(np.uint8([[40]]), cv2.COLORMAP_TURBO),
                                                cv2.COLOR_BGR2RGB)[0, 0])
        hi = tuple(int(v) for v in cv2.cvtColor(cv2.applyColorMap(np.uint8([[240]]), cv2.COLORMAP_TURBO),
                                                cv2.COLOR_BGR2RGB)[0, 0])
        items = [(lo, "little time"), (hi, "most time spent")]
    else:
        for p in paths:
            xy = _smooth(p["pts"][:, 1:])
            if _is_stationary(xy, diag):
                cv2.circle(out, tuple(int(v) for v in xy.mean(axis=0)), int(5 * scale), STATIONARY_COLOUR,
                           thick, cv2.LINE_AA)
                continue
            colour = HEADINGS[_heading(xy)]
            cv2.polylines(out, [xy.astype(np.int32).reshape(-1, 1, 2)], False, colour, thick, cv2.LINE_AA)
            cv2.circle(out, tuple(int(v) for v in xy[0]), thick + 1, colour, -1, cv2.LINE_AA)
            _arrow_end(xy, out, colour, thick)
        heads = Counter()
        for c in moving.values():
            heads.update(c)
        items = [(col, f"{name} {heads[name]}") for name, col in HEADINGS.items() if heads[name]]
        if sum(stationary.values()):
            items.append((STATIONARY_COLOUR, f"stationary {sum(stationary.values())}"))

    what = ", ".join(params["labels"])
    caption = (f"{params['style']} of {what}, {fmt_t(params['t0'])}-{fmt_t(params['t1'])}  ·  "
               f"{len(paths)} tracks  ·  camera motion compensated")
    key = save_picture(video_id, run_id, with_legend(out, items, caption), f"movement_{params['style']}.jpg")
    total = grid.sum() or 1
    busiest = sorted(((grid[i, j] / total, GRID_NAMES[i][j]) for i in range(3) for j in range(3)), reverse=True)
    return {**params, "image_key": key, "tracks_run_id": str(track_run), "tracks_drawn": len(paths),
            "moving_by_heading": {lb: dict(c) for lb, c in moving.items()},
            "moving_total": sum(sum(c.values()) for c in moving.values()),
            "stationary": dict(stationary),
            "busiest_areas": [{"area": n, "share_of_time_pct": round(100 * s, 1)} for s, n in busiest[:3]]}, None


GPU_IMPLS["movement_map"] = _impl_movement


def movement_map(video_id, labels=None, t0=0, t1=None, style="trails") -> ToolOutput:
    style = "heatmap" if str(style).lower().startswith("heat") else "trails"
    labels = labels or ["vehicle", "person"]  # nothing named: the usual movers
    base = _track_params(video_id, labels, t0, t1, 5.0, True)
    params = {**base, "style": style, "version": VERSION}
    result, run_id, _, cached = call_gpu("movement_map", video_id, params)
    heads = Counter()
    for c in result["moving_by_heading"].values():
        heads.update(c)
    stat = sum(result["stationary"].values())
    if style == "trails":
        summary = (f"Trails of {result['moving_total']} moving objects ("
                   + ", ".join(f"{n} {h}" for h, n in heads.most_common()) + f"), {stat} stationary")
    else:
        summary = "Heat map; busiest: " + ", ".join(f"{a['area']} {a['share_of_time_pct']}%"
                                                   for a in result["busiest_areas"])
    data = {k: result[k] for k in ("style", "t0", "t1", "tracks_drawn", "moving_total", "moving_by_heading",
                                   "stationary", "busiest_areas")}
    data["picture"] = "shown to the user in the chat"
    data["note"] = ("Directions are as seen in the picture (up = towards the top of the frame), not compass "
                    "directions. Tracks shorter than 1 s are left out; a track can break when an object is hidden, "
                    "so one object may appear as two trails. busiest_areas = share of tracked object-time per "
                    "third of the frame.")
    if not result["tracks_drawn"]:
        data["warning"] = "No tracks found: the detector may have missed these objects."
    return ToolOutput(("(cached) " if cached else "") + summary, data, run_id, None,
                      image_url=url_or_none(result["image_key"]))


register(Tool(
    name="movement_map", tier="gpu",
    description="PICTURE of where objects moved over a time range: style 'trails' draws each tracked object's "
                "path coloured by heading (right/left/up/down) with stationary ones marked; style 'heatmap' "
                "shows where objects spent the most time. Use for 'where do people walk / cars go', 'show the "
                "paths / routes / movement / flow', 'which areas are busiest', 'where do they stop'. Reuses "
                "cached tracks.",
    parameters={"type": "object", "properties": {
        "labels": {"type": "array", "items": {"type": "string"}},
        "t0": {"type": "number", "default": 0}, "t1": {"type": "number"},
        "style": {"type": "string", "enum": ["trails", "heatmap"], "default": "trails"},
    }},
    fn=lambda video_id, labels=None, t0=0, t1=None, style="trails", **_: movement_map(video_id, labels, t0, t1,
                                                                                      style),
    label=lambda a: f"Mapping movement of {', '.join(_safe_labels(a)) or 'vehicles and people'}"
                    + (" (heat map)" if str(a.get("style", "")).startswith("heat") else ""),
))


# --- junction_flow ----------------------------------------------------------------------------

MOVEMENT_COLOURS = {"straight": (3, 155, 229), "left turn": (255, 152, 0), "right turn": (76, 175, 80),
                    "U-turn": (171, 71, 188)}
INCOMPLETE_COLOUR = (150, 150, 150)
JUNCTION_RADIUS = 0.12  # share of the frame's shorter side treated as "inside the junction"
_NAMES = {0: "right", 90: "bottom", 180: "left", -90: "top"}


def _angle(p, c) -> float:
    """Direction of p from c in degrees; 0 = right, 90 = down (image y grows downwards, so + is clockwise)."""
    return math.degrees(math.atan2(p[1] - c[1], p[0] - c[0]))


def _wrap(a: float) -> float:
    return (a + 180) % 360 - 180


JUNCTION_PROMPT = """This is one frame from a video. Is there a road junction / intersection (where two or more roads
meet) in view? If so, give the centre point of the junction where the roads cross.
Coordinates: x from 0 (left edge) to 1000 (right edge), y from 0 (top edge) to 1000 (bottom edge).
Return only JSON: {"junction": true | false, "x": int, "y": int}"""


def locate_junction(video_id, t: float, frame: np.ndarray) -> dict:
    """Ask the VLM where the junction is in the reference frame; cached on the video (per frame time).
    -> {"found": bool | None, "center": [x, y] in 0..1 | None}; found None = the VLM call failed."""
    key = f"{t:.2f}"
    with session_scope() as s:
        cached = ((s.get(Video, video_id).meta or {}).get("junctions") or {}).get(key)
    if cached:
        return cached
    try:
        out = chat_json([{"role": "user", "content": frames_content([(t, frame)], JUNCTION_PROMPT)}], max_tokens=80)
        found = bool(out.get("junction"))
        x, y = float(out.get("x", 0)) / 1000, float(out.get("y", 0)) / 1000
        res = {"found": found, "center": [round(x, 3), round(y, 3)] if found and 0 < x < 1 and 0 < y < 1 else None}
    except Exception as e:
        log.warning("junction location failed: %s", e)
        return {"found": None, "center": None}  # not cached: try again next time
    with session_scope() as s:
        v = s.get(Video, video_id)
        meta = dict(v.meta or {})
        meta["junctions"] = {**(meta.get("junctions") or {}), key: res}
        v.meta = meta
    return res


def find_centre(paths_xy: list[np.ndarray], w: int, h: int, r: float) -> tuple[float, float]:
    """Where roads meet: the point crossed by paths travelling in the most DIFFERENT directions. A busy straight
    road has many paths but one direction; a junction has several. Paths are rasterised at quarter size into
    4 orientation bins (0/45/90/135 degrees, ignoring which way along the line)."""
    q = 4
    hh, ww = h // q + 1, w // q + 1
    acc = np.zeros((4, hh, ww), np.float32)
    thick = max(2, int(r / q / 3))
    for xy in paths_xy:
        pts = (xy / q).astype(np.int32)
        masks = np.zeros((4, hh, ww), np.uint8)
        for a, b in zip(pts[:-1], pts[1:]):
            if (a == b).all():
                continue
            ang = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) % 180
            cv2.line(masks[int(round(ang / 45)) % 4], tuple(int(v) for v in a), tuple(int(v) for v in b), 1, thick)
        acc += masks
    score = np.sqrt(acc).sum(axis=0) * ((acc > 0).sum(axis=0) >= 2)
    score = cv2.GaussianBlur(score, (0, 0), sigmaX=max(1.0, r / q / 3))
    if score.max() <= 0:
        return w / 2, h / 2
    y, x = np.unravel_index(int(np.argmax(score)), score.shape)
    return float(x * q), float(y * q)


def find_arms(angles: list[float], max_arms: int = 5) -> list[float]:
    """Road directions from the junction: peaks of the (circular, smoothed) histogram of where paths start
    and end. Handles diagonal roads, T-junctions and 5-way junctions."""
    if not angles:
        return []
    hist = np.zeros(72)
    for a in angles:
        hist[int(((a + 180) % 360) // 5)] += 1
    kernel = np.exp(-0.5 * (np.arange(-6, 7) / 2.5) ** 2)
    smooth = np.convolve(np.concatenate([hist[-6:], hist, hist[:6]]), kernel, mode="valid")
    peaks = []
    for i in np.argsort(-smooth):
        if smooth[i] < max(1.0, 0.25 * smooth.max()) or len(peaks) >= max_arms:
            break
        a = i * 5 + 2.5 - 180
        if all(abs(_wrap(a - p)) >= 60 for p in peaks):
            peaks.append(a)
    return peaks


def name_arms(arms: list[float]) -> dict[float, str]:
    """Name each road by the nearest picture direction; clashes get a diagonal name (e.g. 'top-left')."""
    names = {}
    for a in arms:
        base = min(_NAMES, key=lambda d: abs(_wrap(a - d)))
        names[a] = _NAMES[base]
    for a in arms:
        if list(names.values()).count(names[a]) > 1:
            vert = "top" if _wrap(a) < 0 else "bottom"
            horiz = "left" if abs(_wrap(a)) > 90 else "right"
            names[a] = f"{vert}-{horiz}"
    return names


def turn_between(src_angle: float, dst_angle: float) -> str:
    """Turn made entering from the road at src_angle and leaving by the road at dst_angle. Geometric, so it holds
    for left- and right-hand traffic."""
    rel = _wrap(dst_angle - (src_angle + 180))  # 0 = carried straight on; + = clockwise on screen = right
    if abs(rel) <= 40:
        return "straight"
    if abs(rel) >= 140:
        return "U-turn"
    return "right turn" if rel > 0 else "left turn"


def _impl_junction(video_id, params, run_id):
    track_run = _track_run(video_id, params)
    ref, paths = _load_paths(video_id, track_run, params["t0"], params["fps"], params["aerial"])
    h, w = ref.shape[:2]
    diag = math.hypot(w, h)
    r = JUNCTION_RADIUS * min(w, h)
    scale = ui_scale(w)
    thick = max(2, int(2 * scale))
    out = _dim(ref, 0.5)

    moving = []
    stationary = 0
    for p in paths:
        xy = _smooth(p["pts"][:, 1:])
        if _is_stationary(xy, diag):
            stationary += 1
        else:
            moving.append((p, xy))
    located = {"found": None, "center": None}
    if params.get("center"):
        c, centre_source = (params["center"][0] * w, params["center"][1] * h), "given"
    else:
        located = locate_junction(video_id, params["t0"], ref)
        if located["center"]:
            c, centre_source = (located["center"][0] * w, located["center"][1] * h), "located by the vision model"
        else:
            c, centre_source = find_centre([xy for _, xy in moving], w, h, r), "estimated from the paths"
    ends = [_angle(pt, c) for _, xy in moving for pt in (xy[0], xy[-1]) if math.dist(pt, c) >= r]
    arms = find_arms(ends)
    names = name_arms(arms)

    def arm_of(pt):
        if math.dist(pt, c) < r or not arms:
            return None
        a = _angle(pt, c)
        return min(arms, key=lambda x: abs(_wrap(a - x)))

    flows = defaultdict(Counter)    # (from, to) names -> label -> n
    moves = {}
    moments = []                    # one per vehicle through the junction, with its time span
    arm_in, arm_out = Counter(), Counter()
    incomplete = Counter()
    drawn = []
    t_end = params["t1"]
    for p, xy in moving:
        src, dst = arm_of(xy[0]), arm_of(xy[-1])
        passed = float(np.min(np.linalg.norm(xy - np.array(c), axis=1))) < r * 1.3
        last_t, first_t = float(p["pts"][-1, 0]), float(p["pts"][0, 0])
        if dst is None or (src is not None and src == dst and not passed):
            if last_t >= t_end - 1.0:
                reason = "still in or approaching the junction when the clip ends"
            elif dst is None:
                reason = "track lost inside the junction"
            else:
                reason = "did not reach the junction"
            incomplete[reason] += 1
            drawn.append((xy, INCOMPLETE_COLOUR))
            continue
        if src is None:
            incomplete["already in the junction when the clip starts" if first_t <= params["t0"] + 1.0
                       else "track started inside the junction"] += 1
            drawn.append((xy, INCOMPLETE_COLOUR))
            continue
        key = (names[src], names[dst])
        moves[key] = turn_between(src, dst)
        # the manoeuvre itself: while the vehicle is in or near the junction (not the time spent queuing)
        near = p["pts"][np.linalg.norm(p["pts"][:, 1:] - np.array(c), axis=1) < r * 1.6, 0]
        ts = near if len(near) else p["pts"][:, 0]
        moments.append({"movement": moves[key], "from": key[0], "to": key[1], "label": p["label"],
                        "t_start": round(float(ts.min()), 1), "t_end": round(float(ts.max()), 1)})
        flows[key][p["label"]] += 1
        arm_in[names[src]] += 1
        arm_out[names[dst]] += 1
        drawn.append((xy, MOVEMENT_COLOURS[moves[key]]))

    cv2.circle(out, (int(c[0]), int(c[1])), int(r), (255, 255, 255), max(1, thick // 2), cv2.LINE_AA)
    for xy, colour in sorted(drawn, key=lambda d: d[1] != INCOMPLETE_COLOUR):  # grey underneath
        cv2.polylines(out, [xy.astype(np.int32).reshape(-1, 1, 2)], False, colour, thick, cv2.LINE_AA)
        _arrow_end(xy, out, colour, thick)
    by_move = Counter()
    for key, labels in flows.items():
        by_move[moves[key]] += sum(labels.values())
    items = [(col, f"{m} {by_move[m]}") for m, col in MOVEMENT_COLOURS.items() if by_move[m]]
    if sum(incomplete.values()):
        items.append((INCOMPLETE_COLOUR, f"incomplete {sum(incomplete.values())}"))
    caption = f"Junction flow {fmt_t(params['t0'])}-{fmt_t(params['t1'])}  ·  {stationary} stationary not drawn"
    img = with_legend(out, items, caption)
    for a in arms:  # badge on each road, pushed out along its direction and kept inside the frame
        n = names[a]
        if not (arm_in[n] or arm_out[n]):
            continue
        d = 0.36 * min(w, h) + r
        x = min(max(c[0] + d * math.cos(math.radians(a)), 0.12 * w), 0.88 * w)
        y = min(max(c[1] + d * math.sin(math.radians(a)), 0.06 * h), 0.94 * h)
        label_box(img, (x, y), f"{n}: in {arm_in[n]} · out {arm_out[n]}", (33, 33, 36), scale)
    key = save_picture(video_id, run_id, img, "junction_flow.jpg")
    flow_rows = sorted(({"from": s, "to": d, "movement": moves[(s, d)], "count": sum(lb.values()),
                         "by_type": dict(lb)} for (s, d), lb in flows.items()), key=lambda f: -f["count"])
    return {**params, "image_key": key, "tracks_run_id": str(track_run), "flows": flow_rows,
            "by_movement": dict(by_move), "vehicles_through_junction": sum(by_move.values()),
            "roads": [names[a] for a in arms],
            "per_road": {names[a]: {"in": arm_in[names[a]], "out": arm_out[names[a]]} for a in arms},
            "centre_used": [round(c[0] / w, 3), round(c[1] / h, 3)], "centre_source": centre_source,
            "junction_visible": located["found"],
            "incomplete": dict(incomplete), "stationary": stationary,
            "moments": sorted(moments, key=lambda m: m["t_start"])}, None


GPU_IMPLS["junction_flow"] = _impl_junction


def junction_flow(video_id, labels=None, t0=0, t1=None, center=None) -> ToolOutput:
    base = _track_params(video_id, labels or ["vehicle"], t0, t1, 5.0, True)
    if center:
        center = [round(min(max(float(v), 0), 1), 3) for v in list(center)[:2]]
    params = {**base, "center": center or None, "version": VERSION}
    result, run_id, _, cached = call_gpu("junction_flow", video_id, params)
    top = ", ".join(f"{f['from']}→{f['to']} {f['count']} ({f['movement']})" for f in result["flows"][:4])
    summary = f"{result['vehicles_through_junction']} through the junction" + (f": {top}" if top else "")
    if result.get("junction_visible") is False:
        summary = "No junction visible; paths drawn anyway (" + summary + ")"
    data = {k: result[k] for k in ("t0", "t1", "centre_used", "centre_source", "roads", "vehicles_through_junction",
                                   "by_movement", "flows", "per_road", "incomplete", "stationary")}
    data["picture"] = "shown to the user in the chat"
    by_move_times = defaultdict(list)
    for m in result.get("moments", []):
        by_move_times[m["movement"]].append([m["t_start"], m["t_end"]])
    data["moments_by_movement"] = dict(by_move_times)  # pass to make_highlight_clip as segments
    data["moments_note"] = "moments_by_movement: [start, end] in seconds while each vehicle crosses the junction."
    if result.get("junction_visible") is False:
        data["warning"] = ("The vision model sees no road junction in this video, so 'turns' here are not "
                           "meaningful; describe the traffic direction instead (or use movement_map).")
    data["method"] = ("Each tracked object's path is mapped into one camera-stabilised frame. The junction centre "
                      "is pointed out by the vision model on that frame (or, failing that, estimated from where paths "
                      "travelling in different directions cross); the roads are the directions paths most often start and end "
                      "in. The road a vehicle came from is where its path starts, the road it left by is where it "
                      "ends, and the turn follows from the angle between them. Paths that start or end inside the "
                      "junction, or never reach it, are incomplete. Objects that barely move are stationary "
                      "(parked, or waiting the whole time).")
    data["note"] = ("Road names are as seen in the picture (top = top of the frame), not compass directions. "
                    "Tracks can break when vehicles are hidden, so treat counts as approximate. The junction "
                    "centre is found automatically; pass center=[x,y] (0..1) to override it.")
    return ToolOutput(("(cached) " if cached else "") + summary, data, run_id, None,
                      image_url=url_or_none(result["image_key"]))


register(Tool(
    name="junction_flow", tier="gpu",
    description="Turning movements at a road junction / intersection: for each tracked vehicle, the road it "
                "came from and the road it left by (top/right/bottom/left as seen in the picture), giving "
                "straight / left turn / right turn / U-turn counts and a PICTURE of the paths. Use for traffic "
                "flow, 'where do cars come from / go', 'how many turned left', busiest road. labels default "
                "['vehicle']; center = junction centre [x,y] in 0..1 (found automatically if omitted).",
    parameters={"type": "object", "properties": {
        "labels": {"type": "array", "items": {"type": "string"}},
        "t0": {"type": "number", "default": 0}, "t1": {"type": "number"},
        "center": {"type": "array", "items": {"type": "number"}},
    }},
    fn=lambda video_id, labels=None, t0=0, t1=None, center=None, **_: junction_flow(video_id, labels, t0, t1,
                                                                                    center),
    label=lambda a: "Measuring junction flow" + (f" of {', '.join(_safe_labels(a))}" if a.get("labels") else ""),
))
