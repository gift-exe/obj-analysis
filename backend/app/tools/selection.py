"""Click-to-ask: the user clicks an object on a paused frame; SAM outlines it; the agent can then ask what it is,
follow it through the video, or look for it elsewhere.

A selection is stored as a cached tool run ("select_object"), so its id is the tool run id.
"""

import logging
import math
import uuid

import cv2
import numpy as np
from sqlalchemy import select

from app.core.db import session_scope
from app.core.media import FrameReader, fmt_t
from app.core.storage import url_or_none
from app.llm import chat_json, frames_content
from app.models import Detection, ToolRun, Track
from app.tools.pictures import save_picture, ui_scale, with_legend
from app.tools.registry import GPU_IMPLS, Tool, ToolOutput, call_gpu, execute_gpu_tool, register
from app.tools.vision import (MIN_TRACK_S, _clean, _device, _model_lock, _proxy_path, _range, _segmenter,
                              _track_params)

log = logging.getLogger(__name__)
VERSION = 2
SELECTION_TOOLS = {"describe_selection", "follow_selection", "find_selection_elsewhere"}
OUTLINE = (0, 230, 118)


# --- making a selection ------------------------------------------------------------------------

def _impl_select(video_id, params, run_id):
    with FrameReader(_proxy_path(video_id)) as r:
        frame = r.at(params["t"])
    if frame is None:
        raise RuntimeError(f"no frame at t={params['t']}")
    h, w = frame.shape[:2]
    px, py = params["x"] * w, params["y"] * h
    with _model_lock:
        res = _segmenter()(frame, points=[[px, py]], labels=[1], verbose=False, device=_device())[0]
    mask = np.zeros((h, w), bool)
    if res.masks is not None and len(res.masks.data):
        m = res.masks.data[0].cpu().numpy().astype(np.uint8)
        mask = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST) > 0
    # keep only the connected piece under the click
    n, lab = cv2.connectedComponents(mask.astype(np.uint8))
    under = lab[min(h - 1, int(py)), min(w - 1, int(px))]
    if n > 1 and under:
        mask = lab == under
    if not mask.any():  # SAM found nothing: fall back to a small box around the click
        r_ = int(0.03 * min(w, h))
        mask[max(0, int(py) - r_):int(py) + r_, max(0, int(px) - r_):int(px) + r_] = True
    ys, xs = np.where(mask)
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea)
    poly = cv2.approxPolyDP(cnt, 0.004 * cv2.arcLength(cnt, True), True).reshape(-1, 2)[:120]
    # close-up crop with some context around the object
    pad = int(max(16, 0.25 * max(x1 - x0, y1 - y0)))
    cx0, cy0, cx1, cy1 = max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)
    crop = frame[cy0:cy1, cx0:cx1]
    from PIL import Image

    key = save_picture(video_id, run_id, Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)), "selection.jpg")
    return {**params, "bbox": [round(x0 / w, 4), round(y0 / h, 4), round(x1 / w, 4), round(y1 / h, 4)],
            "polygon": [[round(float(x) / w, 4), round(float(y) / h, 4)] for x, y in poly],
            "area_pct": round(100 * float(mask.mean()), 2), "crop_key": key,
            "crop_box": [int(cx0), int(cy0), int(cx1), int(cy1)]}, None


GPU_IMPLS["select_object"] = _impl_select


def select_object(video_id, t: float, x: float, y: float) -> dict:
    """Outline the object at (x, y) (0..1 of the frame) on the frame at t. Returns the selection for the UI."""
    lo, _, duration = _range(video_id, t, None)
    params = {"t": round(min(lo, max(0.0, duration - 0.05)), 2), "x": round(min(max(float(x), 0), 1), 4),
              "y": round(min(max(float(y), 0), 1), 4), "version": VERSION}
    result, run_id, _, _ = call_gpu("select_object", video_id, params)
    return selection_public(run_id, result)


def selection_public(run_id, result: dict) -> dict:
    return {"id": str(run_id), "t": result["t"], "x": result["x"], "y": result["y"], "bbox": result["bbox"],
            "polygon": result["polygon"], "area_pct": result["area_pct"], "crop_url": url_or_none(result["crop_key"])}


def load_selection(video_id, selection_id) -> tuple[uuid.UUID, dict]:
    with session_scope() as s:
        run = s.get(ToolRun, uuid.UUID(str(selection_id)))
        if run is None or run.tool != "select_object" or str(run.video_id) != str(video_id):
            raise ValueError("unknown selection for this video; ask the user to click the object again")
        return run.id, dict(run.result)


def _remember(selection_id, **fields) -> None:
    with session_scope() as s:
        run = s.get(ToolRun, uuid.UUID(str(selection_id)))
        run.result = {**(run.result or {}), **fields}


def context_note(sel: dict) -> str:
    b = sel["bbox"]
    return (f"[The user clicked an object in the video before asking: selection_id={sel['id']}, at "
            f"{fmt_t(sel['t'])} (t={sel['t']}s), box x {b[0]:.2f}-{b[2]:.2f}, y {b[1]:.2f}-{b[3]:.2f} of the frame. "
            "Words like 'this', 'it', 'that' refer to it. Use describe_selection, follow_selection or "
            "find_selection_elsewhere.]")


def _outlined_frame(video_id, sel: dict) -> np.ndarray:
    with FrameReader(_proxy_path(video_id)) as r:
        frame = r.at(sel["t"])
    h, w = frame.shape[:2]
    pts = (np.array(sel["polygon"]) * [w, h]).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(frame, [pts], True, OUTLINE[::-1], max(2, int(w / 400)), cv2.LINE_AA)
    return frame


def _crop(video_id, sel: dict) -> np.ndarray:
    with FrameReader(_proxy_path(video_id)) as r:
        frame = r.at(sel["t"])
    x0, y0, x1, y1 = sel["crop_box"]
    crop = frame[y0:y1, x0:x1]
    if max(crop.shape[:2]) < 320:  # small objects: enlarge so the VLM can see detail
        f = 320 / max(crop.shape[:2])
        crop = cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    return crop


# --- describe_selection ------------------------------------------------------------------------

DESCRIBE_PROMPT = """The user selected the object outlined in green in the first image (a frame from a video);
the second image is a close-up of it. Answer the user's question about THAT object.
Question: {question}
Use only what is clearly visible. If the detail cannot be made out, say so.
Return only JSON: {{"object": "generic type in 1-3 words (e.g. car, person, shopping basket)",
  "answer": "direct answer in 1-3 sentences", "visible": true or false, "confidence": number 0-1}}"""


def describe_selection(video_id, selection_id, question="What is this?") -> ToolOutput:
    sid, sel = load_selection(video_id, selection_id)
    sel["id"] = str(sid)
    frames = [(sel["t"], _outlined_frame(video_id, sel)), (sel["t"], _crop(video_id, sel))]
    data = chat_json([{"role": "user", "content": frames_content(
        frames, DESCRIBE_PROMPT.format(question=question or "What is this?"))}], max_tokens=400)
    obj = _clean(data.get("object") or "")
    if obj and not sel.get("object"):
        _remember(sid, object=obj)
    data = {"t": sel["t"], **data}
    return ToolOutput(f"Selected object at {fmt_t(sel['t'])}: {str(data.get('answer', ''))[:140]}", data)


register(Tool(
    name="describe_selection", tier="vlm",
    description="Look closely at the object the user clicked (outlined, plus a close-up) and answer a question "
                "about it: what it is, its colour, brand, state, what it is doing. Use when the message comes "
                "with a selected object.",
    parameters={"type": "object", "properties": {
        "selection_id": {"type": "string"}, "question": {"type": "string", "default": "What is this?"},
    }, "required": ["selection_id"]},
    fn=lambda video_id, selection_id, question="What is this?", **_: describe_selection(video_id, selection_id,
                                                                                         question),
    label=lambda a: "Looking at the selected object",
))


# --- follow_selection --------------------------------------------------------------------------

def _object_word(video_id, sid, sel) -> str:
    if sel.get("object"):
        return sel["object"]
    out = describe_selection(video_id, sid, "What type of object is this? Answer with its generic type.")
    return out.data.get("object") and _clean(out.data["object"]) or "object"


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _match_in_run(run_id, sel) -> tuple[int, str, float] | None:
    """The track in this run under the selection at its time: (track_no, label, score)."""
    best = None
    with session_scope() as s:
        rows = s.execute(select(Track.track_no, Track.label, Detection.bbox).join(Detection, Detection.track_id == Track.id)
                         .where(Track.tool_run_id == run_id, Detection.t.between(sel["t"] - 0.25, sel["t"] + 0.25))).all()
    for tno, label, bb in rows:
        score = _iou(bb, sel["bbox"])
        if bb[0] <= sel["x"] <= bb[2] and bb[1] <= sel["y"] <= bb[3]:
            score += 0.2  # the click is inside this box
        if score >= 0.3 and (best is None or score > best[2]):
            best = (tno, label, score)
    return best


def _find_track(video_id, sid, sel) -> tuple[uuid.UUID | None, tuple | None, str, bool]:
    """-> (track run, (track_no, label, score) or None, object word, tracking was newly run)."""
    with session_scope() as s:
        runs = s.scalars(select(ToolRun).where(ToolRun.video_id == uuid.UUID(str(video_id)),
                                               ToolRun.tool == "segment_and_track", ToolRun.status == "done")
                         .order_by(ToolRun.created_at.desc())).all()
        runs = [(r.id, r.params) for r in runs]
    best = None  # reuse cached tracks that cover this moment: the best match, then the longest track
    for rid, p in runs:
        if p.get("t0", 0) <= sel["t"] <= p.get("t1", 0) and p.get("fps", 0) >= 2:
            m = _match_in_run(rid, sel)
            if m:
                with session_scope() as s:
                    tr = s.scalars(select(Track).where(Track.tool_run_id == rid, Track.track_no == m[0])).first()
                    span = (tr.t_end - tr.t_start) if tr else 0.0
                key = (round(m[2], 1), span)
                if best is None or key > best[0]:
                    best = (key, rid, m)
    if best:
        return best[1], best[2], best[2][1], False
    word = _object_word(video_id, sid, sel)
    tp = _track_params(video_id, [word], 0, None, 5.0, True)
    tr = execute_gpu_tool("segment_and_track", str(video_id), tp)
    rid = uuid.UUID(tr["tool_run_id"])
    return rid, _match_in_run(rid, sel), word, True


def _chain(paths: dict, start_no: int, diag: float) -> list[int]:
    """Link track fragments of the same object: a track that starts within 2.5 s after another ends, close to
    where it ended (trackers break tracks when an object is hidden or detection flickers)."""
    chain = [start_no]
    for direction in (1, -1):
        cur = start_no
        while True:
            p = paths[cur]
            end_t, end_xy = (p["pts"][-1, 0], p["pts"][-1, 1:]) if direction == 1 else (p["pts"][0, 0], p["pts"][0, 1:])
            best, best_d = None, 0.06 * diag
            for no, q in paths.items():
                if no in chain or q["label"] != p["label"]:
                    continue
                qt, qxy = (q["pts"][0, 0], q["pts"][0, 1:]) if direction == 1 else (q["pts"][-1, 0], q["pts"][-1, 1:])
                gap = (qt - end_t) * direction
                d = float(np.linalg.norm(qxy - end_xy))
                if 0 < gap <= 2.5 and d < best_d:
                    best, best_d = no, d
            if best is None:
                break
            chain.append(best) if direction == 1 else chain.insert(0, best)
            cur = best
    return chain


def _impl_follow(video_id, params, run_id):
    from app.tools.motion import _arrow_end, _dim, _heading, _is_stationary, _load_paths, _smooth

    sid, sel = load_selection(video_id, params["selection_id"])
    track_run, match, word, fresh = _find_track(video_id, sid, sel)
    if match is None:
        return {**params, "found": False, "object": word, "tracking_run": bool(fresh)}, None
    with session_scope() as s:
        aerial = bool((s.get(ToolRun, track_run).params or {}).get("aerial"))
        fps = float((s.get(ToolRun, track_run).params or {}).get("fps") or 5)
    ref, paths = _load_paths(video_id, track_run, sel["t"], fps, aerial)
    by_no = {p["track"]: p for p in paths}
    if match[0] not in by_no:  # matched track too short to be kept by _load_paths
        return {**params, "found": False, "object": word, "reason": "only seen briefly"}, None
    h, w = ref.shape[:2]
    diag = math.hypot(w, h)
    chain = _chain(by_no, match[0], diag)
    pts = np.concatenate([by_no[n]["pts"] for n in chain])
    xy = _smooth(pts[:, 1:])
    t_first, t_last = float(pts[0, 0]), float(pts[-1, 0])
    stationary = _is_stationary(xy, diag)
    scale = ui_scale(w)
    thick = max(3, int(3 * scale))
    out = _dim(ref, 0.55)
    poly = (np.array(sel["polygon"]) * [w, h]).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(out, [poly], True, OUTLINE, max(2, thick - 1), cv2.LINE_AA)
    if not stationary:
        cv2.polylines(out, [xy.astype(np.int32).reshape(-1, 1, 2)], False, (255, 202, 40), thick, cv2.LINE_AA)
        cv2.circle(out, tuple(int(v) for v in xy[0]), thick + 3, (255, 202, 40), -1, cv2.LINE_AA)
        _arrow_end(xy, out, (255, 202, 40), thick)
    items = [(OUTLINE, f"selected at {fmt_t(sel['t'])}"),
             ((255, 202, 40), f"path {fmt_t(t_first)} - {fmt_t(t_last)}" if not stationary else
              f"stayed put {fmt_t(t_first)} - {fmt_t(t_last)}")]
    caption = f"{match[1]} followed through the video  ·  dot = first seen, arrow = last seen"
    key = save_picture(video_id, run_id, with_legend(out, items, caption), "follow.jpg")
    with session_scope() as s:
        t1 = float((s.get(ToolRun, track_run).params or {}).get("t1") or 0)
    return {**params, "found": True, "object": match[1], "image_key": key, "first_seen": round(t_first, 2),
            "last_seen": round(t_last, 2), "on_screen_s": round(t_last - t_first, 1), "stationary": stationary,
            "heading": None if stationary else _heading(xy), "track_pieces": len(chain),
            "segments": [[round(float(by_no[n]["pts"][0, 0]), 2), round(float(by_no[n]["pts"][-1, 0]), 2)]
                         for n in chain],
            "in_view_at_start": t_first <= 0.5, "in_view_at_end": t_last >= t1 - 0.5,
            "tracking_run": str(track_run), "matched": {"track": match[0], "label": match[1]}, "chain": chain}, None


GPU_IMPLS["follow_selection"] = _impl_follow


def follow_selection(video_id, selection_id) -> ToolOutput:
    params = {"selection_id": str(selection_id), "version": VERSION}
    result, run_id, _, cached = call_gpu("follow_selection", video_id, params)
    if not result["found"]:
        data = {"found": False, "object": result.get("object"),
                "note": "The selected object could not be matched to a tracked object (the detector may not "
                        "recognise it, or it is only visible briefly). Use describe_selection or watch_clip."}
        return ToolOutput(f"Could not follow the selected {result.get('object') or 'object'}", data, run_id)
    r = result
    when = f"{fmt_t(r['first_seen'])}-{fmt_t(r['last_seen'])}"
    move = "stayed in place" if r["stationary"] else r["heading"]
    summary = f"Followed the {r['object']}: on screen {when} ({r['on_screen_s']} s), {move}"
    data = {k: r[k] for k in ("object", "first_seen", "last_seen", "on_screen_s", "stationary", "heading",
                              "in_view_at_start", "in_view_at_end", "segments", "track_pieces")}
    data.update(found=True, picture="shown to the user in the chat (path on the frame)",
                note="first_seen/last_seen = when the tracker first and last saw it; in_view_at_start/end mean it "
                     "was already there when the video starts / still there when it ends. Track pieces broken by "
                     "short occlusions are joined, but a long disappearance can end the track early. Directions "
                     "are as seen in the picture. segments can be passed to make_highlight_clip.")
    return ToolOutput(("(cached) " if cached else "") + summary, data, run_id, None,
                      image_url=url_or_none(r["image_key"]))


register(Tool(
    name="follow_selection", tier="gpu",
    description="Follow the object the user clicked through the video: when it was first and last seen, how long "
                "it was on screen, whether it moved and which way, with a PICTURE of its path. Use for 'when did "
                "this arrive / leave', 'where did it go', 'how long was it here'. Reuses cached tracks.",
    parameters={"type": "object", "properties": {"selection_id": {"type": "string"}}, "required": ["selection_id"]},
    fn=lambda video_id, selection_id, **_: follow_selection(video_id, selection_id),
    label=lambda a: "Following the selected object",
))


# --- find_selection_elsewhere ------------------------------------------------------------------

# Appearance match = colour (HSV histogram, rotation-proof; the main cue for telling vehicles apart) blended
# with SigLIP image similarity (shape / type). Measured on road-intersect: the same car seen in another frame
# scored colour 0.73 / SigLIP 0.88; every different car scored colour <= 0.60.
LIKELY_SAME = (0.68, 0.86)   # (colour, siglip) both at least this -> likely the same or an identical-looking one
SIMILAR_COLOUR = 0.55


def _hist(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1, 2], mask, [16, 4, 4], [0, 180, 0, 256, 0, 256])
    return cv2.normalize(h, h).flatten()


def _ellipse_mask(img: np.ndarray) -> np.ndarray:
    """Central ellipse of a box crop: the object, not the road around a rotated car."""
    m = np.zeros(img.shape[:2], np.uint8)
    hh, ww = m.shape
    cv2.ellipse(m, (ww // 2, hh // 2), (max(1, int(ww * 0.38)), max(1, int(hh * 0.38))), 0, 0, 360, 255, -1)
    return m


def _impl_find(video_id, params, run_id):
    from app.embeddings import embed_images

    sid, sel = load_selection(video_id, params["selection_id"])
    track_run, match, word, _ = _find_track(video_id, sid, sel)
    own: set[int] = set()
    if match:
        # nested GPU work must call the implementation directly (call_gpu from inside a GPU job deadlocks the queue)
        fol = execute_gpu_tool("follow_selection", str(video_id), {"selection_id": str(sid), "version": VERSION})["result"]
        own = {match[0], *fol.get("chain", [])}  # every piece of the selected object's own track
    min_frames = max(2, round(MIN_TRACK_S * 5))
    with session_scope() as s:
        cands = []
        for tr in s.scalars(select(Track).where(Track.tool_run_id == track_run)):
            if tr.track_no in own:
                continue
            dets = s.scalars(select(Detection).where(Detection.track_id == tr.id).order_by(Detection.t)).all()
            if len(dets) >= min_frames and (not match or tr.label == match[1]):
                mid = dets[len(dets) // 2]
                cands.append({"track": tr.track_no, "label": tr.label, "t_start": tr.t_start, "t_end": tr.t_end,
                              "t": mid.t, "bbox": mid.bbox})
    if not cands:
        return {**params, "object": word, "matches": [], "candidates": 0}, None
    crops = []
    with FrameReader(_proxy_path(video_id)) as r:
        f = r.at(sel["t"])
        h, w = f.shape[:2]
        x0, y0, x1, y1 = (np.array(sel["bbox"]) * [w, h, w, h]).astype(int)
        sel_crop = f[y0:max(y0 + 2, y1), x0:max(x0 + 2, x1)]  # tight, like the candidates
        poly = np.zeros((h, w), np.uint8)
        cv2.fillPoly(poly, [(np.array(sel["polygon"]) * [w, h]).astype(np.int32)], 255)
        sel_mask = poly[y0:max(y0 + 2, y1), x0:max(x0 + 2, x1)]
        for c in cands:
            g = r.at(c["t"])
            a, b, cc, d = (np.array(c["bbox"]) * [w, h, w, h]).astype(int)
            crops.append(g[max(0, b):max(b + 2, d), max(0, a):max(a + 2, cc)])
    emb = embed_images([cv2.cvtColor(x, cv2.COLOR_BGR2RGB) for x in [sel_crop] + crops])
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    look = emb[1:] @ emb[0]
    hs = _hist(sel_crop, sel_mask if sel_mask.any() else None)
    colour = np.array([1 - cv2.compareHist(hs, _hist(x, _ellipse_mask(x)), cv2.HISTCMP_BHATTACHARYYA) for x in crops])
    score = 0.6 * colour + 0.4 * look
    order = np.argsort(-score)[:6]

    def verdict(i):
        if colour[i] >= LIKELY_SAME[0] and look[i] >= LIKELY_SAME[1]:
            return "likely the same or an identical-looking one"
        return "similar colour, may be different" if colour[i] >= SIMILAR_COLOUR else "different look"

    matches = [{**{k: cands[i][k] for k in ("t_start", "t_end", "label")}, "colour_match": round(float(colour[i]), 2),
                "look_match": round(float(look[i]), 2), "verdict": verdict(i)} for i in order]
    # contact sheet: the selection, then the closest matches
    tile = 220

    def fit(img):
        hh, ww = img.shape[:2]
        k = tile / max(hh, ww)
        img = cv2.resize(img, (max(1, int(ww * k)), max(1, int(hh * k))))
        canvas = np.full((tile, tile, 3), 30, np.uint8)
        y, x = (tile - img.shape[0]) // 2, (tile - img.shape[1]) // 2
        canvas[y:y + img.shape[0], x:x + img.shape[1]] = img
        return canvas

    sheet = cv2.cvtColor(np.hstack([fit(sel_crop)] + [fit(crops[i]) for i in order]), cv2.COLOR_BGR2RGB)
    cv2.rectangle(sheet, (2, 2), (tile - 3, tile - 3), OUTLINE, 4)
    tag = {"likely the same or an identical-looking one": "likely same", "similar colour, may be different": "similar",
           "different look": "different"}
    items = [(OUTLINE, f"selected {fmt_t(sel['t'])}")] + [
        ((255, 202, 40) if m["verdict"] != "different look" else (150, 150, 150),
         f"{fmt_t(m['t_start'])} {tag[m['verdict']]}") for m in matches]
    key = save_picture(video_id, run_id, with_legend(sheet, items, "Closest-looking objects elsewhere, best first "
                                                                    "(colour + appearance)"), "similar.jpg")
    return {**params, "object": match[1] if match else word, "matches": matches, "candidates": len(cands),
            "image_key": key}, None


GPU_IMPLS["find_selection_elsewhere"] = _impl_find


def find_selection_elsewhere(video_id, selection_id) -> ToolOutput:
    params = {"selection_id": str(selection_id), "version": VERSION + 1}
    result, run_id, _, cached = call_gpu("find_selection_elsewhere", video_id, params)
    m = result["matches"]
    if not m:
        return ToolOutput(f"No other {result.get('object') or 'objects'} to compare with", {
            "matches": [], "note": "No other tracked objects of this type were found in the video."}, run_id)
    likely = [x for x in m if x["verdict"].startswith("likely")]
    if likely:
        summary = (f"Possible re-appearance of the selected {result['object']}: "
                   + ", ".join(fmt_t(x["t_start"]) for x in likely))
    else:
        summary = f"No clear re-appearance of the selected {result['object']}; closest looks are different"
    data = {"object": result["object"], "compared_with": result["candidates"], "matches": m,
            "likely_same_count": len(likely),
            "picture": "shown to the user in the chat (selected object, then the closest matches)",
            "note": "Each candidate is another tracked object of the same type, compared by colour and appearance. "
                    "Only a 'likely the same' verdict suggests a re-appearance, and even then an identical-looking "
                    "object (same model and colour) is possible. If likely_same_count is 0, say no clear "
                    "re-appearance was found (the object may simply stay in one continuous track)."}
    return ToolOutput(("(cached) " if cached else "") + summary, data, run_id, None,
                      image_url=url_or_none(result.get("image_key")))


register(Tool(
    name="find_selection_elsewhere", tier="gpu",
    description="Find other appearances of the object the user clicked: compares its look with every other "
                "tracked object of the same type and returns the most similar ones with their times and a "
                "PICTURE. Use for 'is this the same as...', 'does this appear again', 'where else is it'.",
    parameters={"type": "object", "properties": {"selection_id": {"type": "string"}}, "required": ["selection_id"]},
    fn=lambda video_id, selection_id, **_: find_selection_elsewhere(video_id, selection_id),
    label=lambda a: "Looking for the selected object elsewhere",
))
