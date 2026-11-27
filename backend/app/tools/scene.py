"""segment_scene: colour every pixel of one frame by class (sky, tree, building, road, car...) and return a picture.

Detectors draw boxes around objects; that is the wrong tool for regions such as sky, road or a tree canopy.
This uses semantic segmentation instead:
- ground-level / elevated video: SegFormer trained on ADE20K (150 everyday classes);
- aerial (drone, straight-down) video: SegFormer fine-tuned on UAVid drone footage (building, tree, road, car,
  person). ADE20K models trained on ground-level photos mislabel top-down views (roads become "building").
"""

import logging
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image

from app.core.config import get_settings
from app.core.media import FrameReader, fmt_t
from app.core.storage import url_or_none
from app.tools.pictures import save_picture, with_legend
from app.tools.registry import GPU_IMPLS, Tool, ToolOutput, call_gpu, register
from app.tools.vision import _clean, _model_lock, _proxy_path, _range, get_viewpoint

log = logging.getLogger(__name__)

VERSION = 3

# Requested word -> model classes it covers. Words not listed here are matched to a class name directly.
ALIASES = {
    "tree": ["tree", "palm"], "vegetation": ["tree", "palm", "grass", "plant", "flower", "field"],
    "greenery": ["tree", "palm", "grass", "plant", "flower"], "bush": ["plant"], "hedge": ["plant"],
    "shrub": ["plant"], "plant": ["plant", "flower"], "lawn": ["grass"], "grass": ["grass", "field"],
    "building": ["building", "house", "skyscraper", "tower", "hovel"], "house": ["house"],
    "car": ["car", "van"], "vehicle": ["car", "van", "truck", "bus", "minibike", "bicycle"],
    "traffic": ["car", "van", "truck", "bus", "minibike"], "motorcycle": ["minibike"], "motorbike": ["minibike"],
    "lorry": ["truck"], "person": ["person"], "people": ["person"], "pedestrian": ["person"],
    "human": ["person"], "sky": ["sky"], "cloud": ["sky"], "road": ["road"], "street": ["road"],
    "sidewalk": ["sidewalk"], "pavement": ["sidewalk"], "footpath": ["sidewalk", "path"],
    "water": ["water", "sea", "river", "lake", "swimming pool"], "ground": ["earth", "land", "field", "sand"],
}
AERIAL_CLASSES = {"building", "tree", "road", "car", "person"}  # what the UAVid model actually predicts
# On aerial footage, vehicles are also painted from the drone-trained vehicle detector (DOTA classes), which
# finds the grey/white/dark cars the segmentation model misses.
AERIAL_VEHICLES = {"car": ["small vehicle"], "vehicle": ["small vehicle", "large vehicle"],
                   "traffic": ["small vehicle", "large vehicle"], "truck": ["large vehicle"],
                   "bus": ["large vehicle"], "lorry": ["large vehicle"]}

# Conventional colours (RGB) so pictures read naturally; anything else cycles through PALETTE.
COLOURS = {"sky": (66, 165, 245), "tree": (46, 160, 67), "vegetation": (46, 160, 67), "grass": (139, 195, 74),
           "building": (245, 124, 0), "car": (229, 57, 53), "vehicle": (229, 57, 53), "road": (126, 87, 194),
           "person": (255, 214, 0), "people": (255, 214, 0), "water": (0, 172, 193), "sidewalk": (236, 64, 122)}
PALETTE = [(0, 137, 123), (216, 27, 96), (124, 179, 66), (94, 53, 177), (251, 140, 0), (3, 155, 229),
           (192, 202, 51), (109, 76, 65), (84, 110, 122), (244, 81, 30)]
AUTO_MIN_SHARE = 0.01  # with no labels requested, show classes covering at least 1% of the frame
MAX_CLASSES = 10


@lru_cache(maxsize=2)
def _model(name: str):
    import torch
    from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

    from app.embeddings import pick_device

    device = pick_device()
    proc = AutoImageProcessor.from_pretrained(name)
    model = SegformerForSemanticSegmentation.from_pretrained(name).to(device).eval()
    return proc, model, device, torch


def _segment(name: str, frame_bgr: np.ndarray) -> tuple[np.ndarray, dict[int, str]]:
    """-> (H x W array of class ids, id -> class name)."""
    proc, model, device, torch = _model(name)
    img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    inputs = proc(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    seg = proc.post_process_semantic_segmentation(out, target_sizes=[img.size[::-1]])[0].cpu().numpy()
    return seg.astype(np.int32), {int(k): v.strip() for k, v in model.config.id2label.items()}


def resolve_labels(labels: list[str], available: set[str], extra: set[str] = frozenset()
                   ) -> tuple[dict[str, list[str]], list[str]]:
    """Requested words -> ({word: model classes}, words the model cannot segment).
    `extra`: words handled another way (aerial vehicles), kept even with no segmentation class."""
    plan: dict[str, list[str]] = {}
    unknown: list[str] = []
    for raw in labels:
        word = _clean(raw)
        if not word or word in plan:
            continue
        members = [c for c in ALIASES.get(word, [word]) if c in available]
        if members or word in extra:
            plan[word] = members
        else:
            unknown.append(word)
    return plan, unknown


def _aerial_vehicle_masks(frame: np.ndarray, words: list[str]) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Filled oriented boxes from the aerial vehicle detector, per requested word, plus how many it found."""
    from ultralytics import YOLO

    from app.tools.vision import _aerial_names, _device

    names = _aerial_names()
    wanted = sorted({c for w in words for c in AERIAL_VEHICLES[w]})
    res = YOLO(get_settings().aerial_detector_model).predict(
        frame, conf=0.25, imgsz=1024, verbose=False, device=_device(), classes=[names[c] for c in wanted])[0]
    inv = {v: k for k, v in names.items()}
    polys = []
    if res.obb is not None and len(res.obb):
        polys = [(inv[c], p) for c, p in zip(res.obb.cls.int().tolist(), res.obb.xyxyxyxy.cpu().numpy())]
    masks, counts = {}, {}
    for w in words:
        m = np.zeros(frame.shape[:2], np.uint8)
        mine = [p for c, p in polys if c in AERIAL_VEHICLES[w]]
        if mine:
            cv2.fillPoly(m, [p.astype(np.int32) for p in mine], 1)
        masks[w], counts[w] = m.astype(bool), len(mine)
    return masks, counts


def render(frame_bgr: np.ndarray, masks: dict[str, np.ndarray], colours: dict[str, tuple],
           shares: dict[str, float], caption: str) -> Image.Image:
    """Frame with translucent class fills, class outlines and a legend strip underneath."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    fill = rgb.copy()
    for name, m in masks.items():
        fill[m] = colours[name]
    out = cv2.addWeighted(fill, 0.5, rgb, 0.5, 0)
    for name, m in masks.items():
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, colours[name], 2, cv2.LINE_AA)
    return with_legend(out, [(colours[n], f"{n}  {shares[n]:.1f}%") for n in masks], caption)


def _impl_scene(video_id, params, run_id):
    model_name = params["model"]
    with FrameReader(_proxy_path(video_id)) as r:
        frame = r.at(params["t"])
    if frame is None:
        raise RuntimeError(f"no frame at t={params['t']}")
    with _model_lock:
        seg, id2label = _segment(model_name, frame)
    total = seg.size
    class_px = {id2label[int(i)]: int(n) for i, n in zip(*np.unique(seg, return_counts=True))}
    available = set(id2label.values()) & (AERIAL_CLASSES if params["aerial"] else set(id2label.values()))

    vehicle_words: list[str] = []
    if params["labels"]:
        extra = set(AERIAL_VEHICLES) if params["aerial"] else set()
        plan, unknown = resolve_labels(params["labels"], available, extra)
        vehicle_words = [w for w in plan if w in extra]
    else:  # nothing requested: show the main regions of the frame
        top = sorted(((n, c) for c, n in class_px.items() if c in available), reverse=True)
        plan = {c: [c] for n, c in top if n / total >= AUTO_MIN_SHARE}
        plan = dict(list(plan.items())[:MAX_CLASSES])
        unknown = []

    detected: dict[str, int] = {}
    extra_masks: dict[str, np.ndarray] = {}
    if vehicle_words:
        with _model_lock:
            extra_masks, detected = _aerial_vehicle_masks(frame, vehicle_words)
    name_to_id = {v: k for k, v in id2label.items()}
    masks, shares, colours, regions = {}, {}, {}, {}
    k = 0
    for word, members in list(plan.items())[:MAX_CLASSES]:
        m = np.isin(seg, [name_to_id[c] for c in members])
        if word in extra_masks:
            m |= extra_masks[word]
        shares[word] = round(100 * m.sum() / total, 1)
        if not m.any():
            continue
        masks[word] = m
        colours[word] = COLOURS.get(word) or PALETTE[k % len(PALETTE)]
        k += 1
        n_regions, _, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8))
        regions[word] = int(sum(1 for s in stats[1:] if s[cv2.CC_STAT_AREA] >= total * 0.0005))

    model_label = "UAVid (drone footage)" if params["aerial"] else "ADE20K"
    caption = f"t = {fmt_t(params['t'])}  ·  SegFormer {model_label}"
    if detected:
        caption += " + aerial vehicle detector"
    caption += "  ·  % = share of the frame"
    key = save_picture(video_id, run_id, render(frame, masks, colours, shares, caption), "scene.jpg")
    return {**params, "image_key": key, "coverage_pct": shares,
            "classes_used": {w: plan[w] for w in shares}, "separate_regions": regions,
            "not_visible": [w for w, s in shares.items() if s == 0], "unsupported": unknown,
            "vehicles_detected": detected,
            "segmentation_model": model_label}, None


GPU_IMPLS["segment_scene"] = _impl_scene


def segment_scene(video_id, labels=None, t=0.0) -> ToolOutput:
    if isinstance(labels, str):
        labels = [s for s in labels.split(",") if s.strip()]
    lo, _, duration = _range(video_id, t, None)
    t = min(lo, max(0.0, duration - 0.05))
    aerial = get_viewpoint(video_id) == "aerial"
    s = get_settings()
    params = {"t": round(t, 2), "labels": sorted({_clean(x) for x in (labels or []) if _clean(x)}),
              "aerial": aerial, "model": s.aerial_scene_model if aerial else s.scene_model, "version": VERSION}
    result, run_id, _, cached = call_gpu("segment_scene", video_id, params)
    cov = result["coverage_pct"]
    parts = [f"{w} {p}%" for w, p in cov.items() if p > 0]
    summary = f"Highlighted at {fmt_t(result['t'])}: " + (", ".join(parts) or "none of the requested classes found")
    for w in result["not_visible"]:
        summary += f"; no {w} visible"
    if result["unsupported"]:
        summary += f"; can't segment {', '.join(result['unsupported'])}"
    data = {"t": result["t"], "coverage_pct_of_frame": cov, "not_visible": result["not_visible"],
            "unsupported_by_model": result["unsupported"], "separate_regions": result["separate_regions"],
            "segmentation_model": result["segmentation_model"],
            "picture": "shown to the user in the chat (coloured fills per class, with a legend)",
            "note": "Coverage is the share of the frame's pixels. separate_regions counts connected areas, which "
                    "is NOT an object count (touching cars or trees merge into one region): to count objects "
                    "use detect_objects or count_objects."}
    if result.get("vehicles_detected"):
        data["vehicles_detected_in_frame"] = result["vehicles_detected"]
        data["vehicles_note"] = ("Vehicles were also painted from the aerial vehicle detector; "
                                 "vehicles_detected_in_frame is its count of vehicles in this frame.")
    if result["unsupported"]:
        data["unsupported_note"] = ("The aerial segmentation model only knows building, tree, road, car and "
                                    "person; a straight-down drone view normally shows no sky."
                                    if result["aerial"] else
                                    "These words are not classes in the ADE20K segmentation model.")
    return ToolOutput(("(cached) " if cached else "") + summary, data, run_id, None,
                      image_url=url_or_none(result["image_key"]))


register(Tool(
    name="segment_scene", tier="gpu",
    description="Make a PICTURE of one frame with each requested class coloured in (semantic segmentation), "
                "and report the share of the frame each class covers. Use when the user asks to show, "
                "highlight, colour, mark or segment things in an image, especially regions like sky, trees, "
                "buildings, road, grass or water (also works for cars and people). Give simple class words "
                "(['sky','tree','building','car']); leave labels empty to show the main regions. t is the "
                "time of the frame (default the first frame). Not for counting objects.",
    parameters={"type": "object", "properties": {
        "labels": {"type": "array", "items": {"type": "string"}},
        "t": {"type": "number", "default": 0},
    }},
    fn=lambda video_id, labels=None, t=0.0, **_: segment_scene(video_id, labels, t),
    label=lambda a: "Highlighting " + (", ".join(_clean(x) for x in (a.get("labels") or [])) or "the scene"),
))
