"""Draw boxes, masks and track IDs onto frames for the annotated overlay video."""

import colorsys

import cv2
import numpy as np


def color_for(key) -> tuple[int, int, int]:
    h = (hash(str(key)) % 360) / 360.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.75, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def draw(frame: np.ndarray, dets: list[dict], zone_px: np.ndarray | None = None) -> np.ndarray:
    """dets: {bbox (normalised xyxy), label, track (int|None), score, mask (HxW bool|None)}"""
    out = frame.copy()
    h, w = out.shape[:2]
    overlay = out.copy()
    for d in dets:
        if d.get("mask") is not None:
            color = color_for(d.get("track", d["label"]))
            overlay[d["mask"]] = color
    out = cv2.addWeighted(overlay, 0.4, out, 0.6, 0)
    for d in dets:
        x1, y1, x2, y2 = (np.array(d["bbox"]) * [w, h, w, h]).astype(int)
        color = color_for(d.get("track", d["label"]))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        text = f"{d['label']} #{d['track']}" if d.get("track") is not None else f"{d['label']} {d['score']:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), color, -1)
        cv2.putText(out, text, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    if zone_px is not None and len(zone_px) >= 2:
        pts = zone_px.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], len(zone_px) > 2, (0, 255, 255), 2, cv2.LINE_AA)
    return out
