"""Shared helpers for tools that return a still picture to the chat (segment_scene, movement_map, junction_flow)."""

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.core.config import get_settings
from app.core.storage import get_storage, video_prefix

LEGEND_BG = (24, 24, 27)


def font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def ui_scale(width: int) -> float:
    """Text scale so labels stay legible when the picture is shown at chat width."""
    return max(1.0, width / 640)


def with_legend(rgb: np.ndarray, items: list[tuple[tuple, str]], caption: str) -> Image.Image:
    """Picture with a legend strip underneath: colour chips with text, wrapped to the width, then a caption."""
    h, w = rgb.shape[:2]
    scale = ui_scale(w)
    f, small = font(int(18 * scale)), font(int(14 * scale))
    pad, sw, gap, row_h = int(12 * scale), int(16 * scale), int(22 * scale), int(30 * scale)
    chips, rows, x = [], 0, pad
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    for colour, text in items:
        tw = probe.textlength(text, font=f)
        if x + sw + 8 + tw > w - pad and x > pad:
            rows, x = rows + 1, pad
        chips.append((x, rows, colour, text))
        x += sw + 8 + tw + gap
    n_rows = rows + 1 if items else 0
    legend_h = pad * 2 + n_rows * row_h + int(20 * scale)
    canvas = Image.new("RGB", (w, h + legend_h), LEGEND_BG)
    canvas.paste(Image.fromarray(rgb), (0, 0))
    d = ImageDraw.Draw(canvas)
    for x, r, colour, text in chips:
        y = h + pad + r * row_h
        d.rounded_rectangle([x, y + 2 * scale, x + sw, y + 2 * scale + sw], radius=int(3 * scale), fill=colour)
        d.text((x + sw + 8, y), text, fill=(240, 240, 240), font=f)
    d.text((pad, h + pad + n_rows * row_h), caption, fill=(160, 160, 168), font=small)
    return canvas


def label_box(img: Image.Image, xy: tuple[float, float], text: str, colour: tuple, scale: float,
              anchor: str = "mm") -> None:
    """A filled text badge drawn onto the picture (e.g. a road arm's counts)."""
    d = ImageDraw.Draw(img)
    f = font(int(17 * scale))
    l, t, r, b = d.textbbox(xy, text, font=f, anchor=anchor)
    p = int(5 * scale)
    d.rounded_rectangle([l - p, t - p, r + p, b + p], radius=int(5 * scale), fill=colour)
    d.text(xy, text, font=f, anchor=anchor, fill=(255, 255, 255))


def save_picture(video_id, run_id, img: Image.Image, name: str) -> str:
    """Save a picture for a tool run and return its storage key."""
    work = get_settings().work_dir / str(video_id) / "runs" / str(run_id)
    work.mkdir(parents=True, exist_ok=True)
    path = work / name
    img.save(path, quality=90)
    key = f"{video_prefix(video_id)}/runs/{run_id}/{name}"
    get_storage().put_file(key, path, "image/jpeg")
    return key
