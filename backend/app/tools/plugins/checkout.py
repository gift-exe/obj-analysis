"""Plug-in: count units of each item type scanned at a checkout.

Counts *trips*, not moments. Each item is picked from the basket, scanned and bagged, and is handled
continuously in between; the hand goes back to the basket before the next item. So:

1. Label: every sampled frame of the whole video (checkout_fps, default 2 fps) is labelled by the VLM with
   the item the person is handling right then (in hand or on the scanner), or "none". Batches go in order and
   carry the item names seen so far, so the same item keeps the same name.
2. Bridge: short gaps (<= GAP_FILL frames) between frames of the same item are filled: the item was briefly
   out of view, not replaced. For longer gaps (up to BRIDGE_MAX_S) between two trips of the same type, the VLM
   is asked whether that item is still on the scanner or in hand during the gap; if so, it is one trip.
   (Asking "same item or a new one?" directly was unreliable; this perception question is not.)
3. Segment: each run of consecutive handled frames is one trip. A run is split where the item type changes
   and stays changed for SPLIT_MIN frames. Runs shorter than MIN_FRAMES are ignored as flicker.
4. Units: a close-up recount per trip (up to 8 frames at full detail: "how many separate units?"), falling
   back to the most common per-frame unit count. Two cans held together = 2.
5. Totals per item type are summed in code, never by a language model.

The per-frame labels are kept in the result, so the segmentation can be re-tuned without calling the VLM.
"""

import logging
import re
import uuid
from collections import Counter, defaultdict

from app.core.config import get_settings
from app.core.db import session_scope
from app.core.media import FrameReader, fmt_t
from app.llm import chat_json, frames_content
from app.models import Event, Video
from app.tools.registry import GPU_IMPLS, Tool, ToolOutput, call_gpu, register
from app.tools.vision import _proxy_path

log = logging.getLogger(__name__)

VERSION = 4
LABELS_VERSION = 1  # bump when the per-frame labelling (prompt, batch) changes; otherwise labels are reused
BATCH = 8  # frames per VLM request (with 16 the model sometimes labels a whole batch "none")
GAP_FILL = 3  # missed frames (1.5 s at 2 fps) bridged between two frames of the same item
BRIDGE_MAX_S = 4.0  # longer same-type gaps up to this are checked with the VLM
MIN_FRAMES = 2  # shortest run counted as a trip
SPLIT_MIN = 4  # a different label must last this many frames (2 s at 2 fps) to split a run into two trips

LABEL_PROMPT = """These {n} frames are from a camera above a supermarket self-checkout, {t0:.1f}s to {t1:.1f}s.
For EACH frame, name the grocery item the person is handling at that moment: held in their hand(s), or lying on
the scanner/scale. Ignore items still in the basket and items already in a bag.
Use a generic lowercase singular type (e.g. apple, banana bunch, bread, bottle, can, crisps bag), never a brand.
{known}Return only JSON: {{"items": ["...", ...]}} with exactly {n} strings, one per frame in order. Each string
is "none", or the type followed by how many units of it are being handled together, e.g. "apple 1", "bottle 2"."""

_NONE = {"none", "nothing", "no item", "empty", "hand", "hands", "unknown", "n/a", "null", ""}
UNKNOWN = "?"  # frame the VLM gave no label for

GAP_PROMPT = """These frames ({t0:.1f}s to {t1:.1f}s) are from a camera above a supermarket self-checkout.
In these frames, is there a {item} on the scanner/scale or in the person's hands? Ignore anything in the basket
and anything already in a bag.
Return only JSON {{"present": true or false, "reason": "short"}}"""

RECOUNT_PROMPT = """These frames ({t0:.1f}s to {t1:.1f}s) show a shopper at a self-checkout handling {item}.
How many separate units of {item} are they handling together in this period? Count physical units: two held
together = 2, one item turned over or rescanned = 1.
Return only JSON {{"units": integer, "reason": "short"}}"""


def _singular(word: str) -> str:
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith(("ches", "shes", "sses", "xes", "toes")):
        return word[:-2]
    if word.endswith("s") and not word.endswith(("ss", "us", "is")) and len(word) > 3:
        return word[:-1]
    return word


def parse_label(raw) -> tuple[str | None, int]:
    """'bottle 2' -> ('bottle', 2); 'none' -> (None, 0)."""
    text = str(raw or "").strip().lower()
    m = re.match(r"^(.*?)(?:\s*[x×]?\s*(\d+))?\s*$", text)
    name, num = (m.group(1), m.group(2)) if m else (text, None)
    name = re.sub(r"[^a-z ]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    if name in _NONE or (num == "0"):
        return None, 0
    words = name.split()
    words[-1] = _singular(words[-1])
    return " ".join(words), max(1, min(int(num), 12)) if num else 1


def _label_frames(proxy: str, duration: float, fps: float) -> list[dict]:
    states: list[dict] = []
    known: list[str] = []
    span = BATCH / fps
    t0 = 0.0
    with FrameReader(proxy) as r:
        while t0 < duration - 1e-3:
            t1 = min(t0 + span, duration)
            frames = r.sample(t0, t1, fps, max_frames=BATCH)
            if frames:
                prompt = LABEL_PROMPT.format(
                    n=len(frames), t0=t0, t1=t1,
                    known=f"Reuse these names when they fit: {', '.join(known)}.\n" if known else "")
                items = None
                for _attempt in range(2):
                    try:
                        items = chat_json([{"role": "user", "content": frames_content(frames, prompt)}],
                                          max_tokens=500).get("items")
                        if isinstance(items, list) and items:
                            break
                    except Exception as e:  # one bad batch must not sink the whole count
                        log.warning("checkout labels %.1f-%.1fs failed: %s", t0, t1, e)
                items = items if isinstance(items, list) else []
                for i, (t, _f) in enumerate(frames):
                    if i < len(items):
                        item, units = parse_label(items[i])
                    else:
                        item, units = UNKNOWN, 0
                    states.append({"t": round(t, 2), "item": item, "units": units})
                    if item and item != UNKNOWN and item not in known:
                        known.append(item)
            t0 = t1
    return states


def canonical_names(states: list[dict]) -> dict[str, str]:
    """Merge names where one is a word-subset of another ('banana' -> 'banana bunch'), into the more common."""
    freq = Counter(s["item"] for s in states if s["item"] not in (None, UNKNOWN))
    mapping = {}
    for a in freq:
        best = a
        for b in freq:
            wa, wb = set(a.split()), set(b.split())
            if b != a and (wa <= wb or wb <= wa) and (freq[b], len(b)) > (freq[best], len(best)):
                best = b
        mapping[a] = best
    return mapping


def segment_trips(states: list[dict], fps: float, recounts: list[dict] | None = None,
                  bridges: list[dict] | None = None) -> tuple[list[dict], list[dict]]:
    """Per-frame labels -> (trips, ignored blips). Pure function, so it can be tuned offline.
    recounts: close-up unit counts per trip, matched by item and time overlap.
    bridges: gaps the VLM confirmed the item was still present in ({item_type, t_start, t_end, present})."""
    names = canonical_names(states)
    items = [names.get(s["item"], s["item"]) for s in states]
    for b in bridges or []:
        if b.get("present"):
            for k, st in enumerate(states):
                if b["t_start"] <= st["t"] <= b["t_end"] and items[k] in (None, UNKNOWN):
                    items[k] = b["item_type"]
    n = len(items)
    # 2. bridge brief misses between frames of the same item
    i = 0
    while i < n:
        if items[i] in (None, UNKNOWN):
            j = i
            while j < n and items[j] in (None, UNKNOWN):
                j += 1
            misses = sum(1 for k in range(i, j) if items[k] is None)
            if 0 < i and j < n and items[i - 1] == items[j] and misses <= GAP_FILL:
                for k in range(i, j):
                    items[k] = items[i - 1]
            i = j
        else:
            i += 1
    # 3. runs of handled frames
    runs: list[list[int]] = []
    cur: list[int] | None = None
    for k, it in enumerate(items):
        if it in (None, UNKNOWN):
            cur = None
            continue
        if cur is None:
            cur = []
            runs.append(cur)
        cur.append(k)
    trips, blips = [], []
    half = 0.5 / fps
    for run in runs:
        # split where the label changes and the new label lasts; absorb short flickers into a neighbour
        parts: list[list[int]] = []
        for k in run:
            if parts and items[parts[-1][-1]] == items[k]:
                parts[-1].append(k)
            else:
                parts.append([k])
        merged: list[list[int]] = []
        for p in parts:
            if merged and (len(p) < SPLIT_MIN or len(merged[-1]) < SPLIT_MIN):
                merged[-1].extend(p)
            else:
                merged.append(p)
        for p in merged:
            labels = Counter(items[k] for k in p)
            item, votes = labels.most_common(1)[0]
            units_seen = Counter(states[k]["units"] for k in p if items[k] == item and states[k]["units"])
            units = max(units_seen.items(), key=lambda kv: (kv[1], kv[0]))[0] if units_seen else 1
            trip = {"item_type": item, "units": units, "frame_units": units,
                    "t_start": round(max(0.0, states[p[0]]["t"] - half), 2), "t_end": round(states[p[-1]]["t"] + half, 2),
                    "frames": len(p), "label_share": round(votes / len(p), 2)}
            rc = _match_recount(trip, recounts or [])
            if rc is not None:
                trip["units"] = trip["recount_units"] = rc
            if len(p) < MIN_FRAMES:
                blips.append(trip)
                continue
            trip["status"] = "ok" if len(p) >= 3 and trip["label_share"] >= 0.6 else "low_confidence"
            trips.append(trip)
    return trips, blips


def _match_recount(trip: dict, recounts: list[dict]) -> int | None:
    best, best_overlap = None, 0.0
    for rc in recounts:
        overlap = min(trip["t_end"], rc["t_end"]) - max(trip["t_start"], rc["t_start"])
        if rc["item_type"] == trip["item_type"] and overlap > best_overlap and rc.get("units"):
            best, best_overlap = int(rc["units"]), overlap
    return best


def check_gaps(proxy, trips: list[dict]) -> list[dict]:
    """For consecutive same-type trips close together, ask whether the item stayed in view during the gap."""
    out = []
    with FrameReader(proxy) as r:
        for a, b in zip(trips, trips[1:]):
            gap = b["t_start"] - a["t_end"]
            if a["item_type"] != b["item_type"] or gap > BRIDGE_MAX_S:
                continue
            frames = r.sample(a["t_end"], b["t_start"], 2.0, max_frames=8)
            try:
                present = bool(chat_json([{"role": "user", "content": frames_content(frames, GAP_PROMPT.format(
                    t0=a["t_end"], t1=b["t_start"], item=a["item_type"]))}], max_tokens=120).get("present"))
            except Exception as e:
                log.warning("gap check at %.1fs failed: %s", a["t_end"], e)
                continue
            out.append({"item_type": a["item_type"], "t_start": a["t_end"], "t_end": b["t_start"],
                        "present": present})
    return out


def _cached_labels(video_id, fps: float) -> list[dict] | None:
    """Per-frame labels from an earlier run with the same labelling settings (they don't depend on later steps)."""
    from sqlalchemy import select

    from app.models import ToolRun

    with session_scope() as s:
        runs = s.scalars(select(ToolRun).where(ToolRun.video_id == video_id, ToolRun.tool == "checkout_scan_count",
                                               ToolRun.status == "done").order_by(ToolRun.created_at.desc())).all()
        for run in runs:
            res = run.result or {}
            same = (res.get("labels_version") == LABELS_VERSION
                    or (LABELS_VERSION == 1 and (run.params or {}).get("version") == 3))  # v3 used these labels
            if same and float((run.params or {}).get("fps", 0)) == fps and res.get("frames"):
                return res["frames"]
    return None


def recount_units(proxy, trips: list[dict]) -> list[dict]:
    """Close-up count of units per trip (the per-frame labels see small images and undercount pairs)."""
    out = []
    with FrameReader(proxy) as r:
        for tr in trips:
            frames = r.sample(tr["t_start"], tr["t_end"], 2.0, max_frames=8)
            try:
                units = int(chat_json([{"role": "user", "content": frames_content(frames, RECOUNT_PROMPT.format(
                    t0=tr["t_start"], t1=tr["t_end"], item=tr["item_type"]))}], max_tokens=150).get("units"))
            except Exception as e:  # keep the per-frame estimate
                log.warning("recount %s at %.1fs failed: %s", tr["item_type"], tr["t_start"], e)
                continue
            if 1 <= units <= 12:
                out.append({"item_type": tr["item_type"], "t_start": tr["t_start"], "t_end": tr["t_end"],
                            "units": units})
    return out


def _impl(video_id: uuid.UUID, params: dict, run_id: uuid.UUID):
    with session_scope() as s:
        duration = s.get(Video, video_id).duration_s or 0
    fps = float(params["fps"])
    proxy = _proxy_path(video_id)
    states = _cached_labels(video_id, fps) or _label_frames(proxy, duration, fps)
    bridges = check_gaps(proxy, segment_trips(states, fps)[0])
    recounts = recount_units(proxy, segment_trips(states, fps, bridges=bridges)[0])
    trips, blips = segment_trips(states, fps, recounts, bridges)
    totals: dict[str, int] = defaultdict(int)
    with session_scope() as s:
        for tr in trips:
            totals[tr["item_type"]] += tr["units"]
            s.add(Event(video_id=video_id, tool_run_id=run_id, type="scan", label=tr["item_type"],
                        t_start=tr["t_start"], t_end=tr["t_end"], data=tr))
    return {**params, "totals": dict(sorted(totals.items(), key=lambda x: (-x[1], x[0]))),
            "total_units": sum(totals.values()), "distinct_item_types": len(totals),
            "trips": trips, "ignored_blips": blips, "frames": states, "labels_version": LABELS_VERSION,
            "bridges": bridges, "recounts": recounts}, None


GPU_IMPLS["checkout_scan_count"] = _impl

METHOD = ("Every frame was labelled at {fps:g} fps with the item in the person's hands or on the scanner. Each "
          "continuous handling of an item (basket -> scanner -> bag) is one trip (brief gaps where the item is out of "
          "view are joined); each trip's units were recounted "
          "close up (items held together count separately). Totals are summed in code. "
          "Possible errors: an item handled twice in separate trips is counted twice; two identical items "
          "handled back to back without a pause may be counted once.")


def checkout_scan_count(video_id) -> ToolOutput:
    fps = get_settings().checkout_fps
    result, run_id, overlay_url, cached = call_gpu("checkout_scan_count", video_id,
                                                   {"version": VERSION, "fps": fps})
    # trips are rebuilt from the stored per-frame labels (cheap), so segmentation tweaks apply to cached runs
    trips, _blips = segment_trips(result["frames"], fps, result.get("recounts"), result.get("bridges"))
    counts: dict[str, int] = defaultdict(int)
    for t in trips:
        counts[t["item_type"]] += t["units"]
    per_type = dict(sorted(counts.items(), key=lambda x: (-x[1], x[0])))
    total = sum(per_type.values())
    data = {"totals_per_item_type": per_type, "total_units": total, "distinct_item_types": len(per_type),
            "trips": [{"t_start": t["t_start"], "t_end": t["t_end"], "at": fmt_t(t["t_start"]),
                       "item_type": t["item_type"], "units": t["units"], "status": t["status"]} for t in trips],
            "method": METHOD.format(fps=fps)}
    text = ", ".join(f"{k} x{v}" for k, v in per_type.items()) or "no scans found"
    return ToolOutput(("(cached) " if cached else "") + f"{total} units: {text}", data, run_id, overlay_url)


register(Tool(
    name="checkout_scan_count", tier="plugin",
    description="Checkout plug-in: follows every item the person handles at a till (basket -> scanner -> bag), "
                "with its type and the number of units handled together, and returns the units per item type, "
                "the total units and the number of distinct item types (already added up). Use for 'how many "
                "items/units were scanned' and 'how many of each item' questions on checkout footage.",
    parameters={"type": "object", "properties": {}},
    fn=lambda video_id, **_: checkout_scan_count(video_id),
    label=lambda a: "Running the checkout scan counter",
))
