"""Run the evaluation question set against a running API and score it.

    python eval/run_eval.py                       # uses http://localhost:8010, uploads missing sample videos
    python eval/run_eval.py --video shopping.mp4  # one video only

Prints per-question results and writes eval/results-<timestamp>.json.
Requires: pip install httpx pyyaml
"""

import argparse
import json
import statistics
import time
from datetime import datetime
from pathlib import Path

import httpx
import yaml

ROOT = Path(__file__).resolve().parents[1]


def sse_events(resp):
    event, data = None, []
    for line in resp.iter_lines():
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].strip())
        elif line == "" and event:
            yield event, json.loads("\n".join(data)) if data else {}
            event, data = None, []


def ensure_video(client: httpx.Client, filename: str) -> dict:
    for v in client.get("/api/videos").json():
        if v["filename"] == filename and v["status"] == "ready":
            return v
    path = ROOT / filename
    created = client.post("/api/videos", json={"filename": filename, "content_type": "video/mp4",
                                                "size": path.stat().st_size}).json()
    url = created["upload_url"]
    with open(path, "rb") as f:
        (client.put(url, content=f.read(), headers=created["upload_headers"]) if url.startswith("/")
         else httpx.put(url, content=f.read(), headers=created["upload_headers"])).raise_for_status()
    vid = created["video"]["id"]
    client.post(f"/api/videos/{vid}/ingest").raise_for_status()
    print(f"  ingesting {filename} ...", flush=True)
    while True:
        v = client.get(f"/api/videos/{vid}").json()
        if v["status"] in ("ready", "failed"):
            if v["status"] == "failed":
                raise RuntimeError(f"ingest failed: {v['error']}")
            return v
        time.sleep(3)


def iou(a, b) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def ask(client: httpx.Client, video_id: str, question: str) -> tuple[dict, float]:
    conv = client.post(f"/api/videos/{video_id}/conversations").json()
    t0 = time.time()
    with client.stream("POST", f"/api/conversations/{conv['id']}/messages", json={"content": question},
                       timeout=None) as resp:
        for event, data in sse_events(resp):
            if event == "tool_call":
                print(f"    -> {data['step']['label']}", flush=True)
            if event == "final":
                return data["message"], time.time() - t0
            if event == "error":
                raise RuntimeError(data["message"])
    raise RuntimeError("stream ended without an answer")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8010")
    ap.add_argument("--video")
    args = ap.parse_args()
    questions = yaml.safe_load((ROOT / "eval" / "questions.yaml").read_text())
    if args.video:
        questions = [q for q in questions if q["video"] == args.video]
    client = httpx.Client(base_url=args.api, timeout=120)
    videos, rows = {}, []
    for q in questions:
        if q["video"] not in videos:
            videos[q["video"]] = ensure_video(client, q["video"])
        print(f"\n[{q['video']}] {q['question']}")
        msg, latency = ask(client, videos[q["video"]]["id"], q["question"])
        ans = msg["answer"]
        row = {**q, "answer": ans["answer"], "verdict": ans["verdict"], "confidence": ans["confidence"],
               "latency_s": round(latency, 1), "tools": ans["tools_used"],
               "gpu_tools": [s["tool"] for s in msg["steps"] if s["tier"] in ("gpu", "plugin")]}
        if q["type"] == "yesno" and q.get("expected") is not None:
            row["correct"] = ans["verdict"] == str(q["expected"])
        if q.get("window") and ans["evidence"]:
            row["evidence_iou"] = round(max(iou(q["window"], (e["t_start"], e["t_end"])) for e in ans["evidence"]), 2)
        print(f"  {ans['verdict']} ({ans['confidence']}) in {latency:.0f}s: {ans['answer'][:200]}")
        rows.append(row)

    yn = [r for r in rows if "correct" in r]
    ious = [r["evidence_iou"] for r in rows if "evidence_iou" in r]
    lat = [r["latency_s"] for r in rows]
    summary = {
        "questions": len(rows),
        "verdict_accuracy": round(sum(r["correct"] for r in yn) / len(yn), 2) if yn else None,
        "mean_evidence_iou": round(statistics.mean(ious), 2) if ious else None,
        "latency_median_s": round(statistics.median(lat), 1) if lat else None,
        "answered_without_gpu": round(sum(1 for r in rows if not r["gpu_tools"]) / len(rows), 2) if rows else None,
    }
    print("\nSUMMARY", json.dumps(summary, indent=2))
    out = ROOT / "eval" / f"results-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
