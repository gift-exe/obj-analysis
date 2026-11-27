"""ffmpeg / OpenCV helpers for probing, transcoding and frame sampling."""

import base64
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.core.config import get_settings
from app.core.storage import get_storage


@dataclass
class ProbeResult:
    duration_s: float
    fps: float
    width: int
    height: int
    has_audio: bool


def run(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed: {proc.stderr.strip()[-800:]}")
    return proc.stdout


def probe(path: Path) -> ProbeResult:
    out = json.loads(run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams",
                          str(path)]))
    video = next((s for s in out["streams"] if s.get("codec_type") == "video"), None)
    if video is None:
        raise ValueError("file has no video stream")
    num, den = (video.get("avg_frame_rate") or video.get("r_frame_rate") or "0/1").split("/")
    fps = float(num) / float(den) if float(den) else 0.0
    width, height = int(video["width"]), int(video["height"])
    # Respect rotation metadata (phones often record portrait as rotated landscape).
    rotation = 0
    for sd in video.get("side_data_list", []) or []:
        if "rotation" in sd:
            rotation = abs(int(sd["rotation"]))
    if rotation in (90, 270):
        width, height = height, width
    duration = float(out["format"].get("duration") or video.get("duration") or 0)
    has_audio = any(s.get("codec_type") == "audio" for s in out["streams"])
    return ProbeResult(duration, fps, width, height, has_audio)


def make_proxy(src: Path, dest: Path, has_audio: bool) -> None:
    """720p (short side) H.264 proxy with faststart, used for playback and all processing."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    scale = "scale='if(gt(iw,ih),-2,min(720,iw))':'if(gt(iw,ih),min(720,ih),-2)'"
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(src), "-vf", scale, "-c:v", "libx264", "-preset", "veryfast",
           "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
    cmd += ["-c:a", "aac", "-b:a", "128k"] if has_audio else ["-an"]
    run(cmd + [str(dest)])


def encode_frames_to_mp4(frames_dir: Path, fps: float, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    run(["ffmpeg", "-y", "-v", "error", "-framerate", f"{fps:.3f}", "-i", str(frames_dir / "%06d.jpg"),
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
         "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-movflags", "+faststart", str(dest)])


def local_video_path(key: str, video_id) -> Path:
    """Path to a stored file on local disk, downloading it from S3 into the work dir if needed."""
    storage = get_storage()
    path = storage.local_path(key)
    if path:
        return path
    return storage.get_file(key, get_settings().work_dir / str(video_id) / Path(key).name)


class FrameReader:
    """Random-access frame reader over a local video file."""

    def __init__(self, path: Path):
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.frame_count / self.fps if self.fps else 0

    def at(self, t: float) -> np.ndarray | None:
        idx = max(0, min(self.frame_count - 1, int(round(t * self.fps))))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        return frame if ok else None

    def sample(self, t0: float, t1: float, fps: float, max_frames: int | None = None) -> list[tuple[float, np.ndarray]]:
        times = sample_times(t0, t1, fps, max_frames)
        out = []
        for t in times:
            frame = self.at(t)
            if frame is not None:
                out.append((t, frame))
        return out

    def close(self):
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def sample_times(t0: float, t1: float, fps: float, max_frames: int | None = None) -> list[float]:
    span = max(t1 - t0, 1e-3)
    n = max(1, int(span * fps))
    if max_frames:
        n = min(n, max_frames)
    step = span / n
    return [round(t0 + step * (i + 0.5), 3) for i in range(n)]


def resize_max(frame: np.ndarray, max_side: int) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return frame
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def to_jpeg_bytes(frame: np.ndarray, quality: int = 85) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return buf.tobytes()


def to_data_url(frame: np.ndarray, max_side: int | None = None) -> str:
    if max_side:
        frame = resize_max(frame, max_side)
    return "data:image/jpeg;base64," + base64.b64encode(to_jpeg_bytes(frame)).decode()


def fmt_t(t: float) -> str:
    t = max(0.0, t)
    return f"{int(t // 60)}:{t % 60:04.1f}"
