"""Background work dispatch: inline threads for local dev, Celery queues in deployment."""

import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor

from app.core.config import get_settings

log = logging.getLogger(__name__)
# Inline mode: one lane for ingestion so models aren't loaded concurrently by many jobs.
_ingest_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ingest")
_gpu_lock = threading.Lock()


def enqueue_ingest(video_id: str) -> None:
    if get_settings().task_mode == "celery":
        from worker.celery_app import ingest_video

        ingest_video.apply_async(args=[video_id], queue="cpu")
        return
    from app.ingest.pipeline import run_ingest

    fut: Future = _ingest_pool.submit(run_ingest, video_id)
    fut.add_done_callback(lambda f: f.exception() and log.error("ingest failed: %s", f.exception()))


def run_gpu_tool(tool_name: str, video_id: str, params: dict, timeout_s: float = 1800) -> dict:
    """Run a GPU-tier tool and wait for its result (the agent awaits it)."""
    if get_settings().task_mode == "celery":
        from worker.celery_app import gpu_tool

        return gpu_tool.apply_async(args=[tool_name, video_id, params], queue="gpu").get(timeout=timeout_s)
    from app.tools.registry import execute_gpu_tool

    with _gpu_lock:  # one GPU job at a time in inline mode
        return execute_gpu_tool(tool_name, video_id, params)
