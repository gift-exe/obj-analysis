"""Celery worker: `cpu` queue runs ingestion, `gpu` queue runs vision tools.

Run:  celery -A worker.celery_app worker -Q cpu --concurrency 2
      celery -A worker.celery_app worker -Q gpu --concurrency 1 --pool solo
"""

from celery import Celery

from app.core.config import get_settings

settings = get_settings()
celery = Celery("video_intel", broker=settings.redis_url, backend=settings.redis_url)
celery.conf.update(task_acks_late=True, worker_prefetch_multiplier=1, task_track_started=True,
                   result_expires=3600)


@celery.task(name="ingest_video")
def ingest_video(video_id: str) -> None:
    from app.ingest.pipeline import run_ingest

    run_ingest(video_id)


@celery.task(name="gpu_tool")
def gpu_tool(tool_name: str, video_id: str, params: dict) -> dict:
    from app.tools.registry import execute_gpu_tool, load_all

    load_all()
    return execute_gpu_tool(tool_name, video_id, params)
