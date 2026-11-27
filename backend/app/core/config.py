from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BACKEND_DIR / ".env", extra="ignore")

    # Infrastructure
    database_url: str = "postgresql+psycopg://localhost:5432/videointel"
    redis_url: str = "redis://localhost:6379/0"
    # "inline" runs background work in threads inside the API process (no Redis/Celery needed);
    # "celery" sends it to the cpu/gpu Celery queues.
    task_mode: str = "inline"

    # Storage: "local" keeps files on disk and serves them from /media; "s3" uses S3/MinIO.
    storage_backend: str = "local"
    local_storage_dir: Path = BACKEND_DIR / "data" / "storage"
    s3_endpoint_url: str | None = None
    s3_bucket: str = "video-intel"
    s3_region: str = "us-east-1"
    s3_access_key: str | None = None
    s3_secret_key: str | None = None
    work_dir: Path = BACKEND_DIR / "data" / "work"

    # Models. Any OpenAI-compatible endpoint works: vLLM on a GPU box, or Ollama locally.
    vlm_base_url: str = "http://localhost:11434/v1"
    vlm_api_key: str = "not-needed"
    vlm_model: str = "qwen3-vl:8b"
    # The agent can use a different (e.g. larger, hosted) model; defaults to the VLM.
    agent_base_url: str | None = None
    agent_api_key: str | None = None
    agent_model: str | None = None
    llm_timeout_s: float = 180.0
    caption_concurrency: int = 3

    text_embed_model: str = "BAAI/bge-small-en-v1.5"
    text_embed_dim: int = 384
    image_embed_model: str = "google/siglip-base-patch16-224"
    image_embed_dim: int = 768
    detector_model: str = str(BACKEND_DIR / "data" / "weights" / "yolov8s-worldv2.pt")
    segmenter_model: str = str(BACKEND_DIR / "data" / "weights" / "sam2.1_b.pt")
    # Used instead of the open-vocabulary detector when the camera looks straight down from high up
    # (drone footage): trained on aerial imagery (DOTA), where ground-level detectors miss small vehicles.
    aerial_detector_model: str = str(BACKEND_DIR / "data" / "weights" / "yolo11s-obb.pt")
    enable_masks: bool = True  # SAM masks on tracked objects (visual overlay); turn off on slow machines
    # Scene segmentation (colour every pixel by class: sky, tree, building, road, car...). ADE20K covers
    # ground-level / elevated views; the UAVid model is trained on drone footage and is used for aerial video.
    scene_model: str = "nvidia/segformer-b4-finetuned-ade-512-512"
    aerial_scene_model: str = "chribark/segformer-b3-finetuned-UAVid"
    whisper_model: str = "small"
    device: str = "auto"  # auto | cpu | cuda | mps

    # Ingestion
    clip_max_seconds: float = 8.0
    caption_fps: float = 2.0
    caption_max_frames: int = 8
    frame_max_side: int = 768

    # Checkout plug-in: frames per second labelled across the whole video (one VLM image each)
    checkout_fps: float = 2.0

    # Agent
    agent_max_tool_calls: int = 8

    @property
    def agent_url(self) -> str:
        return self.agent_base_url or self.vlm_base_url

    @property
    def agent_key(self) -> str:
        return self.agent_api_key or self.vlm_api_key

    @property
    def agent_model_name(self) -> str:
        return self.agent_model or self.vlm_model


@lru_cache
def get_settings() -> Settings:
    return Settings()
