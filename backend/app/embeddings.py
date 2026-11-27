"""Text (caption/transcript/query) and image (keyframe) embedders, loaded lazily."""

import threading
from functools import lru_cache

import numpy as np

from app.core.config import get_settings

_lock = threading.Lock()


def pick_device() -> str:
    pref = get_settings().device
    if pref != "auto":
        return pref
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@lru_cache
def _text_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(get_settings().text_embed_model, device=pick_device())


@lru_cache
def _siglip():
    from transformers import AutoModel, AutoProcessor

    name = get_settings().image_embed_model
    device = pick_device()
    model = AutoModel.from_pretrained(name).to(device).eval()
    return model, AutoProcessor.from_pretrained(name), device


def embed_texts(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, get_settings().text_embed_dim), dtype=np.float32)
    with _lock:
        vecs = _text_model().encode(texts, normalize_embeddings=True, batch_size=32)
    return np.asarray(vecs, dtype=np.float32)


def embed_query_text(query: str) -> np.ndarray:
    # bge models recommend this prefix for short retrieval queries
    prefix = "Represent this sentence for searching relevant passages: " if "bge" in get_settings().text_embed_model else ""
    return embed_texts([prefix + query])[0]


def _norm(x):
    # transformers 5 returns a ModelOutput from get_*_features; older versions return the tensor
    if not hasattr(x, "norm"):
        x = x.pooler_output
    return x / x.norm(dim=-1, keepdim=True)


def embed_images(images_rgb: list[np.ndarray]) -> np.ndarray:
    import torch

    if not images_rgb:
        return np.zeros((0, get_settings().image_embed_dim), dtype=np.float32)
    model, proc, device = _siglip()
    out = []
    with _lock, torch.no_grad():
        for i in range(0, len(images_rgb), 16):
            inputs = proc(images=images_rgb[i : i + 16], return_tensors="pt").to(device)
            out.append(_norm(model.get_image_features(**inputs)).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def embed_image_query(text: str) -> np.ndarray:
    """SigLIP text tower, for text -> keyframe search."""
    import torch

    model, proc, device = _siglip()
    with _lock, torch.no_grad():
        inputs = proc(text=[f"a photo of {text}"], padding="max_length", return_tensors="pt").to(device)
        return _norm(model.get_text_features(**inputs)).cpu().numpy()[0].astype(np.float32)
