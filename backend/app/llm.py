"""OpenAI-compatible clients for the VLM (captions, watch_clip) and the agent model."""

import json
import logging
import re
from functools import lru_cache

import numpy as np
from openai import OpenAI

from app.core.config import get_settings
from app.core.media import fmt_t, to_data_url

log = logging.getLogger(__name__)


@lru_cache
def vlm_client() -> OpenAI:
    s = get_settings()
    return OpenAI(base_url=s.vlm_base_url, api_key=s.vlm_api_key, timeout=s.llm_timeout_s)


@lru_cache
def agent_client() -> OpenAI:
    s = get_settings()
    return OpenAI(base_url=s.agent_url, api_key=s.agent_key, timeout=s.llm_timeout_s)


def frames_content(frames: list[tuple[float, np.ndarray]], prompt: str, max_side: int | None = None) -> list[dict]:
    """User message content: each frame preceded by its timestamp, then the prompt."""
    max_side = max_side or get_settings().frame_max_side
    parts: list[dict] = []
    for t, frame in frames:
        parts.append({"type": "text", "text": f"Frame at t={t:.1f}s ({fmt_t(t)}):"})
        parts.append({"type": "image_url", "image_url": {"url": to_data_url(frame, max_side)}})
    parts.append({"type": "text", "text": prompt})
    return parts


_THINK = re.compile(r"<think>.*?</think>", re.S)


def strip_thinking(text: str) -> str:
    return _THINK.sub("", text or "").strip()


def parse_json(text: str) -> dict:
    """Parse a JSON object from model output, tolerating code fences and surrounding prose."""
    text = strip_thinking(text)
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        val = json.loads(text)
        if isinstance(val, dict):
            return val
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    raise ValueError(f"model did not return JSON: {text[:300]}")


def chat_json(messages: list[dict], *, agent: bool = False, temperature: float = 0.2,
              max_tokens: int = 1500, retries: int = 1) -> dict:
    s = get_settings()
    client = agent_client() if agent else vlm_client()
    model = s.agent_model_name if agent else s.vlm_model
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        kwargs = dict(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
        try:
            resp = client.chat.completions.create(response_format={"type": "json_object"}, **kwargs)
        except Exception as e:  # some servers reject response_format; retry without it
            log.debug("json mode rejected (%s); retrying without", e)
            resp = client.chat.completions.create(**kwargs)
        try:
            return parse_json(resp.choices[0].message.content or "")
        except ValueError as e:
            last_err = e
            messages = messages + [
                {"role": "assistant", "content": resp.choices[0].message.content or ""},
                {"role": "user", "content": "That was not valid JSON. Reply with only the JSON object."},
            ]
    raise last_err or ValueError("no JSON")


def chat_text(messages: list[dict], *, agent: bool = False, temperature: float = 0.3, max_tokens: int = 800) -> str:
    s = get_settings()
    client = agent_client() if agent else vlm_client()
    model = s.agent_model_name if agent else s.vlm_model
    resp = client.chat.completions.create(model=model, messages=messages, temperature=temperature,
                                          max_tokens=max_tokens)
    return strip_thinking(resp.choices[0].message.content or "")
