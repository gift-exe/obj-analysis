"""Video memory tables (see docs: 'Video memory')."""

import uuid
from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.config import get_settings
from app.core.db import Base

_s = get_settings()
JsonType = JSONB().with_variant(JSON(), "sqlite")


def _uuid() -> uuid.UUID:
    return uuid.uuid4()


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Video(Base):
    __tablename__ = "videos"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    title: Mapped[str] = mapped_column(String(300))
    filename: Mapped[str] = mapped_column(String(300))
    storage_key: Mapped[str] = mapped_column(String(500))
    proxy_key: Mapped[str | None] = mapped_column(String(500))
    thumbnail_key: Mapped[str | None] = mapped_column(String(500))
    # uploading | queued | processing | ready | failed
    status: Mapped[str] = mapped_column(String(20), default="uploading")
    error: Mapped[str | None] = mapped_column(Text)
    duration_s: Mapped[float | None] = mapped_column(Float)
    fps: Mapped[float | None] = mapped_column(Float)
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    has_audio: Mapped[bool | None] = mapped_column(Boolean)
    tag: Mapped[str | None] = mapped_column(String(100))
    analysis: Mapped[dict | None] = mapped_column(JsonType)  # cached general analysis report
    meta: Mapped[dict | None] = mapped_column(JsonType)  # derived facts, e.g. {"viewpoint": "aerial"}
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    jobs: Mapped[list["Job"]] = relationship(back_populates="video", cascade="all, delete-orphan")


class Job(Base):
    """One row per ingestion step; the dashboard streams these as progress."""

    __tablename__ = "jobs"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    video_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"), index=True)
    step: Mapped[str] = mapped_column(String(30))
    order: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(20), default="pending")
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    detail: Mapped[str | None] = mapped_column(Text)
    error: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)

    video: Mapped[Video] = relationship(back_populates="jobs")


class Shot(Base):
    __tablename__ = "shots"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    video_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"), index=True)
    index: Mapped[int] = mapped_column(Integer)
    t_start: Mapped[float] = mapped_column(Float)
    t_end: Mapped[float] = mapped_column(Float)


class Clip(Base):
    __tablename__ = "clips"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    video_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"), index=True)
    shot_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("shots.id", ondelete="CASCADE"))
    index: Mapped[int] = mapped_column(Integer)
    t_start: Mapped[float] = mapped_column(Float)
    t_end: Mapped[float] = mapped_column(Float)
    caption: Mapped[str] = mapped_column(Text, default="")
    objects: Mapped[list] = mapped_column(JsonType, default=list)
    actions: Mapped[list] = mapped_column(JsonType, default=list)
    on_screen_text: Mapped[str] = mapped_column(Text, default="")
    keyframe_key: Mapped[str | None] = mapped_column(String(500))
    caption_emb = mapped_column(Vector(_s.text_embed_dim), nullable=True)
    image_emb = mapped_column(Vector(_s.image_embed_dim), nullable=True)

    __table_args__ = (
        Index("ix_clips_caption_emb", "caption_emb", postgresql_using="hnsw",
              postgresql_ops={"caption_emb": "vector_cosine_ops"}),
        Index("ix_clips_image_emb", "image_emb", postgresql_using="hnsw",
              postgresql_ops={"image_emb": "vector_cosine_ops"}),
    )


class TranscriptSegment(Base):
    __tablename__ = "transcript_segments"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    video_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"), index=True)
    t_start: Mapped[float] = mapped_column(Float)
    t_end: Mapped[float] = mapped_column(Float)
    text: Mapped[str] = mapped_column(Text)
    text_emb = mapped_column(Vector(_s.text_embed_dim), nullable=True)


class Summary(Base):
    __tablename__ = "summaries"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    video_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"), index=True)
    level: Mapped[str] = mapped_column(String(10))  # scene | video
    t_start: Mapped[float] = mapped_column(Float)
    t_end: Mapped[float] = mapped_column(Float)
    text: Mapped[str] = mapped_column(Text)


class ToolRun(Base):
    """Cache of GPU-tier and plug-in tool calls, keyed by (video, tool, params_hash)."""

    __tablename__ = "tool_runs"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    video_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"), index=True)
    tool: Mapped[str] = mapped_column(String(60))
    tier: Mapped[str] = mapped_column(String(10))
    params: Mapped[dict] = mapped_column(JsonType, default=dict)
    params_hash: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(10), default="running")
    result: Mapped[dict | None] = mapped_column(JsonType)
    overlay_key: Mapped[str | None] = mapped_column(String(500))
    gpu_seconds: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Track(Base):
    __tablename__ = "tracks"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    tool_run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tool_runs.id", ondelete="CASCADE"), index=True)
    track_no: Mapped[int] = mapped_column(Integer)
    label: Mapped[str] = mapped_column(String(100))
    t_start: Mapped[float] = mapped_column(Float)
    t_end: Mapped[float] = mapped_column(Float)
    mean_score: Mapped[float] = mapped_column(Float)


class Detection(Base):
    __tablename__ = "detections"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    track_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tracks.id", ondelete="CASCADE"), index=True)
    frame_idx: Mapped[int] = mapped_column(Integer)
    t: Mapped[float] = mapped_column(Float)
    bbox: Mapped[list] = mapped_column(JsonType)  # [x1, y1, x2, y2] normalised 0..1
    score: Mapped[float] = mapped_column(Float)
    mask_key: Mapped[str | None] = mapped_column(String(500))


class Event(Base):
    __tablename__ = "events"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    video_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"), index=True)
    tool_run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tool_runs.id", ondelete="CASCADE"), index=True)
    type: Mapped[str] = mapped_column(String(40))
    label: Mapped[str] = mapped_column(String(100))
    t_start: Mapped[float] = mapped_column(Float)
    t_end: Mapped[float] = mapped_column(Float)
    data: Mapped[dict] = mapped_column(JsonType, default=dict)


class Conversation(Base):
    __tablename__ = "conversations"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    video_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Message(Base):
    __tablename__ = "messages"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String(10))
    content: Mapped[str] = mapped_column(Text)
    answer: Mapped[dict | None] = mapped_column(JsonType)
    steps: Mapped[list] = mapped_column(JsonType, default=list)
    selection: Mapped[dict | None] = mapped_column(JsonType)  # user messages: the object clicked before asking
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
