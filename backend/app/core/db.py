from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import get_settings

engine = create_engine(get_settings().database_url, pool_pre_ping=True, pool_size=10, max_overflow=10)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


@contextmanager
def session_scope() -> Iterator[Session]:
    """Session for background work: commits on success, rolls back on error."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Iterator[Session]:
    """FastAPI dependency."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def init_db() -> None:
    """Create the pgvector extension and all tables. (v0 uses create_all; Alembic comes later.)"""
    from app import models  # noqa: F401  (registers tables)

    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(engine)
    with engine.begin() as conn:  # columns added after v0 (create_all does not alter existing tables)
        if conn.dialect.name == "postgresql":
            conn.execute(text("ALTER TABLE videos ADD COLUMN IF NOT EXISTS meta JSONB"))
            conn.execute(text("ALTER TABLE messages ADD COLUMN IF NOT EXISTS selection JSONB"))
