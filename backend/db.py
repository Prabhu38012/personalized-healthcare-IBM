"""
SQLite database setup using SQLAlchemy.
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase


DB_URL = os.getenv("DATABASE_URL", f"sqlite:///{os.path.join(os.getcwd(), 'healthcare.db')}")


class Base(DeclarativeBase):
    pass


engine = create_engine(
    DB_URL,
    connect_args={"check_same_thread": False} if DB_URL.startswith("sqlite") else {},
    future=True,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def init_db() -> None:
    """Create database tables."""
    from backend.models import Analysis  # noqa: F401 - ensure model is imported
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


