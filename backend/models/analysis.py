from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import Column, Integer, String, Float, DateTime, Text

from backend.db import Base


class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(64), nullable=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    # Core patient inputs (subset for quick querying)
    age = Column(Integer, nullable=True)
    sex = Column(String(1), nullable=True)
    systolic_bp = Column(Integer, nullable=True)
    total_cholesterol = Column(Integer, nullable=True)
    bmi = Column(Float, nullable=True)

    # Raw payloads for full fidelity
    patient_input_json = Column(Text, nullable=False)
    result_json = Column(Text, nullable=False)


