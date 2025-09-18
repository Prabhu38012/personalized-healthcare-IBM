"""
SQLAlchemy database models for user authentication and medical report analysis
"""
from datetime import datetime
from typing import Optional
from sqlalchemy import Column, String, Boolean, DateTime, Integer, Text, Float, JSON
from sqlalchemy.orm import declarative_base
try:
    from backend.db import Base
except ImportError:
    # Fallback for when running from backend directory
    from db import Base

class User(Base):
    """SQLAlchemy User model for persistent storage"""
    __tablename__ = "users"
    
    id = Column(String, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    full_name = Column(String, nullable=False)
    hashed_password = Column(Text, nullable=False)
    role = Column(String, nullable=False, default="patient")
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_login = Column(DateTime, nullable=True)
    login_attempts = Column(Integer, default=0, nullable=False)
    locked_until = Column(DateTime, nullable=True)
    
    def __repr__(self):
        return f"<User(id='{self.id}', email='{self.email}', role='{self.role}')>"

class MedicalReportAnalysis(Base):
    """SQLAlchemy model for storing medical report analysis results"""
    __tablename__ = "medical_report_analyses"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    analysis_id = Column(String, unique=True, index=True, nullable=False)
    patient_name = Column(String, nullable=False)
    filename = Column(String, nullable=False)
    file_size = Column(Integer, nullable=False)
    analysis_date = Column(DateTime, default=datetime.utcnow, nullable=False)
    confidence_score = Column(Float, nullable=False)
    
    # Counts for quick statistics
    conditions_count = Column(Integer, default=0, nullable=False)
    medications_count = Column(Integer, default=0, nullable=False)
    symptoms_count = Column(Integer, default=0, nullable=False)
    lab_values_count = Column(Integer, default=0, nullable=False)
    risks_count = Column(Integer, default=0, nullable=False)
    recommendations_count = Column(Integer, default=0, nullable=False)
    
    # Store full analysis results as JSON
    analysis_data = Column(JSON, nullable=False)
    
    # Store preview of original text for reference
    original_text_preview = Column(Text, nullable=True)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    def __repr__(self):
        return f"<MedicalReportAnalysis(id='{self.analysis_id}', patient='{self.patient_name}', confidence={self.confidence_score:.2f})>"
