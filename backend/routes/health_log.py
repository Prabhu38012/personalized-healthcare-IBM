"""
Health Log Data Persistence API Routes
Handles CRUD operations for user health data logging
"""

import os
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, date
from datetime import date as date_type
import uuid

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field
import sqlalchemy as sa
from sqlalchemy.orm import Session

# Import database and authentication
try:
    from backend.auth.database_store import get_db
    from backend.auth.routes import get_current_user
except ImportError:
    from auth.database_store import get_db
    from auth.routes import get_current_user

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter()

# Pydantic models for health log data
class HealthLogEntry(BaseModel):
    """Health log entry data model"""
    date: date_type = Field(..., description="Date of the health measurement")
    weight: Optional[float] = Field(None, description="Weight in kg")
    height: Optional[float] = Field(None, description="Height in cm")
    systolic_bp: Optional[int] = Field(None, description="Systolic blood pressure")
    diastolic_bp: Optional[int] = Field(None, description="Diastolic blood pressure")
    heart_rate: Optional[int] = Field(None, description="Heart rate in BPM")
    blood_sugar: Optional[float] = Field(None, description="Blood sugar level")
    cholesterol: Optional[float] = Field(None, description="Cholesterol level")
    exercise_minutes: Optional[int] = Field(None, description="Exercise minutes")
    sleep_hours: Optional[float] = Field(None, description="Sleep hours")
    stress_level: Optional[int] = Field(None, ge=1, le=10, description="Stress level (1-10)")
    notes: Optional[str] = Field(None, description="Additional notes")

class HealthLogResponse(BaseModel):
    """Health log response model"""
    id: str
    user_id: str
    date: date
    weight: Optional[float]
    height: Optional[float]
    systolic_bp: Optional[int]
    diastolic_bp: Optional[int]
    heart_rate: Optional[int]
    blood_sugar: Optional[float]
    cholesterol: Optional[float]
    exercise_minutes: Optional[int]
    sleep_hours: Optional[float]
    stress_level: Optional[int]
    notes: Optional[str]
    created_at: datetime
    updated_at: datetime

# Database table creation (if not exists)
def create_health_log_table(db: Session):
    """Create health log table if it doesn't exist"""
    try:
        # Check if table exists
        result = db.execute(sa.text("SELECT name FROM sqlite_master WHERE type='table' AND name='health_logs'"))
        if not result.fetchone():
            # Create table
            db.execute(sa.text("""
                CREATE TABLE health_logs (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    date DATE NOT NULL,
                    weight REAL,
                    height REAL,
                    systolic_bp INTEGER,
                    diastolic_bp INTEGER,
                    heart_rate INTEGER,
                    blood_sugar REAL,
                    cholesterol REAL,
                    exercise_minutes INTEGER,
                    sleep_hours REAL,
                    stress_level INTEGER,
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """))
            db.commit()
            logger.info("Health logs table created successfully")
    except Exception as e:
        logger.error(f"Error creating health logs table: {e}")
        db.rollback()

@router.post("/", response_model=HealthLogResponse)
async def create_health_log_entry(
    entry: HealthLogEntry,
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new health log entry"""
    try:
        # Ensure table exists
        create_health_log_table(db)
        
        entry_id = str(uuid.uuid4())
        now = datetime.utcnow()
        
        # Insert new entry
        db.execute(sa.text("""
            INSERT INTO health_logs (
                id, user_id, date, weight, height, systolic_bp, diastolic_bp,
                heart_rate, blood_sugar, cholesterol, exercise_minutes,
                sleep_hours, stress_level, notes, created_at, updated_at
            ) VALUES (
                :id, :user_id, :date, :weight, :height, :systolic_bp, :diastolic_bp,
                :heart_rate, :blood_sugar, :cholesterol, :exercise_minutes,
                :sleep_hours, :stress_level, :notes, :created_at, :updated_at
            )
        """), {
            "id": entry_id,
            "user_id": current_user.id,
            "date": entry.date,
            "weight": entry.weight,
            "height": entry.height,
            "systolic_bp": entry.systolic_bp,
            "diastolic_bp": entry.diastolic_bp,
            "heart_rate": entry.heart_rate,
            "blood_sugar": entry.blood_sugar,
            "cholesterol": entry.cholesterol,
            "exercise_minutes": entry.exercise_minutes,
            "sleep_hours": entry.sleep_hours,
            "stress_level": entry.stress_level,
            "notes": entry.notes,
            "created_at": now,
            "updated_at": now
        })
        db.commit()
        
        # Return the created entry
        result = db.execute(sa.text("""
            SELECT * FROM health_logs WHERE id = :id
        """), {"id": entry_id}).fetchone()
        
        return HealthLogResponse(
            id=result.id,
            user_id=result.user_id,
            date=result.date,
            weight=result.weight,
            height=result.height,
            systolic_bp=result.systolic_bp,
            diastolic_bp=result.diastolic_bp,
            heart_rate=result.heart_rate,
            blood_sugar=result.blood_sugar,
            cholesterol=result.cholesterol,
            exercise_minutes=result.exercise_minutes,
            sleep_hours=result.sleep_hours,
            stress_level=result.stress_level,
            notes=result.notes,
            created_at=result.created_at,
            updated_at=result.updated_at
        )
        
    except Exception as e:
        logger.error(f"Error creating health log entry: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to create health log entry")

@router.get("/", response_model=List[HealthLogResponse])
async def get_health_log_entries(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    start_date: Optional[date_type] = Query(None),
    end_date: Optional[date_type] = Query(None),
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get health log entries for the current user"""
    try:
        # Ensure table exists
        create_health_log_table(db)
        
        # Build query
        query = "SELECT * FROM health_logs WHERE user_id = :user_id"
        params = {"user_id": current_user.id}
        
        if start_date:
            query += " AND date >= :start_date"
            params["start_date"] = start_date
            
        if end_date:
            query += " AND date <= :end_date"
            params["end_date"] = end_date
            
        query += " ORDER BY date DESC LIMIT :limit OFFSET :offset"
        params["limit"] = limit
        params["offset"] = offset
        
        results = db.execute(sa.text(query), params).fetchall()
        
        return [
            HealthLogResponse(
                id=row.id,
                user_id=row.user_id,
                date=row.date,
                weight=row.weight,
                height=row.height,
                systolic_bp=row.systolic_bp,
                diastolic_bp=row.diastolic_bp,
                heart_rate=row.heart_rate,
                blood_sugar=row.blood_sugar,
                cholesterol=row.cholesterol,
                exercise_minutes=row.exercise_minutes,
                sleep_hours=row.sleep_hours,
                stress_level=row.stress_level,
                notes=row.notes,
                created_at=row.created_at,
                updated_at=row.updated_at
            )
            for row in results
        ]
        
    except Exception as e:
        logger.error(f"Error retrieving health log entries: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve health log entries")

@router.put("/{entry_id}", response_model=HealthLogResponse)
async def update_health_log_entry(
    entry_id: str,
    entry: HealthLogEntry,
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update a health log entry"""
    try:
        # Check if entry exists and belongs to user
        existing = db.execute(sa.text("""
            SELECT * FROM health_logs WHERE id = :id AND user_id = :user_id
        """), {"id": entry_id, "user_id": current_user.id}).fetchone()
        
        if not existing:
            raise HTTPException(status_code=404, detail="Health log entry not found")
        
        now = datetime.utcnow()
        
        # Update entry
        db.execute(sa.text("""
            UPDATE health_logs SET
                date = :date, weight = :weight, height = :height,
                systolic_bp = :systolic_bp, diastolic_bp = :diastolic_bp,
                heart_rate = :heart_rate, blood_sugar = :blood_sugar,
                cholesterol = :cholesterol, exercise_minutes = :exercise_minutes,
                sleep_hours = :sleep_hours, stress_level = :stress_level,
                notes = :notes, updated_at = :updated_at
            WHERE id = :id AND user_id = :user_id
        """), {
            "id": entry_id,
            "user_id": current_user.id,
            "date": entry.date,
            "weight": entry.weight,
            "height": entry.height,
            "systolic_bp": entry.systolic_bp,
            "diastolic_bp": entry.diastolic_bp,
            "heart_rate": entry.heart_rate,
            "blood_sugar": entry.blood_sugar,
            "cholesterol": entry.cholesterol,
            "exercise_minutes": entry.exercise_minutes,
            "sleep_hours": entry.sleep_hours,
            "stress_level": entry.stress_level,
            "notes": entry.notes,
            "updated_at": now
        })
        db.commit()
        
        # Return updated entry
        result = db.execute(sa.text("""
            SELECT * FROM health_logs WHERE id = :id
        """), {"id": entry_id}).fetchone()
        
        return HealthLogResponse(
            id=result.id,
            user_id=result.user_id,
            date=result.date,
            weight=result.weight,
            height=result.height,
            systolic_bp=result.systolic_bp,
            diastolic_bp=result.diastolic_bp,
            heart_rate=result.heart_rate,
            blood_sugar=result.blood_sugar,
            cholesterol=result.cholesterol,
            exercise_minutes=result.exercise_minutes,
            sleep_hours=result.sleep_hours,
            stress_level=result.stress_level,
            notes=result.notes,
            created_at=result.created_at,
            updated_at=result.updated_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating health log entry: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to update health log entry")

@router.delete("/{entry_id}")
async def delete_health_log_entry(
    entry_id: str,
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a health log entry"""
    try:
        # Check if entry exists and belongs to user
        existing = db.execute(sa.text("""
            SELECT * FROM health_logs WHERE id = :id AND user_id = :user_id
        """), {"id": entry_id, "user_id": current_user.id}).fetchone()
        
        if not existing:
            raise HTTPException(status_code=404, detail="Health log entry not found")
        
        # Delete entry
        db.execute(sa.text("""
            DELETE FROM health_logs WHERE id = :id AND user_id = :user_id
        """), {"id": entry_id, "user_id": current_user.id})
        db.commit()
        
        return {"message": "Health log entry deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting health log entry: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to delete health log entry")

@router.get("/statistics")
async def get_health_statistics(
    days: int = Query(30, ge=1, le=365),
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get health statistics for the current user"""
    try:
        # Ensure table exists
        create_health_log_table(db)
        
        # Get statistics for the last N days
        results = db.execute(sa.text("""
            SELECT 
                COUNT(*) as total_entries,
                AVG(weight) as avg_weight,
                AVG(systolic_bp) as avg_systolic_bp,
                AVG(diastolic_bp) as avg_diastolic_bp,
                AVG(heart_rate) as avg_heart_rate,
                AVG(blood_sugar) as avg_blood_sugar,
                AVG(cholesterol) as avg_cholesterol,
                AVG(exercise_minutes) as avg_exercise_minutes,
                AVG(sleep_hours) as avg_sleep_hours,
                AVG(stress_level) as avg_stress_level,
                MIN(date) as earliest_date,
                MAX(date) as latest_date
            FROM health_logs 
            WHERE user_id = :user_id 
            AND date >= date('now', '-' || :days || ' days')
        """), {"user_id": current_user.id, "days": days}).fetchone()
        
        return {
            "period_days": days,
            "total_entries": results.total_entries or 0,
            "averages": {
                "weight": round(results.avg_weight, 2) if results.avg_weight else None,
                "systolic_bp": round(results.avg_systolic_bp, 1) if results.avg_systolic_bp else None,
                "diastolic_bp": round(results.avg_diastolic_bp, 1) if results.avg_diastolic_bp else None,
                "heart_rate": round(results.avg_heart_rate, 1) if results.avg_heart_rate else None,
                "blood_sugar": round(results.avg_blood_sugar, 2) if results.avg_blood_sugar else None,
                "cholesterol": round(results.avg_cholesterol, 2) if results.avg_cholesterol else None,
                "exercise_minutes": round(results.avg_exercise_minutes, 1) if results.avg_exercise_minutes else None,
                "sleep_hours": round(results.avg_sleep_hours, 2) if results.avg_sleep_hours else None,
                "stress_level": round(results.avg_stress_level, 1) if results.avg_stress_level else None
            },
            "date_range": {
                "earliest": results.earliest_date,
                "latest": results.latest_date
            }
        }
        
    except Exception as e:
        logger.error(f"Error retrieving health statistics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve health statistics")

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "health_log",
        "message": "Health log service is running"
    }
