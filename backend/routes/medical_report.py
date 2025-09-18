"""
Medical Report Analysis API Routes
Handles file upload, text extraction, medical analysis, and report generation
"""

import os
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import tempfile
import uuid

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Form
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel
import sqlalchemy as sa
from sqlalchemy.orm import Session

# Import database and models
try:
    from backend.auth.database_store import get_db, MedicalReportAnalysis
except ImportError:
    from auth.database_store import get_db, MedicalReportAnalysis

# Import analysis modules
try:
    from backend.models.medical_report_analyzer import MedicalReportAnalyzer, MedicalReport
    from backend.models.report_generator import MedicalReportGenerator
    from utils.lifestyle_recommender import lifestyle_engine
except ImportError:
    from models.medical_report_analyzer import MedicalReportAnalyzer, MedicalReport
    from models.report_generator import MedicalReportGenerator
    from utils.lifestyle_recommender import lifestyle_engine

# Drug advisor services (NLP extraction with HF/Watson and dosage guidance)
try:
    from backend.services.drug_advisor import (
        extract_drug_info as da_extract,
        dosage_recommendation as da_dosage,
        check_interactions as da_interactions,
    )
except ImportError:
    from services.drug_advisor import (
        extract_drug_info as da_extract,
        dosage_recommendation as da_dosage,
        check_interactions as da_interactions,
    )

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter()

# Initialize analyzers (singleton pattern for performance)
analyzer = None
report_generator = None

def get_analyzer():
    """Get or create medical report analyzer instance"""
    global analyzer
    if analyzer is None:
        try:
            analyzer = MedicalReportAnalyzer()
            logger.info("✓ Medical report analyzer initialized")
        except Exception as e:
            logger.error(f"Failed to initialize analyzer: {e}")
            raise HTTPException(status_code=500, detail=f"Analyzer initialization failed: {e}")
    return analyzer

def get_report_generator():
    """Get or create report generator instance"""
    global report_generator
    if report_generator is None:
        try:
            report_generator = MedicalReportGenerator()
            logger.info("✓ Report generator initialized")
        except Exception as e:
            logger.error(f"Failed to initialize report generator: {e}")
            raise HTTPException(status_code=500, detail=f"Report generator initialization failed: {e}")
    return report_generator

# ----------------------
# Helpers
# ----------------------
def _make_schedule(per_day: Any) -> Optional[list[str]]:
    """Return simple evenly spaced schedule labels based on times per day.
    e.g., 1 -> ["08:00"], 2 -> ["08:00","20:00"], 3 -> ["08:00","14:00","20:00"], 4 -> ["06:00","12:00","18:00","22:00"].
    """
    try:
        if not per_day:
            return None
        n = int(round(float(per_day)))
        if n <= 0:
            return None
        presets = {
            1: ["08:00"],
            2: ["08:00", "20:00"],
            3: ["08:00", "14:00", "20:00"],
            4: ["06:00", "12:00", "18:00", "22:00"],
        }
        if n in presets:
            return presets[n]
        # Fallback: evenly spaced across 24h starting at 08:00
        base = 8
        gap = 24 / n
        times = []
        for i in range(n):
            hour = int((base + i * gap) % 24)
            times.append(f"{hour:02d}:00")
        return times
    except Exception:
        return None

# Pydantic models for API
class ReportAnalysisResponse(BaseModel):
    """Response model for report analysis"""
    analysis_id: str
    patient_name: str
    analysis_date: str
    confidence_score: float
    summary: Dict[str, Any]
    conditions: List[Dict[str, Any]]
    medications: List[Dict[str, Any]]
    lab_values: List[Dict[str, Any]]
    future_risks: List[str]
    recommendations: List[str]
    text_preview: str
    dosage_plan: List[Dict[str, Any]]
    interactions: List[Dict[str, Any]]

class ReportListResponse(BaseModel):
    """Response model for report list"""
    reports: List[Dict[str, Any]]
    total_count: int

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test analyzer initialization
        get_analyzer()
        get_report_generator()
        return {"status": "healthy", "message": "Medical report analysis service is running"}
    except Exception as e:
        return {"status": "unhealthy", "message": f"Service error: {e}"}

@router.post("/upload", response_model=ReportAnalysisResponse)
async def upload_and_analyze_report(
    file: UploadFile = File(...),
    patient_name: str = Form("Patient"),
    age: int | None = Form(None),
    weight: float | None = Form(None),
    condition: str | None = Form(None),
    db: Session = Depends(get_db)
):
    """
    Upload and analyze a medical report
    Supports PDF and image files
    """
    try:
        # Validate file type
        allowed_extensions = {'.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.bmp'}
        file_extension = os.path.splitext(file.filename.lower())[1]
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Validate file size (max 10MB)
        max_size = 10 * 1024 * 1024  # 10MB
        file_content = await file.read()
        
        if len(file_content) > max_size:
            raise HTTPException(
                status_code=400,
                detail="File size too large. Maximum size is 10MB."
            )
        
        if len(file_content) == 0:
            raise HTTPException(
                status_code=400,
                detail="Empty file uploaded."
            )
        
        logger.info(f"Processing file: {file.filename} ({len(file_content)} bytes)")
        
        # Get analyzer and perform analysis
        analyzer_instance = get_analyzer()
        report = analyzer_instance.analyze_report(file_content, file.filename)
        
        # Generate report summary
        generator = get_report_generator()
        summary_data = generator.generate_summary_report(report, patient_name)

        # --- Enrich with Drug Advisor (HuggingFace/Watson NLP) ---
        # Extract drugs from the original text using the same service powering Drug Advisor
        try:
            da_result = da_extract(report.original_text or "")  # {'extracted': [...], 'raw': [...], 'watson': {...}}
        except Exception as e:
            logger.warning(f"Drug extraction failed: {e}")
            da_result = {"extracted": [], "raw": [], "watson": {"used": False, "reason": str(e)}}

        # Prepare dosage plan per extracted drug if age is provided
        dosage_plan: list[dict[str, Any]] = []
        drug_names: list[str] = []
        try:
            extracted = da_result.get("extracted", []) or []
            # Collect unique drug names and nearest normalized frequency from extractor
            seen = set()
            for item in extracted:
                nm = (item.get("name") or "").lower()
                if nm and nm not in seen:
                    seen.add(nm)
                    drug_names.append(nm)
                    # Build per-drug record optionally with dosage rec and schedule
                    rec: dict[str, Any] = {"drug": nm}
                    # Attach parsed context
                    for k in ["strength","frequency","duration","strength_mg","frequency_per_day","duration_days","instructions"]:
                        if k in item:
                            rec[k] = item.get(k)
                    # Compute age-specific dosage if age supplied
                    if isinstance(age, int):
                        try:
                            dr = da_dosage(nm, age)
                            # if weight/condition provided, apply same logic as router did
                            if dr and weight and age < 18:
                                # emulate routers.drug_advisor._add_weight_based_dosing
                                import re
                                mg_kg_match = re.search(r"(\d+(?:\.\d+)?)\s*mg/kg", dr.get("recommendation",""))
                                if mg_kg_match:
                                    mg_per_kg = float(mg_kg_match.group(1))
                                    total_dose = mg_per_kg * float(weight)
                                    dr["calculated_dose_mg"] = total_dose
                                    dr["rounded_dose_mg"] = round(total_dose) if total_dose >= 1 else round(total_dose, 2)
                            # condition warnings not strictly necessary here
                            rec["dosage_recommendation"] = dr
                        except Exception as e:
                            rec["dosage_error"] = str(e)
                    # Suggest schedule from frequency
                    per_day = rec.get("frequency_per_day")
                    schedule = _make_schedule(per_day)
                    if schedule:
                        rec["schedule_times"] = schedule
                    dosage_plan.append(rec)
        except Exception as e:
            logger.warning(f"Dosage plan generation failed: {e}")

        # Optionally check interactions among extracted drugs
        interactions_data: dict[str, Any] = {"interactions": []}
        try:
            if len(drug_names) >= 2:
                interactions_data = da_interactions(drug_names)
        except Exception:
            interactions_data = {"interactions": [], "summary": "interaction check failed"}

        # Store analysis in database
        analysis_id = str(uuid.uuid4())
        db_analysis = MedicalReportAnalysis(
            analysis_id=analysis_id,
            patient_name=patient_name,
            filename=file.filename,
            file_size=len(file_content),
            analysis_date=datetime.now(),
            confidence_score=report.confidence_score,
            conditions_count=len(report.conditions),
            medications_count=len(report.medications),
            symptoms_count=len(report.symptoms),
            lab_values_count=len(report.lab_values),
            risks_count=len(report.future_risks),
            recommendations_count=len(report.recommendations),
            analysis_data={
                **summary_data,
                "drug_nlp": da_result,
                "dosage_plan": dosage_plan,
                "drug_interactions": interactions_data,
                "patient_age": age,
                "patient_weight": weight,
                "patient_condition": condition,
            },  # Store enriched analysis as JSON
            original_text_preview=report.original_text[:1000]  # Store preview
        )
        
        db.add(db_analysis)
        db.commit()
        db.refresh(db_analysis)
        
        logger.info(f"✓ Analysis completed and stored with ID: {analysis_id}")
        
        # Return response
        return ReportAnalysisResponse(
            analysis_id=analysis_id,
            patient_name=patient_name,
            analysis_date=report.analysis_timestamp.isoformat(),
            confidence_score=report.confidence_score,
            summary=summary_data['summary'],
            conditions=summary_data['conditions'],
            medications=summary_data['medications'],
            lab_values=summary_data['lab_values'],
            future_risks=summary_data['future_risks'],
            recommendations=summary_data['recommendations'],
            text_preview=summary_data['text_preview'],
            dosage_plan=dosage_plan,
            interactions=interactions_data.get('interactions', [])
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.get("/analysis/{analysis_id}")
async def get_analysis(analysis_id: str, db: Session = Depends(get_db)):
    """Get stored analysis by ID"""
    try:
        analysis = db.query(MedicalReportAnalysis).filter(
            MedicalReportAnalysis.analysis_id == analysis_id
        ).first()
        
        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        return {
            "analysis_id": analysis.analysis_id,
            "patient_name": analysis.patient_name,
            "filename": analysis.filename,
            "analysis_date": analysis.analysis_date.isoformat(),
            "confidence_score": analysis.confidence_score,
            "summary": {
                "conditions_found": analysis.conditions_count,
                "medications_identified": analysis.medications_count,
                "symptoms_noted": analysis.symptoms_count,
                "lab_values_extracted": analysis.lab_values_count,
                "future_risks": analysis.risks_count,
                "recommendations": analysis.recommendations_count
            },
            "analysis_data": analysis.analysis_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve analysis: {str(e)}")

@router.get("/download/{analysis_id}")
async def download_report_pdf(analysis_id: str, db: Session = Depends(get_db)):
    """Download PDF report for analysis"""
    try:
        # Get analysis from database
        analysis = db.query(MedicalReportAnalysis).filter(
            MedicalReportAnalysis.analysis_id == analysis_id
        ).first()
        
        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        # Reconstruct MedicalReport object from stored data
        analysis_data = analysis.analysis_data
        
        # Create a simplified MedicalReport object for PDF generation
        from backend.models.medical_report_analyzer import MedicalEntity
        
        # Convert stored data back to MedicalEntity objects
        conditions = []
        for cond_data in analysis_data.get('conditions', []):
            conditions.append(MedicalEntity(
                text=cond_data['text'],
                label=cond_data['type'],
                start=0,
                end=len(cond_data['text']),
                confidence=cond_data['confidence'],
                context=cond_data['context']
            ))
        
        medications = []
        for med_data in analysis_data.get('medications', []):
            medications.append(MedicalEntity(
                text=med_data['text'],
                label='MEDICATION',
                start=0,
                end=len(med_data['text']),
                confidence=med_data['confidence'],
                context=med_data['context']
            ))
        
        lab_values = []
        for lab_data in analysis_data.get('lab_values', []):
            lab_values.append(MedicalEntity(
                text=lab_data['value'],
                label=f"LAB_{lab_data['test'].upper().replace(' ', '_')}",
                start=0,
                end=len(lab_data['value']),
                confidence=0.9,
                context=lab_data['context']
            ))
        
        # Create MedicalReport object
        report = MedicalReport(
            original_text=analysis.original_text_preview,
            conditions=conditions,
            medications=medications,
            symptoms=[],  # Simplified for PDF generation
            lab_values=lab_values,
            recommendations=analysis_data.get('recommendations', []),
            risk_factors=[],
            future_risks=analysis_data.get('future_risks', []),
            confidence_score=analysis.confidence_score,
            analysis_timestamp=analysis.analysis_date
        )
        
        # Generate PDF
        generator = get_report_generator()
        pdf_bytes = generator.generate_report(report, analysis.patient_name)
        
        # Return PDF response
        filename = f"medical_report_{analysis.patient_name}_{analysis.analysis_date.strftime('%Y%m%d')}.pdf"
        
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"PDF generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")

@router.get("/list", response_model=ReportListResponse)
async def list_analyses(
    limit: int = 10,
    offset: int = 0,
    patient_name: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """List stored analyses with pagination"""
    try:
        query = db.query(MedicalReportAnalysis)
        
        # Filter by patient name if provided
        if patient_name:
            query = query.filter(MedicalReportAnalysis.patient_name.ilike(f"%{patient_name}%"))
        
        # Get total count
        total_count = query.count()
        
        # Apply pagination and ordering
        analyses = query.order_by(MedicalReportAnalysis.analysis_date.desc()).offset(offset).limit(limit).all()
        
        # Format response
        reports = []
        for analysis in analyses:
            reports.append({
                "analysis_id": analysis.analysis_id,
                "patient_name": analysis.patient_name,
                "filename": analysis.filename,
                "analysis_date": analysis.analysis_date.isoformat(),
                "confidence_score": analysis.confidence_score,
                "conditions_count": analysis.conditions_count,
                "medications_count": analysis.medications_count,
                "file_size": analysis.file_size
            })
        
        return ReportListResponse(
            reports=reports,
            total_count=total_count
        )
        
    except Exception as e:
        logger.error(f"Failed to list analyses: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list analyses: {str(e)}")

@router.delete("/analysis/{analysis_id}")
async def delete_analysis(analysis_id: str, db: Session = Depends(get_db)):
    """Delete stored analysis"""
    try:
        analysis = db.query(MedicalReportAnalysis).filter(
            MedicalReportAnalysis.analysis_id == analysis_id
        ).first()
        
        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        db.delete(analysis)
        db.commit()
        
        return {"message": "Analysis deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete analysis: {str(e)}")

@router.post("/reanalyze/{analysis_id}")
async def reanalyze_report(analysis_id: str, db: Session = Depends(get_db)):
    """Re-analyze a stored report with updated models"""
    try:
        analysis = db.query(MedicalReportAnalysis).filter(
            MedicalReportAnalysis.analysis_id == analysis_id
        ).first()
        
        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        # Note: This would require storing the original file content
        # For now, return the existing analysis
        return {
            "message": "Re-analysis feature requires original file storage",
            "current_analysis": analysis.analysis_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Re-analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Re-analysis failed: {str(e)}")

@router.get("/lifestyle-recommendations/{analysis_id}")
async def get_detailed_lifestyle_recommendations(
    analysis_id: str,
    db: Session = Depends(get_db)
):
    """Get detailed lifestyle recommendations for a specific analysis"""
    try:
        # Get the analysis from database - try both string and UUID formats
        analysis = db.query(MedicalReportAnalysis).filter(
            MedicalReportAnalysis.id == analysis_id
        ).first()
        
        # If not found, try different approaches to find the analysis
        if not analysis:
            # Try to find by any field that might contain this ID
            try:
                # Check if there's an analysis_id column or similar
                analysis = db.query(MedicalReportAnalysis).filter(
                    MedicalReportAnalysis.filename.contains(analysis_id[-8:])  # Last 8 chars as fallback
                ).first()
            except:
                pass
        
        if not analysis:
            # Instead of failing, generate basic lifestyle recommendations
            logger.warning(f"Analysis not found for ID: {analysis_id}. Generating basic recommendations.")
            
            # Generate basic lifestyle plan without specific analysis data
            lifestyle_plan = lifestyle_engine.generate_comprehensive_plan(
                conditions=[],
                lab_values=[],
                patient_age=None,
                patient_sex=None,
                current_medications=[]
            )
            
            # Return basic recommendations
            return {
                "analysis_id": analysis_id,
                "diet_recommendations": [
                    {
                        "category": diet.category,
                        "foods_to_include": diet.foods_to_include,
                        "foods_to_avoid": diet.foods_to_avoid,
                        "meal_suggestions": diet.meal_suggestions,
                        "nutritional_goals": diet.nutritional_goals,
                        "portion_guidelines": diet.portion_guidelines
                    }
                    for diet in lifestyle_plan.diet_recommendations
                ],
                "exercise_recommendations": [
                    {
                        "category": exercise.category,
                        "recommended_activities": exercise.recommended_activities,
                        "frequency": exercise.frequency,
                        "duration": exercise.duration,
                        "intensity": exercise.intensity,
                        "precautions": exercise.precautions,
                        "progression_tips": exercise.progression_tips
                    }
                    for exercise in lifestyle_plan.exercise_recommendations
                ],
                "lifestyle_recommendations": [
                    {
                        "category": lifestyle.category,
                        "recommendations": lifestyle.recommendations,
                        "benefits": lifestyle.benefits,
                        "implementation_tips": lifestyle.implementation_tips
                    }
                    for lifestyle in lifestyle_plan.lifestyle_recommendations
                ],
                "priority_actions": lifestyle_plan.priority_actions,
                "monitoring_suggestions": lifestyle_plan.monitoring_suggestions,
                "professional_consultations": lifestyle_plan.professional_consultations
            }
        
        # Parse the stored analysis data
        import json
        analysis_data = json.loads(analysis.analysis_data)
        
        # Extract conditions and lab values
        conditions = [cond.get('text', '') for cond in analysis_data.get('conditions', [])]
        lab_values = [lab.get('text', '') for lab in analysis_data.get('lab_values', [])]
        
        # Generate comprehensive lifestyle plan
        lifestyle_plan = lifestyle_engine.generate_comprehensive_plan(
            conditions=conditions,
            lab_values=lab_values,
            patient_age=None,  # Could be extracted from report if available
            patient_sex=None,  # Could be extracted from report if available
            current_medications=[med.get('text', '') for med in analysis_data.get('medications', [])]
        )
        
        # Format response
        response = {
            "analysis_id": analysis_id,
            "patient_name": analysis.patient_name,
            "analysis_date": analysis.analysis_date.isoformat(),
            "diet_recommendations": [
                {
                    "category": diet.category,
                    "foods_to_include": diet.foods_to_include,
                    "foods_to_avoid": diet.foods_to_avoid,
                    "meal_suggestions": diet.meal_suggestions,
                    "nutritional_goals": diet.nutritional_goals,
                    "portion_guidelines": diet.portion_guidelines
                }
                for diet in lifestyle_plan.diet_recommendations
            ],
            "exercise_recommendations": [
                {
                    "category": exercise.category,
                    "recommended_activities": exercise.recommended_activities,
                    "frequency": exercise.frequency,
                    "duration": exercise.duration,
                    "intensity": exercise.intensity,
                    "precautions": exercise.precautions,
                    "progression_tips": exercise.progression_tips
                }
                for exercise in lifestyle_plan.exercise_recommendations
            ],
            "lifestyle_recommendations": [
                {
                    "category": lifestyle.category,
                    "recommendations": lifestyle.recommendations,
                    "benefits": lifestyle.benefits,
                    "implementation_tips": lifestyle.implementation_tips
                }
                for lifestyle in lifestyle_plan.lifestyle_recommendations
            ],
            "priority_actions": lifestyle_plan.priority_actions,
            "monitoring_suggestions": lifestyle_plan.monitoring_suggestions,
            "professional_consultations": lifestyle_plan.professional_consultations
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Failed to get lifestyle recommendations: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to get lifestyle recommendations: {str(e)}")

@router.get("/statistics")
async def get_analysis_statistics(db: Session = Depends(get_db)):
    """Get analysis statistics"""
    try:
        total_analyses = db.query(MedicalReportAnalysis).count()
        
        if total_analyses == 0:
            return {
                "total_analyses": 0,
                "average_confidence": 0,
                "most_common_conditions": [],
                "analysis_by_month": []
            }
        
        # Calculate average confidence
        avg_confidence = db.query(sa.func.avg(MedicalReportAnalysis.confidence_score)).scalar()
        
        # Get recent analyses for trends
        recent_analyses = db.query(MedicalReportAnalysis).order_by(
            MedicalReportAnalysis.analysis_date.desc()
        ).limit(100).all()
        
        return {
            "total_analyses": total_analyses,
            "average_confidence": round(float(avg_confidence or 0), 2),
            "recent_analyses_count": len(recent_analyses),
            "high_confidence_analyses": len([a for a in recent_analyses if a.confidence_score >= 0.8])
        }
        
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")
