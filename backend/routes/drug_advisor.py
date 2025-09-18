"""
Drug Advisor API Routes
Endpoints:
- POST /extract_drug_info
- POST /check_interactions
- POST /dosage_recommendation
- POST /alternative_suggestions
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException
import logging

try:
    from services.drug_advisor import (
        extract_drug_info as svc_extract,
        check_interactions as svc_interactions,
        dosage_recommendation as svc_dosage,
        alternative_suggestions as svc_alts,
    )
except ImportError:
    from backend.services.drug_advisor import (
        extract_drug_info as svc_extract,
        check_interactions as svc_interactions,
        dosage_recommendation as svc_dosage,
        alternative_suggestions as svc_alts,
    )

logger = logging.getLogger(__name__)
router = APIRouter()


class ExtractRequest(BaseModel):
    text: str = Field(..., description="Raw prescription or doctor's note text")

class ExtractResponse(BaseModel):
    extracted: List[Dict[str, Any]]
    raw: List[Dict[str, Any]]
    watson: Dict[str, Any]

class InteractionsRequest(BaseModel):
    drugs: List[str] = Field(..., description="List of drug names to check pairwise")

class InteractionsResponse(BaseModel):
    interactions: List[Dict[str, Any]]
    summary: str
    total_drugs: Optional[int] = None
    interaction_count: Optional[int] = None
    severity_breakdown: Optional[Dict[str, int]] = None
    recommendations: Optional[List[str]] = None

class DosageRequest(BaseModel):
    drug: str
    age: int = Field(..., ge=0, le=120)
    weight: Optional[float] = Field(None, ge=0.1, le=500, description="Weight in kg for pediatric dosing")
    condition: Optional[str] = Field(None, description="Medical condition for context-specific dosing")

class DosageResponse(BaseModel):
    drug: str
    age: int
    age_band: Optional[str] = None
    recommendation: str
    source: Optional[str] = None
    weight_based: Optional[bool] = None
    monitoring_required: Optional[List[str]] = None
    warnings: Optional[List[str]] = None

class AlternativeRequest(BaseModel):
    drug: str

class AlternativeResponse(BaseModel):
    drug: str
    category: Optional[str] = None
    alternatives: List[str]
    reason: Optional[str] = None
    source: Optional[str] = None


@router.post("/extract_drug_info", response_model=ExtractResponse)
async def extract_drug_info(req: ExtractRequest):
    try:
        result = svc_extract(req.text)
        return ExtractResponse(**result)
    except Exception as e:
        logger.exception("extract_drug_info failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/check_interactions", response_model=InteractionsResponse)
async def check_interactions(req: InteractionsRequest):
    try:
        result = svc_interactions(req.drugs)
        return InteractionsResponse(**result)
    except Exception as e:
        logger.exception("check_interactions failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/dosage_recommendation", response_model=DosageResponse)
async def dosage_recommendation(req: DosageRequest):
    try:
        result = svc_dosage(req.drug, req.age)
        
        # Add weight-based calculations for pediatric dosing
        if req.weight and req.age < 18:
            result = _add_weight_based_dosing(result, req.weight, req.age)
        
        # Add condition-specific warnings
        if req.condition:
            result = _add_condition_warnings(result, req.condition)
        
        return DosageResponse(**result)
    except Exception as e:
        logger.exception("dosage_recommendation failed")
        raise HTTPException(status_code=500, detail=str(e))

def _add_weight_based_dosing(result: Dict[str, Any], weight: float, age: int) -> Dict[str, Any]:
    """Add weight-based dosing calculations for pediatric patients"""
    if not result.get("weight_based"):
        return result
    
    drug = result.get("drug", "").lower()
    recommendation = result.get("recommendation", "")
    
    # Extract mg/kg dosing from recommendation
    import re
    mg_kg_match = re.search(r'(\d+(?:\.\d+)?)\s*mg/kg', recommendation)
    if mg_kg_match:
        mg_per_kg = float(mg_kg_match.group(1))
        total_dose = mg_per_kg * weight
        
        # Add weight-based calculation to recommendation
        weight_calc = f"\n\n📊 WEIGHT-BASED CALCULATION:\n"
        weight_calc += f"• Patient weight: {weight} kg\n"
        weight_calc += f"• Dose per kg: {mg_per_kg} mg/kg\n"
        weight_calc += f"• Calculated dose: {total_dose:.1f} mg\n"
        weight_calc += f"• Age: {age} years\n"
        
        # Add rounding guidance
        if total_dose < 1:
            rounded_dose = round(total_dose, 2)
            weight_calc += f"• Rounded dose: {rounded_dose} mg (use liquid formulation)\n"
        else:
            rounded_dose = round(total_dose)
            weight_calc += f"• Rounded dose: {rounded_dose} mg\n"
        
        result["recommendation"] = recommendation + weight_calc
        result["calculated_dose_mg"] = total_dose
        result["rounded_dose_mg"] = rounded_dose
    
    return result

def _add_condition_warnings(result: Dict[str, Any], condition: str) -> Dict[str, Any]:
    """Add condition-specific warnings and considerations"""
    condition_lower = condition.lower()
    warnings = result.get("warnings", [])
    
    # Common condition-specific warnings
    if "diabetes" in condition_lower:
        warnings.append("🩸 Monitor blood glucose levels closely")
        warnings.append("⚠️ May affect insulin requirements")
    elif "hypertension" in condition_lower or "high blood pressure" in condition_lower:
        warnings.append("❤️ Monitor blood pressure regularly")
        warnings.append("⚠️ May interact with antihypertensive medications")
    elif "kidney" in condition_lower or "renal" in condition_lower:
        warnings.append("🫘 Monitor renal function and adjust dose if needed")
        warnings.append("⚠️ May require dose reduction")
    elif "liver" in condition_lower or "hepatic" in condition_lower:
        warnings.append("🫀 Monitor liver function tests")
        warnings.append("⚠️ May require dose reduction or alternative medication")
    elif "heart" in condition_lower or "cardiac" in condition_lower:
        warnings.append("❤️ Monitor heart rate and rhythm")
        warnings.append("⚠️ May affect cardiovascular function")
    elif "pregnancy" in condition_lower or "pregnant" in condition_lower:
        warnings.append("🤰 Consult obstetrician before use")
        warnings.append("⚠️ May affect fetal development")
    elif "breastfeeding" in condition_lower:
        warnings.append("🤱 Consult pediatrician before use")
        warnings.append("⚠️ May pass into breast milk")
    
    if warnings:
        result["warnings"] = warnings
    
    return result


@router.post("/alternative_suggestions", response_model=AlternativeResponse)
async def alternative_suggestions(req: AlternativeRequest):
    try:
        result = svc_alts(req.drug)
        return AlternativeResponse(**result)
    except Exception as e:
        logger.exception("alternative_suggestions failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def drug_advisor_health():
    return {"status": "healthy", "message": "Drug advisor endpoints available"}
