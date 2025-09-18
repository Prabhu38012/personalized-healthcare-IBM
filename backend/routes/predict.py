import os
import logging
import asyncio
from typing import Dict, List, Optional, Any, Union

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field, field_validator, model_validator
import joblib
import pandas as pd
import numpy as np

# Import LLM analyzer
from ..utils.llm_analyzer import llm_analyzer

# Import authentication dependencies with absolute imports first
try:
    # Try absolute import first (more reliable)
    from backend.auth.routes import get_current_user, get_doctor_or_admin
    auth_available = True
except ImportError:
    try:
        # Fallback to relative import
        from auth.routes import get_current_user, get_doctor_or_admin
        auth_available = True
    except ImportError:
        # Fallback for when auth is not available
        auth_available = False
        async def get_current_user() -> Any:
            return {"id": "demo_user", "email": "demo@demo.com", "role": "patient"}
        async def get_doctor_or_admin() -> Any:
            return {"id": "demo_admin", "email": "admin@demo.com", "role": "admin"}

# Initialize logger first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# EHRDataProcessor import with proper fallback handling
def get_ehr_processor():
    """Get EHRDataProcessor with fallback"""
    try:
        # Try relative import first (when running from backend directory)
        from utils.ehr_processor import EHRDataProcessor
        return EHRDataProcessor
    except ImportError:
        try:
            # Fallback to absolute import (when running from project root)
            from backend.utils.ehr_processor import EHRDataProcessor
            return EHRDataProcessor
        except ImportError:
            # Final fallback - create a dummy processor
            logger.warning("EHRDataProcessor not found, creating dummy processor")
            class DummyEHRDataProcessor:
                def process_single_record(self, fhir_bundle):
                    return pd.DataFrame()
            return DummyEHRDataProcessor

router = APIRouter()

# Global variables for model and preprocessor
model_data = None

# Try multiple possible model paths to handle different execution contexts
POSSIBLE_MODEL_PATHS = [
    os.path.join(os.path.dirname(__file__), "..", "..", "models", "ehr_risk_model.pkl"),  # From backend/routes/
    os.path.join(os.getcwd(), "models", "ehr_risk_model.pkl"),  # From project root
    r"d:\personalized-healthcare\models\ehr_risk_model.pkl",  # Absolute path
    "models/ehr_risk_model.pkl",  # Relative from current directory
]

# Find the actual model path
MODEL_PATH = None
for path in POSSIBLE_MODEL_PATHS:
    if os.path.exists(path):
        MODEL_PATH = path
        break
        
if MODEL_PATH is None:
    MODEL_PATH = POSSIBLE_MODEL_PATHS[0]  # Use first path as fallback
    logger.warning(f"Model file not found in any expected location. Will use fallback: {MODEL_PATH}")

@router.get("/reload-model")
async def reload_model(current_user = Depends(get_doctor_or_admin) if auth_available else None):
    """Force reload the model (requires doctor or admin access if auth is enabled)"""
    global model_data
    model_data = None
    try:
        load_model()
        return {"message": "Model reloaded successfully", "status": "success"}
    except Exception as e:
        return {"message": f"Failed to reload model: {str(e)}", "status": "error"}

class FHIRPatientData(BaseModel):
    fhir_bundle: Dict

class LabPatientData(BaseModel):
    """Lab-enhanced patient data model with comprehensive lab parameters"""
    # LLM analysis control
    force_llm: bool = Field(default=False, description="Force LLM analysis even if disabled by default")
    
    # Basic demographics and vitals
    age: int = Field(..., ge=1, le=120, description="Patient age in years")
    sex: str = Field(..., description="Patient sex (M/F)")
    weight: float = Field(..., ge=30, le=300, description="Body weight in kg")
    height: float = Field(..., ge=100, le=250, description="Body height in cm")
    systolic_bp: Optional[int] = Field(None, ge=80, le=250, description="Systolic blood pressure in mmHg")
    diastolic_bp: Optional[int] = Field(None, ge=40, le=150, description="Diastolic blood pressure in mmHg")
    
    # Basic chemistry
    total_cholesterol: Optional[int] = Field(None, ge=100, le=600, description="Total cholesterol mg/dL")
    ldl_cholesterol: Optional[int] = Field(None, ge=50, le=300, description="LDL cholesterol mg/dL")
    hdl_cholesterol: Optional[int] = Field(None, ge=20, le=100, description="HDL cholesterol mg/dL")
    triglycerides: Optional[int] = Field(None, ge=50, le=500, description="Triglycerides mg/dL")
    hba1c: Optional[float] = Field(None, ge=4.0, le=15.0, description="HbA1c percentage")
    
    # Medical history
    diabetes: Optional[int] = Field(0, ge=0, le=1, description="Diabetes status")
    smoking: Optional[int] = Field(0, ge=0, le=1, description="Smoking status")
    family_history: Optional[int] = Field(0, ge=0, le=1, description="Family history of heart disease")
    fasting_blood_sugar: Optional[int] = Field(0, ge=0, le=1, description="Fasting blood sugar > 120 mg/dl")
    
    # Complete Blood Count (CBC)
    hemoglobin: Optional[float] = Field(None, ge=5.0, le=20.0, description="Hemoglobin g/dL")
    total_leukocyte_count: Optional[float] = Field(None, ge=1.0, le=50.0, description="Total WBC count 10³/μL")
    red_blood_cell_count: Optional[float] = Field(None, ge=2.0, le=8.0, description="RBC count 10⁶/μL")
    hematocrit: Optional[float] = Field(None, ge=20.0, le=60.0, description="Hematocrit %")
    mean_corpuscular_volume: Optional[float] = Field(None, ge=60.0, le=120.0, description="MCV fL")
    mean_corpuscular_hb: Optional[float] = Field(None, ge=20.0, le=40.0, description="MCH pg")
    mean_corpuscular_hb_conc: Optional[float] = Field(None, ge=25.0, le=40.0, description="MCHC g/dL")
    red_cell_distribution_width: Optional[float] = Field(None, ge=10.0, le=20.0, description="RDW %")
    
    # Differential Count
    neutrophils_percent: Optional[float] = Field(None, ge=0.0, le=100.0, description="Neutrophils %")
    lymphocytes_percent: Optional[float] = Field(None, ge=0.0, le=100.0, description="Lymphocytes %")
    monocytes_percent: Optional[float] = Field(None, ge=0.0, le=100.0, description="Monocytes %")
    eosinophils_percent: Optional[float] = Field(None, ge=0.0, le=100.0, description="Eosinophils %")
    basophils_percent: Optional[float] = Field(None, ge=0.0, le=100.0, description="Basophils %")
    absolute_neutrophil_count: Optional[float] = Field(None, ge=0.0, le=20.0, description="ANC 10³/μL")
    absolute_lymphocyte_count: Optional[float] = Field(None, ge=0.0, le=10.0, description="ALC 10³/μL")
    absolute_monocyte_count: Optional[float] = Field(None, ge=0.0, le=5.0, description="AMC 10³/μL")
    absolute_eosinophil_count: Optional[float] = Field(None, ge=0.0, le=2.0, description="AEC 10³/μL")
    absolute_basophil_count: Optional[float] = Field(None, ge=0.0, le=1.0, description="ABC 10³/μL")
    
    # Platelet Parameters
    platelet_count: Optional[float] = Field(None, ge=50.0, le=1000.0, description="Platelet count 10³/μL")
    mean_platelet_volume: Optional[float] = Field(None, ge=5.0, le=15.0, description="MPV fL")
    platelet_distribution_width: Optional[float] = Field(None, ge=10.0, le=30.0, description="PDW %")
    
    # Additional Tests
    erythrocyte_sedimentation_rate: Optional[float] = Field(None, ge=0.0, le=100.0, description="ESR mm/1st hour")
    
    @field_validator('sex')
    @classmethod
    def validate_sex(cls, v):
        if v not in ['M', 'F']:
            raise ValueError('Sex must be either M or F')
        return v

class SimplePatientData(BaseModel):
    """Patient data model optimized for EHR dataset features
    
    Based on your EHR dataset's top predictive features:
    - birth_date (22.3%) - Most important feature
    - SystolicBloodPressure (16.9%) - Second most important
    - DiastolicBloodPressure (8.5%) - Fourth most important
    - BodyWeight (7.0%) - Fifth most important
    - BodyHeight (5.7%) - Sixth most important
    
    Additional parameters:
    - force_llm: If True, forces LLM analysis even if it's disabled by default
    """
    # LLM analysis control
    force_llm: bool = Field(default=False, description="Force LLM analysis even if disabled by default")
    
    # TOP PRIORITY - Most predictive EHR features (with backward compatibility)
    age: int = Field(..., ge=1, le=120, description="Patient age in years (used to calculate birth_date - 22.3% importance)")
    systolic_bp: Optional[int] = Field(None, ge=80, le=250, description="Systolic blood pressure in mmHg (16.9% importance)")
    weight: float = Field(..., ge=30, le=300, description="Body weight in kg (7.0% importance)")
    height: float = Field(..., ge=100, le=250, description="Body height in cm (5.7% importance)")
    sex: str = Field(..., description="Patient sex (M/F)")
    
    @field_validator('sex')
    @classmethod
    def validate_sex(cls, v):
        if v not in ['M', 'F']:
            raise ValueError('Sex must be either M or F')
        return v
    
    # SECONDARY PRIORITY - Important clinical markers
    diastolic_bp: Optional[int] = Field(None, ge=40, le=150, description="Diastolic blood pressure mmHg (8.5% importance) - calculated if not provided")
    bmi: Optional[float] = Field(None, ge=10, le=50, description="BMI - calculated if not provided (5.0% importance)")
    total_cholesterol: Optional[int] = Field(None, ge=100, le=600, description="Total cholesterol mg/dL (2.8% importance)")
    ldl_cholesterol: Optional[int] = Field(None, ge=50, le=300, description="LDL cholesterol mg/dL (2.9% importance)")
    hdl_cholesterol: Optional[int] = Field(None, ge=20, le=100, description="HDL cholesterol mg/dL")
    triglycerides: Optional[int] = Field(None, ge=50, le=500, description="Triglycerides mg/dL (3.1% importance)")
    
    # MEDICAL CONDITIONS - High impact on risk
    diabetes: Optional[int] = Field(0, ge=0, le=1, description="Diabetes status (1=diabetic, 0=non-diabetic)")
    smoking: Optional[int] = Field(0, ge=0, le=1, description="Smoking status (1=smoker, 0=non-smoker)")
    
    # OPTIONAL - Additional clinical data
    hba1c: Optional[float] = Field(None, ge=4.0, le=15.0, description="HbA1c percentage (diabetes marker)")
    family_history: Optional[int] = Field(0, ge=0, le=1, description="Family history of heart disease (1=yes, 0=no)")
    fasting_blood_sugar: Optional[int] = Field(0, ge=0, le=1, description="Fasting blood sugar > 120 mg/dl (1=true, 0=false)")
    
    # BACKWARD COMPATIBILITY - Legacy fields (automatically map to new fields)
    resting_bp: Optional[int] = Field(None, description="Legacy: Use systolic_bp instead - will auto-map")
    cholesterol: Optional[int] = Field(None, description="Legacy: Use total_cholesterol instead - will auto-map")
    max_heart_rate: Optional[int] = Field(None, ge=60, le=220, description="Maximum heart rate (low importance for EHR model)")
    chest_pain_type: Optional[str] = Field(None, description="Type of chest pain (not in EHR top features)")
    resting_ecg: Optional[str] = Field(None, description="Resting ECG (not in EHR top features)")
    exercise_angina: Optional[int] = Field(None, description="Exercise angina (not in EHR top features)")
    oldpeak: Optional[float] = Field(None, description="ST depression (not in EHR top features)")
    slope: Optional[str] = Field(None, description="ST slope (not in EHR top features)")
    ca: Optional[int] = Field(None, description="Fluoroscopy vessels (not in EHR top features)")
    thal: Optional[str] = Field(None, description="Thalassemia (not in EHR top features)")
    
    @model_validator(mode='after')
    def validate_legacy_fields(self):
        """Auto-map legacy fields to new fields for backward compatibility"""
        # Map resting_bp to systolic_bp if systolic_bp is not provided
        if self.systolic_bp is None and self.resting_bp is not None:
            self.systolic_bp = self.resting_bp
            
        # Map cholesterol to total_cholesterol if total_cholesterol is not provided
        if self.total_cholesterol is None and self.cholesterol is not None:
            self.total_cholesterol = self.cholesterol
            
        # Set default values if not provided
        if self.systolic_bp is None:
            self.systolic_bp = 120  # Default normal systolic BP
        if self.total_cholesterol is None:
            self.total_cholesterol = 200  # Default normal cholesterol
            
        return self

class LLMAnalysis(BaseModel):
    analysis_available: bool
    summary: Optional[str] = None
    key_risk_factors: Optional[List[str]] = None
    health_implications: Optional[str] = None
    recommendations: Optional[List[str]] = None
    urgency_level: Optional[str] = None
    reason: Optional[str] = None

class PredictionResponse(BaseModel):
    risk_probability: float
    risk_level: str
    recommendations: List[str]
    risk_factors: List[str]
    confidence: float
    llm_analysis: Optional[LLMAnalysis] = None
    data: Optional[Dict[str, Any]] = None  # Include original patient data

def load_model():
    """Load the trained model and preprocessor"""
    global model_data
    
    if model_data is None:
        try:
            if MODEL_PATH and os.path.exists(MODEL_PATH):
                model_data = joblib.load(MODEL_PATH)
                logger.info("EHR model loaded successfully")
                
                # Safe model type logging with error handling
                try:
                    if isinstance(model_data, dict) and 'model' in model_data and model_data['model'] is not None:
                        logger.info(f"Model type: {type(model_data['model']).__name__}")
                    else:
                        logger.warning("Model data structure is unexpected or missing 'model' key")
                        logger.info(f"Available keys: {list(model_data.keys()) if isinstance(model_data, dict) else 'Not a dictionary'}")
                except Exception as e:
                    logger.warning(f"Could not determine model type: {e}")
            else:
                logger.warning(f"Model file not found at {MODEL_PATH}")
                # Create a simple fallback model
                from sklearn.ensemble import RandomForestClassifier
                import numpy as np
                
                # Create a dummy model with the expected features
                # Use a more dynamic approach to avoid same results
                X_dummy = np.random.rand(100, 20)  # 20 features
                y_dummy = np.random.randint(0, 2, 100)
                fallback_model = RandomForestClassifier(n_estimators=10, random_state=None)  # Remove fixed seed
                fallback_model.fit(X_dummy, y_dummy)
                
                model_data = {
                    'model': fallback_model,
                    'model_type': 'FallbackRandomForest'
                }
                logger.warning("Using fallback model - predictions may not be accurate. Please train a proper model using train_config.py")
                    
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            # Create emergency fallback
            from sklearn.ensemble import RandomForestClassifier
            import numpy as np
            
            X_dummy = np.random.rand(50, 20)
            y_dummy = np.random.randint(0, 2, 50)
            emergency_model = RandomForestClassifier(n_estimators=5, random_state=42)
            emergency_model.fit(X_dummy, y_dummy)
            
            model_data = {
                'model': emergency_model,
                'model_type': 'EmergencyFallback'
            }
            logger.info("Using emergency fallback model")
    
    return model_data


def calculate_simple_risk_score(patient_data: dict) -> float:
    """Calculate a simple risk score based on patient data (0-1 scale) for fallback prediction
    
    Prioritizes EHR dataset's top predictive features:
    - age/birth_date (22.3%), systolic_bp (16.9%), weight (7.0%), height (5.7%)
    """
    score = 0.0
    
    # Age factor (0-0.3) - Most important feature in EHR dataset
    age = patient_data.get('age', 40)
    if age > 70:
        score += 0.3
    elif age > 60:
        score += 0.25
    elif age > 50:
        score += 0.15
    elif age > 40:
        score += 0.1
    
    # Systolic blood pressure factor (0-0.25) - Second most important in EHR
    systolic_bp = patient_data.get('systolic_bp') or patient_data.get('resting_bp') or 120
    if systolic_bp and systolic_bp >= 160:
        score += 0.25
    elif systolic_bp and systolic_bp >= 140:
        score += 0.2
    elif systolic_bp and systolic_bp >= 130:
        score += 0.1
    
    # Total cholesterol factor (0-0.2) - Important EHR feature
    total_chol = patient_data.get('total_cholesterol') or patient_data.get('cholesterol') or 200
    if total_chol and total_chol >= 280:
        score += 0.2
    elif total_chol and total_chol >= 240:
        score += 0.15
    elif total_chol and total_chol >= 200:
        score += 0.05
    
    # BMI factor (0-0.15) - Calculated from weight/height (important EHR features)
    if 'bmi' in patient_data and patient_data['bmi']:
        bmi = patient_data['bmi']
    else:
        weight = patient_data.get('weight', 70)
        height_m = patient_data.get('height', 170) / 100
        bmi = weight / (height_m ** 2) if weight and height_m else None
    
    if bmi and bmi >= 35:
        score += 0.15
    elif bmi and bmi >= 30:
        score += 0.1
    elif bmi and bmi >= 25:
        score += 0.05
    
    # Smoking factor (0-0.1)
    smoking = patient_data.get('smoking', 0)
    if smoking == 1:  # Smoker
        score += 0.1
    
    # Diabetes factor (0-0.1)
    diabetes = patient_data.get('diabetes', 0)
    if diabetes == 1:  # Diabetes
        score += 0.1
    
    # Family history (0-0.05)
    if patient_data.get('family_history', 0) == 1:
        score += 0.05
    
    # Diastolic BP factor (0-0.08) - Fourth most important EHR feature
    diastolic_bp = patient_data.get('diastolic_bp')
    if not diastolic_bp:
        # Calculate from systolic if not provided
        diastolic_bp = max(60, systolic_bp - 40) if systolic_bp else 80
    
    if diastolic_bp and diastolic_bp >= 100:
        score += 0.08
    elif diastolic_bp and diastolic_bp >= 90:
        score += 0.05
    
    # Legacy clinical assessment fields (lower weight for EHR model)
    chest_pain = patient_data.get('chest_pain_type')
    if chest_pain == 'typical':
        score += 0.03
    elif chest_pain == 'atypical':
        score += 0.02
    
    # Exercise angina (0-0.02)
    if patient_data.get('exercise_angina', 0) == 1:
        score += 0.02
    
    # Ensure score is between 0 and 1
    return min(max(score, 0.05), 0.95)  # Keep between 5% and 95%

def generate_recommendations(patient_data: dict, risk_prob: float) -> List[str]:
    """Generate personalized health recommendations based on patient data"""
    recommendations = []
    
    # Age-based recommendations
    age = patient_data.get('age', 0)
    if age > 65:
        recommendations.append("Annual comprehensive geriatric assessment recommended")
    elif age > 50:
        recommendations.append("Regular health screenings recommended based on age")

    # Blood pressure management (using new field names)
    systolic_bp = patient_data.get('systolic_bp') or patient_data.get('resting_bp') or 0
    if systolic_bp and systolic_bp >= 140:
        recommendations.append("Monitor blood pressure regularly and consult with your healthcare provider")
    elif systolic_bp and systolic_bp >= 130:
        recommendations.append("Lifestyle modifications recommended for blood pressure management")
    
    # Cholesterol management (using new field names)
    total_cholesterol = patient_data.get('total_cholesterol') or patient_data.get('cholesterol') or 0
    if total_cholesterol and total_cholesterol >= 240:
        recommendations.append("Discuss cholesterol management strategies with your doctor")
    elif total_cholesterol and total_cholesterol >= 200:
        recommendations.append("Consider cholesterol screening and dietary modifications")
    
    # BMI recommendations (calculate if needed)
    if 'bmi' in patient_data and patient_data['bmi']:
        bmi = patient_data['bmi']
    else:
        weight = patient_data.get('weight', 70)
        height_m = patient_data.get('height', 170) / 100
        bmi = weight / (height_m ** 2) if weight and height_m else None
    
    if bmi and bmi >= 30:
        recommendations.append("Weight management program recommended for obesity")
    elif bmi and bmi >= 25:
        recommendations.append("Healthy weight maintenance recommended")
    
    # Risk-based recommendations
    if risk_prob > 0.7:
        recommendations.extend([
            "Immediate consultation with a healthcare provider recommended",
            "Consider comprehensive cardiovascular assessment",
            "Close monitoring of vital signs recommended"
        ])
    elif risk_prob > 0.4:
        recommendations.extend([
            "Schedule a follow-up with your healthcare provider",
            "Focus on modifiable risk factors",
            "Regular health monitoring recommended"
        ])
    else:
        recommendations.extend([
            "Maintain regular health check-ups",
            "Continue healthy lifestyle habits",
            "Preventive health measures recommended"
        ])
    
    # Ensure we have at least some recommendations
    if not recommendations:
        recommendations = [
            "Maintain regular health check-ups",
            "Follow a balanced diet and exercise regularly",
            "Monitor your health metrics regularly"
        ]
    
    return recommendations

def identify_ehr_risk_factors(patient_data: dict) -> List[str]:
    """Identify current risk factors based on EHR data"""
    risk_factors = []
    
    # Age risk factors
    if 'birth_date' in patient_data:
        try:
            birth_date = pd.to_datetime(patient_data['birth_date'])
            age = (pd.to_datetime('today') - birth_date).days / 365.25
            if age > 65:
                risk_factors.append(f"Advanced age ({int(age)} years)")
            elif age > 50:
                risk_factors.append(f"Middle age ({int(age)} years)")
        except:
            pass

    # Blood pressure risk factors
    systolic = patient_data.get('SystolicBloodPressure')
    diastolic = patient_data.get('DiastolicBloodPressure')
    
    if systolic and diastolic:
        if systolic >= 140 or diastolic >= 90:
            risk_factors.append(f"Hypertension ({systolic}/{diastolic} mmHg)")
        elif systolic >= 130 or diastolic >= 80:
            risk_factors.append(f"Elevated blood pressure ({systolic}/{diastolic} mmHg)")
    
    # Cholesterol risk factors
    cholesterol = patient_data.get('TotalCholesterol')
    if cholesterol and cholesterol >= 240:
        risk_factors.append(f"High cholesterol ({cholesterol} mg/dL)")

    if 'conditions' in patient_data:
        risk_factors.extend(patient_data['conditions'].split(' '))
    
    return risk_factors

def map_simple_to_fallback_features(patient_dict: dict) -> dict:
    """Map simple patient data to fallback model feature format
    
    Args:
        patient_dict: Dictionary containing simple patient data
        
    Returns:
        Dictionary with features expected by the fallback model
    """
    # Map legacy fields to new fields
    systolic_bp = patient_dict.get('systolic_bp') or patient_dict.get('resting_bp', 120)
    total_cholesterol = patient_dict.get('total_cholesterol') or patient_dict.get('cholesterol', 200)
    
    # Calculate diastolic BP if not provided (typically 60-80% of systolic)
    diastolic_bp = patient_dict.get('diastolic_bp')
    if diastolic_bp is None:
        diastolic_bp = int(systolic_bp * 0.67)  # Rough estimate
    
    # Calculate BMI if not provided
    bmi = patient_dict.get('bmi')
    if bmi is None and patient_dict.get('weight') and patient_dict.get('height'):
        weight_kg = float(patient_dict['weight'])
        height_m = float(patient_dict['height']) / 100  # Convert cm to m
        bmi = weight_kg / (height_m ** 2)
    
    # Create feature mapping for fallback model
    fallback_features = {
        'age': patient_dict.get('age', 50),
        'sex': 1 if patient_dict.get('sex', 'M').upper() == 'M' else 0,
        'systolic_bp': systolic_bp,
        'diastolic_bp': diastolic_bp,
        'cholesterol': total_cholesterol,
        'bmi': bmi or 25.0,
        'smoking': patient_dict.get('smoking', 0),
        'diabetes': patient_dict.get('diabetes', 0),
        'family_history': patient_dict.get('family_history', 0),
        'exercise_hours': patient_dict.get('exercise_hours', 2.0),  # Default moderate exercise
        'stress_level': patient_dict.get('stress_level', 5),  # Default moderate stress
    }
    
    return fallback_features

def map_simple_to_ehr_features(patient_dict: dict) -> dict:
    """Map simple patient data to EHR feature format based on your actual dataset
    
    Uses the top predictive features from your EHR dataset:
    - birth_date (22.3%), SystolicBloodPressure (16.9%), patient_id (12.7%),
    - DiastolicBloodPressure (8.5%), BodyWeight (7.0%)
    
    Args:
        patient_dict: Dictionary containing simple patient data
        
    Returns:
        Dictionary with EHR-formatted features
    """
    # Calculate birth date from age
    from datetime import datetime, timedelta
    current_year = datetime.now().year
    birth_year = current_year - patient_dict.get('age', 50)
    birth_date = f"{birth_year}-01-01"
    
    # Map legacy fields to new fields
    systolic_bp = patient_dict.get('systolic_bp') or patient_dict.get('resting_bp', 120)
    total_cholesterol = patient_dict.get('total_cholesterol') or patient_dict.get('cholesterol', 200)
    
    # Calculate diastolic BP if not provided (typically 60-80% of systolic)
    diastolic_bp = patient_dict.get('diastolic_bp')
    if diastolic_bp is None:
        diastolic_bp = int(systolic_bp * 0.67)  # Rough estimate
    
    # Calculate BMI if not provided
    bmi = patient_dict.get('bmi')
    if bmi is None and patient_dict.get('weight') and patient_dict.get('height'):
        weight_kg = float(patient_dict['weight'])
        height_m = float(patient_dict['height']) / 100  # Convert cm to m
        bmi = weight_kg / (height_m ** 2)
    
    # Create EHR feature mapping
    ehr_features = {
        # Top predictive features from your EHR dataset
        'birth_date': birth_date,  # 22.3% importance
        'SystolicBloodPressure': systolic_bp,  # 16.9% importance
        'patient_id': hash(str(patient_dict)) % 10000,  # 12.7% importance - synthetic ID
        'DiastolicBloodPressure': diastolic_bp,  # 8.5% importance
        'BodyWeight': patient_dict.get('weight', 70.0),  # 7.0% importance
        'BodyHeight': patient_dict.get('height', 170.0),  # 5.7% importance
        'BMI': bmi or 25.0,  # 5.0% importance
        
        # Additional important features
        'TotalCholesterol': total_cholesterol,  # 2.8% importance
        'LDLCholesterol': patient_dict.get('ldl_cholesterol', total_cholesterol * 0.6),  # 2.9%
        'Triglycerides': patient_dict.get('triglycerides', 150),  # 3.1% importance
        'HDLCholesterol': patient_dict.get('hdl_cholesterol', 50),
        'HbA1c': patient_dict.get('hba1c', 5.5),
        
        # Categorical features
        'Sex': 1 if patient_dict.get('sex', 'M').upper() == 'M' else 0,
        'Diabetes': patient_dict.get('diabetes', 0),
        'Smoking': patient_dict.get('smoking', 0),
        'FamilyHistory': patient_dict.get('family_history', 0),
        'FastingBloodSugar': patient_dict.get('fasting_blood_sugar', 0),
        
        # Additional features for compatibility
        'MaxHeartRate': patient_dict.get('max_heart_rate', 150),
        'ChestPainType': patient_dict.get('chest_pain_type', 'typical'),
        'RestingECG': patient_dict.get('resting_ecg', 'normal'),
        'ExerciseAngina': patient_dict.get('exercise_angina', 0),
        'Oldpeak': patient_dict.get('oldpeak', 0.0),
        'Slope': patient_dict.get('slope', 'up'),
        'CA': patient_dict.get('ca', 0),
        'Thal': patient_dict.get('thal', 'normal'),
        
        # Age as numeric feature
        'Age': patient_dict.get('age', 50),
    }
    
    return ehr_features
    
async def _generate_llm_analysis(patient_data: dict, risk_score: float, risk_factors: list[str], force_llm: bool = False) -> Dict[str, Any]:
    """Generate LLM analysis for the prediction
    
    Args:
        patient_data: Dictionary containing patient data
        risk_score: Calculated risk score (0-1)
        risk_factors: List of identified risk factors
        force_llm: If True, forces LLM analysis even if it's disabled by default
        
    Returns:
        Dictionary with the analysis results or error information
    """
    try:
        # Check if LLM analysis is enabled or forced
        if not llm_analyzer.is_enabled() and not force_llm:
            return {
                "analysis_available": False,
                "reason": "LLM analysis is disabled"
            }
        
        # If LLM is disabled but force_llm is True, log a warning
        if not llm_analyzer.is_enabled() and force_llm:
            logger.warning("LLM analysis is disabled but force_llm=True. Attempting to analyze anyway...")
        
        # Generate the analysis using the correct method
        analysis = await llm_analyzer.analyze_risk(
            patient_data=patient_data,
            risk_score=risk_score,
            risk_factors=risk_factors
        )
        
        # Ensure the analysis has the required fields
        if not analysis:
            raise ValueError("Empty analysis result from LLM analyzer")
            
        return {
            "analysis_available": True,
            "summary": analysis.get("summary", "No summary available"),
            "key_risk_factors": analysis.get("key_risk_factors", []),
            "health_implications": analysis.get("health_implications", "No health implications available"),
            "recommendations": analysis.get("recommendations", []),
            "urgency_level": analysis.get("urgency_level", "medium")
        }
        
    except Exception as e:
        logger.error(f"Error generating LLM analysis: {str(e)}", exc_info=True)
        return {
            "analysis_available": False,
            "reason": f"Error during analysis: {str(e)}"
        }

@router.post("/predict/simple", response_model=PredictionResponse)
async def predict_health_risk_simple(patient: SimplePatientData):
    """Predict health risk for a patient using AI directly"""
    try:
        logger.info("=== STARTING AI PREDICTION ===")
        
        # Convert simple data to dictionary format
        patient_dict = patient.dict()
        logger.info(f"Input patient data: {patient_dict}")
        
        # Extract force_llm flag and remove it from patient data
        force_llm = patient_dict.pop('force_llm', False)
        logger.info(f"LLM analysis forced: {force_llm}")
        
        # Use AI for direct prediction
        logger.info("Using AI for health risk prediction")
        ai_prediction = llm_analyzer.predict_health_risk(patient_dict)
        
        if ai_prediction.get("success"):
            # Extract prediction results from AI
            risk_probability = ai_prediction["risk_probability"]
            risk_level = ai_prediction["risk_level"]
            confidence = ai_prediction["confidence"]
            recommendations = ai_prediction["recommendations"]
            risk_factors = ai_prediction["risk_factors"]
            llm_analysis_data = ai_prediction["llm_analysis"]
            
            logger.info(f"AI prediction - Risk Level: {risk_level}, Probability: {risk_probability:.3f}")
            
            # Create LLM analysis object
            llm_analysis_obj = LLMAnalysis(**llm_analysis_data)
            
        else:
            # Fallback to traditional model if AI fails
            logger.warning(f"AI prediction failed: {ai_prediction.get('reason', 'Unknown error')}")
            logger.info("Falling back to traditional model prediction")
            
            # Load traditional model as fallback
            model_data = load_model()
            model = model_data['model']
            
            # Use simple risk calculation as fallback
            risk_probability = calculate_simple_risk_score(patient_dict)
            confidence = abs(risk_probability - 0.5) * 2
            
            # Determine risk level
            if risk_probability > 0.7:
                risk_level = "High"
            elif risk_probability > 0.4:
                risk_level = "Medium"
            else:
                risk_level = "Low"
            
            # Generate recommendations and identify risk factors using traditional methods
            recommendations = generate_recommendations(patient_dict, risk_probability)
            risk_factors = identify_simple_risk_factors(patient_dict)
            
            # Try to generate LLM analysis as additional insight
            llm_analysis_obj = None
            try:
                llm_analysis_result = await _generate_llm_analysis(patient_dict, risk_probability, risk_factors, force_llm)
                if llm_analysis_result and isinstance(llm_analysis_result, dict):
                    llm_analysis_obj = LLMAnalysis(**llm_analysis_result)
            except Exception as e:
                logger.error(f"Error generating fallback LLM analysis: {str(e)}", exc_info=True)
                # Create a fallback LLM analysis object
                llm_analysis_obj = LLMAnalysis(
                    analysis_available=False,
                    reason=f"Both Gemini prediction and LLM analysis failed: {str(e)}"
                )
        
        # Create response
        response = PredictionResponse(
            risk_probability=float(risk_probability),
            risk_level=risk_level,
            recommendations=recommendations,
            risk_factors=risk_factors,
            confidence=float(confidence),
            llm_analysis=llm_analysis_obj,
            data=patient_dict  # Include original patient data
        )
        
        logger.info(f"=== PREDICTION COMPLETED ===")
        logger.info(f"Final Risk Level: {risk_level}, Probability: {risk_probability:.3f}")
        return response
        
    except Exception as e:
        logger.error(f"Simple prediction error: {str(e)}")
        # Return a default response if prediction fails
        return PredictionResponse(
            risk_probability=0.3,
            risk_level="Low",
            recommendations=[
                "Unable to process prediction at this time",
                "Please consult with your healthcare provider",
                "Regular health check-ups recommended"
            ],
            risk_factors=["Assessment temporarily unavailable"],
            confidence=0.5,
            llm_analysis=LLMAnalysis(
                analysis_available=False,
                reason="Prediction error occurred"
            ),
            data=patient.dict() if patient else {}  # Include patient data even in error case
        )
def identify_simple_risk_factors(patient_data: dict) -> List[str]:
    """Identify current risk factors based on simple patient data"""
    risk_factors = []
    
    # Age risk factors
    age = patient_data.get('age', 0)
    if age > 65:
        risk_factors.append(f"Advanced age ({age} years)")
    elif age > 50:
        risk_factors.append(f"Middle age ({age} years)")
    
    # Blood pressure risk factors
    systolic_bp = patient_data.get('systolic_bp', patient_data.get('resting_bp', 0))
    if systolic_bp and systolic_bp >= 140:
        risk_factors.append(f"Hypertension ({systolic_bp} mmHg)")
    elif systolic_bp and systolic_bp >= 130:
        risk_factors.append(f"Elevated blood pressure ({systolic_bp} mmHg)")
    
    # Cholesterol risk factors
    cholesterol = patient_data.get('total_cholesterol', patient_data.get('cholesterol', 0))
    if cholesterol and cholesterol >= 240:
        risk_factors.append(f"High total cholesterol ({cholesterol} mg/dL)")
    elif cholesterol and cholesterol >= 200:
        risk_factors.append(f"Borderline high cholesterol ({cholesterol} mg/dL)")
    
    ldl = patient_data.get('ldl_cholesterol', 0)
    if ldl >= 160:
        risk_factors.append(f"Very high LDL cholesterol ({ldl} mg/dL)")
    elif ldl >= 130:
        risk_factors.append(f"High LDL cholesterol ({ldl} mg/dL)")
    
    hdl = patient_data.get('hdl_cholesterol', 0)
    if hdl < 40:
        risk_factors.append(f"Low HDL cholesterol ({hdl} mg/dL)")
    
    triglycerides = patient_data.get('triglycerides', 0)
    if triglycerides >= 200:
        risk_factors.append(f"High triglycerides ({triglycerides} mg/dL)")
    
    # BMI risk factors
    bmi = patient_data.get('bmi', 0)
    if bmi >= 30:
        risk_factors.append(f"Obesity (BMI: {bmi:.1f})")
    elif bmi >= 25:
        risk_factors.append(f"Overweight (BMI: {bmi:.1f})")
    
    # Blood sugar risk factors
    fasting_glucose = patient_data.get('fasting_blood_sugar', 0)
    if isinstance(fasting_glucose, int) and fasting_glucose > 0:
        risk_factors.append("Elevated fasting blood sugar")
    
    hba1c = patient_data.get('hba1c', 0)
    if hba1c >= 6.5:
        risk_factors.append(f"Diabetes (HbA1c: {hba1c}%)")
    elif hba1c >= 5.7:
        risk_factors.append(f"Prediabetes (HbA1c: {hba1c}%)")
    
    # Clinical assessment risk factors
    if patient_data.get('chest_pain_type') == 'typical':
        risk_factors.append("Typical chest pain")
    
    if patient_data.get('exercise_angina', 0) == 1:
        risk_factors.append("Exercise-induced angina")
    
    oldpeak = patient_data.get('oldpeak', 0)
    if oldpeak > 2.0:
        risk_factors.append(f"Significant ST depression ({oldpeak})")
    
    ca = patient_data.get('ca', 0)
    if ca > 0:
        risk_factors.append(f"Coronary artery disease ({ca} vessels affected)")
    
    if patient_data.get('thal') == 'reversible':
        risk_factors.append("Reversible defect in stress test")
    
    # Lifestyle risk factors (if available)
    smoking = patient_data.get('smoking', 0)
    if smoking == 2:
        risk_factors.append("Current smoker")
    elif smoking == 1:
        risk_factors.append("Former smoker")
    
    diabetes = patient_data.get('diabetes', 0)
    if diabetes == 1:
        risk_factors.append("Diabetes mellitus")
    elif diabetes == 2:
        risk_factors.append("Prediabetes")
    
    if patient_data.get('family_history', 0) == 1:
        risk_factors.append("Family history of heart disease")
    
    return risk_factors

def calculate_lab_enhanced_risk_score(patient_data: dict) -> float:
    """Calculate lab-enhanced risk score incorporating lab parameters"""
    # Start with basic risk calculation
    score = calculate_simple_risk_score(patient_data)
    
    # Lab-based risk adjustments
    hemoglobin = patient_data.get('hemoglobin')
    if hemoglobin is not None and isinstance(hemoglobin, (int, float)):
        sex = patient_data.get('sex', 'M')
        if sex == 'M' and hemoglobin < 13.0:
            score += 0.1  # Anemia in males
        elif sex == 'F' and hemoglobin < 12.0:
            score += 0.1  # Anemia in females
        elif hemoglobin > 18.0:
            score += 0.05  # Polycythemia
    
    # White blood cell count
    wbc = patient_data.get('total_leukocyte_count')
    if wbc is not None and isinstance(wbc, (int, float)):
        if wbc > 11.0:
            score += 0.08  # Leukocytosis (infection/inflammation)
        elif wbc < 4.0:
            score += 0.05  # Leukopenia
    
    # Platelet count
    platelets = patient_data.get('platelet_count')
    if platelets is not None and isinstance(platelets, (int, float)):
        if platelets < 150:
            score += 0.06  # Thrombocytopenia
        elif platelets > 450:
            score += 0.04  # Thrombocytosis
    
    # ESR (inflammation marker)
    esr = patient_data.get('erythrocyte_sedimentation_rate')
    if esr is not None and isinstance(esr, (int, float)) and esr > 20:
        score += 0.05  # Elevated inflammation
    
    # Neutrophil percentage (infection indicator)
    neutrophils = patient_data.get('neutrophils_percent')
    if neutrophils is not None and isinstance(neutrophils, (int, float)):
        if neutrophils > 80:
            score += 0.04  # Neutrophilia
        elif neutrophils < 40:
            score += 0.03  # Neutropenia
    
    # Lymphocyte percentage (immune function)
    lymphocytes = patient_data.get('lymphocytes_percent')
    if lymphocytes is not None and isinstance(lymphocytes, (int, float)) and lymphocytes < 20:
        score += 0.03  # Lymphopenia
    
    return min(max(score, 0.05), 0.95)

def identify_lab_risk_factors(patient_data: dict) -> List[str]:
    """Identify lab-specific risk factors"""
    risk_factors = identify_simple_risk_factors(patient_data)
    
    # Hematology risk factors
    hemoglobin = patient_data.get('hemoglobin')
    if hemoglobin is not None and isinstance(hemoglobin, (int, float)):
        sex = patient_data.get('sex', 'M')
        if sex == 'M' and hemoglobin < 13.0:
            risk_factors.append(f"Anemia (Hemoglobin: {hemoglobin} g/dL)")
        elif sex == 'F' and hemoglobin < 12.0:
            risk_factors.append(f"Anemia (Hemoglobin: {hemoglobin} g/dL)")
        elif hemoglobin > 18.0:
            risk_factors.append(f"Polycythemia (Hemoglobin: {hemoglobin} g/dL)")
    
    # White blood cell abnormalities
    wbc = patient_data.get('total_leukocyte_count')
    if wbc is not None and isinstance(wbc, (int, float)):
        if wbc > 11.0:
            risk_factors.append(f"Leukocytosis (WBC: {wbc} × 10³/μL)")
        elif wbc < 4.0:
            risk_factors.append(f"Leukopenia (WBC: {wbc} × 10³/μL)")
    
    # Platelet abnormalities
    platelets = patient_data.get('platelet_count')
    if platelets is not None and isinstance(platelets, (int, float)):
        if platelets < 150:
            risk_factors.append(f"Thrombocytopenia (Platelets: {platelets} × 10³/μL)")
        elif platelets > 450:
            risk_factors.append(f"Thrombocytosis (Platelets: {platelets} × 10³/μL)")
    
    # Inflammation markers
    esr = patient_data.get('erythrocyte_sedimentation_rate')
    if esr is not None and isinstance(esr, (int, float)) and esr > 20:
        risk_factors.append(f"Elevated inflammation (ESR: {esr} mm/hr)")
    
    # Differential count abnormalities
    neutrophils = patient_data.get('neutrophils_percent')
    if neutrophils is not None and isinstance(neutrophils, (int, float)):
        if neutrophils > 80:
            risk_factors.append(f"Neutrophilia ({neutrophils}%)")
        elif neutrophils < 40:
            risk_factors.append(f"Neutropenia ({neutrophils}%)")
    
    lymphocytes = patient_data.get('lymphocytes_percent')
    if lymphocytes is not None and isinstance(lymphocytes, (int, float)) and lymphocytes < 20:
        risk_factors.append(f"Lymphopenia ({lymphocytes}%)")
    
    return risk_factors

@router.post("/predict/lab", response_model=PredictionResponse)
async def predict_health_risk_lab(patient: LabPatientData):
    """Predict health risk using lab-enhanced data with AI analysis"""
    try:
        logger.info("=== STARTING LAB-ENHANCED AI PREDICTION ===")
        
        # Convert to dict for processing
        patient_dict = patient.dict()
        logger.info(f"Patient data keys: {list(patient_dict.keys())}")
        
        # Calculate BMI if not provided
        if not patient_dict.get('bmi') and patient_dict.get('weight') and patient_dict.get('height'):
            height_m = patient_dict['height'] / 100
            patient_dict['bmi'] = patient_dict['weight'] / (height_m ** 2)
            logger.info(f"Calculated BMI: {patient_dict['bmi']:.2f}")
        
        # Try AI-powered analysis first
        try:
            logger.info("Attempting AI-powered lab analysis...")
            llm_analysis = await _generate_llm_analysis(patient_dict, 0.5, [], force_llm=True)
            
            if llm_analysis.get('analysis_available', False):
                logger.info("✅ AI analysis successful - using AI results")
                return PredictionResponse(
                    risk_probability=llm_analysis.get('risk_score', 0.5),
                    risk_level=_get_risk_level(llm_analysis.get('risk_score', 0.5)),
                    recommendations=llm_analysis.get('recommendations', []),
                    risk_factors=llm_analysis.get('risk_factors', []),
                    confidence=llm_analysis.get('confidence', 0.8),
                    llm_analysis=LLMAnalysis(**llm_analysis),
                    data=patient.dict()  # Include original patient data
                )
            else:
                logger.warning("❌ AI analysis failed - falling back to traditional lab calculation")
                
        except Exception as ai_error:
            logger.error(f"AI analysis error: {str(ai_error)}")
            logger.warning("Falling back to traditional lab risk calculation")
        
        # Fallback to traditional lab-enhanced calculation
        logger.info("Using traditional lab-enhanced risk calculation...")
        
        # Calculate lab-enhanced risk score with detailed logging
        try:
            logger.info("Calling calculate_lab_enhanced_risk_score...")
            risk_score = calculate_lab_enhanced_risk_score(patient_dict)
            logger.info(f"Lab-enhanced risk score: {risk_score}")
        except Exception as calc_error:
            logger.error(f"Error in calculate_lab_enhanced_risk_score: {str(calc_error)}")
            raise calc_error
        
        # Generate recommendations and risk factors with detailed logging
        try:
            logger.info("Generating recommendations...")
            recommendations = generate_recommendations(patient_dict, risk_score)
            logger.info(f"Generated {len(recommendations)} recommendations")
        except Exception as rec_error:
            logger.error(f"Error in generate_recommendations: {str(rec_error)}")
            raise rec_error
            
        try:
            logger.info("Identifying lab risk factors...")
            risk_factors = identify_lab_risk_factors(patient_dict)
            logger.info(f"Identified {len(risk_factors)} risk factors")
        except Exception as rf_error:
            logger.error(f"Error in identify_lab_risk_factors: {str(rf_error)}")
            raise rf_error
        
        # Determine risk level
        risk_level = _get_risk_level(risk_score)
        
        # Calculate confidence based on available lab data
        lab_fields = ['hemoglobin', 'total_leukocyte_count', 'platelet_count', 'neutrophils_percent', 'lymphocytes_percent']
        available_labs = sum(1 for field in lab_fields if patient_dict.get(field) is not None)
        confidence = 0.6 + (available_labs / len(lab_fields)) * 0.3  # 0.6-0.9 range
        
        logger.info("✅ Lab prediction completed successfully")
        
        return PredictionResponse(
            risk_probability=risk_score,
            risk_level=risk_level,
            recommendations=recommendations,
            risk_factors=risk_factors,
            confidence=confidence,
            llm_analysis=llm_analysis_obj,
            data=patient.dict()  # Include original patient data
        )
        
    except Exception as e:
        logger.error(f"Lab prediction error: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Return fallback response
        return PredictionResponse(
            risk_probability=0.3,
            risk_level="Low",
            recommendations=[
                "Unable to process lab prediction at this time",
                "Please consult with your healthcare provider",
                "Regular health check-ups and lab monitoring recommended"
            ],
            risk_factors=["Lab assessment temporarily unavailable"],
            confidence=0.5,
            llm_analysis=LLMAnalysis(
                analysis_available=False,
                summary="Temporary service issue",
                key_risk_factors=[],
                health_implications="Temporary service issue",
                recommendations=["Please try again later or contact support"],
                urgency_level="low",
                reason=f"Lab prediction error: {str(e)}"
            ),
            data=patient.dict() if patient else {}  # Include patient data even in error case
        )

@router.post("/predict", response_model=PredictionResponse)
async def predict_health_risk(patient: FHIRPatientData, current_user = Depends(get_current_user) if auth_available else None):
    """Predict health risk for a patient using EHR data (requires authentication if enabled)"""
    try:
        # Load model
        model_data = load_model()
        model = model_data['model']
        preprocessor = model_data['preprocessor']
        feature_columns = preprocessor.feature_columns
        
        # Process the raw FHIR bundle
        ehr_processor_class = get_ehr_processor()
        ehr_processor = ehr_processor_class()
        patient_df = ehr_processor.process_single_record(patient.fhir_bundle)

        if patient_df.empty:
            raise HTTPException(status_code=400, detail="Could not process the provided FHIR data.")

        # Align columns with the training data
        aligned_df = pd.DataFrame(columns=feature_columns)
        for col in feature_columns:
            if col in patient_df.columns:
                aligned_df[col] = patient_df[col]
            else:
                aligned_df[col] = 0 # or np.nan

        # Prepare features for the model
        processed_data, _ = preprocessor.prepare_features(aligned_df, fit=False)
        
        # Make prediction
        if hasattr(model, 'predict_proba'):
            risk_probability = model.predict_proba(processed_data)[0][1]
        else:
            risk_probability = model.predict(processed_data)[0]
        
        # Calculate confidence (based on probability distance from 0.5)
        confidence = abs(risk_probability - 0.5) * 2
        
        # Determine risk level
        if risk_probability > 0.7:
            risk_level = "High"
        elif risk_probability > 0.4:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        # Generate recommendations and identify risk factors
        patient_dict = aligned_df.to_dict(orient='records')[0]
        recommendations = generate_recommendations(patient_dict, risk_probability)
        risk_factors = identify_ehr_risk_factors(patient_dict)
        
        return PredictionResponse(
            risk_probability=float(risk_probability),
            risk_level=risk_level,
            recommendations=recommendations,
            risk_factors=risk_factors,
            confidence=float(confidence),
            data=patient_dict  # Include original patient data
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.get("/model-info")
async def get_model_info():
    """Get information about the loaded EHR model"""
    try:
        model_data = load_model()
        
        # Get model type safely
        model_type = model_data.get('model_type')
        if not model_type:
            model_type = type(model_data['model']).__name__
        
        # Get feature information safely
        feature_columns = []
        feature_count = 0
        
        if 'feature_columns' in model_data:
            feature_columns = model_data['feature_columns']
            feature_count = len(feature_columns)
        elif 'preprocessor' in model_data and hasattr(model_data['preprocessor'], 'feature_columns'):
            feature_columns = model_data['preprocessor'].feature_columns
            feature_count = len(feature_columns)
        elif hasattr(model_data['model'], 'feature_names_in_'):
            feature_columns = list(model_data['model'].feature_names_in_)
            feature_count = len(feature_columns)
        else:
            # Fallback for models without feature info
            feature_columns = ["Feature information not available"]
            feature_count = 0
        
        # Get additional model information
        model_info = {
            "model_type": model_type,
            "feature_count": feature_count,
            "features": feature_columns[:10] if len(feature_columns) > 10 else feature_columns,  # Limit to first 10 features
            "total_features": feature_count,
            "status": "loaded",
            "dataset": "EHR Data from Kaggle",
            "model_keys": list(model_data.keys())
        }
        
        # Add performance metrics if available
        if 'metrics' in model_data:
            model_info['performance'] = model_data['metrics']
        
        # Add training info if available
        if 'training_info' in model_data:
            model_info['training_info'] = model_data['training_info']
            
        return model_info
        
    except Exception as e:
        logger.error(f"Model info error: {str(e)}")
        # Return a safe fallback response instead of raising error
        return {
            "model_type": "Unknown",
            "feature_count": 0,
            "features": ["Model information temporarily unavailable"],
            "total_features": 0,
            "status": "error",
            "dataset": "EHR Data from Kaggle",
            "error": str(e)
        }

@router.get("/health")
async def health_check():
    """Health check endpoint for the prediction API"""
    try:
        # Try to load model to verify it's working
        model_data = load_model()
        
        return {
            "status": "healthy",
            "model_loaded": True,
            "model_type": model_data.get('model_type', type(model_data['model']).__name__),
            "timestamp": pd.Timestamp.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "degraded",
            "model_loaded": False,
            "error": str(e),
            "timestamp": pd.Timestamp.now().isoformat()
        }

@router.post("/batch-predict")
async def batch_predict_health_risk(patients: List[FHIRPatientData], current_user = Depends(get_doctor_or_admin) if auth_available else None):
    """Predict health risk for multiple patients using EHR data (requires doctor or admin access if auth enabled)"""
    try:
        predictions = []
        
        for patient in patients:
            # Convert single patient data to the expected format
            patient_dict = patient.fhir_bundle if hasattr(patient, 'fhir_bundle') else patient.dict()
            
            # Use the existing prediction logic
            prediction_response = await predict_health_risk(FHIRPatientData(fhir_bundle=patient_dict))
            predictions.append(prediction_response)
        
        return {"predictions": predictions}
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

def _get_risk_level(risk_score: float) -> str:
    """Convert a risk score (0-1) to a risk level string
    
    Args:
        risk_score: Risk probability between 0 and 1
        
    Returns:
        Risk level as string: "Low", "Medium", or "High"
    """
    if risk_score > 0.7:
        return "High"
    elif risk_score > 0.4:
        return "Medium"
    else:
        return "Low"