import os
import logging
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import joblib
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import prediction routes
predict_router = None
predict_available = False

try:
    from routes.predict import router as predict_router
    predict_available = True
    print("✓ Prediction routes imported successfully (relative)")
except ImportError:
    try:
        from backend.routes.predict import router as predict_router
        predict_available = True
        print("✓ Prediction routes imported successfully (absolute)")
    except ImportError as e:
        print(f"✗ Failed to import prediction routes: {e}")
        predict_router = None
        predict_available = False
        print("⚠️  Prediction module not available")

# Import chatbot routes
chatbot_router = None
chatbot_available = False

try:
    from routes.chatbot import router as chatbot_router
    chatbot_available = True
    print("✓ Chatbot routes imported successfully (relative)")
except ImportError:
    try:
        from backend.routes.chatbot import router as chatbot_router
        chatbot_available = True
        print("✓ Chatbot routes imported successfully (absolute)")
    except ImportError as e:
        print(f"Failed to import chatbot routes: {e}")
        chatbot_router = None
        chatbot_available = False
        print("⚠️  Running without AI chatbot functionality")

# Import prescription routes
prescription_router = None
prescription_available = False

try:
    from routes.prescription import router as prescription_router
    prescription_available = True
    print("✓ Prescription routes imported successfully (relative)")
except ImportError:
    try:
        from backend.routes.prescription import router as prescription_router
        prescription_available = True
        print("✓ Prescription routes imported successfully (absolute)")
    except ImportError as e:
        print(f"Failed to import prescription routes: {e}")
        prescription_router = None
        prescription_available = False
        print("⚠️  Prescription analysis module not available")

# Import drug advisor routes (drug extraction, interactions, dosage, alternatives)
drug_advisor_router = None
drug_advisor_available = False

try:
    from routes.drug_advisor import router as drug_advisor_router
    drug_advisor_available = True
    print("✓ Drug advisor routes imported successfully (relative)")
except ImportError:
    try:
        from backend.routes.drug_advisor import router as drug_advisor_router
        drug_advisor_available = True
        print("✓ Drug advisor routes imported successfully (absolute)")
    except ImportError as e:
        print(f"Failed to import drug advisor routes: {e}")
        drug_advisor_router = None
        drug_advisor_available = False
        print("⚠️  Drug advisor module not available")

# Import medical report routes
medical_report_router = None
medical_report_available = False

try:
    from routes.medical_report import router as medical_report_router
    medical_report_available = True
    print("✓ Medical report routes imported successfully (relative)")
except ImportError:
    try:
        from backend.routes.medical_report import router as medical_report_router
        medical_report_available = True
        print("✓ Medical report routes imported successfully (absolute)")
    except ImportError as e:
        print(f"Failed to import medical report routes: {e}")
        medical_report_router = None
        medical_report_available = False
        print("⚠️  Medical report analysis module not available")

# Import health log routes
health_log_router = None
health_log_available = False

try:
    from routes.health_log import router as health_log_router
    health_log_available = True
    print("✓ Health log routes imported successfully (relative)")
except ImportError:
    try:
        from backend.routes.health_log import router as health_log_router
        health_log_available = True
        print("✓ Health log routes imported successfully (absolute)")
    except ImportError as e:
        print(f"Failed to import health log routes: {e}")
        health_log_router = None
        health_log_available = False
        print("⚠️  Health log module not available")

# Import authentication routes with absolute imports first
auth_router = None
auth_available = False

try:
    # Try relative import first when running from backend directory
    from auth.database_store import get_db
    from auth.routes import router as auth_router
    auth_available = True
    print("✓ Authentication routes imported successfully (relative)")
except ImportError as e1:
    try:
        # Fallback to absolute import
        from backend.auth.database_store import get_db
        from backend.auth.routes import router as auth_router
        auth_available = True
        print("✓ Authentication routes imported successfully (absolute)")
    except ImportError as e2:
        print(f"Failed to import auth routes (relative): {e1}")
        print(f"Failed to import auth routes (absolute): {e2}")
        auth_router = None
        auth_available = False
        print("⚠️  Authentication module not available")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Personalized Healthcare Recommendation API",
    description="AI-powered healthcare recommendation system",
    version="1.0.0"
)

# CORS middleware for frontend integration
# Support both comma-separated and JSON-style list strings in CORS_ORIGINS
cors_env = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8501,http://localhost:8503,http://localhost:8000,http://localhost:8002")
try:
    import json as _json
    if cors_env.strip().startswith("["):
        cors_origins = _json.loads(cors_env)
    else:
        cors_origins = [o.strip() for o in cors_env.split(",") if o.strip()]
except Exception:
    cors_origins = [o.strip() for o in cors_env.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,  # Load from environment variables
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
if predict_available and predict_router:
    app.include_router(predict_router, prefix="/api", tags=["predictions"])
    print("✓ Prediction routes registered at /api")
else:
    print("⚠️  Running without prediction functionality")

if chatbot_available and chatbot_router:
    app.include_router(chatbot_router, prefix="/api", tags=["chatbot"])
    print("✓ Chatbot routes registered at /api")
    print("Available chatbot endpoints:")
    print("  - POST /api/chat")
    print("  - GET /api/chat/health")
    print("  - POST /api/chat/explain-risk")
else:
    print("⚠️  Running without AI chatbot functionality")

if prescription_available and prescription_router:
    app.include_router(prescription_router, prefix="/api/prescription", tags=["prescription"])
    print("✓ Prescription routes registered at /api/prescription")
    print("Available prescription endpoints:")
    print("  - POST /api/prescription/upload")
    print("  - POST /api/prescription/medicine-info")
    print("  - GET /api/prescription/health")
else:
    print("⚠️  Running without prescription analysis functionality")

if drug_advisor_available and drug_advisor_router:
    app.include_router(drug_advisor_router, prefix="/api/drug-advisor", tags=["drug-advisor"])
    print("✓ Drug advisor routes registered at /api/drug-advisor")
    print("Available drug advisor endpoints:")
    print("  - POST /api/drug-advisor/extract_drug_info")
    print("  - POST /api/drug-advisor/check_interactions")
    print("  - POST /api/drug-advisor/dosage_recommendation")
    print("  - POST /api/drug-advisor/alternative_suggestions")
else:
    print("⚠️  Running without drug advisor functionality")

if medical_report_available and medical_report_router:
    app.include_router(medical_report_router, prefix="/api/medical-report", tags=["medical-report"])
    print("✓ Medical report routes registered at /api/medical-report")
    print("Available medical report endpoints:")
    print("  - POST /api/medical-report/upload")
    print("  - GET /api/medical-report/analysis/{analysis_id}")
    print("  - GET /api/medical-report/download/{analysis_id}")
    print("  - GET /api/medical-report/list")
    print("  - DELETE /api/medical-report/analysis/{analysis_id}")
    print("  - GET /api/medical-report/lifestyle-recommendations/{analysis_id}")
    print("  - GET /api/medical-report/statistics")
    print("  - GET /api/medical-report/health")
else:
    print("⚠️  Running without medical report analysis functionality")

if health_log_available and health_log_router:
    app.include_router(health_log_router, prefix="/api/health-log", tags=["health-log"])
    print("✓ Health log routes registered at /api/health-log")
    print("Available health log endpoints:")
    print("  - POST /api/health-log/")
    print("  - GET /api/health-log/")
    print("  - PUT /api/health-log/{entry_id}")
    print("  - DELETE /api/health-log/{entry_id}")
    print("  - GET /api/health-log/statistics")
    print("  - GET /api/health-log/health")
else:
    print("⚠️  Running without health log functionality")

if auth_available and auth_router:
    app.include_router(auth_router, prefix="/api/auth", tags=["authentication"])
    print("✓ Authentication routes registered at /api/auth")
    print("Available auth endpoints:")
    print("  - POST /api/auth/login")
    print("  - GET /api/auth/default-users")
    print("  - GET /api/auth/health")
    print("  - GET /api/auth/test-connection")
else:
    print("⚠️  Running in demo mode without authentication")
    print("  - Authentication endpoints will return 404")

@app.get("/")
async def root():
    return {"message": "Healthcare Recommendation API is running"}

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "message": "Backend service is running"}

@app.get("/api/version")
async def version_info():
    """Get API version information"""
    try:
        # Import here to avoid circular imports
        try:
            from routes.predict import load_model
        except ImportError:
            from backend.routes.predict import load_model
        
        model_data = load_model()
        return {
            "version": "1.0.0",
            "model_status": "loaded" if model_data else "not_loaded",
            "features": model_data['feature_columns'] if model_data else []
        }
    except:
        return {
            "version": "1.0.0",
            "model_status": "not_loaded",
            "features": []
        }

@app.get("/api/test")
async def test_endpoint():
    return {"message": "This is a test endpoint"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
