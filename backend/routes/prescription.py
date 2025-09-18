"""
Prescription Analysis API Routes
Handles prescription upload, OCR processing, and medical analysis
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging
import base64
import io
from datetime import datetime

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    from utils.prescription_ocr import PrescriptionOCR, MedicineDatabase
except ImportError:
    from backend.utils.prescription_ocr import PrescriptionOCR, MedicineDatabase

try:
    from auth.database_store import get_current_user
except ImportError:
    from backend.auth.database_store import get_current_user

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize OCR and medicine database
ocr_processor = PrescriptionOCR()
medicine_db = MedicineDatabase()

# Pydantic models
class PrescriptionAnalysisResponse(BaseModel):
    success: bool
    analysis_id: str
    extracted_text: List[str]
    identified_medicines: List[Dict[str, Any]]
    dosage_info: Dict[str, List[str]]
    analysis_summary: str
    confidence_score: float
    total_medicines_found: int
    professional_advice: str
    health_recommendations: List[str]
    detailed_analysis: str
    downloadable_report: str
    timestamp: datetime

class MedicineInfoRequest(BaseModel):
    medicine_name: str = Field(..., description="Name of the medicine to get information about")

class MedicineInfoResponse(BaseModel):
    medicine_name: str
    information: str
    professional_advice: str
    safety_warnings: List[str]

@router.post("/upload", response_model=PrescriptionAnalysisResponse)
async def upload_prescription(
    file: UploadFile = File(..., description="Prescription image file")
):
    """
    Upload and analyze prescription image
    """
    try:
        # Validate file type - handle None content_type
        content_type = getattr(file, 'content_type', None) or ''
        filename = getattr(file, 'filename', '') or ''
        
        # Check content type or file extension for images and PDFs
        valid_image_types = ['image/jpeg', 'image/jpg', 'image/png', 'image/bmp', 'image/tiff', 'image/webp']
        valid_pdf_types = ['application/pdf']
        valid_image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        valid_pdf_extensions = ['.pdf']
        
        is_valid_image_type = any(content_type.lower().startswith(img_type) for img_type in valid_image_types) if content_type else False
        is_valid_pdf_type = any(content_type.lower().startswith(pdf_type) for pdf_type in valid_pdf_types) if content_type else False
        is_valid_image_ext = any(filename.lower().endswith(ext) for ext in valid_image_extensions) if filename else False
        is_valid_pdf_ext = any(filename.lower().endswith(ext) for ext in valid_pdf_extensions) if filename else False
        
        is_image = is_valid_image_type or is_valid_image_ext
        is_pdf = is_valid_pdf_type or is_valid_pdf_ext
        
        if not (is_image or is_pdf):
            raise HTTPException(
                status_code=400,
                detail=f"Only image files (JPEG, PNG, BMP, TIFF, WebP) and PDF files are supported. Received: {content_type or 'unknown'}"
            )
        
        # Read file data
        file_data = await file.read()
        
        if len(file_data) > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(
                status_code=400,
                detail="File size too large. Maximum size is 10MB."
            )
        
        # Determine file type
        file_type = 'pdf' if is_pdf else 'image'
        logger.info(f"Processing prescription upload - file: {filename}, content_type: {content_type}, type: {file_type}, size: {len(file_data)} bytes")
        
        # Analyze prescription with file type
        analysis_result = ocr_processor.analyze_prescription(file_data, file_type)
        
        if not analysis_result['success']:
            raise HTTPException(
                status_code=500,
                detail=f"Prescription analysis failed: {analysis_result.get('error', 'Unknown error')}"
            )
        
        # Generate professional medical advice
        professional_advice = generate_professional_advice(
            analysis_result['identified_medicines'],
            analysis_result['dosage_info']
        )
        
        # Generate health recommendations
        health_recommendations = generate_health_recommendations(
            analysis_result['identified_medicines'],
            {}
        )
        
        # Generate unique analysis ID
        analysis_id = f"RX_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Generate detailed analysis paragraph
        detailed_analysis = generate_detailed_analysis_paragraph(
            analysis_result['identified_medicines'],
            analysis_result['dosage_info'],
            analysis_result['extracted_text'],
            analysis_result['confidence_score']
        )
        
        # Generate downloadable report
        downloadable_report = generate_downloadable_report(
            analysis_id,
            analysis_result,
            professional_advice,
            health_recommendations,
            detailed_analysis
        )
        
        return PrescriptionAnalysisResponse(
            success=True,
            analysis_id=analysis_id,
            extracted_text=analysis_result['extracted_text'],
            identified_medicines=analysis_result['identified_medicines'],
            dosage_info=analysis_result['dosage_info'],
            analysis_summary=analysis_result['analysis_summary'],
            confidence_score=analysis_result['confidence_score'],
            total_medicines_found=analysis_result['total_medicines_found'],
            professional_advice=professional_advice,
            health_recommendations=health_recommendations,
            detailed_analysis=detailed_analysis,
            downloadable_report=downloadable_report,
            timestamp=datetime.now()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prescription upload error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@router.post("/medicine-info", response_model=MedicineInfoResponse)
async def get_medicine_info(
    request: MedicineInfoRequest
):
    """
    Get detailed information about a specific medicine
    """
    try:
        medicine_name = request.medicine_name.strip()
        
        if not medicine_name:
            raise HTTPException(
                status_code=400,
                detail="Medicine name is required"
            )
        
        # Get medicine information
        medicine_info = medicine_db.get_medicine_info(medicine_name)
        
        if not medicine_info:
            return MedicineInfoResponse(
                medicine_name=medicine_name,
                information=f"Detailed information about {medicine_name} is not available in our database.",
                professional_advice="Please consult your healthcare provider or pharmacist for comprehensive information about this medication.",
                safety_warnings=[
                    "Always follow your doctor's prescribed dosage",
                    "Read the medication label carefully",
                    "Report any unusual side effects to your healthcare provider",
                    "Do not share medications with others"
                ]
            )
        
        # Generate professional advice
        professional_advice = medicine_db.generate_medicine_advice(medicine_name)
        
        # Generate safety warnings
        safety_warnings = generate_safety_warnings(medicine_info)
        
        return MedicineInfoResponse(
            medicine_name=medicine_name,
            information=format_medicine_information(medicine_info),
            professional_advice=professional_advice,
            safety_warnings=safety_warnings
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Medicine info error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve medicine information: {str(e)}"
        )

@router.get("/health")
async def prescription_service_health():
    """Health check for prescription analysis service"""
    try:
        # Test OCR functionality
        test_successful = True
        error_message = None
        
        try:
            # Simple test to ensure OCR components are working
            if not ocr_processor.easyocr_reader:
                test_successful = False
                error_message = "EasyOCR not initialized"
        except Exception as e:
            test_successful = False
            error_message = str(e)
        
        return {
            "status": "healthy" if test_successful else "degraded",
            "message": "Prescription analysis service is running" if test_successful else f"Service issues: {error_message}",
            "ocr_available": test_successful,
            "medicine_database_loaded": len(medicine_db.medicine_info) > 0,
            "supported_formats": ["JPEG", "PNG", "BMP", "TIFF", "PDF"],
            "max_file_size": "10MB"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Service health check failed: {str(e)}",
            "ocr_available": False,
            "medicine_database_loaded": False
        }

def generate_professional_advice(medicines: List[Dict[str, Any]], dosage_info: Dict[str, List[str]]) -> str:
    """Generate professional medical advice based on identified medicines"""
    
    if not medicines:
        return """
**Professional Medical Assessment:**

I was unable to clearly identify specific medications from this prescription image. This could be due to:
- Handwriting clarity
- Image quality
- Lighting conditions

**Recommendations:**
1. Please ensure the prescription image is clear and well-lit
2. Try taking the photo from directly above the prescription
3. Consult your pharmacist for assistance in reading the prescription
4. Contact your healthcare provider if you have questions about your medications

**Important:** Never guess about medication names or dosages. Always verify with a healthcare professional.
        """.strip()
    
    advice_parts = []
    
    # Header
    advice_parts.append("**Professional Medical Assessment:**")
    advice_parts.append(f"I have identified {len(medicines)} medication(s) from your prescription:")
    
    # Medicine-specific advice
    for i, medicine in enumerate(medicines, 1):
        medicine_name = medicine['medicine_name']
        category = medicine['category'].replace('_', ' ').title()
        
        advice_parts.append(f"\n**{i}. {medicine_name}** ({category})")
        
        # Get detailed advice from database
        detailed_advice = medicine_db.generate_medicine_advice(medicine_name)
        if detailed_advice and "not available" not in detailed_advice:
            advice_parts.append(detailed_advice)
        else:
            advice_parts.append(f"This medication is commonly used for {category.lower()} conditions. Please follow your doctor's instructions carefully.")
    
    # General recommendations
    advice_parts.append("\n**General Recommendations:**")
    advice_parts.append("1. Take medications exactly as prescribed by your healthcare provider")
    advice_parts.append("2. Complete the full course of treatment, especially for antibiotics")
    advice_parts.append("3. Store medications in a cool, dry place away from children")
    advice_parts.append("4. Set reminders to take medications at the correct times")
    advice_parts.append("5. Report any unusual side effects to your healthcare provider immediately")
    
    # Dosage reminders if found
    if dosage_info.get('frequencies'):
        advice_parts.append(f"\n**Dosage Schedule:** {', '.join(dosage_info['frequencies'])}")
    
    advice_parts.append("\n**Disclaimer:** This analysis is for informational purposes only and does not replace professional medical advice.")
    
    return '\n'.join(advice_parts)

def generate_health_recommendations(medicines: List[Dict[str, Any]], user_data: Dict[str, Any]) -> List[str]:
    """Generate personalized health recommendations based on medicines and user profile"""
    
    recommendations = []
    
    if not medicines:
        return [
            "Maintain regular check-ups with your healthcare provider",
            "Keep an updated list of all your medications",
            "Store medications safely and check expiration dates regularly"
        ]
    
    # Category-based recommendations
    categories = [med['category'] for med in medicines]
    
    if 'antibiotics' in categories:
        recommendations.extend([
            "Complete the full course of antibiotics even if you feel better",
            "Take probiotics to support gut health during antibiotic treatment",
            "Avoid alcohol while taking antibiotics"
        ])
    
    if 'pain_relievers' in categories:
        recommendations.extend([
            "Use pain relievers only as needed and as directed",
            "Consider non-medication pain management techniques like ice/heat therapy",
            "Monitor for any stomach upset or unusual symptoms"
        ])
    
    if 'cardiovascular' in categories:
        recommendations.extend([
            "Monitor your blood pressure regularly",
            "Maintain a heart-healthy diet low in sodium",
            "Engage in regular, doctor-approved physical activity"
        ])
    
    if 'diabetes' in categories:
        recommendations.extend([
            "Monitor blood sugar levels as directed by your healthcare provider",
            "Follow a consistent meal schedule",
            "Keep glucose tablets or snacks available for low blood sugar episodes"
        ])
    
    # General health recommendations
    recommendations.extend([
        "Set up medication reminders to ensure consistent timing",
        "Keep a medication diary to track effectiveness and side effects",
        "Schedule regular follow-up appointments with your healthcare provider"
    ])
    
    return recommendations[:6]  # Limit to 6 recommendations

def format_medicine_information(medicine_info: Dict[str, Any]) -> str:
    """Format medicine information for display"""
    
    info_parts = []
    
    info_parts.append(f"**Generic Name:** {medicine_info['generic_name']}")
    info_parts.append(f"**Category:** {medicine_info['category']}")
    
    if medicine_info.get('brand_names'):
        info_parts.append(f"**Brand Names:** {', '.join(medicine_info['brand_names'])}")
    
    if medicine_info.get('uses'):
        info_parts.append(f"**Primary Uses:** {', '.join(medicine_info['uses'])}")
    
    if medicine_info.get('common_dosage'):
        info_parts.append(f"**Typical Dosage:** {medicine_info['common_dosage']}")
    
    if medicine_info.get('side_effects'):
        info_parts.append(f"**Common Side Effects:** {', '.join(medicine_info['side_effects'])}")
    
    return '\n'.join(info_parts)

def generate_safety_warnings(medicine_info: Dict[str, Any]) -> List[str]:
    """Generate safety warnings for a medicine"""
    
    warnings = []
    
    if medicine_info.get('contraindications'):
        warnings.append(f"Avoid if you have: {', '.join(medicine_info['contraindications'])}")
    
    if medicine_info.get('interactions'):
        warnings.append(f"May interact with: {', '.join(medicine_info['interactions'])}")
    
    if medicine_info.get('max_daily_dose'):
        warnings.append(f"Do not exceed {medicine_info['max_daily_dose']} per day")
    
    # General warnings
    warnings.extend([
        "Always follow your doctor's prescribed dosage",
        "Report any unusual side effects immediately",
        "Do not share this medication with others",
        "Store as directed on the label"
    ])
    
    return warnings

def generate_detailed_analysis_paragraph(medicines: List[Dict[str, Any]], dosage_info: Dict[str, List[str]], extracted_text: List[str], confidence_score: float) -> str:
    """Generate a comprehensive detailed analysis paragraph"""
    
    analysis_parts = []
    
    # Header
    analysis_parts.append("## 📋 Comprehensive Prescription Analysis Report")
    analysis_parts.append(f"**Analysis Date:** {datetime.now().strftime('%B %d, %Y at %I:%M %p')}")
    analysis_parts.append(f"**Confidence Score:** {confidence_score:.1%}")
    analysis_parts.append("")
    
    # OCR Results
    analysis_parts.append("### 🔍 Optical Character Recognition Results")
    if extracted_text:
        analysis_parts.append("The following text was successfully extracted from your prescription image:")
        for i, text in enumerate(extracted_text[:5], 1):  # Limit to first 5 lines
            analysis_parts.append(f"{i}. {text}")
        if len(extracted_text) > 5:
            analysis_parts.append(f"... and {len(extracted_text) - 5} more lines")
    else:
        analysis_parts.append("No clear text could be extracted from the prescription image. This may be due to handwriting clarity, image quality, or lighting conditions.")
    analysis_parts.append("")
    
    # Medicine Analysis
    analysis_parts.append("### 💊 Identified Medications")
    if medicines:
        analysis_parts.append(f"I have successfully identified **{len(medicines)} medication(s)** from your prescription:")
        analysis_parts.append("")
        
        for i, medicine in enumerate(medicines, 1):
            medicine_name = medicine.get('medicine_name', 'Unknown')
            category = medicine.get('category', 'general').replace('_', ' ').title()
            confidence = medicine.get('confidence', 0.0)
            
            analysis_parts.append(f"**{i}. {medicine_name}**")
            analysis_parts.append(f"   - **Category:** {category}")
            analysis_parts.append(f"   - **Detection Confidence:** {confidence:.1%}")
            
            # Get medicine info from database
            medicine_info = medicine_db.get_medicine_info(medicine_name)
            if medicine_info:
                if medicine_info.get('uses'):
                    analysis_parts.append(f"   - **Primary Uses:** {', '.join(medicine_info['uses'][:3])}")
                if medicine_info.get('common_dosage'):
                    analysis_parts.append(f"   - **Typical Dosage:** {medicine_info['common_dosage']}")
            analysis_parts.append("")
    else:
        analysis_parts.append("No medications could be clearly identified from the prescription image. This could be due to:")
        analysis_parts.append("- Handwriting legibility issues")
        analysis_parts.append("- Image quality or resolution")
        analysis_parts.append("- Lighting or angle of the photograph")
        analysis_parts.append("- Prescription format or layout")
        analysis_parts.append("")
    
    # Dosage Information
    if dosage_info and any(dosage_info.values()):
        analysis_parts.append("### ⏰ Dosage Schedule Information")
        if dosage_info.get('frequencies'):
            analysis_parts.append(f"**Identified Frequencies:** {', '.join(dosage_info['frequencies'])}")
        if dosage_info.get('durations'):
            analysis_parts.append(f"**Treatment Duration:** {', '.join(dosage_info['durations'])}")
        if dosage_info.get('instructions'):
            analysis_parts.append(f"**Special Instructions:** {', '.join(dosage_info['instructions'])}")
        analysis_parts.append("")
    
    # Quality Assessment
    analysis_parts.append("### 📊 Image Quality Assessment")
    if confidence_score >= 0.8:
        analysis_parts.append("**Excellent:** The prescription image quality is very good, allowing for high-confidence text extraction and medication identification.")
    elif confidence_score >= 0.6:
        analysis_parts.append("**Good:** The prescription image quality is acceptable, though some details may be unclear. Consider retaking the photo with better lighting if needed.")
    elif confidence_score >= 0.4:
        analysis_parts.append("**Fair:** The prescription image quality has some issues that may affect accuracy. Please ensure the image is clear, well-lit, and taken from directly above.")
    else:
        analysis_parts.append("**Poor:** The prescription image quality is challenging for accurate analysis. Please retake the photo with better lighting, focus, and angle for improved results.")
    analysis_parts.append("")
    
    # Next Steps
    analysis_parts.append("### 📝 Recommended Next Steps")
    analysis_parts.append("1. **Verify with Healthcare Provider:** Always confirm medication details with your doctor or pharmacist")
    analysis_parts.append("2. **Check Dosage Instructions:** Ensure you understand the correct dosage and timing")
    analysis_parts.append("3. **Review Side Effects:** Be aware of potential side effects and interactions")
    analysis_parts.append("4. **Set Medication Reminders:** Use alarms or apps to maintain consistent timing")
    analysis_parts.append("5. **Store Properly:** Keep medications in appropriate conditions as directed")
    analysis_parts.append("")
    
    # Disclaimer
    analysis_parts.append("### ⚠️ Important Disclaimer")
    analysis_parts.append("This analysis is provided for informational purposes only and should not replace professional medical advice. Always consult with your healthcare provider or pharmacist for medication guidance, dosage confirmation, and any questions about your treatment plan.")
    
    return '\n'.join(analysis_parts)

def generate_downloadable_report(analysis_id: str, analysis_result: Dict[str, Any], professional_advice: str, health_recommendations: List[str], detailed_analysis: str) -> str:
    """Generate a comprehensive downloadable report"""
    
    report_parts = []
    
    # Header
    report_parts.append("# PRESCRIPTION ANALYSIS REPORT")
    report_parts.append("=" * 50)
    report_parts.append(f"Report ID: {analysis_id}")
    report_parts.append(f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}")
    report_parts.append(f"Analysis Confidence: {analysis_result.get('confidence_score', 0.0):.1%}")
    report_parts.append("")
    
    # Executive Summary
    report_parts.append("## EXECUTIVE SUMMARY")
    report_parts.append("-" * 20)
    medicines_count = len(analysis_result.get('identified_medicines', []))
    if medicines_count > 0:
        report_parts.append(f"✓ Successfully identified {medicines_count} medication(s)")
        report_parts.append(f"✓ Extracted {len(analysis_result.get('extracted_text', []))} lines of text")
        report_parts.append(f"✓ Generated {len(health_recommendations)} personalized recommendations")
    else:
        report_parts.append("⚠ No medications could be clearly identified")
        report_parts.append("⚠ Image quality may need improvement")
    report_parts.append("")
    
    # Detailed Analysis
    report_parts.append(detailed_analysis)
    report_parts.append("")
    
    # Professional Advice
    report_parts.append("## PROFESSIONAL MEDICAL ADVICE")
    report_parts.append("-" * 30)
    report_parts.append(professional_advice)
    report_parts.append("")
    
    # Health Recommendations
    report_parts.append("## PERSONALIZED HEALTH RECOMMENDATIONS")
    report_parts.append("-" * 40)
    for i, recommendation in enumerate(health_recommendations, 1):
        report_parts.append(f"{i}. {recommendation}")
    report_parts.append("")
    
    # Technical Details
    report_parts.append("## TECHNICAL ANALYSIS DETAILS")
    report_parts.append("-" * 30)
    report_parts.append("### Extracted Text:")
    for i, text in enumerate(analysis_result.get('extracted_text', []), 1):
        report_parts.append(f"{i}. {text}")
    report_parts.append("")
    
    if analysis_result.get('identified_medicines'):
        report_parts.append("### Identified Medications:")
        for i, medicine in enumerate(analysis_result['identified_medicines'], 1):
            report_parts.append(f"{i}. {medicine.get('medicine_name', 'Unknown')} ({medicine.get('category', 'general')})")
            report_parts.append(f"   Confidence: {medicine.get('confidence', 0.0):.1%}")
    report_parts.append("")
    
    # Footer
    report_parts.append("=" * 50)
    report_parts.append("IMPORTANT: This report is for informational purposes only.")
    report_parts.append("Always consult healthcare professionals for medical advice.")
    report_parts.append("Generated by Healthcare AI Assistant")
    
    return '\n'.join(report_parts)
