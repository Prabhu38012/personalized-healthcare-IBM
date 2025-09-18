import os
import logging
import json
import base64
from typing import Dict, List, Optional, Any, Union
import google.generativeai as genai
from PIL import Image
import io
import PyPDF2
import fitz  # PyMuPDF for better PDF handling

logger = logging.getLogger(__name__)

class ReportAnalyzer:
    """AI-powered lab report analyzer for images and PDFs"""
    
    def __init__(self):
        self.model = None
        self.vision_model = None
        self.is_configured = False
        self._initialize_ai()
    
    def _initialize_ai(self):
        """Initialize AI models"""
        try:
            api_key = os.getenv('GEMINI_API_KEY')
            
            if not api_key:
                logger.warning("GEMINI_API_KEY not found in environment variables")
                return
            
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
            self.vision_model = genai.GenerativeModel('gemini-2.0-flash-exp')
            self.is_configured = True
            logger.info("Report Analyzer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize AI models: {str(e)}")
            self.is_configured = False
    
    def is_enabled(self) -> bool:
        """Check if report analysis is available"""
        return self.is_configured and self.model is not None
    
    async def analyze_uploaded_report(self, file_content: bytes, file_type: str, filename: str) -> Dict[str, Any]:
        """
        Analyze uploaded lab report (image or PDF)
        
        Args:
            file_content: Raw file bytes
            file_type: MIME type of the file
            filename: Original filename
            
        Returns:
            Dictionary containing extracted lab data and analysis
        """
        if not self.is_enabled():
            # Return sample data when AI is not available
            return self._get_fallback_sample_data(filename)
        
        try:
            logger.info(f"Analyzing uploaded report: {filename} ({file_type})")
            
            if file_type.startswith('image/'):
                return await self._analyze_image_report(file_content, filename)
            elif file_type == 'application/pdf':
                return await self._analyze_pdf_report(file_content, filename)
            else:
                return {
                    "success": False,
                    "error": f"Unsupported file type: {file_type}",
                    "extracted_data": {}
                }
                
        except Exception as e:
            logger.error(f"Error analyzing report: {str(e)}")
            # Check if it's a quota error and provide fallback
            if "quota" in str(e).lower() or "429" in str(e):
                logger.warning("API quota exceeded, using fallback sample data")
                return self._get_fallback_sample_data(filename)
            
            return {
                "success": False,
                "error": f"Analysis failed: {str(e)}",
                "extracted_data": {}
            }
    
    async def _analyze_image_report(self, image_content: bytes, filename: str) -> Dict[str, Any]:
        """Analyze lab report from image"""
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_content))
            
            # Create prompt for lab report extraction
            prompt = self._create_extraction_prompt()
            
            # Analyze with vision model
            response = self.vision_model.generate_content([prompt, image])
            
            # Parse the response
            return self._parse_extraction_response(response.text, filename)
            
        except Exception as e:
            logger.error(f"Error analyzing image report: {str(e)}")
            # Check if it's a quota error and provide fallback
            if "quota" in str(e).lower() or "429" in str(e):
                logger.warning("API quota exceeded during image analysis, using fallback sample data")
                return self._get_fallback_sample_data(filename)
            
            return {
                "success": False,
                "error": f"Image analysis failed: {str(e)}",
                "extracted_data": {}
            }
    
    async def _analyze_pdf_report(self, pdf_content: bytes, filename: str) -> Dict[str, Any]:
        """Analyze lab report from PDF"""
        try:
            # Extract text from PDF
            pdf_text = self._extract_pdf_text(pdf_content)
            
            if not pdf_text.strip():
                # If no text extracted, try to convert PDF pages to images
                return await self._analyze_pdf_as_images(pdf_content, filename)
            
            # Create prompt for text-based extraction
            prompt = self._create_text_extraction_prompt(pdf_text)
            
            # Analyze with text model
            response = self.model.generate_content(prompt)
            
            # Parse the response
            return self._parse_extraction_response(response.text, filename)
            
        except Exception as e:
            logger.error(f"Error analyzing PDF report: {str(e)}")
            # Check if it's a quota error and provide fallback
            if "quota" in str(e).lower() or "429" in str(e):
                logger.warning("API quota exceeded during PDF analysis, using fallback sample data")
                return self._get_fallback_sample_data(filename)
            
            return {
                "success": False,
                "error": f"PDF analysis failed: {str(e)}",
                "extracted_data": {}
            }
    
    def _extract_pdf_text(self, pdf_content: bytes) -> str:
        """Extract text from PDF using PyMuPDF"""
        try:
            # Use PyMuPDF for better text extraction
            pdf_document = fitz.open(stream=pdf_content, filetype="pdf")
            text = ""
            
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                text += page.get_text()
            
            pdf_document.close()
            return text
            
        except Exception as e:
            logger.error(f"Error extracting PDF text: {str(e)}")
            # Fallback to PyPDF2
            try:
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
                return text
            except Exception as e2:
                logger.error(f"Fallback PDF extraction also failed: {str(e2)}")
                return ""
    
    async def _analyze_pdf_as_images(self, pdf_content: bytes, filename: str) -> Dict[str, Any]:
        """Convert PDF pages to images and analyze"""
        try:
            pdf_document = fitz.open(stream=pdf_content, filetype="pdf")
            
            # Convert first page to image (most lab reports are single page)
            page = pdf_document[0]
            pix = page.get_pixmap()
            img_data = pix.tobytes("png")
            
            pdf_document.close()
            
            # Analyze as image
            return await self._analyze_image_report(img_data, filename)
            
        except Exception as e:
            logger.error(f"Error converting PDF to image: {str(e)}")
            return {
                "success": False,
                "error": f"PDF to image conversion failed: {str(e)}",
                "extracted_data": {}
            }
    
    def _create_extraction_prompt(self) -> str:
        """Create prompt for extracting lab values from image"""
        return """
You are an expert medical laboratory technician and data extraction specialist. Analyze this lab report image and extract all the laboratory values and patient information.

Please extract the following information and return it in JSON format:

{
    "patient_info": {
        "name": "Patient name if visible",
        "age": "Age in years (number only)",
        "sex": "M or F",
        "date_of_collection": "Date when sample was collected"
    },
    "complete_blood_count": {
        "hemoglobin": "Value in g/dL",
        "total_leukocyte_count": "Value in 10³/μL or thousands/μL",
        "red_blood_cell_count": "Value in 10⁶/μL or millions/μL", 
        "hematocrit": "Value in %",
        "mean_corpuscular_volume": "MCV value in fL",
        "mean_corpuscular_hb": "MCH value in pg",
        "mean_corpuscular_hb_conc": "MCHC value in g/dL",
        "red_cell_distribution_width": "RDW value in %"
    },
    "differential_count": {
        "neutrophils_percent": "Neutrophils percentage",
        "lymphocytes_percent": "Lymphocytes percentage",
        "monocytes_percent": "Monocytes percentage", 
        "eosinophils_percent": "Eosinophils percentage",
        "basophils_percent": "Basophils percentage"
    },
    "absolute_counts": {
        "absolute_neutrophil_count": "ANC in 10³/μL",
        "absolute_lymphocyte_count": "ALC in 10³/μL",
        "absolute_monocyte_count": "AMC in 10³/μL",
        "absolute_eosinophil_count": "AEC in 10³/μL",
        "absolute_basophil_count": "ABC in 10³/μL"
    },
    "platelet_parameters": {
        "platelet_count": "Platelet count in 10³/μL",
        "mean_platelet_volume": "MPV in fL",
        "platelet_distribution_width": "PDW in %"
    },
    "other_parameters": {
        "erythrocyte_sedimentation_rate": "ESR in mm/hr",
        "total_cholesterol": "Total cholesterol in mg/dL",
        "ldl_cholesterol": "LDL cholesterol in mg/dL", 
        "hdl_cholesterol": "HDL cholesterol in mg/dL",
        "triglycerides": "Triglycerides in mg/dL",
        "hba1c": "HbA1c in %"
    },
    "report_metadata": {
        "lab_name": "Laboratory name",
        "report_type": "Type of report (e.g., Complete Blood Count, Lipid Profile)",
        "reference_ranges_provided": "true/false"
    }
}

IMPORTANT INSTRUCTIONS:
1. Extract ONLY the numerical values without units
2. If a value is not found or not visible, use null
3. Convert all values to the specified units
4. Be very careful with decimal points and number precision
5. Look for common abbreviations (Hb for Hemoglobin, WBC for White Blood Cells, etc.)
6. Pay attention to normal ranges provided in the report
7. Return ONLY the JSON object, no additional text
"""
    
    def _create_text_extraction_prompt(self, pdf_text: str) -> str:
        """Create prompt for extracting lab values from PDF text"""
        return f"""
You are an expert medical laboratory technician and data extraction specialist. Analyze this lab report text and extract all the laboratory values and patient information.

LAB REPORT TEXT:
{pdf_text}

Please extract the following information and return it in JSON format:

{{
    "patient_info": {{
        "name": "Patient name if visible",
        "age": "Age in years (number only)",
        "sex": "M or F",
        "date_of_collection": "Date when sample was collected"
    }},
    "complete_blood_count": {{
        "hemoglobin": "Value in g/dL",
        "total_leukocyte_count": "Value in 10³/μL or thousands/μL",
        "red_blood_cell_count": "Value in 10⁶/μL or millions/μL", 
        "hematocrit": "Value in %",
        "mean_corpuscular_volume": "MCV value in fL",
        "mean_corpuscular_hb": "MCH value in pg",
        "mean_corpuscular_hb_conc": "MCHC value in g/dL",
        "red_cell_distribution_width": "RDW value in %"
    }},
    "differential_count": {{
        "neutrophils_percent": "Neutrophils percentage",
        "lymphocytes_percent": "Lymphocytes percentage",
        "monocytes_percent": "Monocytes percentage", 
        "eosinophils_percent": "Eosinophils percentage",
        "basophils_percent": "Basophils percentage"
    }},
    "absolute_counts": {{
        "absolute_neutrophil_count": "ANC in 10³/μL",
        "absolute_lymphocyte_count": "ALC in 10³/μL",
        "absolute_monocyte_count": "AMC in 10³/μL",
        "absolute_eosinophil_count": "AEC in 10³/μL",
        "absolute_basophil_count": "ABC in 10³/μL"
    }},
    "platelet_parameters": {{
        "platelet_count": "Platelet count in 10³/μL",
        "mean_platelet_volume": "MPV in fL",
        "platelet_distribution_width": "PDW in %"
    }},
    "other_parameters": {{
        "erythrocyte_sedimentation_rate": "ESR in mm/hr",
        "total_cholesterol": "Total cholesterol in mg/dL",
        "ldl_cholesterol": "LDL cholesterol in mg/dL", 
        "hdl_cholesterol": "HDL cholesterol in mg/dL",
        "triglycerides": "Triglycerides in mg/dL",
        "hba1c": "HbA1c in %"
    }},
    "report_metadata": {{
        "lab_name": "Laboratory name",
        "report_type": "Type of report",
        "reference_ranges_provided": "true/false"
    }}
}}

IMPORTANT INSTRUCTIONS:
1. Extract ONLY the numerical values without units
2. If a value is not found, use null
3. Convert all values to the specified units
4. Be very careful with decimal points and number precision
5. Look for common abbreviations and medical terminology
6. Return ONLY the JSON object, no additional text
"""
    
    def _parse_extraction_response(self, response_text: str, filename: str) -> Dict[str, Any]:
        """Parse AI response and convert to lab data format"""
        try:
            # Extract JSON from response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON found in response")
            
            json_str = response_text[start_idx:end_idx]
            extracted_data = json.loads(json_str)
            
            # Convert to our lab data format
            lab_data = self._convert_to_lab_format(extracted_data)
            
            return {
                "success": True,
                "extracted_data": lab_data,
                "source_file": filename,
                "extraction_metadata": {
                    "patient_info": extracted_data.get("patient_info", {}),
                    "report_metadata": extracted_data.get("report_metadata", {})
                }
            }
            
        except Exception as e:
            logger.error(f"Error parsing extraction response: {str(e)}")
            return {
                "success": False,
                "error": f"Failed to parse extracted data: {str(e)}",
                "extracted_data": {}
            }
    
    def _convert_to_lab_format(self, extracted_data: Dict) -> Dict[str, Any]:
        """Convert extracted data to our LabPatientData format"""
        lab_data = {}
        
        # Patient info
        patient_info = extracted_data.get("patient_info", {})
        if patient_info.get("age"):
            try:
                lab_data["age"] = int(float(patient_info["age"]))
            except (ValueError, TypeError):
                pass
        
        if patient_info.get("sex"):
            sex = patient_info["sex"].upper()
            if sex in ["M", "F"]:
                lab_data["sex"] = sex
        
        # Default values for required fields
        lab_data.update({
            "weight": 70.0,  # Default weight
            "height": 170,   # Default height
            "systolic_bp": 120,  # Default BP
            "diastolic_bp": 80,
            "total_cholesterol": 200,  # Default values
            "ldl_cholesterol": 120,
            "hdl_cholesterol": 50,
            "triglycerides": 150,
            "hba1c": 5.5,
            "diabetes": 0,
            "smoking": 0,
            "family_history": 0,
            "fasting_blood_sugar": 0
        })
        
        # Complete Blood Count
        cbc = extracted_data.get("complete_blood_count", {})
        cbc_mapping = {
            "hemoglobin": "hemoglobin",
            "total_leukocyte_count": "total_leukocyte_count", 
            "red_blood_cell_count": "red_blood_cell_count",
            "hematocrit": "hematocrit",
            "mean_corpuscular_volume": "mean_corpuscular_volume",
            "mean_corpuscular_hb": "mean_corpuscular_hb",
            "mean_corpuscular_hb_conc": "mean_corpuscular_hb_conc",
            "red_cell_distribution_width": "red_cell_distribution_width"
        }
        
        for extracted_key, lab_key in cbc_mapping.items():
            value = cbc.get(extracted_key)
            if value is not None:
                try:
                    lab_data[lab_key] = float(value)
                except (ValueError, TypeError):
                    pass
        
        # Differential Count
        diff = extracted_data.get("differential_count", {})
        diff_mapping = {
            "neutrophils_percent": "neutrophils_percent",
            "lymphocytes_percent": "lymphocytes_percent",
            "monocytes_percent": "monocytes_percent",
            "eosinophils_percent": "eosinophils_percent", 
            "basophils_percent": "basophils_percent"
        }
        
        for extracted_key, lab_key in diff_mapping.items():
            value = diff.get(extracted_key)
            if value is not None:
                try:
                    lab_data[lab_key] = float(value)
                except (ValueError, TypeError):
                    pass
        
        # Absolute Counts
        abs_counts = extracted_data.get("absolute_counts", {})
        abs_mapping = {
            "absolute_neutrophil_count": "absolute_neutrophil_count",
            "absolute_lymphocyte_count": "absolute_lymphocyte_count",
            "absolute_monocyte_count": "absolute_monocyte_count",
            "absolute_eosinophil_count": "absolute_eosinophil_count",
            "absolute_basophil_count": "absolute_basophil_count"
        }
        
        for extracted_key, lab_key in abs_mapping.items():
            value = abs_counts.get(extracted_key)
            if value is not None:
                try:
                    lab_data[lab_key] = float(value)
                except (ValueError, TypeError):
                    pass
        
        # Platelet Parameters
        platelets = extracted_data.get("platelet_parameters", {})
        plt_mapping = {
            "platelet_count": "platelet_count",
            "mean_platelet_volume": "mean_platelet_volume",
            "platelet_distribution_width": "platelet_distribution_width"
        }
        
        for extracted_key, lab_key in plt_mapping.items():
            value = platelets.get(extracted_key)
            if value is not None:
                try:
                    lab_data[lab_key] = float(value)
                except (ValueError, TypeError):
                    pass
        
        # Other Parameters
        other = extracted_data.get("other_parameters", {})
        if other.get("erythrocyte_sedimentation_rate"):
            try:
                lab_data["erythrocyte_sedimentation_rate"] = float(other["erythrocyte_sedimentation_rate"])
            except (ValueError, TypeError):
                pass
        
        # Override with extracted cholesterol values if available
        cholesterol_mapping = {
            "total_cholesterol": "total_cholesterol",
            "ldl_cholesterol": "ldl_cholesterol", 
            "hdl_cholesterol": "hdl_cholesterol",
            "triglycerides": "triglycerides",
            "hba1c": "hba1c"
        }
        
        for extracted_key, lab_key in cholesterol_mapping.items():
            value = other.get(extracted_key)
            if value is not None:
                try:
                    if lab_key == "hba1c":
                        lab_data[lab_key] = float(value)
                    else:
                        lab_data[lab_key] = int(float(value))
                except (ValueError, TypeError):
                    pass
        
        return lab_data
    
    def _get_fallback_sample_data(self, filename: str) -> Dict[str, Any]:
        """Return sample lab data when AI analysis is not available"""
        return {
            "success": True,
            "extracted_data": {
                # Basic Demographics
                "age": 27,
                "sex": "M",
                "weight": 70.0,
                "height": 175,
                "systolic_bp": 130,
                "diastolic_bp": 85,
                
                # Basic Chemistry
                "total_cholesterol": 200,
                "ldl_cholesterol": 120,
                "hdl_cholesterol": 45,
                "triglycerides": 150,
                "hba1c": 5.5,
                
                # Medical History
                "diabetes": 0,
                "smoking": 0,
                "family_history": 0,
                "fasting_blood_sugar": 0,
                
                # Complete Blood Count (sample data)
                "hemoglobin": 13.9,
                "total_leukocyte_count": 12.1,
                "red_blood_cell_count": 5.00,
                "hematocrit": 43.7,
                "mean_corpuscular_volume": 88.0,
                "mean_corpuscular_hb": 27.9,
                "mean_corpuscular_hb_conc": 31.7,
                "red_cell_distribution_width": 14.5,
                
                # Differential Count
                "neutrophils_percent": 70.9,
                "lymphocytes_percent": 22.1,
                "monocytes_percent": 5.5,
                "eosinophils_percent": 1.1,
                "basophils_percent": 0.4,
                
                # Absolute Counts
                "absolute_neutrophil_count": 8.58,
                "absolute_lymphocyte_count": 2.67,
                "absolute_monocyte_count": 0.67,
                "absolute_eosinophil_count": 0.13,
                "absolute_basophil_count": 0.05,
                
                # Platelet Parameters
                "platelet_count": 580.0,
                "mean_platelet_volume": 8.5,
                "platelet_distribution_width": 12.0,
                
                # Additional Chemistry
                "erythrocyte_sedimentation_rate": 2.0
            },
            "source_file": filename,
            "extraction_metadata": {
                "patient_info": {
                    "name": "Sample Patient",
                    "age": "27",
                    "sex": "M"
                },
                "report_metadata": {
                    "lab_name": "Sample Lab Report",
                    "report_type": "Complete Blood Count with Differential",
                    "reference_ranges_provided": "true"
                }
            },
            "fallback_used": True,
            "message": "AI analysis temporarily unavailable. Using sample lab data for demonstration."
        }

# Global instance
report_analyzer = ReportAnalyzer()
