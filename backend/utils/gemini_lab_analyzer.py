import os
import logging
import json
from typing import Dict, List, Optional, Any
import google.generativeai as genai

logger = logging.getLogger(__name__)

class GeminiLabAnalyzer:
    """Gemini AI-powered comprehensive lab analysis and risk prediction"""
    
    def __init__(self):
        self.model = None
        self.is_configured = False
        self._initialize_gemini()
    
    def _initialize_gemini(self):
        """Initialize Gemini AI model"""
        try:
            # Try to get API key from environment
            api_key = os.getenv('GEMINI_API_KEY')
            
            if not api_key:
                logger.warning("GEMINI_API_KEY not found in environment variables")
                return
            
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
            self.is_configured = True
            logger.info("Gemini Lab Analyzer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {str(e)}")
            self.is_configured = False
    
    def is_enabled(self) -> bool:
        """Check if Gemini analysis is available"""
        return self.is_configured and self.model is not None
    
    async def analyze_comprehensive_lab_data(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive lab analysis using Gemini AI
        
        Args:
            patient_data: Dictionary containing patient demographics, vitals, and lab values
            
        Returns:
            Dictionary containing comprehensive analysis results
        """
        if not self.is_enabled():
            return {
                "success": False,
                "error": "Gemini AI is not available",
                "risk_probability": 0.3,
                "risk_level": "Unknown",
                "confidence": 0.1
            }
        
        try:
            # Prepare comprehensive prompt for lab analysis
            prompt = self._create_lab_analysis_prompt(patient_data)
            
            # Generate analysis using Gemini
            response = await self._generate_gemini_response(prompt)
            
            # Parse and structure the response
            analysis_result = self._parse_gemini_lab_response(response)
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error in Gemini lab analysis: {str(e)}")
            return {
                "success": False,
                "error": f"Analysis failed: {str(e)}",
                "risk_probability": 0.3,
                "risk_level": "Unknown",
                "confidence": 0.1
            }
    
    def _create_lab_analysis_prompt(self, patient_data: Dict[str, Any]) -> str:
        """Create comprehensive prompt for Gemini lab analysis"""
        
        # Extract patient information
        age = patient_data.get('age', 'Unknown')
        sex = patient_data.get('sex', 'Unknown')
        
        # Format lab values for analysis
        lab_values = self._format_lab_values(patient_data)
        
        prompt = f"""
You are an expert clinical pathologist and internal medicine physician. Analyze the following comprehensive lab report and patient data to provide a detailed health risk assessment.

PATIENT INFORMATION:
- Age: {age} years
- Sex: {sex}
- Weight: {patient_data.get('weight', 'Not provided')} kg
- Height: {patient_data.get('height', 'Not provided')} cm
- BMI: {patient_data.get('bmi', 'Not calculated')}
- Blood Pressure: {patient_data.get('systolic_bp', 'Unknown')}/{patient_data.get('diastolic_bp', 'Unknown')} mmHg

LABORATORY VALUES:
{lab_values}

MEDICAL HISTORY:
- Diabetes: {'Yes' if patient_data.get('diabetes') == 1 else 'No'}
- Smoking: {'Yes' if patient_data.get('smoking') == 1 else 'No'}
- Family History of Heart Disease: {'Yes' if patient_data.get('family_history') == 1 else 'No'}
- Fasting Blood Sugar >120 mg/dL: {'Yes' if patient_data.get('fasting_blood_sugar') == 1 else 'No'}

ANALYSIS REQUIREMENTS:
Please provide a comprehensive analysis in the following JSON format:

{{
    "risk_assessment": {{
        "overall_risk_probability": <float between 0.0 and 1.0>,
        "risk_level": "<Low/Medium/High>",
        "confidence_score": <float between 0.0 and 1.0>
    }},
    "clinical_findings": {{
        "significant_abnormalities": [
            "List of significant lab abnormalities with clinical significance"
        ],
        "normal_findings": [
            "List of reassuring normal findings"
        ],
        "patterns_identified": [
            "Clinical patterns or syndromes identified"
        ]
    }},
    "risk_factors": [
        "Specific risk factors identified from the data"
    ],
    "health_implications": {{
        "immediate_concerns": [
            "Any immediate health concerns requiring attention"
        ],
        "long_term_risks": [
            "Potential long-term health risks"
        ],
        "protective_factors": [
            "Positive health factors identified"
        ]
    }},
    "recommendations": {{
        "immediate_actions": [
            "Actions needed immediately"
        ],
        "follow_up_tests": [
            "Recommended follow-up laboratory tests"
        ],
        "lifestyle_modifications": [
            "Specific lifestyle recommendations"
        ],
        "monitoring_schedule": [
            "Recommended monitoring frequency"
        ]
    }},
    "clinical_summary": "A comprehensive clinical summary of the patient's health status based on all available data",
    "urgency_level": "<low/medium/high>",
    "differential_diagnosis": [
        "Possible conditions to consider based on the lab pattern"
    ]
}}

Focus on:
1. Complete Blood Count interpretation (anemia, infection, hematologic disorders)
2. Differential count analysis (immune status, infection patterns)
3. Platelet function and bleeding risk
4. Inflammatory markers (ESR)
5. Integration with vital signs and medical history
6. Age and sex-specific normal ranges
7. Clinical correlation and pattern recognition
8. Risk stratification for cardiovascular, infectious, and hematologic conditions

Provide evidence-based analysis with clinical reasoning for all assessments.
"""
        
        return prompt
    
    def _format_lab_values(self, patient_data: Dict[str, Any]) -> str:
        """Format lab values for the prompt"""
        lab_sections = []
        
        # Complete Blood Count
        cbc_values = []
        cbc_params = {
            'hemoglobin': ('Hemoglobin', 'g/dL', '13.0-17.0 (M), 12.0-15.5 (F)'),
            'total_leukocyte_count': ('Total WBC', '×10³/μL', '4.0-10.0'),
            'red_blood_cell_count': ('RBC Count', '×10⁶/μL', '4.5-5.5'),
            'hematocrit': ('Hematocrit', '%', '40.0-50.0 (M), 36.0-46.0 (F)'),
            'mean_corpuscular_volume': ('MCV', 'fL', '83.0-101.0'),
            'mean_corpuscular_hb': ('MCH', 'pg', '27.0-32.0'),
            'mean_corpuscular_hb_conc': ('MCHC', 'g/dL', '31.5-34.5'),
            'red_cell_distribution_width': ('RDW', '%', '11.6-14.0')
        }
        
        for key, (name, unit, normal) in cbc_params.items():
            value = patient_data.get(key)
            if value is not None:
                cbc_values.append(f"  {name}: {value} {unit} (Normal: {normal})")
        
        if cbc_values:
            lab_sections.append("Complete Blood Count:\n" + "\n".join(cbc_values))
        
        # Differential Count
        diff_values = []
        diff_params = {
            'neutrophils_percent': ('Neutrophils', '%', '40-80'),
            'lymphocytes_percent': ('Lymphocytes', '%', '20-40'),
            'monocytes_percent': ('Monocytes', '%', '2-10'),
            'eosinophils_percent': ('Eosinophils', '%', '1-6'),
            'basophils_percent': ('Basophils', '%', '0-2')
        }
        
        for key, (name, unit, normal) in diff_params.items():
            value = patient_data.get(key)
            if value is not None:
                diff_values.append(f"  {name}: {value}{unit} (Normal: {normal})")
        
        # Absolute counts
        abs_params = {
            'absolute_neutrophil_count': ('ANC', '×10³/μL', '2.0-7.0'),
            'absolute_lymphocyte_count': ('ALC', '×10³/μL', '1.0-3.0'),
            'absolute_monocyte_count': ('AMC', '×10³/μL', '0.2-1.0'),
            'absolute_eosinophil_count': ('AEC', '×10³/μL', '0.02-0.5'),
            'absolute_basophil_count': ('ABC', '×10³/μL', '0.02-0.10')
        }
        
        for key, (name, unit, normal) in abs_params.items():
            value = patient_data.get(key)
            if value is not None:
                diff_values.append(f"  {name}: {value} {unit} (Normal: {normal})")
        
        if diff_values:
            lab_sections.append("Differential Leukocyte Count:\n" + "\n".join(diff_values))
        
        # Platelet Parameters
        plt_values = []
        plt_params = {
            'platelet_count': ('Platelet Count', '×10³/μL', '150-410'),
            'mean_platelet_volume': ('MPV', 'fL', '7-9'),
            'platelet_distribution_width': ('PDW', '%', '11.6-14.0')
        }
        
        for key, (name, unit, normal) in plt_params.items():
            value = patient_data.get(key)
            if value is not None:
                plt_values.append(f"  {name}: {value} {unit} (Normal: {normal})")
        
        if plt_values:
            lab_sections.append("Platelet Parameters:\n" + "\n".join(plt_values))
        
        # Additional Chemistry
        chem_values = []
        chem_params = {
            'total_cholesterol': ('Total Cholesterol', 'mg/dL', '<200'),
            'ldl_cholesterol': ('LDL Cholesterol', 'mg/dL', '<100'),
            'hdl_cholesterol': ('HDL Cholesterol', 'mg/dL', '>40 (M), >50 (F)'),
            'triglycerides': ('Triglycerides', 'mg/dL', '<150'),
            'hba1c': ('HbA1c', '%', '<5.7'),
            'erythrocyte_sedimentation_rate': ('ESR', 'mm/hr', '0-10')
        }
        
        for key, (name, unit, normal) in chem_params.items():
            value = patient_data.get(key)
            if value is not None:
                chem_values.append(f"  {name}: {value} {unit} (Normal: {normal})")
        
        if chem_values:
            lab_sections.append("Blood Chemistry:\n" + "\n".join(chem_values))
        
        return "\n\n".join(lab_sections) if lab_sections else "No lab values provided"
    
    async def _generate_gemini_response(self, prompt: str) -> str:
        """Generate response from Gemini"""
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Gemini generation error: {str(e)}")
            raise
    
    def _parse_gemini_lab_response(self, response_text: str) -> Dict[str, Any]:
        """Parse Gemini response and extract structured data"""
        try:
            # Try to extract JSON from the response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON found in response")
            
            json_str = response_text[start_idx:end_idx]
            parsed_data = json.loads(json_str)
            
            # Extract key values for the API response format
            risk_assessment = parsed_data.get('risk_assessment', {})
            
            result = {
                "success": True,
                "risk_probability": risk_assessment.get('overall_risk_probability', 0.3),
                "risk_level": risk_assessment.get('risk_level', 'Medium'),
                "confidence": risk_assessment.get('confidence_score', 0.7),
                "risk_factors": parsed_data.get('risk_factors', []),
                "recommendations": self._format_recommendations(parsed_data.get('recommendations', {})),
                "clinical_findings": parsed_data.get('clinical_findings', {}),
                "health_implications": parsed_data.get('health_implications', {}),
                "clinical_summary": parsed_data.get('clinical_summary', ''),
                "urgency_level": parsed_data.get('urgency_level', 'medium'),
                "differential_diagnosis": parsed_data.get('differential_diagnosis', []),
                "gemini_analysis": {
                    "analysis_available": True,
                    "summary": parsed_data.get('clinical_summary', ''),
                    "key_risk_factors": parsed_data.get('risk_factors', []),
                    "health_implications": self._format_health_implications(parsed_data.get('health_implications', {})),
                    "recommendations": self._format_ai_recommendations(parsed_data.get('recommendations', {})),
                    "urgency_level": parsed_data.get('urgency_level', 'medium')
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error parsing Gemini response: {str(e)}")
            # Fallback response
            return {
                "success": False,
                "error": f"Failed to parse AI response: {str(e)}",
                "risk_probability": 0.3,
                "risk_level": "Medium",
                "confidence": 0.5,
                "risk_factors": ["Analysis parsing failed"],
                "recommendations": ["Please consult with your healthcare provider"],
                "gemini_analysis": {
                    "analysis_available": False,
                    "reason": f"Response parsing failed: {str(e)}"
                }
            }
    
    def _format_recommendations(self, recommendations_dict: Dict) -> List[str]:
        """Format recommendations for the API response"""
        formatted = []
        
        for category, items in recommendations_dict.items():
            if isinstance(items, list):
                for item in items:
                    formatted.append(f"{category.replace('_', ' ').title()}: {item}")
        
        return formatted if formatted else ["Consult with your healthcare provider"]
    
    def _format_health_implications(self, health_implications: Dict) -> str:
        """Format health implications as a string"""
        implications = []
        
        for category, items in health_implications.items():
            if isinstance(items, list) and items:
                category_name = category.replace('_', ' ').title()
                implications.append(f"{category_name}: {', '.join(items)}")
        
        return "; ".join(implications) if implications else "No specific health implications identified"
    
    def _format_ai_recommendations(self, recommendations_dict: Dict) -> List[str]:
        """Format AI recommendations for LLM analysis section"""
        formatted = []
        
        for category, items in recommendations_dict.items():
            if isinstance(items, list):
                formatted.extend(items)
        
        return formatted if formatted else ["Consult with your healthcare provider"]

# Global instance
gemini_lab_analyzer = GeminiLabAnalyzer()
