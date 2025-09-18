import os
from typing import Dict, Any, Optional
import google.generativeai as genai
from dotenv import load_dotenv
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class LLMRiskAnalyzer:
    """Handles LLM-based risk analysis for healthcare predictions using Google's Gemini API"""
    
    def __init__(self):
        # Get API key from environment variable
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model_name = "gemini-2.0-flash-exp"
        self.temperature = 0.3
        
        # Configure the Gemini client if API key is available
        if not self.api_key:
            logger.warning("GEMINI_API_KEY environment variable not set. LLM analysis will be disabled.")
            self.model = None
            return
            
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
            logger.info(f"Initialized Gemini model: {self.model_name}")
        except Exception as e:
            logger.error(f"Error initializing Gemini client: {str(e)}")
            self.model = None
    
    def is_available(self) -> bool:
        """Check if LLM service is available"""
        return self.model is not None and bool(self.api_key)
    
    def is_enabled(self) -> bool:
        """Alias for is_available for backward compatibility"""
        return self.is_available()
    
    async def analyze_risk(
        self, 
        patient_data: Dict[str, Any], 
        risk_score: float,
        risk_factors: list[str]
    ) -> Dict[str, Any]:
        """Alias for analyze_risk_factors for backward compatibility"""
        return await self.analyze_risk_factors(patient_data, risk_score, risk_factors)
    
    async def predict_health_risk(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use Gemini AI to predict health risk directly from patient data
        
        Args:
            patient_data: Patient data dictionary
            
        Returns:
            Dictionary containing risk prediction and analysis
        """
        if not self.is_available():
            return {
                "success": False,
                "reason": "Gemini service not configured"
            }
        
        try:
            # Prepare the prediction prompt
            prompt = self._create_prediction_prompt(patient_data)
            
            # Call the Gemini API
            response = await self._call_llm_api(prompt)
            
            # Parse and return the response
            return self._parse_prediction_response(response)
            
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "quota" in error_msg.lower():
                logger.warning(f"LLM API quota exceeded: {error_msg}")
                return {
                    "success": False,
                    "reason": "AI prediction temporarily unavailable due to API limits. Please try again later."
                }
            else:
                logger.error(f"Error in LLM prediction: {error_msg}", exc_info=True)
                return {
                    "success": False,
                    "reason": f"Error during prediction: {error_msg}"
                }

    async def analyze_risk_factors(
        self, 
        patient_data: Dict[str, Any], 
        risk_score: float,
        risk_factors: list[str]
    ) -> Dict[str, Any]:
        """
        Analyze patient data and risk factors using Gemini
        
        Args:
            patient_data: Patient data dictionary
            risk_score: Numeric risk score (0-1)
            risk_factors: List of identified risk factors
            
        Returns:
            Dictionary containing LLM analysis
        """
        if not self.is_available():
            return {
                "analysis_available": False,
                "reason": "Gemini service not configured"
            }
        
        try:
            # Prepare the prompt
            prompt = self._create_analysis_prompt(patient_data, risk_score, risk_factors)
            
            # Call the Gemini API
            response = await self._call_llm_api(prompt)
            
            # Parse and return the response
            return self._parse_llm_response(response)
            
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "quota" in error_msg.lower():
                logger.warning(f"LLM API quota exceeded: {error_msg}")
                return {
                    "analysis_available": False,
                    "reason": "AI analysis temporarily unavailable due to API limits. Please try again later."
                }
            else:
                logger.error(f"Error in LLM analysis: {error_msg}", exc_info=True)
                return {
                    "analysis_available": False,
                    "reason": f"Error during analysis: {error_msg}"
                }
    
    def _create_prediction_prompt(self, patient_data: Dict[str, Any]) -> str:
        """Create a prompt for Gemini to predict health risk directly"""
        # Format patient data for the prompt
        patient_info = "\n".join([f"- {k}: {v}" for k, v in patient_data.items()])
        
        prompt = f"""You are an expert medical AI assistant specializing in cardiovascular risk assessment. Analyze the following patient data and provide a comprehensive health risk prediction.

Patient Data:
{patient_info}

Based on this information, provide a complete cardiovascular risk assessment. Consider factors like:
- Age and demographic factors
- Blood pressure readings (systolic/diastolic)
- Cholesterol levels (total, HDL, LDL, triglycerides)
- BMI and weight status
- Diabetes and blood sugar levels
- Smoking history and lifestyle factors
- Family history of heart disease
- Other relevant clinical indicators

Format your response as a JSON object with these exact keys:
- "risk_probability" (number): Risk probability as a decimal between 0.0 and 1.0
- "risk_level" (string): "Low", "Medium", or "High"
- "confidence" (number): Confidence in prediction as decimal between 0.0 and 1.0
- "risk_factors" (list of strings): 3-7 identified risk factors
- "recommendations" (list of strings): 5-8 specific, actionable recommendations
- "summary" (string): Brief overview of the patient's cardiovascular health status
- "key_risk_factors" (list of strings): Top 3-5 most significant risk factors
- "health_implications" (string): Potential health consequences and outlook
- "urgency_level" (string): "low", "medium", or "high" based on risk level

Provide evidence-based medical assessment. Be thorough but concise. Respond with valid JSON only, no additional text or markdown formatting."""
        
        return prompt

    def _create_analysis_prompt(
        self, 
        patient_data: Dict[str, Any],
        risk_score: float,
        risk_factors: list[str]
    ) -> str:
        """Create a prompt for the LLM based on patient data"""
        # Format patient data for the prompt
        patient_info = "\n".join([f"- {k}: {v}" for k, v in patient_data.items()])
        
        prompt = f"""You are a medical AI assistant analyzing patient health risks. 
        
        Patient Data:
        {patient_info}
        
        Risk Score: {risk_score:.1%}
        Identified Risk Factors: {', '.join(risk_factors) if risk_factors else 'None'}
        
        Please provide a detailed analysis including:
        1. A brief summary of the patient's health status
        2. Explanation of key risk factors
        3. Potential health implications
        4. Recommended next steps
        
        Format your response as a JSON object with these keys:
        - "summary" (string): Brief overview of the patient's health status
        - "key_risk_factors" (list of strings): Top 3-5 most significant risk factors
        - "health_implications" (string): Potential health consequences
        - "recommendations" (list of strings): 3-5 specific, actionable recommendations
        - "urgency_level" (string): "low", "medium", or "high" based on risk
        
        Respond with valid JSON only, no additional text or markdown formatting.
        """
        
        return prompt
    
    async def _call_llm_api(self, prompt: str) -> str:
        """Call the Gemini API with the given prompt"""
        if not self.model:
            raise ValueError("Gemini model not initialized")
            
        try:
            # Create chat session
            chat = self.model.start_chat(history=[])
            
            # Send the prompt (use synchronous call for compatibility)
            response = chat.send_message(
                prompt,
                generation_config={
                    "temperature": self.temperature,
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_output_tokens": 2048,
                },
            )
            
            # Extract the text response
            if hasattr(response, 'text'):
                return response.text
            elif hasattr(response, 'candidates') and response.candidates:
                return response.candidates[0].content.parts[0].text
            else:
                raise ValueError("Unexpected response format from Gemini API")
            
        except Exception as e:
            logger.error(f"Error calling Gemini API: {str(e)}")
            raise
    
    def _parse_prediction_response(self, response_text: str) -> Dict[str, Any]:
        """Parse the Gemini prediction response into a structured format"""
        try:
            # Clean the response (remove markdown code blocks if present)
            clean_text = response_text.strip()
            if '```json' in clean_text:
                clean_text = clean_text.split('```json')[1].split('```')[0].strip()
            elif '```' in clean_text:
                clean_text = clean_text.split('```')[1].split('```')[0].strip()
            
            # Try to parse the response as JSON
            result = json.loads(clean_text)
            
            # Ensure all required fields are present for prediction
            required_fields = ["risk_probability", "risk_level", "confidence", "risk_factors", "recommendations"]
            for field in required_fields:
                if field not in result:
                    raise ValueError(f"Missing required field in prediction response: {field}")
            
            # Validate data types and ranges
            risk_prob = float(result["risk_probability"])
            confidence = float(result["confidence"])
            
            if not (0.0 <= risk_prob <= 1.0):
                raise ValueError(f"Risk probability must be between 0.0 and 1.0, got {risk_prob}")
            if not (0.0 <= confidence <= 1.0):
                raise ValueError(f"Confidence must be between 0.0 and 1.0, got {confidence}")
            
            if result["risk_level"] not in ["Low", "Medium", "High"]:
                raise ValueError(f"Risk level must be Low, Medium, or High, got {result['risk_level']}")
            
            return {
                "success": True,
                "risk_probability": risk_prob,
                "risk_level": result["risk_level"],
                "confidence": confidence,
                "risk_factors": result["risk_factors"],
                "recommendations": result["recommendations"],
                "llm_analysis": {
                    "analysis_available": True,
                    "summary": result.get("summary", "AI-generated health risk assessment"),
                    "key_risk_factors": result.get("key_risk_factors", result["risk_factors"][:5]),
                    "health_implications": result.get("health_implications", "Please consult with healthcare provider"),
                    "recommendations": result["recommendations"],
                    "urgency_level": result.get("urgency_level", "medium")
                }
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse prediction response: {response_text}")
            return {
                "success": False,
                "reason": "Invalid JSON response from AI service"
            }
        except Exception as e:
            logger.error(f"Error parsing prediction response: {str(e)}")
            return {
                "success": False,
                "reason": f"Error processing AI response: {str(e)}"
            }

    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """Parse the LLM response into a structured format"""
        try:
            # Clean the response (remove markdown code blocks if present)
            clean_text = response_text.strip()
            if '```json' in clean_text:
                clean_text = clean_text.split('```json')[1].split('```')[0].strip()
            elif '```' in clean_text:
                clean_text = clean_text.split('```')[1].split('```')[0].strip()
            
            # Try to parse the response as JSON
            result = json.loads(clean_text)
            
            # Ensure all required fields are present
            required_fields = ["summary", "key_risk_factors", "health_implications", "recommendations", "urgency_level"]
            for field in required_fields:
                if field not in result:
                    raise ValueError(f"Missing required field in LLM response: {field}")
            
            return {
                "analysis_available": True,
                **result
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {response_text}")
            return {
                "analysis_available": False,
                "reason": "Invalid JSON response from AI service"
            }
        except Exception as e:
            logger.error(f"Error parsing LLM response: {str(e)}")
            return {
                "analysis_available": False,
                "reason": f"Error processing AI response: {str(e)}"
            }

# Create a singleton instance
llm_analyzer = LLMRiskAnalyzer()
