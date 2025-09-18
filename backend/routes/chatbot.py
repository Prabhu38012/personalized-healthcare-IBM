from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
import os
import json
import logging
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter()

# Pydantic models for request/response
class ChatMessage(BaseModel):
    role: str = Field(..., description="Role: 'user', 'assistant', or 'system'")
    content: str = Field(..., description="Message content")
    timestamp: Optional[datetime] = None

class ChatRequest(BaseModel):
    message: str = Field(..., description="User's message")
    conversation_history: Optional[List[ChatMessage]] = Field(default=[], description="Previous conversation")
    patient_context: Optional[Dict[str, Any]] = Field(default=None, description="Current patient data context")
    include_health_data: Optional[bool] = Field(default=False, description="Include patient health data in context")

class ChatResponse(BaseModel):
    response: str = Field(..., description="Chatbot response")
    conversation_id: Optional[str] = None
    suggestions: Optional[List[str]] = Field(default=[], description="Suggested follow-up questions")
    requires_medical_attention: Optional[bool] = Field(default=False, description="Whether response suggests medical consultation")

# Initialize Groq API client
def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_api_key_here" or api_key == "REPLACE_WITH_YOUR_ACTUAL_GROQ_API_KEY":
        raise HTTPException(
            status_code=500, 
            detail="Groq API key not configured. Please set GROQ_API_KEY environment variable with a valid Groq API key from https://console.groq.com/keys"
        )
    
    # Validate API key format (Groq API keys start with 'gsk_')
    if not api_key.startswith('gsk_') or len(api_key) < 20:
        raise HTTPException(
            status_code=500,
            detail="Invalid Groq API key format. Please check your API key from https://console.groq.com/keys"
        )
    
    # Initialize OpenAI client configured for Groq API
    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1"
        )
        return client
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize Groq API client: {str(e)}"
        )

# System prompts for the healthcare chatbot using Groq
HEALTHCARE_SYSTEM_PROMPT = """
You are an intelligent healthcare assistant AI designed to help users understand their health data and provide general health information. You are powered by Groq's fast inference technology, allowing for quick and accurate responses.

IMPORTANT GUIDELINES:
1. You are NOT a replacement for professional medical advice, diagnosis, or treatment
2. Always encourage users to consult with healthcare professionals for serious concerns
3. Provide educational information about health conditions, symptoms, and general wellness
4. Help explain medical terminology and health metrics in simple terms
5. Offer evidence-based lifestyle and wellness recommendations
6. When discussing risk assessments, explain what the numbers mean and their implications
7. Be empathetic and supportive while maintaining accuracy
8. If asked about specific medications or treatments, always recommend consulting a healthcare provider

RESPONSE STYLE:
- Clear and conversational
- Use simple language to explain complex medical concepts
- Provide actionable insights when appropriate
- Include disclaimers about medical advice when necessary
- Be encouraging and supportive

MEDICAL DISCLAIMER:
Remember to include appropriate medical disclaimers when discussing health conditions or providing health advice.
"""

def create_healthcare_context(patient_data: Optional[Dict[str, Any]] = None) -> str:
    """Create context string from patient health data"""
    if not patient_data:
        return ""
    
    context_parts = ["Current patient health data context:"]
    
    # Basic demographics
    if patient_data.get('age'):
        context_parts.append(f"- Age: {patient_data['age']} years")
    if patient_data.get('sex'):
        context_parts.append(f"- Sex: {patient_data['sex']}")
    
    # Vital signs
    if patient_data.get('systolic_bp') or patient_data.get('resting_bp'):
        bp = patient_data.get('systolic_bp') or patient_data.get('resting_bp', 0)
        context_parts.append(f"- Blood Pressure: {bp} mmHg (systolic)")
    
    if patient_data.get('diastolic_bp'):
        context_parts.append(f"- Diastolic BP: {patient_data['diastolic_bp']} mmHg")
    
    if patient_data.get('cholesterol'):
        context_parts.append(f"- Cholesterol: {patient_data['cholesterol']} mg/dL")
    
    if patient_data.get('max_heart_rate'):
        context_parts.append(f"- Max Heart Rate: {patient_data['max_heart_rate']} bpm")
    
    # Physical measurements
    if patient_data.get('height'):
        context_parts.append(f"- Height: {patient_data['height']} cm")
    if patient_data.get('weight'):
        context_parts.append(f"- Weight: {patient_data['weight']} kg")
    
    # Risk assessment if available
    if patient_data.get('risk_level'):
        context_parts.append(f"- Current Risk Assessment: {patient_data['risk_level']} risk")
    if patient_data.get('risk_probability'):
        prob = float(patient_data['risk_probability']) * 100
        context_parts.append(f"- Risk Probability: {prob:.1f}%")
    
    return "\n".join(context_parts)

def generate_follow_up_suggestions(response: str, patient_context: Optional[Dict] = None) -> List[str]:
    """Generate contextual follow-up suggestions based on response and patient data"""
    suggestions = []
    
    # General health suggestions
    base_suggestions = [
        "What does my risk assessment mean?",
        "How can I improve my cardiovascular health?",
        "What lifestyle changes would you recommend?",
        "Can you explain my health metrics?",
    ]
    
    # Context-specific suggestions
    if patient_context:
        if patient_context.get('risk_level') == 'High':
            suggestions.extend([
                "What should I do about my high risk level?",
                "When should I see a doctor?",
                "What are the warning signs to watch for?"
            ])
        elif patient_context.get('risk_level') == 'Medium':
            suggestions.extend([
                "How can I reduce my risk level?",
                "What preventive measures should I take?",
                "How often should I monitor my health?"
            ])
        
        # Blood pressure specific
        bp = patient_context.get('systolic_bp') or patient_context.get('resting_bp', 0)
        if bp > 140:
            suggestions.append("What can I do about high blood pressure?")
        
        # Cholesterol specific  
        if patient_context.get('cholesterol', 0) > 200:
            suggestions.append("How can I manage my cholesterol levels?")
    
    # Return a mix of base and context suggestions
    all_suggestions = list(set(base_suggestions + suggestions))
    return all_suggestions[:4]  # Limit to 4 suggestions

@router.post("/chat", response_model=ChatResponse)
async def chat_with_assistant(request: ChatRequest):
    """
    Main chat endpoint for the healthcare assistant using Groq AI
    """
    try:
        client = get_groq_client()
        
        # Prepare conversation messages
        messages: List[ChatCompletionMessageParam] = [{"role": "system", "content": HEALTHCARE_SYSTEM_PROMPT}]
        
        # Add patient context if provided
        if request.include_health_data and request.patient_context:
            health_context = create_healthcare_context(request.patient_context)
            if health_context:
                context_message = f"\n\n{health_context}\n\nPlease use this health data context when relevant to answer the user's questions."
                messages[0] = {"role": "system", "content": HEALTHCARE_SYSTEM_PROMPT + context_message}
        
        # Add conversation history
        if request.conversation_history:
            for msg in request.conversation_history[-10:]:  # Limit to last 10 messages
                # Ensure proper typing for message roles
                if msg.role == "user":
                    messages.append({"role": "user", "content": msg.content})
                elif msg.role == "assistant":
                    messages.append({"role": "assistant", "content": msg.content})
                elif msg.role == "system":
                    messages.append({"role": "system", "content": msg.content})
        
        # Add current user message
        messages.append({"role": "user", "content": request.message})
        
        # Call Groq API - using a current model
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # Groq's current Llama model
            messages=messages,
            max_tokens=500,
            temperature=0.7,
            stream=False
        )
        
        assistant_response = response.choices[0].message.content or ""
        
        # Check if response suggests medical attention
        requires_attention = any(phrase in assistant_response.lower() for phrase in [
            "see a doctor", "consult", "medical attention", "healthcare provider",
            "emergency", "urgent", "immediately", "serious"
        ])
        
        # Generate follow-up suggestions
        suggestions = generate_follow_up_suggestions(assistant_response, request.patient_context)
        
        return ChatResponse(
            response=assistant_response,
            suggestions=suggestions,
            requires_medical_attention=requires_attention
        )
        
    except Exception as e:
        logger.error(f"Groq API error: {e}")
        raise HTTPException(status_code=500, detail=f"AI service error: {str(e)}")

@router.get("/chat/health")
async def chatbot_health():
    """Health check endpoint for the chatbot service using Groq"""
    try:
        # Test Groq API connection
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key or api_key == "REPLACE_WITH_YOUR_ACTUAL_GROQ_API_KEY":
            return {
                "status": "error", 
                "message": "Groq API key not configured. Please set a valid API key from https://console.groq.com/keys",
                "groq_available": False
            }
        
        return {
            "status": "healthy",
            "message": "Chatbot service is running with Groq AI",
            "groq_available": True,
            "model": "llama-3.3-70b-versatile"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Chatbot service error: {str(e)}",
            "groq_available": False
        }

@router.post("/chat/explain-risk")
async def explain_risk_assessment(request: Dict[str, Any]):
    """
    Specialized endpoint to explain risk assessment results using Groq AI
    """
    try:
        client = get_groq_client()
        
        risk_level = request.get('risk_level', 'Unknown')
        risk_probability = request.get('risk_probability', 0)
        patient_data = request.get('patient_data', {})
        
        # Create explanation prompt
        prompt = f"""
        Explain this cardiovascular risk assessment result to a patient in simple, clear terms:
        
        Risk Level: {risk_level}
        Risk Probability: {risk_probability:.1%}
        
        {create_healthcare_context(patient_data)}
        
        Please explain:
        1. What this risk level means
        2. What factors contributed to this assessment
        3. What actions they should consider
        4. When they should seek medical advice
        
        Keep the explanation encouraging and actionable while being medically accurate.
        """
        
        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": HEALTHCARE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=400,
            temperature=0.6
        )
        
        return {"explanation": response.choices[0].message.content or ""}
        
    except Exception as e:
        logger.error(f"Risk explanation error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to explain risk assessment: {str(e)}")