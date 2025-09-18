"""
API utilities for frontend communication with backend services
"""
from typing import Dict, Any, Optional
import requests
import streamlit as st
import time
from .caching import cached_health_check, cached_model_info


class APIClient:
    """General API client for backend communication"""
    
    def __init__(self, base_url="http://localhost:8002"):
        self.base_url = base_url
        self.auth_headers = {}
    
    def set_auth_headers(self, headers):
        """Set authentication headers"""
        self.auth_headers = headers or {}


class HealthcareAPI:
    """Healthcare API client for making requests to the backend"""
    
    def __init__(self, base_url="http://localhost:8002/api"):
        self.base_url = base_url
        self.auth_headers = {}
    
    def set_auth_headers(self, headers):
        """Set authentication headers"""
        self.auth_headers = headers or {}
    
    def _make_request(self, method, endpoint, **kwargs):
        """Make HTTP request with comprehensive error handling and token refresh"""
        try:
            headers = {**self.auth_headers, "Content-Type": "application/json"}
            if 'headers' in kwargs:
                headers.update(kwargs['headers'])
            kwargs['headers'] = headers
            
            if 'timeout' not in kwargs:
                kwargs['timeout'] = 30
            
            url = f"{self.base_url}{endpoint}"
            response = requests.request(method, url, **kwargs)
            
            # Handle authentication errors specifically
            if response.status_code == 401:
                st.error("🔒 Authentication expired. Please log in again.")
                # Clear authentication state
                if 'authenticated' in st.session_state:
                    st.session_state.authenticated = False
                st.rerun()
                return None
            elif response.status_code == 403:
                st.error("🚫 Access denied. You don't have permission for this action.")
                return None
            
            response.raise_for_status()
            
            # Handle different content types
            content_type = response.headers.get('content-type', '')
            if 'application/json' in content_type:
                return response.json()
            else:
                return response.content
                
        except requests.exceptions.HTTPError as http_err:
            status_code = getattr(http_err.response, 'status_code', None)
            
            # Get detailed error information for debugging
            error_detail = None
            if hasattr(http_err, 'response') and http_err.response.text:
                try:
                    error_detail = http_err.response.json().get('detail', http_err.response.text)
                except:
                    error_detail = http_err.response.text[:200]
            
            # Provide user-friendly error messages based on status codes
            if status_code == 422:
                st.error("📝 Invalid data provided. Please check your input and try again.")
                if error_detail:
                    st.error(f"Details: {error_detail}")
            elif status_code == 404:
                st.error("🔍 Requested resource not found.")
            elif status_code == 500:
                st.error("🔧 Server error occurred. Please try again later.")
                if error_detail:
                    st.error(f"Error details: {error_detail}")
            else:
                error_msg = f"Request failed (HTTP {status_code})"
                if error_detail:
                    error_msg += f": {error_detail}"
                st.error(error_msg)
            return None
            
        except requests.exceptions.Timeout:
            st.error("⏱️ Request timed out. The server may be busy. Please try again.")
            return None
            
        except requests.exceptions.ConnectionError:
            st.error("🌐 Cannot connect to backend service. Please check if the server is running.")
            return None
            
        except requests.exceptions.RequestException as e:
            st.error(f"🔌 Network error: {str(e)}")
            return None
            
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")
            return None

    def make_lab_prediction(self, patient_data, force_llm=False):
        """Make API call to get lab-enhanced AI prediction"""
        filtered_data = dict(patient_data)
        filtered_data['force_llm'] = force_llm
        
        # Convert numeric fields to appropriate types
        numeric_fields = ['age', 'resting_bp', 'systolic_bp', 'diastolic_bp', 'cholesterol', 
                         'total_cholesterol', 'hdl_cholesterol', 'ldl_cholesterol',
                         'triglycerides', 'fasting_blood_sugar', 'max_heart_rate', 'ca',
                         'hemoglobin', 'total_leukocyte_count', 'red_blood_cell_count',
                         'hematocrit', 'platelet_count']
        
        for field in numeric_fields:
            if field in filtered_data and filtered_data[field] is not None:
                try:
                    filtered_data[field] = int(float(filtered_data[field]))
                except (ValueError, TypeError):
                    pass
        
        # Convert float fields including lab parameters
        float_fields = ['bmi', 'hba1c', 'oldpeak', 'weight', 'height',
                       'mean_corpuscular_volume', 'mean_corpuscular_hb', 'mean_corpuscular_hb_conc',
                       'red_cell_distribution_width', 'mean_platelet_volume', 'platelet_distribution_width',
                       'erythrocyte_sedimentation_rate', 'neutrophils_percent', 'lymphocytes_percent',
                       'monocytes_percent', 'eosinophils_percent', 'basophils_percent',
                       'absolute_neutrophil_count', 'absolute_lymphocyte_count', 'absolute_monocyte_count',
                       'absolute_eosinophil_count', 'absolute_basophil_count']
        for field in float_fields:
            if field in filtered_data and filtered_data[field] is not None:
                try:
                    filtered_data[field] = float(filtered_data[field])
                except (ValueError, TypeError):
                    pass
        
        # Map legacy field names
        if 'resting_bp' in filtered_data and 'systolic_bp' not in filtered_data:
            filtered_data['systolic_bp'] = filtered_data['resting_bp']
        if 'cholesterol' in filtered_data and 'total_cholesterol' not in filtered_data:
            filtered_data['total_cholesterol'] = filtered_data['cholesterol']
        
        with st.spinner("🔬 Analyzing lab data..."):
            result = self._make_request('POST', '/predict/lab', json=filtered_data)
            
        if result:
            result['powered_by_ai'] = False
            result['lab_enhanced'] = True
            result['analysis_type'] = 'Lab Analysis'
            
        return result

    def make_prediction(self, patient_data, force_llm=False):
        """Make API call to get AI-powered prediction"""
        filtered_data = dict(patient_data)
        filtered_data['force_llm'] = force_llm
        
        # Convert numeric fields
        numeric_fields = ['age', 'resting_bp', 'systolic_bp', 'diastolic_bp', 'cholesterol', 
                         'total_cholesterol', 'hdl_cholesterol', 'ldl_cholesterol',
                         'triglycerides', 'fasting_blood_sugar', 'max_heart_rate', 'ca']
        
        for field in numeric_fields:
            if field in filtered_data and filtered_data[field] is not None:
                try:
                    filtered_data[field] = int(float(filtered_data[field]))
                except (ValueError, TypeError):
                    pass
        
        # Convert float fields
        float_fields = ['bmi', 'hba1c', 'oldpeak', 'weight', 'height']
        for field in float_fields:
            if field in filtered_data and filtered_data[field] is not None:
                try:
                    filtered_data[field] = float(filtered_data[field])
                except (ValueError, TypeError):
                    pass
        
        # Map legacy field names
        if 'resting_bp' in filtered_data and 'systolic_bp' not in filtered_data:
            filtered_data['systolic_bp'] = filtered_data['resting_bp']
        if 'cholesterol' in filtered_data and 'total_cholesterol' not in filtered_data:
            filtered_data['total_cholesterol'] = filtered_data['cholesterol']
        
        with st.spinner("🤖 Analyzing with AI..."):
            result = self._make_request('POST', '/predict/simple', json=filtered_data, timeout=45)
            
        if result:
            result['powered_by_ai'] = True
            
        return result
    
    def check_backend_health(self):
        """Check if backend is healthy"""
        # Backend exposes health at /api/health
        return cached_health_check(self.base_url)
    
    def get_model_info(self):
        """Get model information from backend"""
        return cached_model_info(self.base_url)
    
    def reload_model(self):
        """Force reload the model on backend"""
        return self._make_request('GET', '/reload-model', timeout=10)
    
    # Medical Report Analysis Methods
    def upload_medical_report(self, file_data, patient_name, age: Optional[int] = None, weight: Optional[float] = None, condition: Optional[str] = None):
        """Upload and analyze medical report with optional patient context (age/weight/condition) for dosage planning"""
        try:
            headers = {**self.auth_headers}
            files = {"file": file_data}
            data = {"patient_name": patient_name}
            if age is not None:
                data["age"] = str(int(age))
            if weight is not None:
                data["weight"] = str(float(weight))
            if condition:
                data["condition"] = condition
            
            with st.spinner("📄 Uploading and analyzing medical report..."):
                response = requests.post(
                    f"{self.base_url}/medical-report/upload",
                    files=files,
                    data=data,
                    headers=headers,
                    timeout=120
                )
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            st.error(f"Error uploading medical report: {str(e)}")
            return None
    
    def get_analysis(self, analysis_id):
        """Get medical report analysis by ID"""
        return self._make_request('GET', f'/medical-report/analysis/{analysis_id}')
    
    def list_analyses(self, patient_name=None, limit=50):
        """List medical report analyses"""
        params = {"limit": limit}
        if patient_name:
            params["patient_name"] = patient_name
        return self._make_request('GET', '/medical-report/list', params=params)
    
    def download_report(self, analysis_id):
        """Download PDF report"""
        return self._make_request('GET', f'/medical-report/download/{analysis_id}', timeout=60)
    
    def delete_analysis(self, analysis_id):
        """Delete medical report analysis"""
        return self._make_request('DELETE', f'/medical-report/analysis/{analysis_id}')
    
    def get_lifestyle_recommendations(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """Get lifestyle recommendations for analysis
        
        Args:
            analysis_id: The ID of the analysis to get recommendations for
            
        Returns:
            Dict containing lifestyle recommendations or None if not available
        """
        try:
            response = self._make_request('GET', f'/medical-report/lifestyle-recommendations/{analysis_id}')
            if not response:
                return None
                
            # Transform the response to match the expected format
            return {
                'diet_recommendations': response.get('diet_recommendations', []),
                'exercise_recommendations': response.get('exercise_recommendations', []),
                'lifestyle_recommendations': response.get('lifestyle_recommendations', [])
            }
            
        except Exception as e:
            st.error(f"Error fetching lifestyle recommendations: {str(e)}")
            return None
    
    # Health Log Data Persistence Methods
    def save_health_data(self, health_data):
        """Save health log data to backend"""
        return self._make_request('POST', '/health-log/', json=health_data)
    
    def get_health_data(self, user_id=None, limit=100):
        """Get health log data from backend"""
        params = {"limit": limit}
        if user_id:
            params["user_id"] = user_id
        return self._make_request('GET', '/health-log/', params=params)
    
    def update_health_data(self, entry_id, health_data):
        """Update health log entry"""
        return self._make_request('PUT', f'/health-log/{entry_id}', json=health_data)
    
    def delete_health_data(self, entry_id):
        """Delete health log entry"""
        return self._make_request('DELETE', f'/health-log/{entry_id}')
    
    # User Management Methods (Admin)
    def get_users(self):
        """Get all users (admin only)"""
        return self._make_request('GET', '/auth/users')
    
    def create_user(self, user_data):
        """Create new user (admin only)"""
        return self._make_request('POST', '/auth/register', json=user_data)
    
    def update_user(self, user_id, user_data):
        """Update user (admin only)"""
        return self._make_request('PUT', f'/auth/users/{user_id}', json=user_data)
    
    def delete_user(self, user_id):
        """Delete user (admin only)"""
        return self._make_request('DELETE', f'/auth/users/{user_id}')
    
    # Chatbot Methods
    def send_chat_message(self, message, context=None):
        """Send message to chatbot"""
        data = {"message": message}
        if context:
            data["context"] = context
        return self._make_request('POST', '/chat', json=data)
    
    def explain_risk(self, prediction_data):
        """Get risk explanation from chatbot"""
        return self._make_request('POST', '/chat/explain-risk', json=prediction_data)
    
    # Prescription Analysis Methods
    def upload_prescription(self, file_data, patient_name):
        """Upload and analyze prescription"""
        try:
            headers = {**self.auth_headers}
            files = {"file": file_data}
            data = {"patient_name": patient_name}
            
            with st.spinner("💊 Analyzing prescription..."):
                response = requests.post(
                    f"{self.base_url}/prescription/upload",
                    files=files,
                    data=data,
                    headers=headers,
                    timeout=60
                )
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            st.error(f"Error uploading prescription: {str(e)}")
            return None
    
    def get_medicine_info(self, medicine_name):
        """Get medicine information"""
        return self._make_request('POST', '/prescription/medicine-info', json={"medicine_name": medicine_name})

    # Drug Advisor Endpoints
    def extract_drug_info(self, text: str):
        """Extract structured drug entities from free text using backend drug advisor"""
        # Loading a Transformers model may take longer on first call; allow a higher timeout
        return self._make_request('POST', '/drug-advisor/extract_drug_info', json={"text": text}, timeout=120)

    def check_interactions(self, drugs):
        """Check pairwise drug interactions for a list of drug names"""
        return self._make_request('POST', '/drug-advisor/check_interactions', json={"drugs": drugs})

    def dosage_recommendation(self, drug: str, age: int, weight: Optional[float] = None, condition: Optional[str] = None):
        """Get age-specific dosage recommendation for a drug with optional weight and condition

        Args:
            drug: Drug name
            age: Age in years
            weight: Optional weight in kg (for pediatric weight-based dosing)
            condition: Optional condition string for context-specific warnings
        """
        payload = {"drug": drug, "age": age}
        if weight is not None:
            payload["weight"] = weight
        if condition:
            payload["condition"] = condition
        return self._make_request('POST', '/drug-advisor/dosage_recommendation', json=payload)

    def alternative_suggestions(self, drug: str):
        """Get safer/equivalent alternative medication suggestions"""
        return self._make_request('POST', '/drug-advisor/alternative_suggestions', json={"drug": drug})