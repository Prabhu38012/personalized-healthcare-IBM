"""
Medical Report Analysis Page
Upload and analyze medical reports with comprehensive results display
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io
import base64
from typing import Dict, List, Any, Optional

# Import utilities
try:
    from frontend.utils.api_client import HealthcareAPI
    from frontend.utils.caching import cache_data
    from frontend.components.auth import get_auth_headers
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from utils.api_client import HealthcareAPI
    from utils.caching import cache_data
    from components.auth import get_auth_headers

class MedicalReportAnalyzer:
    """Frontend class for medical report analysis"""
    
    def __init__(self):
        self.api_client = HealthcareAPI()
        # Set authentication headers
        self.api_client.set_auth_headers(get_auth_headers())
    
    def upload_and_analyze_report(self, uploaded_file, patient_name: str, age: Optional[int] = None, weight: Optional[float] = None, condition: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Upload and analyze medical report using the integrated API client
        Accepts optional age/weight/condition to enable dosage planning in backend.
        """
        return self.api_client.upload_medical_report(uploaded_file, patient_name, age=age, weight=weight, condition=condition)
    
    def get_analysis_list(self, limit: int = 10, patient_name: str = None) -> Optional[Dict[str, Any]]:
        """Get list of previous analyses using integrated API client"""
        return self.api_client.list_analyses(patient_name, limit)
    
    def get_analysis_details(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed analysis results using integrated API client"""
        return self.api_client.get_analysis(analysis_id)
    
    def download_report(self, analysis_id: str) -> Optional[bytes]:
        """Download PDF report using integrated API client"""
        return self.api_client.download_report(analysis_id)
    
    def delete_analysis(self, analysis_id: str) -> bool:
        """Delete analysis using integrated API client"""
        result = self.api_client.delete_analysis(analysis_id)
        return result is not None
    
    def get_lifestyle_recommendations(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """Get lifestyle recommendations using integrated API client"""
        return self.api_client.get_lifestyle_recommendations(analysis_id)
    
    def download_pdf_report(self, analysis_id: str, patient_name: str):
        """Download PDF report using integrated API client"""
        try:
            pdf_content = self.api_client.download_report(analysis_id)
            
            if pdf_content:
                # Create download button
                st.download_button(
                    label="📄 Download PDF Report",
                    data=pdf_content,
                    file_name=f"medical_report_{patient_name}_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            else:
                st.error("Failed to generate PDF report")
                
        except Exception as e:
            st.error(f"PDF download failed: {str(e)}")
    
    def display_confidence_meter(self, confidence: float):
        """Display confidence score as a meter"""
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=confidence * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Analysis Confidence"},
            delta={'reference': 80},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    def display_summary_metrics(self, summary: Dict[str, Any]):
        """Display summary metrics in columns"""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Conditions Found", summary.get('conditions_found', 0))
            st.metric("Lab Values", summary.get('lab_values_extracted', 0))
        
        with col2:
            st.metric("Medications", summary.get('medications_identified', 0))
            st.metric("Future Risks", summary.get('future_risks', 0))
        
        with col3:
            st.metric("Symptoms", summary.get('symptoms_noted', 0))
            st.metric("Recommendations", summary.get('recommendations', 0))
    
    def display_conditions_analysis(self, conditions: List[Dict[str, Any]]):
        """Display medical conditions analysis"""
        if not conditions:
            st.info("No medical conditions were identified in the report.")
            return
        
        st.subheader("🏥 Medical Conditions Identified")
        
        # Create DataFrame for better display
        conditions_df = pd.DataFrame([
            {
                'Condition': condition['text'],
                'Type': condition['type'].replace('CONDITION_', '').replace('_', ' ').title(),
                'Confidence': f"{condition['confidence']:.1%}",
                'Context Preview': condition['context'][:100] + "..." if len(condition['context']) > 100 else condition['context']
            }
            for condition in conditions
        ])
        
        st.dataframe(conditions_df, use_container_width=True)
        
        # Condition type distribution
        if len(conditions) > 1:
            type_counts = pd.DataFrame([
                condition['type'].replace('CONDITION_', '').replace('_', ' ').title() 
                for condition in conditions
            ], columns=['Type']).value_counts().reset_index()
            
            fig = px.pie(type_counts, values='count', names='Type', 
                        title="Distribution of Condition Types")
            st.plotly_chart(fig, use_container_width=True)
    
    def display_medications_analysis(self, medications: List[Dict[str, Any]]):
        """Display medications analysis"""
        if not medications:
            st.info("No medications were identified in the report.")
            return
        
        st.subheader("💊 Medications and Treatments")
        
        # Create DataFrame
        medications_df = pd.DataFrame([
            {
                'Medication/Treatment': med['text'],
                'Confidence': f"{med['confidence']:.1%}",
                'Context Preview': med['context'][:100] + "..." if len(med['context']) > 100 else med['context']
            }
            for med in medications
        ])
        
        st.dataframe(medications_df, use_container_width=True)
    
    def display_lab_values_analysis(self, lab_values: List[Dict[str, Any]]):
        """Display lab values analysis"""
        if not lab_values:
            st.info("No laboratory values were identified in the report.")
            return
        
        st.subheader("🔬 Laboratory Values")
        
        # Create DataFrame
        lab_df = pd.DataFrame([
            {
                'Test/Parameter': lab['test'],
                'Value': lab['value'],
                'Context Preview': lab['context'][:80] + "..." if len(lab['context']) > 80 else lab['context']
            }
            for lab in lab_values
        ])
        
        st.dataframe(lab_df, use_container_width=True)
    
    def display_risk_assessment(self, future_risks: List[str]):
        """Display future risk assessment"""
        if not future_risks:
            st.info("No specific future health risks were identified.")
            return
        
        st.subheader("⚠️ Future Health Risk Assessment")
        
        st.warning("""
        **Important:** The following risk assessments are based on statistical correlations and medical literature. 
        They are not definitive predictions and should be discussed with your doctor.
        """)
        
        for i, risk in enumerate(future_risks, 1):
            st.write(f"**{i}.** {risk}")
    
    def display_recommendations(self, recommendations: List[str]):
        """Display health recommendations"""
        if not recommendations:
            st.info("No specific recommendations were generated.")
            return
        
        st.subheader("💡 Health Recommendations")
        
        st.success("""
        **Note:** These recommendations are based on identified conditions and established medical guidelines. 
        Please consult with your doctor before making any changes to your treatment plan.
        """)
        
        for i, recommendation in enumerate(recommendations, 1):
            st.write(f"**{i}.** {recommendation}")
    
    def _get_risk_based_recommendations(self, future_risks: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """Generate comprehensive diet and exercise recommendations based on future health risks"""
        recommendations = {
            'diet_recommendations': [],
            'exercise_recommendations': []
        }
        
        # Risk to condition mapping with specific risk factors
        risk_mapping = {
            'diabetes': {
                'keywords': ['diabetes', 'prediabetes', 'insulin resistance', 'high blood sugar', 'hyperglycemia'],
                'description': 'Based on your risk of developing diabetes, these recommendations focus on blood sugar control and metabolic health.'
            },
            'heart_disease': {
                'keywords': ['heart disease', 'cardiovascular disease', 'heart attack', 'stroke', 'coronary artery'],
                'description': 'These recommendations aim to improve heart health and reduce cardiovascular risk factors.'
            },
            'hypertension': {
                'keywords': ['hypertension', 'high blood pressure', 'elevated bp'],
                'description': 'Focused on reducing blood pressure through diet and lifestyle modifications.'
            },
            'obesity': {
                'keywords': ['obesity', 'weight gain', 'overweight', 'metabolic syndrome'],
                'description': 'Designed to support healthy weight management and metabolic health.'
            },
            'osteoporosis': {
                'keywords': ['osteoporosis', 'bone loss', 'fracture risk', 'low bone density'],
                'description': 'Focused on improving bone health and reducing fracture risk.'
            },
            'cognitive_decline': {
                'keywords': ['dementia', 'alzheimer', 'cognitive decline', 'memory loss'],
                'description': 'Aimed at supporting brain health and cognitive function.'
            }
        }
        
        # Map risks to specific conditions
        risk_conditions = {}
        for condition, data in risk_mapping.items():
            for risk in future_risks:
                if any(keyword in risk.lower() for keyword in data['keywords']):
                    risk_conditions[condition] = {
                        'description': data['description'],
                        'specific_risks': [r for r in future_risks if any(kw in r.lower() for kw in data['keywords'])]
                    }
                    break
        
        # Generate recommendations for each identified risk condition
        for condition, data in risk_conditions.items():
            # Get base recommendations
            diet_recs = self._generate_diet_recommendations([condition])
            exercise_recs = self._generate_exercise_recommendations([condition])
            
            # Add risk-specific context
            if diet_recs:
                for rec in diet_recs:
                    rec['risk_based'] = True
                    rec['risk_description'] = data['description']
                    rec['specific_risks'] = data['specific_risks']
                recommendations['diet_recommendations'].extend(diet_recs)
            
            if exercise_recs:
                for rec in exercise_recs:
                    rec['risk_based'] = True
                    rec['risk_description'] = data['description']
                    rec['specific_risks'] = data['specific_risks']
                recommendations['exercise_recommendations'].extend(exercise_recs)
        
        # Add general preventive recommendations if no specific risks identified
        if not risk_conditions and future_risks:
            general_risk_desc = "Based on your health profile, these general preventive recommendations may help reduce your future health risks."
            
            general_diet = {
                'category': 'General Preventive Nutrition',
                'description': general_risk_desc,
                'risk_based': True,
                'specific_risks': future_risks,
                'foods_to_include': [
                    'Colorful fruits and vegetables (at least 5 servings daily)',
                    'Whole grains (brown rice, quinoa, whole wheat)',
                    'Lean proteins (fish, poultry, legumes)',
                    'Healthy fats (avocados, nuts, olive oil)'
                ],
                'foods_to_avoid': [
                    'Processed and fried foods',
                    'Added sugars and sugary beverages',
                    'Excessive alcohol',
                    'Trans fats and hydrogenated oils'
                ],
                'meal_timing': [
                    'Eat regular, balanced meals',
                    'Avoid late-night snacking',
                    'Stay hydrated throughout the day'
                ]
            }
            
            general_exercise = {
                'category': 'General Preventive Exercise',
                'description': general_risk_desc,
                'risk_based': True,
                'specific_risks': future_risks,
                'recommended_activities': [
                    'Brisk walking (30 minutes most days)',
                    'Strength training (2-3 times weekly)',
                    'Flexibility exercises (daily stretching or yoga)'
                ],
                'frequency': '5-7 days per week',
                'duration': '30-60 minutes per session',
                'intensity': 'Moderate (able to talk but not sing during activity)',
                'benefits': [
                    'Improves cardiovascular health',
                    'Helps maintain healthy weight',
                    'Reduces stress and improves mood',
                    'Strengthens muscles and bones',
                    'Boosts energy levels'
                ]
            }
            
            recommendations['diet_recommendations'].append(general_diet)
            recommendations['exercise_recommendations'].append(general_exercise)
        
        return recommendations

    def _generate_exercise_recommendations(self, conditions: List[str]) -> List[Dict[str, Any]]:
        """Generate exercise recommendations based on conditions.
        This complements _generate_diet_recommendations and is used by risk-based planner.
        """
        recs: List[Dict[str, Any]] = []
        lower = [c.lower() for c in conditions]

        # Cardio-metabolic focus for diabetes/prediabetes
        if any(c in lower for c in ["diabetes", "prediabetes", "insulin resistance", "metabolic syndrome"]):
            recs.append({
                'category': 'Cardio & Strength for Glycemic Control',
                'recommended_activities': [
                    'Brisk walking 30–45 min/day',
                    'Cycling or swimming (low-impact) 3–5x/week',
                    'Resistance training 2–3x/week (full-body)'
                ],
                'frequency': '5–7 days/week aerobic; 2–3 days/week strength',
                'duration': '150+ minutes/week moderate aerobic',
                'intensity': 'Moderate (can talk, not sing)',
                'precautions': [
                    'Monitor glucose before/after sessions',
                    'Carry fast-acting carbohydrate source',
                    'Proper footwear and hydration'
                ],
                'progression_tips': [
                    'Start with 10–15 min and add 5 min/week',
                    'Prioritize consistency over intensity'
                ]
            })

        # Blood pressure focused (DASH-aligned exercise)
        if any(c in lower for c in ["hypertension", "high blood pressure"]):
            recs.append({
                'category': 'BP-Friendly Exercise Plan',
                'recommended_activities': [
                    'Walking or light jogging',
                    'Elliptical or swimming',
                    'Light–moderate resistance training'
                ],
                'frequency': '5–7 days/week aerobic; 2 days/week strength',
                'duration': '30–45 min/session',
                'intensity': 'Light to moderate',
                'precautions': [
                    'Avoid Valsalva (breath holding) during lifts',
                    'Gradual warm-up and cool-down'
                ]
            })

        # General plan if nothing specific
        if not recs:
            recs.append({
                'category': 'General Fitness Plan',
                'recommended_activities': [
                    '150 minutes/week moderate aerobic (e.g., brisk walking)',
                    '2+ strength sessions/week',
                    'Flexibility & balance 2–3x/week'
                ],
                'frequency': 'Most days',
                'duration': '30–45 min/session',
                'intensity': 'Moderate',
                'precautions': [
                    'Progress gradually',
                    'Stop if chest pain, dizziness, or severe shortness of breath'
                ]
            })

        return recs
    
    def display_lifestyle_plan(self, analysis_data: Dict[str, Any]):
        """Display comprehensive lifestyle recommendations based on health conditions and future risks"""
        st.markdown("### 🏃‍♀️ Personalized Lifestyle Plan")
        
        # Get conditions and future risks from analysis data
        conditions = [c['text'].lower() for c in analysis_data.get('conditions', [])]
        future_risks = analysis_data.get('future_risks', [])
        
        # Show loading state
        with st.spinner("Generating personalized lifestyle recommendations..."):
            # Try to get lifestyle recommendations from API first
            analysis_id = analysis_data.get('analysis_id')
            lifestyle_data = None
            
            if analysis_id:
                lifestyle_data = self.get_lifestyle_recommendations(analysis_id)
            
            # If no data from API, generate from analysis
            if not lifestyle_data:
                # Generate base recommendations from conditions
                base_diet = self._generate_diet_recommendations(conditions) if conditions else []
                base_exercise = self._generate_exercise_recommendations(conditions) if conditions else []
                base_lifestyle = self._generate_general_recommendations(conditions) if conditions else []
                
                # Generate risk-based recommendations
                risk_recommendations = self._get_risk_based_recommendations(future_risks) if future_risks else {}
                
                lifestyle_data = {
                    'diet_recommendations': base_diet + risk_recommendations.get('diet_recommendations', []),
                    'exercise_recommendations': base_exercise + risk_recommendations.get('exercise_recommendations', []),
                    'lifestyle_recommendations': base_lifestyle,
                    'priority_actions': [],
                    'professional_consultations': []
                }
            
            # Ensure all required keys exist in lifestyle_data
            if 'priority_actions' not in lifestyle_data:
                lifestyle_data['priority_actions'] = []
            if 'professional_consultations' not in lifestyle_data:
                lifestyle_data['professional_consultations'] = []
            
            if not lifestyle_data or not any(lifestyle_data.values()):
                st.warning("Unable to generate detailed lifestyle recommendations. Showing general guidelines.")
                self.display_fallback_lifestyle_recommendations()
                return
            
            # Add priority actions based on conditions
            if any(cond in conditions for cond in ['diabetes', 'prediabetes']):
                lifestyle_data['priority_actions'].extend([
                    "Monitor blood sugar levels regularly",
                    "Maintain consistent meal timing"
                ])
            if any(cond in conditions for cond in ['hypertension', 'high blood pressure']):
                lifestyle_data['priority_actions'].extend([
                    "Monitor blood pressure regularly",
                    "Reduce sodium intake"
                ])
            if any(cond in conditions for cond in ['heart disease', 'chd', 'cad']):
                lifestyle_data['priority_actions'].extend([
                    "Follow a heart-healthy diet",
                    "Engage in regular physical activity as tolerated"
                ])
            
            # Display recommendations in tabs
            tab1, tab2, tab3 = st.tabs(["🥗 Diet & Nutrition", "🏃‍♂️ Exercise Plan", "🧘‍♀️ Wellness & Monitoring"])
            
            with tab1:
                self._display_diet_plan(lifestyle_data.get('diet_recommendations', []))
            
            with tab2:
                self._display_exercise_plan(lifestyle_data.get('exercise_recommendments', []))
            
            with tab3:
                if lifestyle_data.get('lifestyle_recommendations'):
                    with st.expander("🌿 General Wellness", expanded=True):
                        self._display_wellness_plan(lifestyle_data['lifestyle_recommendations'])
                
                if lifestyle_data.get('priority_actions'):
                    with st.expander("🎯 Priority Actions", expanded=True):
                        for action in lifestyle_data['priority_actions']:
                            st.markdown(f"- ✅ {action}")
                
                self.display_monitoring_recommendations(lifestyle_data)
                
                # Add a note about consulting healthcare providers
                st.markdown("""
                <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; margin-top: 20px;">
                    <p style="font-size: 0.9em; margin-bottom: 0;">
                        <strong>Note:</strong> These recommendations are general guidelines. Please consult with your 
                        healthcare provider before making any significant changes to your diet or exercise routine, 
                        especially if you have any medical conditions or concerns.
                    </p>
                </div>
                """, unsafe_allow_html=True)
    
    def _generate_diet_recommendations(self, conditions: List[str]) -> List[Dict[str, Any]]:
        """Generate diet recommendations based on conditions"""
        recommendations = []
        
        # Heart Health
        heart_conditions = ['hypertension', 'high blood pressure', 'heart disease', 'high cholesterol', 'cad', 'chd']
        if any(cond.lower() in heart_conditions for cond in conditions):
            recommendations.append({
                'category': 'Blood Sugar Management',
                'foods_to_include': [
                    'Non-starchy vegetables (broccoli, spinach, peppers)',
                    'High-fiber foods (beans, lentils, whole grains)',
                    'Lean proteins (chicken, turkey, fish, tofu)',
                    'Healthy fats (avocados, nuts, olive oil)',
                    'Low-glycemic fruits (berries, apples, pears)'
                ],
                'foods_to_avoid': [
                    'Sugary drinks and fruit juices',
                    'White bread, white rice, and pasta',
                    'Processed snacks and sweets',
                    'Fried foods',
                    'High-sugar fruits (grapes, mangoes, bananas in excess)'
                ],
                'nutritional_goals': [
                    'Balance carbohydrates with protein and healthy fats',
                    'Eat smaller, more frequent meals',
                    'Monitor carbohydrate intake',
                    'Stay hydrated with water and unsweetened beverages'
                ]
            })
        
        # General healthy eating (always included)
        recommendations.append({
            'category': 'General Healthy Eating',
            'foods_to_include': [
                'Colorful fruits and vegetables',
                'Whole grains',
                'Lean proteins',
                'Healthy fats',
                'Low-fat dairy or alternatives'
            ],
            'foods_to_avoid': [
                'Processed and packaged foods',
                'Added sugars',
                'Excessive alcohol',
                'Trans fats',
                'High-sodium foods'
            ],
            'nutritional_goals': [
                'Eat a variety of foods from all food groups',
                'Stay hydrated with water',
                'Practice portion control',
                'Limit added sugars and salt'
            ]
        })
        
        # Diabetes/Blood Sugar Management
        if any(cond.lower() in ['diabetes', 'prediabetes', 'insulin resistance', 'metabolic syndrome'] for cond in conditions):
            recommendations.append({
                'category': 'Blood Sugar Management',
                'foods_to_include': [
                    'Non-starchy vegetables (broccoli, spinach, peppers, zucchini)',
                    'High-fiber foods (beans, lentils, whole grains, chia seeds)',
                    'Lean proteins (chicken, turkey, fish, eggs, tofu)',
                    'Healthy fats (avocados, nuts, seeds, olive oil)',
                    'Low-glycemic fruits (berries, apples, pears, citrus)'
                ],
                'foods_to_avoid': [
                    'Sugary beverages (sodas, fruit juices, sweet tea)',
                    'Refined carbohydrates (white bread, white rice, pastries)',
                    'Processed snacks and desserts',
                    'Fried foods and trans fats',
                    'High-sugar fruits (mangoes, grapes, bananas in excess)'
                ],
                'meal_pattern': [
                    'Eat at regular intervals (every 3-4 hours)',
                    'Balance each meal with protein, healthy fats, and fiber',
                    'Consider smaller, more frequent meals if needed',
                    'Avoid skipping meals to prevent blood sugar fluctuations'
                ]
            })
            
            # Add specific exercise recommendations for diabetes management
            recommendations.append({
                'category': 'Exercise for Blood Sugar Control',
                'recommended_activities': [
                    'Brisk walking (30 minutes daily)',
                    'Resistance training (2-3 times per week)',
                    'Yoga or tai chi for stress reduction',
                    'Swimming or water aerobics (low-impact option)'
                ],
                'intensity': 'Moderate to vigorous (as tolerated)',
                'precautions': [
                    'Check blood sugar before and after exercise',
                    'Have a fast-acting carbohydrate source available',
                    'Wear proper footwear and check feet daily',
                    'Stay well-hydrated with water',
                    'Avoid exercise if blood sugar is very high (>250 mg/dL with ketones) or very low (<100 mg/dL)'
                ]
            })
        
        # General fitness (always included)
        recommendations.append({
            'category': 'General Fitness',
            'recommended_activities': [
                '150 minutes of moderate aerobic activity per week',
                'Muscle-strengthening activities 2+ days per week',
                'Flexibility exercises 2-3 times per week',
                'Balance exercises (especially for older adults)'
            ],
            'intensity': 'Varies by activity and fitness level',
            'precautions': [
                'Start slowly and gradually increase intensity',
                'Listen to your body and rest when needed',
                'Stay hydrated before, during, and after exercise',
                'Use proper form to prevent injury',
                'Consult a doctor before starting a new exercise program'
            ]
        })
        
        return recommendations
    
    def _generate_general_recommendations(self, conditions: List[str]) -> List[Dict[str, Any]]:
        """Generate general lifestyle recommendations"""
        recommendations = []
        
        # Stress management
        recommendations.append({
            'category': 'Stress Management',
            'recommendations': [
                'Practice deep breathing exercises daily',
                'Try meditation or mindfulness',
                'Get 7-9 hours of quality sleep',
                'Maintain social connections',
                'Set aside time for hobbies and relaxation'
            ]
        })
        
        # Smoking cessation
        if any(cond.lower() in ['copd', 'asthma', 'heart disease', 'high blood pressure'] for cond in conditions):
            recommendations.append({
                'category': 'Smoking Cessation',
                'recommendations': [
                    'Talk to your doctor about smoking cessation programs',
                    'Consider nicotine replacement therapy if needed',
                    'Avoid secondhand smoke',
                    'Seek support from friends and family',
                    'Use stress management techniques to cope with cravings'
                ]
            })
        
        # Regular health monitoring
        monitoring = {
            'category': 'Health Monitoring',
            'recommendations': [
                'Schedule regular check-ups with your doctor',
                'Monitor your blood pressure at home if recommended',
                'Keep track of your symptoms and any changes'
            ]
        }
        
        if any(cond.lower() in ['diabetes', 'prediabetes'] for cond in conditions):
            monitoring['recommendations'].append('Regularly monitor your blood sugar as directed')
        
        if any(cond.lower() in ['high blood pressure', 'heart disease'] for cond in conditions):
            monitoring['recommendations'].append('Monitor your blood pressure regularly')
        
        recommendations.append(monitoring)
        
        return recommendations
    
    def _display_diet_plan(self, recommendations: List[Dict[str, Any]]):
        """Display the diet plan with improved formatting and risk-based recommendations"""
        if not recommendations:
            # Show default healthy eating guidelines if no specific recommendations
            st.info("## 🍽️ General Healthy Eating Guidelines")
            st.markdown("""
            While we don't have specific dietary recommendations based on your current health data, 
            here are some general healthy eating guidelines:
            
            - **Eat a variety** of fruits and vegetables (aim for 5+ servings per day)
            - Choose **whole grains** over refined grains
            - Include **lean proteins** like fish, poultry, beans, and nuts
            - Limit **added sugars** and **saturated fats**
            - Stay **hydrated** with water instead of sugary drinks
            - Practice **portion control** and mindful eating
            - Limit **processed foods** and high-sodium items
            - Include **healthy fats** like olive oil, avocados, and nuts
            - Choose **low-fat dairy** or dairy alternatives
            - Limit **alcohol** consumption
            
            For personalized dietary advice, please consult with a registered dietitian or healthcare provider.
            """)
            return
            
        for rec in recommendations:
            with st.container():
                # Add emoji based on category
                category = rec.get('category', 'Dietary Recommendations')
                emoji = ""
                if 'diabetes' in category.lower():
                    emoji = "🩸 "
                elif 'heart' in category.lower():
                    emoji = "❤️ "
                elif 'general' in category.lower():
                    emoji = "🍎 "
                
                # Show risk context if this is a risk-based recommendation
                if rec.get('risk_based', False):
                    with st.container():
                        st.markdown(f"#### 🔍 **Risk-Based Recommendations**")
                        st.markdown(f"*{rec.get('risk_description', 'Based on identified health risks')}*")
                        
                        if 'specific_risks' in rec and rec['specific_risks']:
                            with st.expander("⚠️ Specific Risks Addressed", expanded=False):
                                for risk in rec['specific_risks']:
                                    st.markdown(f"- {risk}")
                        st.markdown("---")
                
                st.markdown(f"### {emoji}{category}")
                
                # Show general description if available
                if 'description' in rec and rec['description'] and not rec.get('risk_based', False):
                    st.markdown(f"*{rec['description']}*")
                    st.markdown("")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'foods_to_include' in rec and rec['foods_to_include']:
                        with st.expander("✅ Foods to Include", expanded=True):
                            for food in rec['foods_to_include']:
                                if isinstance(food, dict):
                                    st.markdown(f"- **{food.get('item', '')}**: {food.get('reason', '')}")
                                else:
                                    st.markdown(f"- {food}")
                
                with col2:
                    if 'foods_to_avoid' in rec and rec['foods_to_avoid']:
                        with st.expander("❌ Foods to Limit or Avoid", expanded=True):
                            for food in rec['foods_to_avoid']:
                                if isinstance(food, dict):
                                    st.markdown(f"- **{food.get('item', '')}**: {food.get('reason', '')}")
                                else:
                                    st.markdown(f"- {food}")
                
                # Additional sections with improved formatting
                sections = [
                    ('nutritional_goals', "🎯 Nutritional Goals"),
                    ('meal_suggestions', "🍴 Sample Meal Ideas"),
                    ('portion_guidelines', "📏 Portion Guidelines"),
                    ('meal_timing', "⏰ Meal Timing"),
                    ('shopping_tips', "🛒 Shopping Tips"),
                    ('preparation_tips', "👩‍🍳 Food Preparation Tips")
                ]
                
                for field, title in sections:
                    if field in rec and rec[field]:
                        with st.expander(title, expanded=False):
                            if isinstance(rec[field], str):
                                st.markdown(rec[field])
                            else:
                                for item in rec[field]:
                                    st.markdown(f"- {item}")
                
                st.markdown("---")
    
    def _display_exercise_plan(self, recommendations: List[Dict[str, Any]]):
        """Display the exercise plan with improved formatting and risk-based recommendations"""
        if not recommendations:
            # Show default exercise guidelines if no specific recommendations
            st.info("## 🏃‍♂️ General Exercise Guidelines")
            st.markdown("""
            While we don't have specific exercise recommendations based on your current health data, 
            here are some general exercise guidelines for adults:
            
            - **Aerobic Activity**: At least 150 minutes of moderate-intensity or 75 minutes of vigorous-intensity per week
            - **Strength Training**: Include muscle-strengthening activities 2+ days per week
            - **Flexibility**: Stretch major muscle groups at least 2-3 times per week
            - **Balance Exercises**: Especially important for older adults to prevent falls
            - **Start Slow**: If you're new to exercise, start with 10-15 minutes and gradually increase
            - **Listen to Your Body**: Stop if you experience pain, dizziness, or shortness of breath
            - **Stay Hydrated**: Drink water before, during, and after exercise
            - **Warm Up/Cool Down**: Include 5-10 minutes of each in your routine
            
            For personalized exercise recommendations, please consult with a healthcare provider or certified fitness professional.
            """)
            return
            
        for rec in recommendations:
            with st.container():
                # Add emoji based on category
                category = rec.get('category', 'Exercise Recommendations')
                emoji = ""
                if 'cardio' in category.lower() or 'aerobic' in category.lower():
                    emoji = "🏃‍♂️ "
                elif 'strength' in category.lower() or 'resistance' in category.lower():
                    emoji = "💪 "
                elif 'flexibility' in category.lower() or 'balance' in category.lower():
                    emoji = "🧘‍♀️ "
                elif 'rehabilitation' in category.lower():
                    emoji = "🩺 "
                
                # Show risk context if this is a risk-based recommendation
                if rec.get('risk_based', False):
                    with st.container():
                        st.markdown(f"#### 🔍 **Risk-Based Exercise Plan**")
                        st.markdown(f"*{rec.get('risk_description', 'Based on identified health risks')}*")
                        
                        if 'specific_risks' in rec and rec['specific_risks']:
                            with st.expander("⚠️ Specific Risks Addressed", expanded=False):
                                for risk in rec['specific_risks']:
                                    st.markdown(f"- {risk}")
                        st.markdown("---")
                
                st.markdown(f"### {emoji}{category}")
                
                if 'description' in rec and rec['description']:
                    st.markdown(f"*{rec['description']}*")
                    st.markdown("")
                
                # Main exercise details
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'recommended_activities' in rec and rec['recommended_activities']:
                        with st.expander("🏃‍♂️ Recommended Activities", expanded=True):
                            for activity in rec['recommended_activities']:
                                if isinstance(activity, dict):
                                    st.markdown(f"- **{activity.get('activity', '')}**: {activity.get('details', '')}")
                                else:
                                    st.markdown(f"- {activity}")
                    
                    # Exercise parameters
                    params = [
                        ('frequency', "⏱️ Frequency"),
                        ('duration', "⏳ Duration"),
                        ('intensity', "⚡ Intensity"),
                        ('progression', "📈 Progression"),
                        ('equipment_needed', "🧰 Equipment Needed")
                    ]
                    
                    for param, label in params:
                        if param in rec and rec[param]:
                            st.info(f"{label}: {rec[param]}")
                
                with col2:
                    # Benefits section
                    if 'benefits' in rec and rec['benefits']:
                        with st.expander("💪 Benefits", expanded=True):
                            for benefit in rec['benefits']:
                                if isinstance(benefit, dict):
                                    st.markdown(f"- **{benefit.get('benefit', '')}**: {benefit.get('details', '')}")
                                else:
                                    st.markdown(f"- {benefit}")
                    
                    # Precautions and safety
                    if 'precautions' in rec and rec['precautions']:
                        with st.expander("⚠️ Precautions", expanded=True):
                            for precaution in rec['precautions']:
                                st.markdown(f"- {precaution}")
                    
                    # Sample workout if available
                    if 'sample_workout' in rec and rec['sample_workout']:
                        with st.expander("📋 Sample Workout", expanded=False):
                            if isinstance(rec['sample_workout'], str):
                                st.markdown(rec['sample_workout'])
                            else:
                                for item in rec['sample_workout']:
                                    st.markdown(f"- {item}")
                
                if 'precautions' in rec and rec['precautions']:
                    with st.expander("⚠️ Important Precautions", expanded=False):
                        for precaution in rec['precautions']:
                            st.markdown(f"- {precaution}")
                
                if 'progression_tips' in rec and rec['progression_tips']:
                    with st.expander("📈 Progression Tips", expanded=False):
                        for tip in rec['progression_tips']:
                            st.markdown(f"- {tip}")
    
    def _display_wellness_plan(self, recommendations: List[Dict[str, Any]]):
        """Display the general wellness plan with improved formatting"""
        if not recommendations:
            st.info("No general wellness recommendations available.")
            return
            
        for rec in recommendations:
            with st.container():
                st.markdown(f"#### 🧘 {rec.get('category', 'Wellness Recommendations')}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'recommendations' in rec and rec['recommendations']:
                        with st.expander("💡 Key Recommendations", expanded=True):
                            for item in rec['recommendations']:
                                st.markdown(f"- {item}")
                    
                    if 'implementation_tips' in rec and rec['implementation_tips']:
                        with st.expander("📝 How to Implement", expanded=False):
                            for tip in rec['implementation_tips']:
                                st.markdown(f"- {tip}")
                
                with col2:
                    if 'benefits' in rec and rec['benefits']:
                        with st.expander("🌟 Benefits", expanded=True):
                            for benefit in rec['benefits']:
                                st.markdown(f"- {benefit}")
                    
                    if 'resources' in rec and rec['resources']:
                        with st.expander("📚 Helpful Resources", expanded=False):
                            for resource in rec['resources']:
                                st.markdown(f"- {resource}")
                
                if 'tracking_suggestions' in rec and rec['tracking_suggestions']:
                    with st.expander("📊 Tracking Your Progress", expanded=False):
                        for item in rec['tracking_suggestions']:
                            st.markdown(f"- {item}")
    
    def display_monitoring_recommendations(self, lifestyle_data: Dict[str, Any]):
        """Display monitoring and professional consultation recommendations with improved formatting"""
        # Priority Actions
        with st.expander("📊 Priority Actions", expanded=True):
            priority_actions = lifestyle_data.get('priority_actions', [])
            if priority_actions:
                for action in priority_actions:
                    st.markdown(f"- ✅ **{action}**")
            else:
                st.info("No specific priority actions identified at this time.")
        
        # Professional Consultations
        with st.expander("👨‍⚕️ Recommended Professional Consultations", expanded=False):
            consultations = lifestyle_data.get('professional_consultations', [])
            if consultations:
                for consult in consultations:
                    st.markdown(f"- **{consult.get('type', 'Specialist')}**")
                    if 'reason' in consult:
                        st.markdown(f"  - *Reason:* {consult['reason']}")
                    if 'frequency' in consult:
                        st.markdown(f"  - *Frequency:* {consult['frequency']}")
                    if 'urgency' in consult:
                        st.markdown(f"  - *Priority:* {consult['urgency'].capitalize()}")
            else:
                st.info("No specific professional consultations recommended at this time.")
        
        # Monitoring Plan
        with st.expander("📋 Recommended Monitoring Plan", expanded=False):
            monitoring = lifestyle_data.get('monitoring_suggestions', [])
            if monitoring:
                st.markdown("### Health Metrics to Track")
                for item in monitoring:
                    if isinstance(item, dict):
                        st.markdown(f"- **{item.get('metric', 'Health metric')}**")
                        if 'frequency' in item:
                            st.markdown(f"  - *Frequency:* {item['frequency']}")
                        if 'target' in item:
                            st.markdown(f"  - *Target Range:* {item['target']}")
                        if 'notes' in item:
                            st.markdown(f"  - *Notes:* {item['notes']}")
                    else:
                        st.markdown(f"- {item}")
            else:
                st.info("No specific monitoring suggestions available.")
        
        # General Follow-up
        with st.expander("🔄 Follow-up Plan", expanded=False):
            st.markdown("### Recommended Follow-up")
            st.markdown("- Schedule a follow-up appointment with your primary care provider in 3-6 months")
            st.markdown("- Keep a health journal to track symptoms, diet, and exercise")
            st.markdown("- Report any new or worsening symptoms to your healthcare provider immediately")
            
            if 'follow_up_notes' in lifestyle_data and lifestyle_data['follow_up_notes']:
                st.markdown("\n**Additional Notes:**")
                st.markdown(lifestyle_data['follow_up_notes'])
    
    def display_fallback_lifestyle_recommendations(self):
        """Display fallback lifestyle recommendations when API fails"""
        st.markdown("### 🥗 General Diet Guidelines")
        st.write("• Eat a balanced diet with plenty of fruits and vegetables")
        st.write("• Choose whole grains over refined grains")
        st.write("• Include lean proteins in your meals")
        st.write("• Limit processed foods and added sugars")
        st.write("• Stay hydrated with plenty of water")
        
        st.markdown("### 🏃‍♂️ Exercise Recommendations")
        st.write("• Aim for 150 minutes of moderate aerobic activity weekly")
        st.write("• Include strength training exercises 2-3 times per week")
        st.write("• Start slowly and gradually increase intensity")
        st.write("• Choose activities you enjoy to stay motivated")
        st.write("• Consult your doctor before starting new exercise programs")
        
        st.markdown("### 🧘‍♀️ Lifestyle Tips")
        st.write("• Get 7-9 hours of quality sleep each night")
        st.write("• Practice stress management techniques")
        st.write("• Avoid smoking and limit alcohol consumption")
        st.write("• Maintain regular medical check-ups")
        st.write("• Build strong social connections")
    
    def generate_lifestyle_recommendations_from_analysis(self, analysis_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate lifestyle recommendations directly from analysis data"""
        try:
            # Extract conditions from analysis data
            conditions = analysis_data.get('conditions', [])
            condition_texts = []
            
            if isinstance(conditions, list):
                for condition in conditions:
                    if isinstance(condition, dict):
                        condition_texts.append(condition.get('text', ''))
                    else:
                        condition_texts.append(str(condition))
            
            # Generate basic lifestyle recommendations based on conditions
            diet_recommendations = []
            exercise_recommendations = []
            lifestyle_recommendations = []
            priority_actions = []
            monitoring_suggestions = []
            professional_consultations = []
            
            # Check for diabetes
            if any('diabetes' in str(cond).lower() for cond in condition_texts):
                diet_recommendations.append({
                    "category": "Diabetes Management Diet",
                    "foods_to_include": [
                        "Non-starchy vegetables (broccoli, spinach, peppers)",
                        "Lean proteins (chicken, fish, tofu, legumes)",
                        "Whole grains (quinoa, brown rice, oats)",
                        "Low-glycemic fruits (berries, apples, citrus)"
                    ],
                    "foods_to_avoid": [
                        "Refined sugars and sweets",
                        "White bread and refined grains",
                        "Sugary beverages and fruit juices",
                        "Processed and packaged foods"
                    ],
                    "meal_suggestions": [
                        "Breakfast: Greek yogurt with berries and nuts",
                        "Lunch: Grilled chicken salad with olive oil dressing",
                        "Dinner: Baked salmon with quinoa and steamed vegetables"
                    ],
                    "nutritional_goals": [
                        "Maintain stable blood sugar levels",
                        "Aim for 25-35g fiber per day",
                        "Include protein in every meal"
                    ],
                    "portion_guidelines": [
                        "Use plate method: 1/2 vegetables, 1/4 protein, 1/4 whole grains",
                        "Monitor carbohydrate portions carefully"
                    ]
                })
                
                exercise_recommendations.append({
                    "category": "Diabetes Exercise Program",
                    "recommended_activities": [
                        "Brisk walking (30-45 minutes daily)",
                        "Swimming or water aerobics",
                        "Resistance training (2-3 times weekly)"
                    ],
                    "frequency": "5-7 days per week for aerobic, 2-3 days for strength",
                    "duration": "150 minutes moderate aerobic activity weekly",
                    "intensity": "Moderate (can talk while exercising)",
                    "precautions": [
                        "Monitor blood glucose before, during, and after exercise",
                        "Carry glucose tablets for hypoglycemia",
                        "Check feet daily for injuries"
                    ],
                    "progression_tips": [
                        "Begin with 10-minute sessions if sedentary",
                        "Gradually increase duration before intensity"
                    ]
                })
                
                priority_actions.extend([
                    "Start blood glucose monitoring routine",
                    "Schedule appointment with endocrinologist or diabetes educator"
                ])
                
                monitoring_suggestions.extend([
                    "Daily blood glucose monitoring",
                    "Quarterly HbA1c testing"
                ])
                
                professional_consultations.extend([
                    "Endocrinologist for diabetes management",
                    "Registered dietitian for meal planning"
                ])
            
            # Check for hypertension
            if any('hypertension' in str(cond).lower() or 'blood pressure' in str(cond).lower() for cond in condition_texts):
                diet_recommendations.append({
                    "category": "DASH Diet for Blood Pressure",
                    "foods_to_include": [
                        "Fruits and vegetables (8-10 servings daily)",
                        "Whole grains (6-8 servings daily)",
                        "Low-fat dairy products",
                        "Potassium-rich foods (bananas, oranges, spinach)"
                    ],
                    "foods_to_avoid": [
                        "High-sodium processed foods",
                        "Canned soups and sauces",
                        "Fast food and restaurant meals",
                        "Excessive alcohol"
                    ],
                    "meal_suggestions": [
                        "Breakfast: Oatmeal with banana and low-fat milk",
                        "Lunch: Turkey and vegetable wrap with whole grain tortilla",
                        "Dinner: Grilled fish with roasted vegetables and brown rice"
                    ],
                    "nutritional_goals": [
                        "Limit sodium to less than 2,300mg daily",
                        "Increase potassium intake to 3,500-4,700mg daily"
                    ],
                    "portion_guidelines": [
                        "Read nutrition labels for sodium content",
                        "Use herbs and spices instead of salt"
                    ]
                })
                
                priority_actions.extend([
                    "Begin daily blood pressure monitoring",
                    "Implement DASH diet principles immediately"
                ])
                
                monitoring_suggestions.extend([
                    "Daily blood pressure readings",
                    "Weekly weight monitoring"
                ])
            
            # Add general lifestyle recommendations
            lifestyle_recommendations.extend([
                {
                    "category": "Stress Management",
                    "recommendations": [
                        "Practice daily meditation or mindfulness (10-20 minutes)",
                        "Maintain regular sleep schedule (7-9 hours nightly)",
                        "Engage in hobbies and enjoyable activities"
                    ],
                    "benefits": [
                        "Reduces cortisol levels",
                        "Improves immune function",
                        "Better sleep quality"
                    ],
                    "implementation_tips": [
                        "Start with 5-minute meditation sessions",
                        "Use smartphone apps for guided meditation"
                    ]
                },
                {
                    "category": "Sleep Optimization",
                    "recommendations": [
                        "Maintain consistent sleep-wake schedule",
                        "Create dark, cool, quiet sleep environment",
                        "Avoid screens 1 hour before bedtime"
                    ],
                    "benefits": [
                        "Improved glucose metabolism",
                        "Better blood pressure control",
                        "Enhanced immune function"
                    ],
                    "implementation_tips": [
                        "Use blackout curtains or eye mask",
                        "Keep bedroom temperature 65-68°F"
                    ]
                }
            ])
            
            return {
                "diet_recommendations": diet_recommendations,
                "exercise_recommendations": exercise_recommendations,
                "lifestyle_recommendations": lifestyle_recommendations,
                "priority_actions": priority_actions,
                "monitoring_suggestions": monitoring_suggestions,
                "professional_consultations": professional_consultations
            }
            
        except Exception as e:
            st.error(f"Error generating lifestyle recommendations: {str(e)}")
            return None
    
    def display_text_preview(self, text_preview: str):
        """Display extracted text preview"""
        with st.expander("📄 Extracted Text Preview"):
            st.text_area(
                "Original Text (Preview)",
                value=text_preview,
                height=200,
                disabled=True
            )

    def _display_dosage_plan(self, analysis_data: Dict[str, Any]):
        """Render age-specific dosage recommendations and suggested schedule from enriched backend data.
        Expects 'dosage_plan' (list) in analysis_data.
        """
        st.subheader("💊 Age-Specific Dosage Plan & Schedule")
        dosage_plan = analysis_data.get('dosage_plan') or analysis_data.get('analysis_data', {}).get('dosage_plan', [])
        if not dosage_plan:
            st.info("No dosage recommendations available. Provide patient age when uploading to enable this feature.")
            return
        import pandas as pd
        rows = []
        for rec in dosage_plan:
            drug = rec.get('drug')
            dr = rec.get('dosage_recommendation', {}) or {}
            rows.append({
                'drug': drug,
                'strength': rec.get('strength'),
                'frequency': rec.get('frequency'),
                'duration': rec.get('duration'),
                'age_band': dr.get('age_band'),
                'recommendation': dr.get('recommendation'),
                'source': dr.get('source'),
                'calc_dose_mg': dr.get('calculated_dose_mg'),
                'rounded_dose_mg': dr.get('rounded_dose_mg'),
                'schedule_times': ", ".join(rec.get('schedule_times', [])) if rec.get('schedule_times') else None,
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        # Pretty schedule cards
        st.markdown("#### ⏰ Suggested Schedule")
        for rec in dosage_plan:
            times = rec.get('schedule_times')
            if not times:
                continue
            st.markdown(f"**{rec.get('drug','').title()}**: " + ", ".join(times))

    def _display_interactions(self, analysis_data: Dict[str, Any]):
        """Render interaction results if available."""
        st.subheader("🔗 Drug-Drug Interactions")
        inter = analysis_data.get('interactions') or analysis_data.get('analysis_data', {}).get('drug_interactions', {}).get('interactions', [])
        if not inter:
            st.success("No interactions detected from available sources. Always confirm with a clinician.")
            return
        import pandas as pd
        df = pd.DataFrame(inter)
        st.dataframe(df, use_container_width=True)
    
    def render_upload_section(self):
        """Render the file upload section"""
        st.header("🏥 Medical Report Analysis")
        st.markdown("""
        Upload your medical report (PDF or image) to get comprehensive analysis including:
        - **Medical conditions** identification and classification
        - **Medications and treatments** extraction
        - **Laboratory values** analysis
        - **Future health risks** prediction
        - **Personalized recommendations**
        - **Downloadable PDF report**
        """)
        
        # Upload form
        with st.form("upload_form"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                uploaded_file = st.file_uploader(
                    "Choose medical report file",
                    type=['pdf', 'png', 'jpg', 'jpeg', 'tiff', 'bmp'],
                    help="Supported formats: PDF, PNG, JPG, JPEG, TIFF, BMP (Max 10MB)"
                )
            
            with col2:
                patient_name = st.text_input(
                    "Patient Name",
                    value="",
                    help="Enter patient name for the report"
                )
                age = st.number_input("Age (years)", min_value=0, max_value=120, value=30, key="mra_age")
                weight = st.number_input("Weight (kg)", min_value=0.0, max_value=500.0, value=0.0, step=0.1, key="mra_weight")
                condition = st.text_input("Condition (optional)", value="", placeholder="e.g., diabetes, renal impairment", key="mra_condition")
            
            submit_button = st.form_submit_button(
                "🔍 Analyze Report",
                use_container_width=True
            )
            
            if submit_button and uploaded_file is not None:
                # Validate file size
                if uploaded_file.size > 10 * 1024 * 1024:  # 10MB
                    st.error("File size too large. Maximum size is 10MB.")
                    return
                
                # Show progress
                with st.spinner("Analyzing medical report... This may take a few minutes."):
                    progress_bar = st.progress(0)
                    
                    # Simulate progress
                    import time
                    for i in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)
                    
                    # Perform analysis
                    result = self.upload_and_analyze_report(uploaded_file, patient_name, int(age) if age is not None else None, float(weight) if weight else None, condition if condition else None)
                    
                    if result:
                        st.success("✅ Analysis completed successfully!")
                        
                        # Store result in session state for display
                        st.session_state['current_analysis'] = result
                        st.session_state['show_results'] = True
                        st.rerun()
    
    def render_results_section(self, analysis_data: Dict[str, Any]):
        """Render the analysis results section"""
        st.header("📊 Analysis Results")
        
        # Analysis metadata
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**Patient:** {analysis_data['patient_name']}")
        with col2:
            st.info(f"**Analysis ID:** {analysis_data['analysis_id'][:8]}...")
        with col3:
            analysis_date = datetime.fromisoformat(analysis_data['analysis_date'].replace('Z', '+00:00'))
            st.info(f"**Date:** {analysis_date.strftime('%Y-%m-%d %H:%M')}")
        
        # Confidence score
        st.subheader("🎯 Analysis Confidence")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            self.display_confidence_meter(analysis_data['confidence_score'])
        
        with col2:
            confidence_pct = analysis_data['confidence_score'] * 100
            if confidence_pct >= 80:
                st.success(f"**High Confidence ({confidence_pct:.1f}%)**\n\nThe analysis is based on clear, well-structured medical text with identifiable medical terminology.")
            elif confidence_pct >= 60:
                st.warning(f"**Moderate Confidence ({confidence_pct:.1f}%)**\n\nThe analysis found some medical information, but there may be ambiguity in the source text.")
            else:
                st.error(f"**Low Confidence ({confidence_pct:.1f}%)**\n\nLimited medical information was extracted. Consider uploading a clearer document.")
        
        # Summary metrics
        st.subheader("📈 Summary Overview")
        self.display_summary_metrics(analysis_data['summary'])
        
        # Detailed analysis sections
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
            "🏥 Conditions", "💊 Medications", "🔬 Lab Values", 
            "⚠️ Risk Assessment", "💡 Recommendations", 
            "🏃‍♀️ Lifestyle Plan", "📄 Text Preview", "💊 Dosage Plan", "🔗 Interactions"
        ])
        
        with tab1:
            self.display_conditions_analysis(analysis_data['conditions'])
        
        with tab2:
            self.display_medications_analysis(analysis_data['medications'])
        
        with tab3:
            self.display_lab_values_analysis(analysis_data['lab_values'])
        
        with tab4:
            self.display_risk_assessment(analysis_data['future_risks'])
        
        with tab5:
            self.display_recommendations(analysis_data['recommendations'])
        
        with tab6:
            self.display_lifestyle_plan(analysis_data)
        
        with tab7:
            self.display_text_preview(analysis_data['text_preview'])

        with tab8:
            self._display_dosage_plan(analysis_data)

        with tab9:
            self._display_interactions(analysis_data)
        
        # Download PDF report
        st.subheader("📄 Download Report")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            self.download_pdf_report(analysis_data['analysis_id'], analysis_data['patient_name'])
        
        with col2:
            if st.button("🔄 Analyze Another Report", use_container_width=True):
                # Clear session state
                if 'current_analysis' in st.session_state:
                    del st.session_state['current_analysis']
                if 'show_results' in st.session_state:
                    del st.session_state['show_results']
                st.rerun()
    
    def render_history_section(self):
        """Render analysis history section"""
        st.header("📚 Analysis History")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            patient_filter = st.text_input("Filter by Patient Name", placeholder="Enter patient name...")
        
        with col2:
            limit = st.selectbox("Number of Results", [5, 10, 20, 50], index=1)
        
        with col3:
            if st.button("🔍 Search", use_container_width=True):
                st.session_state['refresh_history'] = True
        
        # Get analysis list
        if st.session_state.get('refresh_history', True):
            analysis_list = self.get_analysis_list(limit, patient_filter if patient_filter else None)
            st.session_state['analysis_history'] = analysis_list
            st.session_state['refresh_history'] = False
        else:
            analysis_list = st.session_state.get('analysis_history')
        
        if analysis_list and analysis_list['reports']:
            # Display results
            st.subheader(f"Found {analysis_list['total_count']} analyses")
            
            for report in analysis_list['reports']:
                with st.expander(f"📋 {report['patient_name']} - {report['filename']} ({report['analysis_date'][:10]})"):
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Confidence", f"{report['confidence_score']:.1%}")
                    
                    with col2:
                        st.metric("Conditions", report['conditions_count'])
                    
                    with col3:
                        st.metric("Medications", report['medications_count'])
                    
                    with col4:
                        file_size_mb = report['file_size'] / (1024 * 1024)
                        st.metric("File Size", f"{file_size_mb:.1f} MB")
                    
                    # Action buttons
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.button(f"👁️ View Details", key=f"view_{report['analysis_id']}"):
                            details = self.get_analysis_details(report['analysis_id'])
                            if details:
                                st.session_state['current_analysis'] = details['analysis_data']
                                st.session_state['current_analysis']['analysis_id'] = report['analysis_id']
                                st.session_state['show_results'] = True
                                st.rerun()
                    
                    with col2:
                        self.download_pdf_report(report['analysis_id'], report['patient_name'])
                    
                    with col3:
                        if st.button(f"🗑️ Delete", key=f"delete_{report['analysis_id']}", type="secondary"):
                            if st.session_state.get(f"confirm_delete_{report['analysis_id']}", False):
                                # Perform deletion
                                try:
                                    response = requests.delete(
                                        f"{self.api_client.base_url}/api/medical-report/analysis/{report['analysis_id']}"
                                    )
                                    if response.status_code == 200:
                                        st.success("Analysis deleted successfully")
                                        st.session_state['refresh_history'] = True
                                        st.rerun()
                                    else:
                                        st.error("Failed to delete analysis")
                                except Exception as e:
                                    st.error(f"Deletion failed: {str(e)}")
                            else:
                                st.session_state[f"confirm_delete_{report['analysis_id']}"] = True
                                st.warning("Click delete again to confirm")
        else:
            st.info("No previous analyses found.")
    
    def run(self):
        """Main application runner"""
        # Initialize session state
        if 'show_results' not in st.session_state:
            st.session_state['show_results'] = False
        
        # Sidebar navigation
        st.sidebar.title("🏥 Medical Report Analysis")
        
        page = st.sidebar.radio(
            "Navigation",
            ["📤 Upload & Analyze", "📊 View Results", "📚 Analysis History"],
            index=1 if st.session_state.get('show_results', False) else 0
        )
        
        # Sidebar info
        st.sidebar.markdown("---")
        st.sidebar.markdown("""
        ### 🔍 Supported Analysis
        - Medical conditions identification
        - Medication extraction
        - Lab values analysis
        - Risk assessment
        - Health recommendations
        - PDF report generation
        
        ### 📁 Supported Formats
        - PDF documents
        - Image files (PNG, JPG, TIFF, BMP)
        - Maximum file size: 10MB
        """)
        
        # Main content based on navigation
        if page == "📤 Upload & Analyze":
            self.render_upload_section()
        
        elif page == "📊 View Results":
            if st.session_state.get('show_results', False) and 'current_analysis' in st.session_state:
                self.render_results_section(st.session_state['current_analysis'])
            else:
                st.info("No analysis results to display. Please upload and analyze a medical report first.")
                if st.button("📤 Go to Upload", use_container_width=True):
                    st.session_state['show_results'] = False
                    st.rerun()
        
        elif page == "📚 Analysis History":
            self.render_history_section()

def main():
    """Main function to run the medical report analysis app"""
    analyzer = MedicalReportAnalyzer()
    analyzer.run()

if __name__ == "__main__":
    main()
