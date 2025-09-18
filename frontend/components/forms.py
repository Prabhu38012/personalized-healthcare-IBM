import streamlit as st
import numpy as np

class PatientInputForm:
    def __init__(self):
        self.patient_data = {}
    
    def render(self):
        """Render the enhanced patient input form with comprehensive health metrics"""
        
        # Professional Basic Information Section
        st.markdown("""
        <div style="background: var(--background-primary); padding: 2rem; border-radius: var(--border-radius-lg); 
                    margin-bottom: 2rem; border: 1px solid var(--border-color); box-shadow: var(--shadow-sm);">
            <h3 style="color: var(--primary-color); margin: 0 0 1rem 0; display: flex; align-items: center; gap: 0.75rem;">
                <span style="background: linear-gradient(135deg, var(--primary-color), var(--accent-color)); 
                             color: white; padding: 0.5rem; border-radius: 50%; display: flex; 
                             align-items: center; justify-content: center;">👤</span>
                Personal Information
            </h3>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age (years)", min_value=18, max_value=120, value=45)
            sex = st.selectbox("Sex", ["M", "F"], format_func=lambda x: "Male" if x == "M" else "Female")
            height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
            weight = st.number_input("Weight (kg)", min_value=30.0, max_value=300.0, value=70.0, step=0.1)
            
            # Calculate BMI
            bmi = weight / ((height/100) ** 2)
            st.metric("BMI", f"{bmi:.1f}", 
                     help="Underweight: <18.5, Normal: 18.5-24.9, Overweight: 25-29.9, Obese: ≥30")
        
        with col2:
            smoking = st.select_slider("Smoking Status", 
                                     options=[0, 1, 2], 
                                     format_func=lambda x: ["Never", "Former", "Current"][x])
            
            diabetes = st.select_slider("Diabetes Status", 
                                      options=[0, 2, 1], 
                                      format_func=lambda x: ["No", "Pre-diabetes", "Yes"][x//2])
            
            family_history = st.selectbox("Family History of Heart Disease", 
                                        [0, 1], 
                                        format_func=lambda x: "No" if x == 0 else "Yes")
            
            hypertension = st.checkbox("History of Hypertension")
            kidney_disease = st.checkbox("History of Kidney Disease")
        
        # Professional Vital Signs Section
        st.markdown("""
        <div style="background: var(--background-primary); padding: 2rem; border-radius: var(--border-radius-lg); 
                    margin: 2rem 0; border: 1px solid var(--border-color); box-shadow: var(--shadow-sm);">
            <h3 style="color: var(--primary-color); margin: 0 0 1rem 0; display: flex; align-items: center; gap: 0.75rem;">
                <span style="background: linear-gradient(135deg, var(--secondary-color), var(--accent-color)); 
                             color: white; padding: 0.5rem; border-radius: 50%; display: flex; 
                             align-items: center; justify-content: center;">💓</span>
                Vital Signs
            </h3>
        </div>
        """, unsafe_allow_html=True)
        
        col3, col4 = st.columns(2)
        
        with col3:
            resting_bp = st.number_input("Resting Blood Pressure (mmHg)", 
                                       min_value=80, max_value=250, value=120)
            
            max_heart_rate = st.number_input("Maximum Heart Rate (bpm)", 
                                           min_value=40, max_value=220, value=150)
            
            sleep_duration = st.number_input("Sleep Duration (hours/night)", 
                                           min_value=0.0, max_value=24.0, value=7.0, step=0.5)
        
        with col4:
            cholesterol = st.number_input("Total Cholesterol (mg/dL)", 
                                        min_value=100, max_value=500, value=200)
            
            hdl = st.number_input("HDL Cholesterol (mg/dL)", 
                                 min_value=10, max_value=150, value=50)
            
            ldl = st.number_input("LDL Cholesterol (mg/dL)", 
                                 min_value=30, max_value=400, value=120)
            
            triglycerides = st.number_input("Triglycerides (mg/dL)", 
                                          min_value=30, max_value=1000, value=150)
        
        # Professional Blood Work Section
        st.markdown("""
        <div style="background: var(--background-primary); padding: 2rem; border-radius: var(--border-radius-lg); 
                    margin: 2rem 0; border: 1px solid var(--border-color); box-shadow: var(--shadow-sm);">
            <h3 style="color: var(--primary-color); margin: 0 0 1rem 0; display: flex; align-items: center; gap: 0.75rem;">
                <span style="background: linear-gradient(135deg, #dc2626, #ef4444); 
                             color: white; padding: 0.5rem; border-radius: 50%; display: flex; 
                             align-items: center; justify-content: center;">🩸</span>
                Blood Work & Laboratory Results
            </h3>
        </div>
        """, unsafe_allow_html=True)
        
        col5, col6 = st.columns(2)
        
        with col5:
            fasting_blood_sugar = st.number_input("Fasting Blood Sugar (mg/dL)", 
                                                min_value=50, max_value=500, value=90)
            
            hba1c = st.number_input("HbA1c (%)", 
                                  min_value=3.0, max_value=15.0, value=5.5, step=0.1)
        
        with col6:
            exercise_angina = st.selectbox("Exercise Induced Angina", 
                                         [0, 1], 
                                         format_func=lambda x: "No" if x == 0 else "Yes")
            
            oldpeak = st.number_input("ST Depression (Oldpeak)", 
                                    min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        
        # Professional Lifestyle Section
        st.markdown("""
        <div style="background: var(--background-primary); padding: 2rem; border-radius: var(--border-radius-lg); 
                    margin: 2rem 0; border: 1px solid var(--border-color); box-shadow: var(--shadow-sm);">
            <h3 style="color: var(--primary-color); margin: 0 0 1rem 0; display: flex; align-items: center; gap: 0.75rem;">
                <span style="background: linear-gradient(135deg, #059669, #10b981); 
                             color: white; padding: 0.5rem; border-radius: 50%; display: flex; 
                             align-items: center; justify-content: center;">🏃</span>
                Lifestyle Factors
            </h3>
        </div>
        """, unsafe_allow_html=True)
        
        col7, col8 = st.columns(2)
        
        with col7:
            physical_activity = st.number_input("Weekly Physical Activity (minutes)", 
                                              min_value=0, max_value=1000, value=150)
            
            alcohol_consumption = st.number_input("Alcoholic Drinks per Week", 
                                                min_value=0, max_value=50, value=0)
        
        with col8:
            stress_level = st.slider("Stress Level (1-10)", 
                                   min_value=1, max_value=10, value=5)
            
            on_meds = st.checkbox("Currently on Medications")
        
        # Professional Clinical Assessment Section
        st.markdown("""
        <div style="background: var(--background-primary); padding: 2rem; border-radius: var(--border-radius-lg); 
                    margin: 2rem 0; border: 1px solid var(--border-color); box-shadow: var(--shadow-sm);">
            <h3 style="color: var(--primary-color); margin: 0 0 1rem 0; display: flex; align-items: center; gap: 0.75rem;">
                <span style="background: linear-gradient(135deg, #7c3aed, #a855f7); 
                             color: white; padding: 0.5rem; border-radius: 50%; display: flex; 
                             align-items: center; justify-content: center;">🔬</span>
                Clinical Assessment
            </h3>
        </div>
        """, unsafe_allow_html=True)
        
        col9, col10 = st.columns(2)
        
        with col9:
            chest_pain_type = st.selectbox("Chest Pain Type", 
                                         ["typical", "atypical", "non_anginal", "asymptomatic"],
                                         format_func=lambda x: x.replace("_", " ").title())
            
            resting_ecg = st.selectbox("Resting ECG", 
                                     ["normal", "abnormal", "hypertrophy"],
                                     format_func=lambda x: x.title())
        
        with col10:
            slope = st.selectbox("ST Slope", 
                               ["upsloping", "flat", "downsloping"],
                               format_func=lambda x: x.title())
            
            ca = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
            
            thal = st.selectbox("Thalassemia", 
                              ["normal", "fixed", "reversible"],
                              format_func=lambda x: x.title())
        
        # Compile patient data
        self.patient_data = {
            # Basic Information
            "age": age,
            "sex": sex,
            "height": height,
            "weight": weight,
            "bmi": bmi,
            
            # Vital Signs
            "resting_bp": resting_bp,
            "max_heart_rate": max_heart_rate,
            "sleep_duration": sleep_duration,
            
            # Lipid Profile
            "cholesterol": cholesterol,
            "hdl_cholesterol": hdl,
            "ldl_cholesterol": ldl,
            "triglycerides": triglycerides,
            
            # Blood Sugar
            "fasting_blood_sugar": 1 if fasting_blood_sugar > 120 else 0,
            "fasting_glucose_value": fasting_blood_sugar,  # Keep original value for display
            "hba1c": hba1c,
            
            # Clinical Assessment
            "chest_pain_type": chest_pain_type,
            "resting_ecg": resting_ecg,
            "exercise_angina": exercise_angina,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal,
            
            # Lifestyle Factors
            "smoking": smoking,
            "alcohol_consumption": alcohol_consumption,
            "physical_activity": physical_activity,
            "stress_level": stress_level,
            
            # Medical History
            "diabetes": diabetes,
            "family_history": family_history,
            "hypertension": 1 if hypertension else 0,
            "kidney_disease": 1 if kidney_disease else 0,
            
            # Medications
            "on_blood_pressure_meds": on_meds,
            "on_cholesterol_meds": on_meds,
            "on_diabetes_meds": on_meds if diabetes == 1 else False
        }
        
        return self.patient_data
    
    def get_sample_data(self):
        """Get sample patient data for testing with comprehensive health metrics"""
        return {
            # Basic Information
            "age": 52,
            "sex": "M",
            "height": 175,
            "weight": 80.5,
            "bmi": 26.3,
            
            # Vital Signs
            "resting_bp": 145,
            "max_heart_rate": 148,
            "sleep_duration": 6.5,
            
            # Lipid Profile
            "cholesterol": 240,
            "hdl_cholesterol": 42,
            "ldl_cholesterol": 165,
            "triglycerides": 180,
            
            # Blood Sugar
            "fasting_blood_sugar": 1,  # > 120 mg/dL
            "hba1c": 6.1,
            
            # Clinical Assessment
            "chest_pain_type": "atypical",
            "resting_ecg": "normal",
            "exercise_angina": 0,
            "oldpeak": 1.5,
            "slope": "flat",
            "ca": 1,
            "thal": "reversible",
            
            # Lifestyle Factors
            "smoking": 1,  # Former smoker
            "alcohol_consumption": 3,
            "physical_activity": 120,
            "stress_level": 6,
            
            # Medical History
            "diabetes": 2,  # Pre-diabetes
            "family_history": 1,
            "hypertension": 1,
            "kidney_disease": 0,
            
            # Medications
            "on_blood_pressure_meds": True,
            "on_cholesterol_meds": False,
            "on_diabetes_meds": False
        }
