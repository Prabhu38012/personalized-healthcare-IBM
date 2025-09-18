import streamlit as st
import numpy as np

class LabPatientInputForm:
    def __init__(self):
        self.patient_data = {}
        # Initialize session state for form data persistence
        if 'lab_form_data' not in st.session_state:
            st.session_state.lab_form_data = {}
    
    def render(self):
        """Render comprehensive lab-enhanced patient input form"""
        
        st.markdown("### Comprehensive Health Assessment with Lab Values")
        st.markdown("*Enhanced risk assessment using complete blood count and hematology data*")
        
        # Create tabs for better organization
        basic_tab, cbc_tab, differential_tab, platelets_tab = st.tabs([
            "Basic Info", "Complete Blood Count", "Differential Count", "Platelets & Other"
        ])
        
        # Always render all sections to maintain state, but show them in tabs
        with basic_tab:
            self._render_basic_info()
        
        with cbc_tab:
            self._render_cbc_section()
        
        with differential_tab:
            self._render_differential_section()
        
        with platelets_tab:
            self._render_platelets_section()
        
        # Store form data in session state to persist across tab switches
        st.session_state.lab_form_data = self.patient_data.copy()
        
        return self.patient_data
    
    def _render_basic_info(self):
        """Render basic demographic and vital information"""
        st.markdown("#### Demographics & Vital Signs")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age (years)", min_value=18, max_value=120, value=45, key="lab_age")
            sex = st.selectbox("Sex", ["M", "F"], format_func=lambda x: "Male" if x == "M" else "Female", key="lab_sex")
            
            # Vital Signs
            systolic_bp = st.number_input("Systolic Blood Pressure (mmHg)", 
                                        min_value=80, max_value=250, value=120, key="lab_systolic_bp")
            diastolic_bp = st.number_input("Diastolic Blood Pressure (mmHg)", 
                                         min_value=40, max_value=150, value=80, key="lab_diastolic_bp")
        
        with col2:
            # Body measurements
            weight = st.number_input("Weight (kg)", min_value=30.0, max_value=300.0, value=70.0, step=0.1, key="lab_weight")
            height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170, key="lab_height")
            
            # Calculate and display BMI
            bmi = weight / ((height/100) ** 2)
            st.metric("BMI", f"{bmi:.1f}")
        
        st.markdown("#### Basic Chemistry")
        col3, col4 = st.columns(2)
        
        with col3:
            total_cholesterol = st.number_input("Total Cholesterol (mg/dL)", 
                                              min_value=100, max_value=600, value=200, key="lab_total_cholesterol")
            ldl_cholesterol = st.number_input("LDL Cholesterol (mg/dL)", 
                                            min_value=50, max_value=300, value=120, key="lab_ldl_cholesterol")
            triglycerides = st.number_input("Triglycerides (mg/dL)", 
                                          min_value=50, max_value=500, value=150, key="lab_triglycerides")
        
        with col4:
            hdl_cholesterol = st.number_input("HDL Cholesterol (mg/dL)", 
                                            min_value=20, max_value=100, value=50, key="lab_hdl_cholesterol")
            hba1c = st.number_input("HbA1c (%)", 
                                  min_value=4.0, max_value=15.0, value=5.5, step=0.1, key="lab_hba1c")
        
        st.markdown("#### Medical History")
        col5, col6 = st.columns(2)
        
        with col5:
            diabetes = st.selectbox("Diabetes Status", 
                                  [0, 1], 
                                  format_func=lambda x: "No" if x == 0 else "Yes", key="lab_diabetes")
            smoking = st.selectbox("Smoking Status", 
                                 [0, 1], 
                                 format_func=lambda x: "Non-smoker" if x == 0 else "Smoker", key="lab_smoking")
        
        with col6:
            family_history = st.selectbox("Family History of Heart Disease", 
                                        [0, 1], 
                                        format_func=lambda x: "No" if x == 0 else "Yes", key="lab_family_history")
            fasting_blood_sugar = st.selectbox("Fasting Blood Sugar > 120 mg/dL", 
                                             [0, 1], 
                                             format_func=lambda x: "No" if x == 0 else "Yes", key="lab_fasting_blood_sugar")
        
        # Store basic data
        self.patient_data.update({
            "age": age,
            "sex": sex,
            "weight": weight,
            "height": height,
            "systolic_bp": systolic_bp,
            "diastolic_bp": diastolic_bp,
            "total_cholesterol": total_cholesterol,
            "ldl_cholesterol": ldl_cholesterol,
            "hdl_cholesterol": hdl_cholesterol,
            "triglycerides": triglycerides,
            "hba1c": hba1c,
            "diabetes": diabetes,
            "smoking": smoking,
            "family_history": family_history,
            "fasting_blood_sugar": fasting_blood_sugar
        })
    
    def _render_cbc_section(self):
        """Render Complete Blood Count section"""
        st.markdown("#### Complete Blood Count (CBC)")
        st.markdown("*Optional but highly valuable for enhanced risk assessment*")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Primary CBC parameters
            hemoglobin = st.number_input("Hemoglobin (g/dL)", 
                                       min_value=5.0, max_value=20.0, value=None, 
                                       help="Normal: 13.0-17.0 (M), 12.0-15.5 (F)", key="lab_hemoglobin")
            
            total_leukocyte_count = st.number_input("Total WBC Count (10³/μL)", 
                                                  min_value=1.0, max_value=50.0, value=None,
                                                  help="Normal: 4.0-10.0", key="lab_total_leukocyte_count")
            
            red_blood_cell_count = st.number_input("RBC Count (10⁶/μL)", 
                                                 min_value=2.0, max_value=8.0, value=None,
                                                 help="Normal: 4.5-5.5", key="lab_red_blood_cell_count")
            
            hematocrit = st.number_input("Hematocrit (%)", 
                                       min_value=20.0, max_value=60.0, value=None,
                                       help="Normal: 40.0-50.0", key="lab_hematocrit")
        
        with col2:
            # RBC Indices
            mean_corpuscular_volume = st.number_input("MCV (fL)", 
                                                    min_value=60.0, max_value=120.0, value=None,
                                                    help="Normal: 83.0-101.0", key="lab_mean_corpuscular_volume")
            
            mean_corpuscular_hb = st.number_input("MCH (pg)", 
                                                min_value=20.0, max_value=40.0, value=None,
                                                help="Normal: 27.0-32.0", key="lab_mean_corpuscular_hb")
            
            mean_corpuscular_hb_conc = st.number_input("MCHC (g/dL)", 
                                                     min_value=25.0, max_value=40.0, value=None,
                                                     help="Normal: 31.5-34.5", key="lab_mean_corpuscular_hb_conc")
            
            red_cell_distribution_width = st.number_input("RDW (%)", 
                                                        min_value=10.0, max_value=20.0, value=None,
                                                        help="Normal: 11.6-14.0", key="lab_red_cell_distribution_width")
        
        # Store CBC data
        self.patient_data.update({
            "hemoglobin": hemoglobin,
            "total_leukocyte_count": total_leukocyte_count,
            "red_blood_cell_count": red_blood_cell_count,
            "hematocrit": hematocrit,
            "mean_corpuscular_volume": mean_corpuscular_volume,
            "mean_corpuscular_hb": mean_corpuscular_hb,
            "mean_corpuscular_hb_conc": mean_corpuscular_hb_conc,
            "red_cell_distribution_width": red_cell_distribution_width
        })
    
    def _render_differential_section(self):
        """Render Differential Leukocyte Count section"""
        st.markdown("#### Differential Leukocyte Count")
        st.markdown("*Percentages and absolute counts of white blood cell types*")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Percentages (%)**")
            neutrophils_percent = st.number_input("Neutrophils (%)", 
                                                min_value=0.0, max_value=100.0, value=None,
                                                help="Normal: 40-80%", key="lab_neutrophils_percent")
            
            lymphocytes_percent = st.number_input("Lymphocytes (%)", 
                                                min_value=0.0, max_value=100.0, value=None,
                                                help="Normal: 20-40%", key="lab_lymphocytes_percent")
            
            monocytes_percent = st.number_input("Monocytes (%)", 
                                              min_value=0.0, max_value=100.0, value=None,
                                              help="Normal: 2-10%", key="lab_monocytes_percent")
            
            eosinophils_percent = st.number_input("Eosinophils (%)", 
                                                min_value=0.0, max_value=100.0, value=None,
                                                help="Normal: 1-6%", key="lab_eosinophils_percent")
            
            basophils_percent = st.number_input("Basophils (%)", 
                                              min_value=0.0, max_value=100.0, value=None,
                                              help="Normal: 0-2%", key="lab_basophils_percent")
        
        with col2:
            st.markdown("**Absolute Counts (10³/μL)**")
            absolute_neutrophil_count = st.number_input("ANC (10³/μL)", 
                                                      min_value=0.0, max_value=20.0, value=None,
                                                      help="Normal: 2.0-7.0", key="lab_absolute_neutrophil_count")
            
            absolute_lymphocyte_count = st.number_input("ALC (10³/μL)", 
                                                      min_value=0.0, max_value=10.0, value=None,
                                                      help="Normal: 1.0-3.0", key="lab_absolute_lymphocyte_count")
            
            absolute_monocyte_count = st.number_input("AMC (10³/μL)", 
                                                    min_value=0.0, max_value=5.0, value=None,
                                                    help="Normal: 0.2-1.0", key="lab_absolute_monocyte_count")
            
            absolute_eosinophil_count = st.number_input("AEC (10³/μL)", 
                                                      min_value=0.0, max_value=2.0, value=None,
                                                      help="Normal: 0.02-0.5", key="lab_absolute_eosinophil_count")
            
            absolute_basophil_count = st.number_input("ABC (10³/μL)", 
                                                    min_value=0.0, max_value=1.0, value=None,
                                                    help="Normal: 0.02-0.10", key="lab_absolute_basophil_count")
        
        # Store differential data
        self.patient_data.update({
            "neutrophils_percent": neutrophils_percent,
            "lymphocytes_percent": lymphocytes_percent,
            "monocytes_percent": monocytes_percent,
            "eosinophils_percent": eosinophils_percent,
            "basophils_percent": basophils_percent,
            "absolute_neutrophil_count": absolute_neutrophil_count,
            "absolute_lymphocyte_count": absolute_lymphocyte_count,
            "absolute_monocyte_count": absolute_monocyte_count,
            "absolute_eosinophil_count": absolute_eosinophil_count,
            "absolute_basophil_count": absolute_basophil_count
        })
    
    def _render_platelets_section(self):
        """Render Platelet Parameters and Additional Tests section"""
        st.markdown("#### Platelet Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            platelet_count = st.number_input("Platelet Count (10³/μL)", 
                                           min_value=50.0, max_value=1000.0, value=None,
                                           help="Normal: 150-410", key="lab_platelet_count")
            
            mean_platelet_volume = st.number_input("MPV (fL)", 
                                                 min_value=5.0, max_value=15.0, value=None,
                                                 help="Normal: 7-9", key="lab_mean_platelet_volume")
        
        with col2:
            platelet_distribution_width = st.number_input("PDW (%)", 
                                                        min_value=10.0, max_value=30.0, value=None,
                                                        help="Normal: 11.6-14.0", key="lab_platelet_distribution_width")
        
        st.markdown("#### Additional Blood Chemistry")
        
        erythrocyte_sedimentation_rate = st.number_input("ESR (mm/1st hour)", 
                                                       min_value=0.0, max_value=100.0, value=None,
                                                       help="Normal: 0-10", key="lab_erythrocyte_sedimentation_rate")
        
        # Store platelet and additional data
        self.patient_data.update({
            "platelet_count": platelet_count,
            "mean_platelet_volume": mean_platelet_volume,
            "platelet_distribution_width": platelet_distribution_width,
            "erythrocyte_sedimentation_rate": erythrocyte_sedimentation_rate
        })
    
    def get_sample_lab_data(self):
        """Get sample lab data based on the provided lab report"""
        return {
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
            
            # Complete Blood Count (from the lab report)
            "hemoglobin": 13.9,
            "total_leukocyte_count": 12.1,
            "red_blood_cell_count": 5.00,
            "hematocrit": 43.7,
            "mean_corpuscular_volume": 88.0,
            "mean_corpuscular_hb": 27.9,
            "mean_corpuscular_hb_conc": 31.7,
            "red_cell_distribution_width": 14.5,
            
            # Differential Count (from the lab report)
            "neutrophils_percent": 70.9,
            "lymphocytes_percent": 22.1,
            "monocytes_percent": 5.5,
            "eosinophils_percent": 1.1,
            "basophils_percent": 0.4,
            
            # Absolute Counts (calculated from percentages and total WBC)
            "absolute_neutrophil_count": 8.58,  # 70.9% of 12.1
            "absolute_lymphocyte_count": 2.67,  # 22.1% of 12.1
            "absolute_monocyte_count": 0.67,    # 5.5% of 12.1
            "absolute_eosinophil_count": 0.13,  # 1.1% of 12.1
            "absolute_basophil_count": 0.05,    # 0.4% of 12.1
            
            # Platelet Parameters (from the lab report)
            "platelet_count": 580.0,
            "mean_platelet_volume": None,  # Not provided in report
            "platelet_distribution_width": None,  # Not provided in report
            
            # Additional Chemistry
            "erythrocyte_sedimentation_rate": 2.0  # From the report
        }
