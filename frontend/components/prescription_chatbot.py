"""
Enhanced Prescription Analysis Chatbot Component
Professional UI for prescription upload, OCR analysis, and medical recommendations
"""

import streamlit as st
import requests
import json
import base64
from typing import List, Dict, Any, Optional
from datetime import datetime
import time
from PIL import Image
import io

class PrescriptionChatbot:
    def __init__(self, api_base_url: str = "http://localhost:8002/api"):
        self.api_base_url = api_base_url
        self.prescription_endpoint = f"{api_base_url}/prescription"
        self.chat_endpoint = f"{api_base_url}/chat"
    
    def upload_prescription(self, image_file) -> Dict[str, Any]:
        """Upload and analyze prescription image"""
        try:
            files = {"file": image_file}
            response = requests.post(
                f"{self.prescription_endpoint}/upload",
                files=files,
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                error_detail = response.json().get("detail", "Unknown error")
                return {
                    "success": False,
                    "error": f"Upload failed ({response.status_code}): {error_detail}"
                }
                
        except requests.exceptions.Timeout:
            return {"success": False, "error": "Upload timeout - please try again"}
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Connection error: {str(e)}"}
        except Exception as e:
            return {"success": False, "error": f"Unexpected error: {str(e)}"}
    
    def get_medicine_info(self, medicine_name: str) -> Dict[str, Any]:
        """Get detailed medicine information"""
        try:
            payload = {"medicine_name": medicine_name}
            response = requests.post(
                f"{self.prescription_endpoint}/medicine-info",
                json=payload,
                timeout=30,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {st.session_state.get('auth_token', 'demo_token')}"
                }
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"Failed to get medicine info: {response.status_code}"}
                
        except Exception as e:
            return {"error": f"Error getting medicine info: {str(e)}"}

def render_prescription_chatbot():
    """Render the enhanced prescription analysis chatbot interface"""
    
    # Initialize chatbot in session state
    if 'prescription_chatbot' not in st.session_state:
        st.session_state.prescription_chatbot = PrescriptionChatbot()
    
    if 'prescription_messages' not in st.session_state:
        st.session_state.prescription_messages = []
    
    # Professional header with updated styling
    st.markdown("""
    <div style="background: linear-gradient(135deg, var(--primary-color) 0%, var(--accent-color) 100%); 
                padding: 2.5rem; border-radius: var(--border-radius-lg); margin-bottom: 2rem; 
                color: white; text-align: center; box-shadow: var(--shadow-lg);">
        <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 50%; 
                    display: inline-flex; align-items: center; justify-content: center; 
                    font-size: 2.5rem; margin-bottom: 1rem;">🩺</div>
        <h1 style="margin: 0; font-size: 2.5rem; font-weight: 700;">AI Prescription Analyst</h1>
        <p style="margin: 0.75rem 0 0 0; font-size: 1.2rem; opacity: 0.95; line-height: 1.4;">
            Professional Medical Document Analysis & AI-Powered Recommendations
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main interface tabs
    tab1, tab2, tab3 = st.tabs(["📋 Upload Prescription", "💬 Chat Analysis", "📊 Medicine Database"])
    
    with tab1:
        render_prescription_upload()
    
    with tab2:
        render_chat_interface()
    
    with tab3:
        render_medicine_lookup()

def render_prescription_upload():
    """Render prescription upload interface"""
    st.markdown("""
    <div style="background: var(--background-primary); padding: 2rem; border-radius: var(--border-radius-lg); 
                margin-bottom: 2rem; border: 1px solid var(--border-color); box-shadow: var(--shadow-sm);">
        <h3 style="color: var(--primary-color); margin: 0 0 1rem 0; display: flex; align-items: center; gap: 0.75rem;">
            <span style="background: linear-gradient(135deg, var(--accent-color), var(--secondary-color)); 
                         color: white; padding: 0.5rem; border-radius: 50%; display: flex; 
                         align-items: center; justify-content: center;">📤</span>
            Upload Your Prescription
        </h3>
        <p style="color: var(--text-secondary); margin: 0; line-height: 1.6;">
            Upload a clear image of your prescription for AI-powered analysis and professional medical recommendations.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Professional upload area
    st.markdown("""
    <div style="border: 2px dashed var(--accent-color); border-radius: var(--border-radius-lg); 
                padding: 3rem; text-align: center; background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); 
                margin: 2rem 0; transition: all 0.3s ease;">
        <div style="background: var(--accent-color); color: white; padding: 1rem; border-radius: 50%; 
                    display: inline-flex; align-items: center; justify-content: center; 
                    font-size: 2rem; margin-bottom: 1rem;">📸</div>
        <h4 style="color: var(--primary-color); margin-bottom: 1rem; font-size: 1.25rem;">Prescription Upload Zone</h4>
        <p style="color: var(--text-secondary); margin: 0; font-size: 1rem;">
            Supported formats: JPG, PNG, BMP, TIFF, PDF • Maximum size: 10MB
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose prescription file",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'pdf'],
        help="Upload a clear image or PDF of your prescription for best results"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if uploaded_file.type == "application/pdf":
                st.markdown("#### 📄 Uploaded PDF")
                st.info("📋 PDF file uploaded successfully. The system will extract and analyze all pages.")
                st.markdown(f"**File:** {uploaded_file.name}")
                st.markdown(f"**Size:** {len(uploaded_file.getvalue()) / 1024:.1f} KB")
            else:
                st.markdown("#### 🖼️ Uploaded Image")
                image = Image.open(uploaded_file)
                st.image(image, caption="Prescription Image", use_column_width=True)
            
            # Image quality tips
            if uploaded_file.type == "application/pdf":
                st.info("""
                **📋 PDF Upload Tips:**
                - Ensure PDF contains clear, readable text
                - Scanned prescriptions work best
                - Multiple pages will be analyzed automatically
                - Original quality PDFs provide better results
                """)
            else:
                st.info("""
                **📋 For Best Results:**
                - Ensure good lighting
                - Keep image straight and focused
                - Include the entire prescription
                - Avoid shadows and glare
                """)
        
        with col2:
            st.markdown("#### 🔍 Analysis Options")
            
            # Analysis settings
            include_dosage = st.checkbox("📊 Extract dosage information", value=True)
            include_recommendations = st.checkbox("💡 Generate health recommendations", value=True)
            detailed_analysis = st.checkbox("🔬 Detailed medicine analysis", value=True)
            
            # Analyze button
            file_type_text = "PDF" if uploaded_file.type == "application/pdf" else "Image"
            if st.button(f"🚀 Analyze Prescription {file_type_text}", type="primary", use_container_width=True):
                analyze_prescription(uploaded_file, include_dosage, include_recommendations, detailed_analysis)

def analyze_prescription(uploaded_file, include_dosage, include_recommendations, detailed_analysis):
    """Analyze the uploaded prescription"""
    
    with st.spinner("🔍 Analyzing prescription... This may take a moment."):
        # Reset file pointer
        uploaded_file.seek(0)
        
        # Upload and analyze
        result = st.session_state.prescription_chatbot.upload_prescription(uploaded_file)
        
        if result.get('success'):
            st.success("✅ Prescription analyzed successfully!")
            
            # Store result in session state
            st.session_state.last_prescription_analysis = result
            
            # Display results
            display_prescription_results(result, include_dosage, include_recommendations, detailed_analysis)
            
            # Add to chat history
            add_prescription_to_chat(result)
            
        else:
            st.error(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
            st.info("💡 **Troubleshooting Tips:**\n- Check image quality and lighting\n- Ensure prescription text is clearly visible\n- Try a different image format\n- Make sure file size is under 10MB")

def display_prescription_results(result, include_dosage, include_recommendations, detailed_analysis):
    """Display prescription analysis results"""
    
    # Analysis summary
    st.markdown("### 📋 Analysis Summary")
    st.markdown(f"""
    <div style='background: #e8f5e8; padding: 1.5rem; border-radius: 10px; border-left: 5px solid #4CAF50;'>
        <h4 style='color: #2E7D32; margin-top: 0;'>🎯 Analysis Results</h4>
        <p style='margin-bottom: 0; font-size: 1.1rem;'>{result['analysis_summary']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Confidence score
    confidence = result.get('confidence_score', 0) * 100
    confidence_color = "#4CAF50" if confidence > 70 else "#FF9800" if confidence > 40 else "#F44336"
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🎯 Confidence Score", f"{confidence:.1f}%")
    with col2:
        st.metric("💊 Medicines Found", result.get('total_medicines_found', 0))
    with col3:
        st.metric("📄 Text Extracted", len(result.get('extracted_text', [])))
    
    # Identified medicines
    if result.get('identified_medicines'):
        st.markdown("### 💊 Identified Medicines")
        
        for i, medicine in enumerate(result['identified_medicines'], 1):
            with st.expander(f"🔍 {medicine['medicine_name']} - {medicine['category'].replace('_', ' ').title()}", expanded=True):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**Medicine:** {medicine['medicine_name']}")
                    st.markdown(f"**Category:** {medicine['category'].replace('_', ' ').title()}")
                    st.markdown(f"**Original Text:** {medicine['original_text']}")
                
                with col2:
                    confidence_badge = f"{medicine['confidence']*100:.1f}%"
                    st.markdown(f"**Confidence:** {confidence_badge}")
                    
                    if st.button(f"ℹ️ More Info", key=f"info_{i}"):
                        get_detailed_medicine_info(medicine['medicine_name'])
    
    # Dosage information
    if include_dosage and result.get('dosage_info'):
        st.markdown("### 📊 Dosage Information")
        dosage_info = result['dosage_info']
        
        col1, col2 = st.columns(2)
        
        with col1:
            if dosage_info.get('dosages'):
                st.markdown("**💊 Dosages Found:**")
                for dosage in dosage_info['dosages']:
                    st.markdown(f"• {dosage}")
            
            if dosage_info.get('frequencies'):
                st.markdown("**⏰ Frequencies:**")
                for freq in dosage_info['frequencies']:
                    st.markdown(f"• {freq}")
        
        with col2:
            if dosage_info.get('durations'):
                st.markdown("**📅 Durations:**")
                for duration in dosage_info['durations']:
                    st.markdown(f"• {duration}")
            
            if dosage_info.get('instructions'):
                st.markdown("**📋 Instructions:**")
                for instruction in dosage_info['instructions']:
                    st.markdown(f"• {instruction}")
    
    # Detailed Analysis Paragraph
    if result.get('detailed_analysis'):
        st.markdown("### 📋 Detailed Analysis Report")
        st.markdown(result['detailed_analysis'])
        
        # Download button for detailed report
        if result.get('downloadable_report'):
            st.download_button(
                label="📥 Download Full Report",
                data=result['downloadable_report'],
                file_name=f"prescription_analysis_{result.get('analysis_id', 'report')}.txt",
                mime="text/plain",
                help="Download a comprehensive analysis report"
            )
    
    # Enhanced medical advice
    if result.get('medical_advice'):
        st.markdown("### 👨‍⚕️ AI Medical Analysis")
        st.markdown(f"""
        <div style='background: #f8f9ff; padding: 1.5rem; border-radius: 10px; border-left: 5px solid #3f51b5; font-family: 'Segoe UI', sans-serif;'>
            {result['medical_advice'].replace('\n', '<br>').replace('**', '<strong>').replace('**', '</strong>')}
        </div>
        """, unsafe_allow_html=True)
    
    # Professional advice (legacy support)
    elif result.get('professional_advice'):
        st.markdown("### 👨‍⚕️ Professional Medical Advice")
        st.markdown(f"""
        <div style='background: #fff3e0; padding: 1.5rem; border-radius: 10px; border-left: 5px solid #FF9800;'>
            {result['professional_advice'].replace('\n', '<br>')}
        </div>
        """, unsafe_allow_html=True)
    
    # Health recommendations
    if include_recommendations and result.get('health_recommendations'):
        st.markdown("### 💡 Personalized Health Recommendations")
        
        for i, recommendation in enumerate(result['health_recommendations'], 1):
            st.markdown(f"**{i}.** {recommendation}")

def get_detailed_medicine_info(medicine_name):
    """Get and display detailed medicine information"""
    
    with st.spinner(f"Getting detailed information for {medicine_name}..."):
        info_result = st.session_state.prescription_chatbot.get_medicine_info(medicine_name)
        
        if 'error' not in info_result:
            st.markdown(f"### 📖 Detailed Information: {medicine_name}")
            
            # Medicine information
            st.markdown("#### 📋 General Information")
            st.markdown(info_result['information'])
            
            # Professional advice
            st.markdown("#### 👨‍⚕️ Professional Advice")
            st.markdown(info_result['professional_advice'])
            
            # Safety warnings
            if info_result.get('safety_warnings'):
                st.markdown("#### ⚠️ Safety Warnings")
                for warning in info_result['safety_warnings']:
                    st.warning(f"⚠️ {warning}")
        else:
            st.error(f"Failed to get medicine information: {info_result['error']}")

def render_chat_interface():
    """Render chat interface for prescription discussions"""
    st.markdown("### 💬 Prescription Discussion")
    st.markdown("Ask questions about your prescription analysis or get additional medical guidance.")
    
    # Display chat messages
    if st.session_state.prescription_messages:
        for message in st.session_state.prescription_messages:
            if message['role'] == 'user':
                st.markdown(f"""
                <div style='background: #e3f2fd; padding: 1rem; border-radius: 10px; margin: 0.5rem 0; margin-left: 2rem;'>
                    <strong>👤 You:</strong> {message['content']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style='background: #f3e5f5; padding: 1rem; border-radius: 10px; margin: 0.5rem 0; margin-right: 2rem;'>
                    <strong>🩺 AI Doctor:</strong> {message['content']}
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("💡 Upload a prescription first, then ask questions about your medications, dosages, or health recommendations.")
    
    # Chat input
    with st.form("prescription_chat_form", clear_on_submit=True):
        user_question = st.text_area(
            "Ask about your prescription:",
            placeholder="e.g., What are the side effects of my medications? How should I take these medicines? Are there any interactions?",
            height=100
        )
        
        col1, col2 = st.columns([3, 1])
        with col2:
            submit_chat = st.form_submit_button("💬 Ask", type="primary", use_container_width=True)
        
        if submit_chat and user_question.strip():
            process_prescription_question(user_question.strip())

def render_medicine_lookup():
    """Render medicine database lookup interface"""
    st.markdown("### 🔍 Medicine Information Lookup")
    st.markdown("Search our comprehensive medicine database for detailed information about any medication.")
    
    # Search interface
    with st.form("medicine_search_form"):
        medicine_name = st.text_input(
            "Enter medicine name:",
            placeholder="e.g., Paracetamol, Ibuprofen, Amoxicillin",
            help="Enter the generic or brand name of the medicine"
        )
        
        search_button = st.form_submit_button("🔍 Search Medicine", type="primary")
        
        if search_button and medicine_name.strip():
            get_detailed_medicine_info(medicine_name.strip())
    
    # Common medicines quick access
    st.markdown("#### 🏥 Common Medicines")
    
    common_medicines = [
        "Paracetamol", "Ibuprofen", "Amoxicillin", "Aspirin", 
        "Metformin", "Lisinopril", "Omeprazole", "Atorvastatin"
    ]
    
    cols = st.columns(4)
    for i, medicine in enumerate(common_medicines):
        with cols[i % 4]:
            if st.button(f"💊 {medicine}", key=f"common_{medicine}"):
                get_detailed_medicine_info(medicine)

def process_prescription_question(question):
    """Process user question about prescription"""
    
    # Add user message
    st.session_state.prescription_messages.append({
        'role': 'user',
        'content': question,
        'timestamp': datetime.now()
    })
    
    # Generate AI response (this would integrate with the chatbot API)
    with st.spinner("🤔 AI Doctor is thinking..."):
        # For now, provide a template response
        # In production, this would call the chatbot API with prescription context
        
        response = generate_prescription_response(question)
        
        st.session_state.prescription_messages.append({
            'role': 'assistant',
            'content': response,
            'timestamp': datetime.now()
        })
    
    st.rerun()

def generate_prescription_response(question):
    """Generate enhanced AI response for prescription questions"""
    
    # Get last prescription analysis if available
    last_analysis = st.session_state.get('last_prescription_analysis')
    
    if not last_analysis:
        return """**🩺 AI Medical Assistant:**

I'd be happy to help with your prescription questions! However, I don't see any recent prescription analysis. Please upload your prescription first so I can provide specific guidance about your medications.

**📋 General Medical Guidance:**
• **Medication Adherence:** Always take medications exactly as prescribed by your doctor
• **Dosage Timing:** Don't skip doses or stop medications without consulting your doctor
• **Side Effect Monitoring:** Report any unusual symptoms immediately to your doctor
• **Safe Storage:** Keep medications in original containers, away from children and pets
• **Expiration Dates:** Check and dispose of expired medications properly

**⚠️ Important:** For specific medication questions, always consult your doctor or pharmacist."""
    
    # Extract context from last analysis
    medicines = last_analysis.get('identified_medicines', [])
    medicine_names = [med['medicine_name'] for med in medicines]
    dosage_info = last_analysis.get('dosage_info', {})
    
    # Generate contextual response based on question content
    question_lower = question.lower()
    
    if 'side effect' in question_lower:
        return f"""**🩺 Side Effects Analysis for Your Medications:**

**📋 Your Identified Medications:** {', '.join(medicine_names)}

**⚠️ Common Side Effects to Monitor:**
• **Gastrointestinal:** Nausea, stomach upset, diarrhea (especially with oral medications)
• **Neurological:** Dizziness, drowsiness, headache, confusion
• **Dermatological:** Skin rash, itching, allergic reactions
• **Cardiovascular:** Changes in blood pressure, heart rate irregularities

**🚨 Serious Side Effects - Seek Immediate Medical Attention:**
• **Allergic Reactions:** Difficulty breathing, swelling of face/throat, severe rash
• **Severe GI Issues:** Persistent vomiting, severe abdominal pain, blood in stool
• **Cardiovascular:** Chest pain, severe dizziness, fainting, irregular heartbeat
• **Neurological:** Severe confusion, seizures, severe headache

**📞 When to Contact Your Doctor:**
• Any side effect that interferes with daily activities
• Side effects that worsen over time
• New symptoms after starting medication
• Questions about continuing medication

**💡 Management Tips:**
• Keep a medication diary to track side effects
• Take medications with food if stomach upset occurs
• Stay hydrated and maintain regular sleep schedule
• Don't stop medications abruptly without medical consultation

**⚕️ Medical Disclaimer:** This information is for educational purposes only. Always consult your doctor for personalized medical advice about your specific medications and health condition."""
    
    elif 'interaction' in question_lower:
        return f"""**🩺 Drug Interaction Analysis for Your Medications:**

**📋 Your Current Medications:** {', '.join(medicine_names)}

**⚠️ Important Drug Interaction Precautions:**

**🏥 Healthcare Communication:**
• **Complete Medication List:** Inform ALL doctors about every medication, supplement, and herbal product you take
• **Pharmacy Records:** Use one primary pharmacy to maintain complete medication records
• **Medical History:** Always mention allergies and previous adverse reactions

**🚫 Substances to Avoid or Use Cautiously:**
• **Alcohol:** Can interact dangerously with many medications - consult your doctor
• **OTC Medications:** Pain relievers, cold medicines, antacids can interact
• **Herbal Supplements:** St. John's Wort, ginkgo, garlic supplements can affect drug metabolism
• **Grapefruit Products:** Can interfere with many medications' effectiveness

**✅ Always Consult Before Adding:**
• New prescription medications from different doctors
• Over-the-counter pain relievers (ibuprofen, acetaminophen, aspirin)
• Vitamins and mineral supplements
• Herbal or natural products
• Someone else's medications (never share prescriptions)

**📱 Interaction Checking Tools:**
• Use pharmacy apps or websites to check interactions
• Consult your pharmacist - they are drug interaction experts
• Keep an updated medication list on your phone

**🚨 Warning Signs of Drug Interactions:**
• Unusual side effects after starting new medication
• Existing medication seems less effective
• Unexpected symptoms or feeling unwell
• Changes in how you normally feel on your medications

**⚕️ Professional Recommendation:** Always consult your pharmacist or doctor before adding any new medications, supplements, or making changes to your current regimen. They can perform comprehensive interaction checks specific to your medication profile."""
    
    elif 'how to take' in question_lower or 'dosage' in question_lower:
        dosage_details = ""
        if dosage_info.get('dosages'):
            dosage_details += f"\n**🔍 Detected Dosages:** {', '.join(dosage_info['dosages'][:3])}"
        if dosage_info.get('frequencies'):
            dosage_details += f"\n**⏰ Detected Frequencies:** {', '.join(dosage_info['frequencies'][:3])}"
        
        return f"""**🩺 Medication Administration Guide:**

**📋 Your Prescribed Medications:** {', '.join(medicine_names)}{dosage_details}

**💊 Dosage Instructions:**
• **Exact Dosing:** Take exactly as prescribed - never adjust doses without medical consultation
• **Consistent Timing:** Take at the same times each day to maintain steady medication levels
• **Complete Course:** Finish entire prescription, especially antibiotics, even if feeling better
• **Medication Reminders:** Use pill organizers, phone alarms, or medication apps

**🍽️ Food and Medication Timing:**
• **With Food:** Reduces stomach irritation for many medications
• **Empty Stomach:** Some medications absorb better without food
• **Specific Instructions:** Always follow label directions for food timing
• **Consistency:** Take the same way each time (always with food or always without)

**⏰ Missed Dose Protocol:**
• **Soon After:** Take as soon as you remember if it's within a few hours
• **Close to Next Dose:** Skip missed dose if it's almost time for the next one
• **Never Double:** Don't take two doses at once to make up for missed dose
• **Frequent Misses:** Contact your doctor if you often forget doses

**📱 Medication Management Tips:**
• **Pill Organizer:** Weekly organizers help track daily medications
• **Phone Apps:** Medication reminder apps with alarm features
• **Routine Integration:** Link medication times to daily activities (meals, bedtime)
• **Travel Planning:** Bring extra medication when traveling

**🚨 When to Contact Your Doctor:**
• Difficulty swallowing pills or keeping medication down
• Frequent missed doses affecting treatment
• Questions about timing with other medications
• Side effects that interfere with taking medication

**⚕️ Important:** These are general guidelines. Always follow the specific instructions provided by your doctor and pharmacist for your individual medications."""
    
    else:
        return f"""**🩺 Comprehensive Medication Guidance:**

**📋 Your Current Prescription:** {', '.join(medicine_names)}

**💊 Essential Medication Management:**

**1. ⏰ Timing & Consistency**
• Take medications at the same times each day for optimal effectiveness
• Set multiple alarms or use medication reminder apps
• Maintain consistent intervals between doses as prescribed

**2. 🏠 Proper Storage**
• Keep medications in original labeled containers
• Store in cool, dry place away from bathroom humidity
• Protect from direct sunlight and extreme temperatures
• Keep away from children and pets

**3. 📊 Health Monitoring**
• Track both positive effects and any side effects
• Monitor symptoms your medication is treating
• Keep a medication diary for doctor visits
• Note any changes in how you feel

**4. 💬 Healthcare Communication**
• Inform all doctors about current medications
• Report side effects promptly to your prescribing doctor
• Ask questions during pharmacy visits
• Keep an updated medication list in your wallet/phone

**📞 When to Contact Medical Professionals:**

**🏥 Your Prescribing Doctor:**
• Medical concerns or worsening symptoms
• Side effects affecting quality of life
• Questions about treatment effectiveness
• Need for dosage adjustments

**💊 Your Pharmacist:**
• Drug interaction questions
• Proper administration techniques
• Generic vs. brand name questions
• Storage and handling instructions

**🚨 Emergency Situations:**
• Severe allergic reactions
• Difficulty breathing after taking medication
• Severe side effects or poisoning symptoms
• Accidental overdose

**✅ Medication Best Practices:**
• Never share prescription medications
• Don't stop medications abruptly without medical approval
• Check expiration dates regularly
• Dispose of unused medications safely
• Bring medication list to all medical appointments

**⚕️ Medical Disclaimer:** This AI guidance is for educational purposes only and does not replace professional medical advice. Always consult your doctor or pharmacist for personalized medical guidance regarding your specific medications and health conditions."""

def add_prescription_to_chat(analysis_result):
    """Add enhanced prescription analysis to chat history"""
    
    medicines = analysis_result.get('identified_medicines', [])
    medicine_names = [med['medicine_name'] for med in medicines]
    confidence = analysis_result.get('confidence_score', 0) * 100
    
    if medicines:
        summary_message = f"""**🩺 Prescription Analysis Complete - Medical Report**

**✅ Successfully Identified Medications:** {len(medicines)} medication(s)
**💊 Medications Found:** {', '.join(medicine_names)}
**📊 Analysis Confidence:** {confidence:.1f}%

**🏥 Therapeutic Categories Detected:**
{chr(10).join([f'• {med["category"].replace("_", " ").title()}: {med["medicine_name"]}' for med in medicines[:5]])}

**📋 Analysis Summary:**
{analysis_result.get('analysis_summary', 'Analysis completed successfully')}

**💬 Ask Me About:**
• Medication side effects and interactions
• Proper dosage and administration instructions
• Storage and safety guidelines
• Drug interaction warnings
• General health recommendations

**⚕️ Ready to answer your medical questions!**"""
    else:
        summary_message = f"""**🩺 Prescription Analysis Complete**

**⚠️ Analysis Result:** No medications were clearly identified from the uploaded prescription.

**📊 Analysis Confidence:** {confidence:.1f}%

**🔍 Possible Reasons:**
• Image quality or lighting issues
• Handwriting clarity challenges
• Document format or angle
• OCR technical limitations

**💡 Recommendations:**
• Try retaking the photo with better lighting
• Ensure the prescription is clearly visible
• Take the photo from directly above
• Consider uploading a PDF if available

**💬 I can still help with:**
• General medication questions
• Drug interaction information
• Medication safety guidelines
• Health recommendations

**⚕️ Feel free to ask any medical questions!**"""
    
    st.session_state.prescription_messages.append({
        'role': 'assistant',
        'content': summary_message,
        'timestamp': datetime.now()
    })
