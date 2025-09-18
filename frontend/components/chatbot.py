import streamlit as st
import requests
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import time

class HealthcareChatbot:
    def __init__(self, api_base_url: str = "http://localhost:8002/api"):
        self.api_base_url = api_base_url
        self.chat_endpoint = f"{api_base_url}/chat"
        self.health_endpoint = f"{api_base_url}/chat/health"
        self.explain_endpoint = f"{api_base_url}/chat/explain-risk"
    
    def check_chatbot_health(self) -> Dict[str, Any]:
        """Check if chatbot service is available"""
        try:
            response = requests.get(self.health_endpoint, timeout=5)
            return response.json()
        except requests.exceptions.RequestException:
            return {"status": "error", "message": "Chatbot service unavailable"}
    
    def send_message(self, message: str, conversation_history: Optional[List[Dict]] = None, 
                    patient_context: Optional[Dict] = None, include_health_data: bool = False) -> Dict[str, Any]:
        """Send a message to the chatbot"""
        try:
            payload = {
                "message": message,
                "conversation_history": conversation_history or [],
                "patient_context": patient_context,
                "include_health_data": include_health_data
            }
            
            response = requests.post(
                self.chat_endpoint,
                json=payload,
                timeout=30,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                error_detail = response.json().get("detail", "Unknown error")
                return {
                    "error": f"API Error ({response.status_code}): {error_detail}",
                    "response": "I'm sorry, I'm having trouble processing your request right now. Please try again later."
                }
                
        except requests.exceptions.Timeout:
            return {
                "error": "Request timeout",
                "response": "I'm taking longer than usual to respond. Please try again."
            }
        except requests.exceptions.RequestException as e:
            return {
                "error": f"Connection error: {str(e)}",
                "response": "I'm having trouble connecting to the AI service. Please check your connection and try again."
            }
        except Exception as e:
            return {
                "error": f"Unexpected error: {str(e)}",
                "response": "Something unexpected happened. Please try again."
            }
    
    def explain_risk_assessment(self, risk_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get explanation for risk assessment results"""
        try:
            response = requests.post(
                self.explain_endpoint,
                json=risk_data,
                timeout=20,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"Failed to get risk explanation: {response.status_code}"}
                
        except Exception as e:
            return {"error": f"Error explaining risk: {str(e)}"}

def process_message_automatically(message: str, include_health_data: bool = True, use_conversation_context: bool = True):
    """Helper function to automatically process a message through the AI"""
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = HealthcareChatbot()
    
    # Add user message to history
    st.session_state.chat_messages.append({
        "role": "user",
        "content": message,
        "timestamp": datetime.now()
    })
    
    # Prepare conversation history for API
    conversation_history = []
    if use_conversation_context:
        for msg in st.session_state.chat_messages[-10:]:  # Last 10 messages
            if msg["role"] in ["user", "assistant"]:
                conversation_history.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
    
    # Get patient context if available
    patient_context = st.session_state.get('patient_data') if include_health_data else None
    
    try:
        # Send message to chatbot
        response_data = st.session_state.chatbot.send_message(
            message=message,
            conversation_history=conversation_history[:-1] if use_conversation_context else [],
            patient_context=patient_context,
            include_health_data=include_health_data
        )
        
        # Add assistant response to history
        assistant_msg = {
            "role": "assistant",
            "content": response_data.get("response", "I'm sorry, I couldn't process your request."),
            "timestamp": datetime.now(),
            "suggestions": response_data.get("suggestions", []),
            "requires_medical_attention": response_data.get("requires_medical_attention", False)
        }
        
        st.session_state.chat_messages.append(assistant_msg)
        
        # Show success toast
        if "error" not in response_data:
            st.toast("✅ AI response received!", icon="🤖")
        else:
            st.error(f"⚠️ {response_data['error']}")
    
    except Exception as e:
        st.error(f"Error processing message: {str(e)}")

def render_chatbot_interface():
    """Render the chatbot interface in Streamlit"""
    
    # Initialize chatbot in session state
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = HealthcareChatbot()
    
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []
    
    if 'chatbot_ready' not in st.session_state:
        st.session_state.chatbot_ready = False
    
    # Check chatbot health on first load
    if not st.session_state.chatbot_ready:
        health_status = st.session_state.chatbot.check_chatbot_health()
        st.session_state.chatbot_ready = health_status.get("status") == "healthy"
        st.session_state.chatbot_error = health_status.get("message", "") if not st.session_state.chatbot_ready else None
    
    # Professional status indicator
    if st.session_state.chatbot_ready:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); 
                    padding: 1.5rem; border-radius: var(--border-radius-lg); margin-bottom: 2rem; 
                    border: 1px solid #10b981; box-shadow: var(--shadow-sm);">
            <div style="display: flex; align-items: center; gap: 1rem;">
                <div style="background: #10b981; color: white; padding: 0.75rem; border-radius: 50%; 
                            display: flex; align-items: center; justify-content: center; font-size: 1.25rem;">🤖</div>
                <div>
                    <h3 style="color: #065f46; margin: 0; font-size: 1.25rem;">AI Assistant Ready</h3>
                    <p style="color: #047857; margin: 0; font-size: 0.95rem;">Your intelligent healthcare companion is online and ready to assist</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%); 
                    padding: 1.5rem; border-radius: var(--border-radius-lg); margin-bottom: 2rem; 
                    border: 1px solid #f87171; box-shadow: var(--shadow-sm);">
            <div style="display: flex; align-items: center; gap: 1rem;">
                <div style="background: #ef4444; color: white; padding: 0.75rem; border-radius: 50%; 
                            display: flex; align-items: center; justify-content: center; font-size: 1.25rem;">⚠️</div>
                <div>
                    <h3 style="color: #dc2626; margin: 0; font-size: 1.25rem;">AI Assistant Unavailable</h3>
                    <p style="color: #b91c1c; margin: 0; font-size: 0.95rem;">{}</p>
                </div>
            </div>
        </div>
        """.format(st.session_state.get('chatbot_error', 'Unknown error')), unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: var(--background-primary); padding: 2rem; border-radius: var(--border-radius-lg); 
                    border: 1px solid var(--border-color); box-shadow: var(--shadow-sm);">
            <h4 style="color: var(--primary-color); margin-top: 0;">🔧 Setup Instructions</h4>
            <ol style="color: var(--text-secondary); line-height: 1.6;">
                <li>Install required packages: <code>pip install openai</code></li>
                <li>Set your OpenAI API key: <code>export OPENAI_API_KEY=your_key_here</code></li>
                <li>Restart the backend server</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Professional chat interface
    st.markdown("""
    <div style="background: var(--background-primary); padding: 1.5rem; border-radius: var(--border-radius-lg); 
                margin-bottom: 1.5rem; border: 1px solid var(--border-color); box-shadow: var(--shadow-sm);">
        <h3 style="color: var(--primary-color); margin: 0; display: flex; align-items: center; gap: 0.75rem;">
            <span style="background: linear-gradient(135deg, var(--primary-color), var(--accent-color)); 
                         -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
                         background-clip: text;">💬 Conversation</span>
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Display conversation history
    chat_container = st.container()
    with chat_container:
        if not st.session_state.chat_messages:
            st.markdown("""
            <div style="background: linear-gradient(135deg, var(--background-primary) 0%, #f8fafc 100%); 
                        padding: 3rem; border-radius: var(--border-radius-lg); text-align: center; 
                        border: 1px solid var(--border-color); box-shadow: var(--shadow-sm); margin-bottom: 2rem;">
                <div style="background: linear-gradient(135deg, var(--primary-color), var(--accent-color)); 
                            color: white; padding: 1.5rem; border-radius: 50%; display: inline-flex; 
                            align-items: center; justify-content: center; font-size: 2rem; margin-bottom: 1.5rem;">🤖</div>
                <h3 style="color: var(--primary-color); margin-bottom: 1rem; font-size: 1.5rem;">Welcome to Your AI Health Assistant</h3>
                <p style="color: var(--text-secondary); margin: 0; font-size: 1.1rem; line-height: 1.6; max-width: 600px; margin: 0 auto;">
                    I'm here to help you understand your health data, explain medical terms, and provide personalized health guidance. 
                    Start by asking me anything about your health!
                </p>
                <div style="margin-top: 2rem; display: flex; gap: 1rem; justify-content: center; flex-wrap: wrap;">
                    <span style="background: var(--accent-color); color: white; padding: 0.5rem 1rem; 
                                 border-radius: var(--border-radius-md); font-size: 0.9rem;">Risk Assessment</span>
                    <span style="background: var(--secondary-color); color: white; padding: 0.5rem 1rem; 
                                 border-radius: var(--border-radius-md); font-size: 0.9rem;">Health Tips</span>
                    <span style="background: var(--primary-color); color: white; padding: 0.5rem 1rem; 
                                 border-radius: var(--border-radius-md); font-size: 0.9rem;">Medical Guidance</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        for i, msg in enumerate(st.session_state.chat_messages):
            if msg["role"] == "user":
                st.markdown(f"""
                <div style="display: flex; justify-content: flex-end; margin-bottom: 1.5rem;">
                    <div style="background: linear-gradient(135deg, var(--primary-color), var(--accent-color)); 
                                color: white; padding: 1rem 1.5rem; border-radius: 1.5rem 1.5rem 0.5rem 1.5rem; 
                                max-width: 70%; box-shadow: var(--shadow-sm); position: relative;">
                        <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem; 
                                    font-size: 0.85rem; opacity: 0.9;">
                            <span>👤</span> <strong>You</strong>
                        </div>
                        <div style="line-height: 1.5; font-size: 1rem;">{msg['content']}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Check if this is a medical attention response
                requires_attention = msg.get('requires_medical_attention', False)
                bg_color = "linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%)" if requires_attention else "var(--background-primary)"
                border_color = "#f87171" if requires_attention else "var(--border-color)"
                icon = "⚠️" if requires_attention else "🤖"
                
                st.markdown(f"""
                <div style="display: flex; justify-content: flex-start; margin-bottom: 1.5rem;">
                    <div style="background: {bg_color}; color: var(--text-primary); 
                                padding: 1rem 1.5rem; border-radius: 1.5rem 1.5rem 1.5rem 0.5rem; 
                                max-width: 75%; box-shadow: var(--shadow-sm); border: 1px solid {border_color};">
                        <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem; 
                                    font-size: 0.85rem; color: var(--text-secondary);">
                            <span>{icon}</span> <strong>AI Assistant</strong>
                            {"<span style='margin-left: 0.5rem; font-size: 0.75rem; background: #dc2626; color: white; padding: 0.25rem 0.5rem; border-radius: 0.75rem;'>Medical Attention</span>" if requires_attention else ""}
                        </div>
                        <div style="line-height: 1.6; font-size: 1rem; color: var(--text-primary);">{msg['content']}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Show suggestions if available
                if msg.get('suggestions'):
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); 
                                padding: 1.5rem; border-radius: var(--border-radius-lg); margin: 1rem 0; 
                                border: 1px solid #0ea5e9; box-shadow: var(--shadow-sm);">
                        <h4 style="color: #0369a1; margin: 0 0 1rem 0; display: flex; align-items: center; gap: 0.5rem;">
                            💡 Suggested Follow-up Questions
                        </h4>
                        <p style="color: #0284c7; margin: 0 0 1rem 0; font-size: 0.95rem;">
                            Click any question below to ask it automatically:
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Create columns for better layout
                    cols = st.columns(2) if len(msg['suggestions']) > 2 else [st.container()]
                    for idx, suggestion in enumerate(msg['suggestions']):
                        col_idx = idx % len(cols)
                        with cols[col_idx]:
                            suggestion_key = f"suggestion_{i}_{hash(suggestion)}"
                            if st.button(
                                f"💭 {suggestion}", 
                                key=suggestion_key, 
                                use_container_width=True,
                                help=f"Click to automatically ask: '{suggestion}'"
                            ):
                                # Show processing indicator
                                with st.spinner("🤔 Processing your question..."):
                                    # Automatically process the suggestion through AI
                                    process_message_automatically(suggestion, include_health_data=True, use_conversation_context=True)
                                st.rerun()
    
    # Professional input area
    st.markdown("""
    <div style="background: var(--background-primary); padding: 2rem; border-radius: var(--border-radius-lg); 
                margin: 2rem 0; border: 1px solid var(--border-color); box-shadow: var(--shadow-sm);">
        <h3 style="color: var(--primary-color); margin: 0 0 1.5rem 0; display: flex; align-items: center; gap: 0.75rem;">
            <span style="background: linear-gradient(135deg, var(--primary-color), var(--accent-color)); 
                         color: white; padding: 0.5rem; border-radius: 50%; display: flex; 
                         align-items: center; justify-content: center;">📝</span>
            Send a Message
        </h3>
    """, unsafe_allow_html=True)
    
    with st.form("chat_form", clear_on_submit=True):
        # Message input with professional styling
        user_input = st.text_area(
            "Ask me anything about your health...", 
            placeholder="e.g., What does my risk assessment mean? How can I improve my heart health? What do these symptoms suggest?",
            key="chat_input",
            height=120,
            help="Type your health-related question here. I can help explain medical terms, analyze your health data, and provide general health guidance."
        )
        
        # Professional options row
        st.markdown("""
        <div style="background: #f8fafc; padding: 1rem; border-radius: var(--border-radius-md); 
                    margin: 1rem 0; border: 1px solid #e2e8f0;">
            <h5 style="color: var(--text-primary); margin: 0 0 0.75rem 0; font-size: 0.95rem;">⚙️ Message Options</h5>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            include_health_data = st.checkbox(
                "📈 Include my health data", 
                value=True, 
                help="Allow the AI to reference your current health metrics for personalized responses"
            )
        
        with col2:
            conversation_context = st.checkbox(
                "📜 Use conversation history", 
                value=True,
                help="Allow the AI to reference previous messages in this conversation"
            )
            
        with col3:
            submit_button = st.form_submit_button(
                "💬 Send Message", 
                use_container_width=True,
                type="primary"
            )
        
        if submit_button and user_input.strip():
            # Add user message to history
            st.session_state.chat_messages.append({
                "role": "user",
                "content": user_input.strip(),
                "timestamp": datetime.now()
            })
            
            # Get patient context if available
            patient_context = st.session_state.get('patient_data') if include_health_data else None
            
            # Show professional thinking indicator
            thinking_placeholder = st.empty()
            with thinking_placeholder:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); 
                            padding: 2rem; border-radius: var(--border-radius-lg); text-align: center; 
                            border: 1px solid #0ea5e9; margin: 1rem 0;">
                    <div style="background: linear-gradient(135deg, #0ea5e9, #0284c7); color: white; 
                                padding: 1rem; border-radius: 50%; display: inline-flex; 
                                align-items: center; justify-content: center; font-size: 1.5rem; 
                                margin-bottom: 1rem; animation: pulse 2s infinite;">🤔</div>
                    <h4 style="color: #0369a1; margin: 0;">AI is analyzing your question...</h4>
                    <p style="color: #0284c7; margin: 0.5rem 0 0 0; font-size: 0.9rem;">Processing your health inquiry with advanced AI</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Send message to chatbot
                response_data = st.session_state.chatbot.send_message(
                    message=user_input.strip(),
                    conversation_history=[{"role": "user", "content": msg["content"]} for msg in st.session_state.chat_messages[:-1]],
                    patient_context=patient_context,
                    include_health_data=include_health_data
                )
            
            thinking_placeholder.empty()
            
            # Add assistant response to history
            assistant_msg = {
                "role": "assistant",
                "content": response_data.get("response", "I'm sorry, I couldn't process your request."),
                "timestamp": datetime.now(),
                "suggestions": response_data.get("suggestions", []),
                "requires_medical_attention": response_data.get("requires_medical_attention", False)
            }
            
            st.session_state.chat_messages.append(assistant_msg)
            
            # Show error if any
            if "error" in response_data:
                st.error(f"⚠️ {response_data['error']}")
            
            st.rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Professional quick action buttons
    if st.session_state.chat_messages:
        st.markdown("""
        <div style="background: var(--background-primary); padding: 2rem; border-radius: var(--border-radius-lg); 
                    margin: 2rem 0; border: 1px solid var(--border-color); box-shadow: var(--shadow-sm);">
            <h3 style="color: var(--primary-color); margin: 0 0 1.5rem 0; display: flex; align-items: center; gap: 0.75rem;">
                <span style="background: linear-gradient(135deg, var(--accent-color), var(--secondary-color)); 
                             color: white; padding: 0.5rem; border-radius: 50%; display: flex; 
                             align-items: center; justify-content: center;">⚡</span>
                Quick Actions
            </h3>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Clear Chat", help="Clear conversation history", use_container_width=True):
                st.session_state.chat_messages = []
                st.rerun()
        
        with col2:
            if st.button("📊 Explain My Risk", help="Get detailed explanation of your risk assessment", use_container_width=True):
                if 'last_prediction' in st.session_state:
                    # Create a detailed risk explanation request
                    risk_data = st.session_state.last_prediction
                    risk_message = f"Please explain my cardiovascular risk assessment. My risk level is {risk_data.get('risk_level', 'Unknown')} with a probability of {risk_data.get('risk_probability', 0):.1%}. What does this mean and what should I do?"
                    # Automatically process the risk explanation request
                    process_message_automatically(risk_message, include_health_data=True, use_conversation_context=True)
                    st.rerun()
                else:
                    st.warning("No risk assessment available. Please run a prediction first.")
        
        with col3:
            if st.button("💡 Health Tips", help="Get personalized health tips", use_container_width=True):
                tip_message = "Can you give me some personalized health tips based on my current health data?"
                # Automatically process the health tips request
                process_message_automatically(tip_message, include_health_data=True, use_conversation_context=True)
                st.rerun()

def render_chatbot_sidebar():
    """Render a compact chatbot in the sidebar"""
    st.sidebar.markdown("""
    <div style="background: linear-gradient(135deg, var(--primary-color), var(--accent-color)); 
                padding: 1.5rem; border-radius: var(--border-radius-lg); margin-bottom: 1rem; color: white;">
        <h3 style="margin: 0; display: flex; align-items: center; gap: 0.5rem;">
            🤖 Quick Health Chat
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize if needed
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = HealthcareChatbot()
    
    # Quick health check
    health_status = st.session_state.chatbot.check_chatbot_health()
    if health_status.get("status") != "healthy":
        st.sidebar.error("🔴 AI Assistant offline")
        return
    
    st.sidebar.success("🟢 AI Ready")
    
    # Simple chat input
    with st.sidebar.form("sidebar_chat"):
        quick_question = st.text_input(
            "Ask a quick question:", 
            placeholder="e.g., Is my blood pressure normal?",
            help="Quick health questions for instant answers"
        )
        if st.form_submit_button("💬 Ask", use_container_width=True):
            if quick_question.strip():
                # Process the question through the main chatbot and add to chat history
                process_message_automatically(quick_question.strip(), include_health_data=True, use_conversation_context=True)
                st.sidebar.success("✅ Question sent!")
                st.sidebar.info("📱 Check AI Assistant page for full conversation")