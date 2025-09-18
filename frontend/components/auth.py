"""
Authentication components for the Streamlit frontend
"""
import streamlit as st
import requests
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import time


class AuthManager:
    """Handles authentication state and API calls"""
    
    def __init__(self, backend_url: str = "http://localhost:8002"):
        self.backend_url = backend_url
        self.auth_endpoint = f"{backend_url}/api/auth"
    
    def login(self, email: str, password: str) -> tuple[bool, str, dict]:
        """
        Attempt to log in user
        Returns: (success, message, user_data)
        """
        try:
            response = requests.post(
                f"{self.auth_endpoint}/login",
                json={"email": email, "password": password},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return True, "Login successful!", data
            elif response.status_code == 401:
                return False, "Invalid email or password", {}
            elif response.status_code == 423:
                return False, "Account is temporarily locked due to too many failed attempts", {}
            elif response.status_code == 403:
                return False, "Account is inactive. Please contact administrator.", {}
            else:
                error_detail = response.json().get("detail", "Login failed")
                return False, error_detail, {}
                
        except requests.RequestException as e:
            return False, f"Connection error: {str(e)}", {}
        except Exception as e:
            return False, f"Unexpected error: {str(e)}", {}
    
    def logout(self, token: str) -> bool:
        """Log out user"""
        try:
            headers = {"Authorization": f"Bearer {token}"}
            response = requests.post(
                f"{self.auth_endpoint}/logout",
                headers=headers,
                timeout=5
            )
            return response.status_code == 200
        except:
            return True  # Always allow logout on frontend
    
    def verify_token(self, token: str) -> tuple[bool, dict]:
        """Verify if token is still valid"""
        try:
            headers = {"Authorization": f"Bearer {token}"}
            response = requests.get(
                f"{self.auth_endpoint}/verify-token",
                headers=headers,
                timeout=5
            )
            
            if response.status_code == 200:
                return True, response.json()
            else:
                return False, {}
        except:
            return False, {}
    
    def get_user_info(self, token: str) -> Optional[dict]:
        """Get current user information"""
        try:
            headers = {"Authorization": f"Bearer {token}"}
            response = requests.get(
                f"{self.auth_endpoint}/me",
                headers=headers,
                timeout=5
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return None
        except:
            return None
    
    def get_default_credentials(self) -> dict:
        """Get default test credentials - returns empty dict as we don't use default credentials in production"""
        return {}
    
    def signup(self, email: str, password: str, full_name: str, role: str = "patient") -> tuple[bool, str, dict]:
        """
        Attempt to sign up a new user
        Returns: (success, message, user_data)
        """
        try:
            # Use the public signup endpoint
            response = requests.post(
                f"{self.auth_endpoint}/signup",
                json={
                    "email": email, 
                    "password": password, 
                    "full_name": full_name, 
                    "role": role
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return True, "Account created successfully! Please log in.", data
            elif response.status_code == 400:
                error_detail = response.json().get("detail", "Signup failed")
                return False, error_detail, {}
            else:
                error_detail = response.json().get("detail", "Signup failed")
                return False, error_detail, {}
                
        except requests.RequestException as e:
            return False, f"Connection error: {str(e)}", {}
        except Exception as e:
            return False, f"Unexpected error: {str(e)}", {}


def init_session_state():
    """Initialize authentication-related session state"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user_data' not in st.session_state:
        st.session_state.user_data = {}
    if 'access_token' not in st.session_state:
        st.session_state.access_token = None
    if 'login_time' not in st.session_state:
        st.session_state.login_time = None
    if 'auth_manager' not in st.session_state:
        st.session_state.auth_manager = AuthManager()
    
    # Initialize health log data
    if 'health_log_data' not in st.session_state:
        import pandas as pd
        st.session_state.health_log_data = pd.DataFrame()


def check_session_validity():
    """Check if current session is still valid with comprehensive error handling"""
    if not st.session_state.authenticated:
        return False
    
    try:
        # Check token expiry (basic check based on login time)
        if st.session_state.login_time:
            login_time = datetime.fromisoformat(st.session_state.login_time)
            if datetime.now() - login_time > timedelta(hours=1):
                logout_user()
                return False
        
        # Verify token with backend (with error handling)
        if st.session_state.access_token:
            try:
                valid, _ = st.session_state.auth_manager.verify_token(st.session_state.access_token)
                if not valid:
                    logout_user()
                    return False
            except Exception as e:
                # If token verification fails due to network issues, allow session to continue
                # but log the error for debugging
                print(f"Token verification failed: {e}")
                # Only logout if it's clearly an authentication error
                if "401" in str(e) or "403" in str(e):
                    logout_user()
                    return False
        
        return True
        
    except Exception as e:
        # Handle any unexpected errors in session validation
        print(f"Session validation error: {e}")
        return False


def logout_user():
    """Log out the current user"""
    if st.session_state.access_token:
        st.session_state.auth_manager.logout(st.session_state.access_token)
    
    # Clear session state
    st.session_state.authenticated = False
    st.session_state.user_data = {}
    st.session_state.access_token = None
    st.session_state.login_time = None
    
    st.success("Logged out successfully!")
    st.rerun()


def show_development_credentials():
    """Show database user info instead of test credentials"""
    pass


def render_login_page():
    """Render a professional login page with modern design"""
    # Professional login page styling
    st.markdown("""
    <style>
    :root {
        --primary-color: #059669;
        --primary-dark: #047857;
        --secondary-color: #f0fdf4;
        --accent-color: #10b981;
        --background-primary: #ffffff;
        --background-secondary: #f7fef9;
        --text-primary: #064e3b;
        --text-secondary: #065f46;
        --border-color: #d1fae5;
        --shadow-sm: 0 1px 2px 0 rgb(5 150 105 / 0.05);
        --shadow-md: 0 4px 6px -1px rgb(5 150 105 / 0.1), 0 2px 4px -2px rgb(5 150 105 / 0.1);
        --shadow-lg: 0 10px 15px -3px rgb(5 150 105 / 0.1), 0 4px 6px -4px rgb(5 150 105 / 0.1);
        --border-radius-sm: 0.375rem;
        --border-radius-md: 0.5rem;
        --border-radius-lg: 0.75rem;
        --border-radius-xl: 1rem;
    }
    .login-container {
        max-width: 450px;
        margin: 3rem auto;
        padding: 3rem 2.5rem;
        background: linear-gradient(135deg, var(--background-primary) 0%, var(--secondary-color) 100%);
        border-radius: var(--border-radius-lg);
        box-shadow: var(--shadow-lg);
        border: 1px solid var(--border-color);
    }
    .login-header {
        text-align: center;
        margin-bottom: 2.5rem;
    }
    .login-title {
        font-size: 2.5rem;
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-dark) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
        margin-bottom: 0.75rem;
        font-family: 'Inter', sans-serif;
    }
    .login-subtitle {
        color: var(--text-secondary);
        font-size: 1.1rem;
        margin-bottom: 0;
        font-weight: 400;
    }
    .form-container {
        margin-top: 2rem;
    }
    .tab-container {
        margin-bottom: 2rem;
    }
    .welcome-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.5rem 1rem;
        background: linear-gradient(135deg, var(--accent-color), var(--primary-color));
        color: white;
        border-radius: 50px;
        font-size: 0.875rem;
        font-weight: 500;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    
    # Center the login form
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        
        # Header section
        st.markdown("""
        <div class="login-header">
            <h1 class="login-title">💚 MyVitals</h1>
            <p class="login-subtitle">Your Personal Health Companion</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Tab selection for Login/Signup
        tab1, tab2 = st.tabs(["🔐 Sign In", "📝 Sign Up"])
        
        with tab1:
            # Login form
            with st.form("login_form", clear_on_submit=False):
                st.markdown('<div class="form-container">', unsafe_allow_html=True)
                
                # Disable autofill and clear any default values
                st.markdown("""
                <style>
                /* Disable all autofill */
                input:-webkit-autofill,
                input:-webkit-autofill:hover, 
                input:-webkit-autofill:focus, 
                input:-webkit-autofill:active  {
                    -webkit-box-shadow: 0 0 0 30px white inset !important;
                }
                /* Hide autofill icons */
                input::-webkit-credentials-auto-fill-button {
                    visibility: hidden;
                    pointer-events: none;
                    position: absolute;
                    right: 0;
                }
                </style>
                """, unsafe_allow_html=True)
                
                # Create form with autocomplete off and clear default values
                email = st.text_input(
                    "Email Address",
                    value="",  # Explicitly set to empty
                    help="Use your registered healthcare system email",
                    label_visibility="visible",
                    key="login_email",
                    autocomplete="off"  # Disable autocomplete
                )
                
                password = st.text_input(
                    "Password",
                    value="",  # Explicitly set to empty
                    type="password",
                    help="Your secure password",
                    label_visibility="visible",
                    key="login_password",
                    autocomplete="new-password"  # Prevent autofill
                )
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Submit button with custom styling
                submitted = st.form_submit_button(
                    "Sign In",
                    type="primary",
                    use_container_width=True
                )
                
                if submitted:
                    if not email or not password:
                        st.error("⚠️ Please enter both email and password")
                    else:
                        with st.spinner("🔐 Authenticating..."):
                            success, message, user_data = st.session_state.auth_manager.login(email, password)
                        
                        if success:
                            # Store authentication data
                            st.session_state.authenticated = True
                            st.session_state.user_data = user_data.get('user', {})
                            st.session_state.access_token = user_data.get('token', {}).get('access_token')
                            st.session_state.login_time = datetime.now().isoformat()
                            
                            # Set default page based on role
                            role = st.session_state.user_data.get('role', '').lower()
                            if role == 'doctor':
                                st.session_state.page = '👥 Patient Management'
                            else:
                                st.session_state.page = '🏠 Risk Assessment'
                            
                            st.success(f"✅ Welcome, {st.session_state.user_data.get('full_name', 'User')}!")
                            time.sleep(0.5)
                            st.rerun()
                        else:
                            st.error(f"❌ {message}")
        
        with tab2:
            # Signup form
            with st.form("signup_form", clear_on_submit=False):
                st.markdown('<div class="form-container">', unsafe_allow_html=True)
                
                full_name = st.text_input(
                    "Full Name",
                    help="Your complete name as it should appear in the system",
                    label_visibility="visible"
                )
                
                signup_email = st.text_input(
                    "Email Address",
                    help="This will be your login email",
                    label_visibility="visible",
                    key="signup_email",
                    autocomplete="email"
                )
                
                signup_password = st.text_input(
                    "Password",
                    type="password",
                    help="Must be at least 8 characters with uppercase, lowercase, and numbers",
                    label_visibility="visible",
                    key="signup_password",
                    autocomplete="new-password"
                )
                
                confirm_password = st.text_input(
                    "Confirm Password",
                    type="password",
                    help="Re-enter your password to confirm",
                    label_visibility="visible",
                    key="confirm_password",
                    autocomplete="new-password"
                )
                
                role = st.selectbox(
                    "Account Type",
                    ["patient", "doctor"],
                    help="Select your role in the healthcare system",
                    label_visibility="visible"
                )
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Submit button with custom styling
                signup_submitted = st.form_submit_button(
                    "Create Account",
                    type="primary",
                    use_container_width=True
                )
                
                if signup_submitted:
                    if not all([full_name, signup_email, signup_password, confirm_password]):
                        st.error("⚠️ Please fill in all fields")
                    elif signup_password != confirm_password:
                        st.error("⚠️ Passwords do not match")
                    elif len(signup_password) < 8:
                        st.error("⚠️ Password must be at least 8 characters long")
                    else:
                        with st.spinner("📝 Creating your account..."):
                            success, message, user_data = st.session_state.auth_manager.signup(
                                signup_email, signup_password, full_name, role
                            )
                        
                        if success:
                            # Show success message and reset form
                            st.success("✅ Account created successfully! Please sign in with your new credentials.")
                            
                            # Clear the form
                            st.session_state.signup_form_full_name = ""
                            st.session_state.signup_form_email = ""
                            st.session_state.signup_form_password = ""
                            st.session_state.signup_form_confirm_password = ""
                            
                            # Switch to the login tab
                            st.session_state.active_tab = "Sign In"
                            time.sleep(0.5)
                            st.rerun()
                        else:
                            st.error(f"❌ {message}")
        
        st.markdown('</div>', unsafe_allow_html=True)  # Close login-container
    
    # System information in an expandable section below
    st.markdown("<br>", unsafe_allow_html=True)
    
    with st.expander("ℹ️ About the Healthcare System", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🎯 Purpose**
            - AI-powered health risk assessment
            - Cardiovascular disease prediction
            - Personalized health recommendations
            
            **👥 User Roles**
            - **Admin**: System management
            - **Doctor**: Patient care & analytics
            - **Patient**: Personal health insights
            """)
        
        with col2:
            st.markdown("""
            **🔒 Security Features**
            - JWT token authentication
            - Role-based access control
            - Secure password encryption
            - Session management
            
            **📊 Key Features**
            - EHR data processing
            - Interactive dashboards
            - Real-time predictions
            - Health data visualization
            """)


def render_user_info():
    """Render user information in sidebar"""
    if st.session_state.authenticated and st.session_state.user_data:
        user = st.session_state.user_data
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 👤 User Info")
        
        # User details
        st.sidebar.markdown(f"**Name:** {user.get('full_name', 'N/A')}")
        st.sidebar.markdown(f"**Email:** {user.get('email', 'N/A')}")
        st.sidebar.markdown(f"**Role:** {user.get('role', 'N/A').title()}")
        
        # Role badge
        role = user.get('role', '').lower()
        if role == 'admin':
            st.sidebar.markdown("🛡️ **Administrator**")
        elif role == 'doctor':
            st.sidebar.markdown("👨‍⚕️ **Medical Professional**")
        elif role == 'patient':
            st.sidebar.markdown("👤 **Patient**")
        
        # Login time
        if st.session_state.login_time:
            login_time = datetime.fromisoformat(st.session_state.login_time)
            st.sidebar.markdown(f"**Logged in:** {login_time.strftime('%H:%M:%S')}")
        
        st.sidebar.markdown("---")
        
        # Logout button
        if st.sidebar.button("🚪 Logout", type="primary", use_container_width=True, key="user_info_logout_btn"):
            logout_user()


def require_auth(required_roles: Optional[list[str]] = None):
    """
    Decorator function to require authentication
    
    Args:
        required_roles: List of roles that can access the page (None = any authenticated user)
    """
    if not st.session_state.authenticated or not check_session_validity():
        render_login_page()
        st.stop()
    
    if required_roles:
        user_role = st.session_state.user_data.get('role', '').lower()
        if user_role not in [role.lower() for role in required_roles]:
            st.error(f"Access denied. Required role: {', '.join(required_roles)}")
            st.stop()


def get_auth_headers():
    """Get authentication headers for API requests"""
    if st.session_state.access_token:
        return {"Authorization": f"Bearer {st.session_state.access_token}"}
    return {}


def is_admin():
    """Check if current user is admin"""
    return st.session_state.user_data.get('role', '').lower() == 'admin'


def is_doctor():
    """Check if current user is doctor or admin"""
    role = st.session_state.user_data.get('role', '').lower()
    return role in ['doctor', 'admin']


def is_patient():
    """Check if current user is patient"""
    return st.session_state.user_data.get('role', '').lower() == 'patient'