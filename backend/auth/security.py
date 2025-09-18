"""
Security utilities for authentication
"""
import os
import jwt
from datetime import datetime, timedelta
from typing import Optional, Union
import secrets
import warnings

# Suppress bcrypt version warnings for Python 3.13 compatibility
warnings.filterwarnings("ignore", message=".*bcrypt.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*__about__.*")

try:
    from passlib.context import CryptContext
    from passlib.hash import bcrypt
    PASSLIB_AVAILABLE = True
except ImportError:
    PASSLIB_AVAILABLE = False

# Password hashing context with enhanced bcrypt compatibility fix
try:
    # First try the standard configuration
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    # Test the context to ensure it works
    test_hash = pwd_context.hash("test")
    pwd_context.verify("test", test_hash)
except Exception as e:
    try:
        # Enhanced fallback for bcrypt compatibility issues (Python 3.13+)
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning, module="passlib")
        pwd_context = CryptContext(
            schemes=["bcrypt"], 
            deprecated="auto", 
            bcrypt__rounds=12,
            bcrypt__ident="2b"  # Force bcrypt 2b format for compatibility
        )
        # Test the fallback context
        test_hash = pwd_context.hash("test")
        pwd_context.verify("test", test_hash)
    except Exception as fallback_error:
        # Ultimate fallback using direct bcrypt
        import bcrypt as direct_bcrypt
        print(f"Warning: Using direct bcrypt due to passlib compatibility issues: {fallback_error}")
        
        class DirectBcryptContext:
            def hash(self, password: str) -> str:
                return direct_bcrypt.hashpw(password.encode('utf-8'), direct_bcrypt.gensalt()).decode('utf-8')
            
            def verify(self, password: str, hashed: str) -> bool:
                return direct_bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
        
        pwd_context = DirectBcryptContext()

# JWT settings
SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_urlsafe(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60  # 1 hour


def get_password_hash(password: str) -> str:
    """Hash a password"""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(token: str) -> Optional[dict]:
    """Verify and decode JWT token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


def generate_reset_token() -> str:
    """Generate a secure reset token"""
    return secrets.token_urlsafe(32)


def hash_reset_token(token: str) -> str:
    """Hash a reset token for storage"""
    return get_password_hash(token)


def verify_reset_token(token: str, hashed_token: str) -> bool:
    """Verify a reset token"""
    return verify_password(token, hashed_token)


class SecurityConfig:
    """Security configuration constants"""
    MIN_PASSWORD_LENGTH = 8
    MAX_LOGIN_ATTEMPTS = 5
    ACCOUNT_LOCK_DURATION_MINUTES = 30
    TOKEN_EXPIRE_MINUTES = 60
    RESET_TOKEN_EXPIRE_MINUTES = 15
    
    # Rate limiting
    MAX_REQUESTS_PER_MINUTE = 100
    
    # Session settings
    SESSION_EXPIRE_HOURS = 8
    
    # Security headers
    SECURITY_HEADERS = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Content-Security-Policy": "default-src 'self'"
    }


def generate_session_id() -> str:
    """Generate a secure session ID"""
    return secrets.token_urlsafe(32)


def is_strong_password(password: str) -> tuple[bool, list[str]]:
    """
    Check if password meets security requirements
    Returns: (is_valid, list_of_errors)
    """
    errors = []
    
    if len(password) < SecurityConfig.MIN_PASSWORD_LENGTH:
        errors.append(f"Password must be at least {SecurityConfig.MIN_PASSWORD_LENGTH} characters long")
    
    if not any(c.isupper() for c in password):
        errors.append("Password must contain at least one uppercase letter")
    
    if not any(c.islower() for c in password):
        errors.append("Password must contain at least one lowercase letter")
    
    if not any(c.isdigit() for c in password):
        errors.append("Password must contain at least one digit")
    
    if not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
        errors.append("Password must contain at least one special character")
    
    return len(errors) == 0, errors


def sanitize_input(input_string: str) -> str:
    """Basic input sanitization"""
    if not isinstance(input_string, str):
        return ""
    
    # Remove potentially dangerous characters
    dangerous_chars = ["<", ">", "&", "\"", "'", "/", "\\"]
    sanitized = input_string
    
    for char in dangerous_chars:
        sanitized = sanitized.replace(char, "")
    
    return sanitized.strip()


def generate_api_key() -> str:
    """Generate an API key for service-to-service communication"""
    return f"hc_{secrets.token_urlsafe(32)}"


def mask_email(email: str) -> str:
    """Mask email for logging purposes"""
    if "@" not in email:
        return "***"
    
    local, domain = email.split("@", 1)
    if len(local) <= 2:
        masked_local = "*" * len(local)
    else:
        masked_local = local[0] + "*" * (len(local) - 2) + local[-1]
    
    return f"{masked_local}@{domain}"


def log_security_event(event_type: str, user_email: str, additional_info: str = ""):
    """Log security events for monitoring"""
    timestamp = datetime.utcnow().isoformat()
    masked_email = mask_email(user_email)
    
    print(f"[SECURITY] {timestamp} - {event_type} - User: {masked_email} - {additional_info}")
    
    # In production, this should write to a secure log file or monitoring system