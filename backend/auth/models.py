"""
Authentication models and schemas for the healthcare system
"""
from datetime import datetime, timedelta
from typing import Optional, List
from enum import Enum
from pydantic import BaseModel, EmailStr, field_validator
import hashlib
import secrets


class UserRole(str, Enum):
    """User roles for role-based access control"""
    ADMIN = "admin"
    DOCTOR = "doctor"
    NURSE = "nurse"
    PATIENT = "patient"


class UserBase(BaseModel):
    """Base user model with common fields"""
    email: EmailStr
    full_name: str
    role: UserRole
    is_active: bool = True
    created_at: datetime = datetime.utcnow()


class UserCreate(BaseModel):
    """Model for user creation"""
    email: EmailStr
    full_name: str
    password: str
    role: UserRole = UserRole.PATIENT
    
    @field_validator('password')
    @classmethod
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        return v


class UserUpdate(BaseModel):
    """Model for user updates"""
    full_name: Optional[str] = None
    role: Optional[UserRole] = None
    is_active: Optional[bool] = None


class UserInDB(UserBase):
    """User model as stored in database"""
    id: str
    hashed_password: str
    last_login: Optional[datetime] = None
    login_attempts: int = 0
    locked_until: Optional[datetime] = None


class UserResponse(UserBase):
    """User model for API responses (without sensitive data)"""
    id: str
    last_login: Optional[datetime] = None


class Token(BaseModel):
    """JWT token response model"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int = 3600  # 1 hour


class TokenData(BaseModel):
    """Token payload data"""
    user_id: Optional[str] = None
    email: Optional[str] = None
    role: Optional[str] = None


class LoginRequest(BaseModel):
    """Login request model"""
    email: EmailStr
    password: str


class LoginResponse(BaseModel):
    """Login response model"""
    user: UserResponse
    token: Token
    message: str = "Login successful"


class PasswordReset(BaseModel):
    """Password reset model"""
    email: EmailStr


class PasswordChange(BaseModel):
    """Password change model"""
    current_password: str
    new_password: str
    
    @field_validator('new_password')
    @classmethod
    def validate_new_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        return v


class SessionData(BaseModel):
    """Session data model for frontend"""
    user_id: str
    email: str
    full_name: str
    role: str
    is_active: bool
    login_time: datetime
    last_activity: datetime


# In-memory user store (replace with actual database in production)
class UserStore:
    """Simple in-memory user store for demonstration"""
    
    def __init__(self):
        self.users: dict[str, UserInDB] = {}
        self._create_default_users()
    
    def _create_default_users(self):
        """Create default users for testing"""
        # Import here to avoid circular imports
        import hashlib
        import secrets as secrets_module
        
        def get_password_hash(password: str) -> str:
            """Simple password hashing for default users"""
            try:
                import bcrypt
                return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            except Exception as e:
                # Fallback to hashlib if bcrypt fails
                import hashlib
                import secrets
                salt = secrets.token_hex(16)
                return hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 100000).hex() + ':' + salt
        
        # Default admin user
        admin_id = secrets_module.token_urlsafe(16)
        admin_user = UserInDB(
            id=admin_id,
            email="admin@healthcare.com",
            full_name="System Administrator",
            role=UserRole.ADMIN,
            hashed_password=get_password_hash("Admin123!"),
            is_active=True,
            created_at=datetime.utcnow()
        )
        self.users[admin_id] = admin_user
        
        # Default doctor user
        doctor_id = secrets_module.token_urlsafe(16)
        doctor_user = UserInDB(
            id=doctor_id,
            email="doctor@healthcare.com",
            full_name="Dr. Jane Smith",
            role=UserRole.DOCTOR,
            hashed_password=get_password_hash("Doctor123!"),
            is_active=True,
            created_at=datetime.utcnow()
        )
        self.users[doctor_id] = doctor_user
        
        # Default patient user
        patient_id = secrets_module.token_urlsafe(16)
        patient_user = UserInDB(
            id=patient_id,
            email="patient@healthcare.com",
            full_name="John Doe",
            role=UserRole.PATIENT,
            hashed_password=get_password_hash("Patient123!"),
            is_active=True,
            created_at=datetime.utcnow()
        )
        self.users[patient_id] = patient_user
    
    def get_user_by_email(self, email: str) -> Optional[UserInDB]:
        """Get user by email"""
        for user in self.users.values():
            if user.email == email:
                return user
        return None
    
    def get_user_by_id(self, user_id: str) -> Optional[UserInDB]:
        """Get user by ID"""
        return self.users.get(user_id)
    
    def create_user(self, user_data: UserCreate) -> UserInDB:
        """Create a new user"""
        # Import here to avoid circular imports
        import bcrypt
        import secrets
        
        def get_password_hash(password: str) -> str:
            try:
                import bcrypt
                return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            except Exception as e:
                # Fallback to hashlib if bcrypt fails
                import hashlib
                import secrets
                salt = secrets.token_hex(16)
                return hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 100000).hex() + ':' + salt
        
        # Check if user already exists
        if self.get_user_by_email(user_data.email):
            raise ValueError("User with this email already exists")
        
        user_id = secrets.token_urlsafe(16)
        user = UserInDB(
            id=user_id,
            email=user_data.email,
            full_name=user_data.full_name,
            role=user_data.role,
            hashed_password=get_password_hash(user_data.password),
            is_active=True,
            created_at=datetime.utcnow()
        )
        
        self.users[user_id] = user
        return user
    
    def update_user(self, user_id: str, user_data: UserUpdate) -> Optional[UserInDB]:
        """Update user data"""
        user = self.users.get(user_id)
        if not user:
            return None
        
        update_data = user_data.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(user, field, value)
        
        return user
    
    def update_last_login(self, user_id: str):
        """Update user's last login time"""
        user = self.users.get(user_id)
        if user:
            user.last_login = datetime.utcnow()
            user.login_attempts = 0  # Reset login attempts on successful login
    
    def increment_login_attempts(self, email: str):
        """Increment failed login attempts"""
        user = self.get_user_by_email(email)
        if user:
            user.login_attempts += 1
            # Lock account after 5 failed attempts for 30 minutes
            if user.login_attempts >= 5:
                user.locked_until = datetime.utcnow() + timedelta(minutes=30)
    
    def is_account_locked(self, email: str) -> bool:
        """Check if account is locked"""
        user = self.get_user_by_email(email)
        if user and user.locked_until:
            if datetime.utcnow() < user.locked_until:
                return True
            else:
                # Unlock account if lock period has passed
                user.locked_until = None
                user.login_attempts = 0
        return False
    
    def get_all_users(self) -> List[UserInDB]:
        """Get all users (admin only)"""
        return list(self.users.values())


# Global user store instance
user_store = UserStore()