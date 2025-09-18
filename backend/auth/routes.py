"""
Authentication routes for the healthcare API
"""
from datetime import datetime, timedelta
try:
    from .models import (
        LoginRequest, LoginResponse, UserCreate, UserResponse, 
        Token, PasswordChange, PasswordReset, UserUpdate,
        UserRole, SessionData
    )
    from .database_store import database_user_store
    from .security import (
        create_access_token, verify_token,
        log_security_event, generate_session_id, SecurityConfig
    )
except ImportError:
    # Fallback for when running from backend directory
    from auth.models import (
        LoginRequest, LoginResponse, UserCreate, UserResponse, 
        Token, PasswordChange, PasswordReset, UserUpdate,
        UserRole, SessionData
    )
    from auth.database_store import database_user_store
    from auth.security import (
        create_access_token, verify_token,
        log_security_event, generate_session_id, SecurityConfig
    )

from typing import Optional
from fastapi import APIRouter, HTTPException, Depends, status, Request, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import secrets

router = APIRouter()
security = HTTPBearer()


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current authenticated user from JWT token"""
    token = credentials.credentials
    payload = verify_token(token)
    
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token or token expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user_id = payload.get("user_id")
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user = database_user_store.get_user_by_id(user_id)
    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user


async def get_admin_user(current_user = Depends(get_current_user)):
    """Require admin role"""
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user


async def get_doctor_or_admin(current_user = Depends(get_current_user)):
    """Require doctor or admin role"""
    if current_user.role not in [UserRole.DOCTOR, UserRole.ADMIN]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Doctor or admin access required"
        )
    return current_user


@router.post("/login", response_model=LoginResponse)
async def login(request: Request, login_data: LoginRequest):
    """User login endpoint"""
    try:
        # Check if account is locked
        if database_user_store.is_account_locked(login_data.email):
            log_security_event("LOGIN_BLOCKED", login_data.email, "Account locked")
            raise HTTPException(
                status_code=status.HTTP_423_LOCKED,
                detail="Account is temporarily locked due to too many failed login attempts"
            )
        
        # Get user by email
        user = database_user_store.get_user_by_email(login_data.email)
        if not user:
            database_user_store.increment_login_attempts(login_data.email)
            log_security_event("LOGIN_FAILED", login_data.email, "User not found")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        
        # Verify password
        def verify_password(plain_password: str, hashed_password: str) -> bool:
            """Verify password with fallback for different hash types"""
            try:
                import bcrypt
                return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))
            except Exception:
                # Fallback for hashlib-based passwords
                if ':' in hashed_password:
                    hash_part, salt = hashed_password.split(':', 1)
                    import hashlib
                    computed_hash = hashlib.pbkdf2_hmac('sha256', plain_password.encode('utf-8'), salt.encode('utf-8'), 100000).hex()
                    return computed_hash == hash_part
                return False
        
        if not verify_password(login_data.password, user.hashed_password):
            database_user_store.increment_login_attempts(login_data.email)
            log_security_event("LOGIN_FAILED", login_data.email, "Invalid password")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        
        # Check if user is active
        if not user.is_active:
            log_security_event("LOGIN_BLOCKED", login_data.email, "Account inactive")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account is inactive. Please contact administrator."
            )
        
        # Create access token
        access_token_expires = timedelta(minutes=SecurityConfig.TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"user_id": user.id, "email": user.email, "role": user.role},
            expires_delta=access_token_expires
        )
        
        # Update last login
        database_user_store.update_last_login(user.id)
        
        # Log successful login
        log_security_event("LOGIN_SUCCESS", login_data.email, f"Role: {user.role}")
        
        # Create response
        user_response = UserResponse(
            id=user.id,
            email=user.email,
            full_name=user.full_name,
            role=user.role,
            is_active=user.is_active,
            created_at=user.created_at,
            last_login=user.last_login
        )
        
        token_response = Token(
            access_token=access_token,
            token_type="bearer",
            expires_in=SecurityConfig.TOKEN_EXPIRE_MINUTES * 60
        )
        
        return LoginResponse(
            user=user_response,
            token=token_response,
            message="Login successful"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        log_security_event("LOGIN_ERROR", login_data.email, f"System error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed due to system error"
        )


@router.get("/test-connection")
async def test_connection():
    """Test if authentication service is working"""
    try:
        users = database_user_store.get_all_users()
        return {
            "status": "ok",
            "message": "Authentication service is running with database storage",
            "users_available": len(users),
            "storage_type": "SQLite Database"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Authentication service error: {str(e)}",
            "users_available": 0,
            "storage_type": "SQLite Database"
        }

@router.get("/health")
async def auth_health():
    """Health check for auth service"""
    try:
        users = database_user_store.get_all_users()
        return {
            "status": "healthy",
            "service": "authentication",
            "storage_type": "SQLite Database",
            "users_loaded": len(users) > 0,
            "total_users": len(users)
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "authentication",
            "error": str(e),
            "storage_type": "SQLite Database",
            "users_loaded": False,
            "total_users": 0
        }

@router.post("/logout")
async def logout(current_user = Depends(get_current_user)):
    """User logout endpoint"""
    log_security_event("LOGOUT", current_user.email, "User logged out")
    return {"message": "Logout successful"}


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user = Depends(get_current_user)):
    """Get current user information"""
    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        full_name=current_user.full_name,
        role=current_user.role,
        is_active=current_user.is_active,
        created_at=current_user.created_at,
        last_login=current_user.last_login
    )


@router.post("/register", response_model=UserResponse)
async def register_user(user_data: UserCreate, admin_user = Depends(get_admin_user)):
    """Register a new user (admin only)"""
    try:
        # Create new user
        new_user = database_user_store.create_user(user_data)
        
        log_security_event("USER_CREATED", user_data.email, f"Created by: {admin_user.email}")
        
        return UserResponse(
            id=new_user.id,
            email=new_user.email,
            full_name=new_user.full_name,
            role=new_user.role,
            is_active=new_user.is_active,
            created_at=new_user.created_at,
            last_login=new_user.last_login
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        log_security_event("USER_CREATION_ERROR", user_data.email, f"Error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="User creation failed"
        )


@router.post("/signup", response_model=UserResponse)
async def signup_user(user_data: UserCreate):
    """Public signup endpoint for new users"""
    try:
        # Only allow patient or doctor role for public signup
        if user_data.role not in [UserRole.PATIENT, UserRole.DOCTOR]:
            user_data.role = UserRole.PATIENT  # Default to patient if invalid role
        
        # Create new user
        new_user = database_user_store.create_user(user_data)
        
        log_security_event("USER_SIGNUP", user_data.email, f"Self-registered as: {user_data.role}")
        
        return UserResponse(
            id=new_user.id,
            email=new_user.email,
            full_name=new_user.full_name,
            role=new_user.role,
            is_active=new_user.is_active,
            created_at=new_user.created_at,
            last_login=new_user.last_login
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        log_security_event("USER_SIGNUP_ERROR", user_data.email, f"Error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Signup failed due to system error"
        )


@router.put("/users/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: str, 
    user_data: UserUpdate, 
    admin_user = Depends(get_admin_user)
):
    """Update user information (admin only)"""
    user = database_user_store.get_user_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    updated_user = database_user_store.update_user(user_id, user_data)
    if not updated_user:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="User update failed"
        )
    
    log_security_event("USER_UPDATED", user.email, f"Updated by: {admin_user.email}")
    
    return UserResponse(
        id=updated_user.id,
        email=updated_user.email,
        full_name=updated_user.full_name,
        role=updated_user.role,
        is_active=updated_user.is_active,
        created_at=updated_user.created_at,
        last_login=updated_user.last_login
    )


@router.post("/change-password")
async def change_password(
    password_data: PasswordChange,
    current_user = Depends(get_current_user)
):
    """Change user password"""
    # Verify current password
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify password with fallback for different hash types"""
        try:
            import bcrypt
            return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))
        except Exception:
            # Fallback for hashlib-based passwords
            if ':' in hashed_password:
                hash_part, salt = hashed_password.split(':', 1)
                import hashlib
                computed_hash = hashlib.pbkdf2_hmac('sha256', plain_password.encode('utf-8'), salt.encode('utf-8'), 100000).hex()
                return computed_hash == hash_part
            return False
    
    if not verify_password(password_data.current_password, current_user.hashed_password):
        log_security_event("PASSWORD_CHANGE_FAILED", current_user.email, "Invalid current password")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect"
        )
    
    # Update password
    def get_password_hash(password: str) -> str:
        try:
            import bcrypt
            return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        except Exception:
            # Fallback to hashlib if bcrypt fails
            import hashlib
            import secrets
            salt = secrets.token_hex(16)
            return hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 100000).hex() + ':' + salt
    
    current_user.hashed_password = get_password_hash(password_data.new_password)
    
    log_security_event("PASSWORD_CHANGED", current_user.email, "Password changed successfully")
    
    return {"message": "Password changed successfully"}


@router.get("/users", response_model=list[UserResponse])
async def list_users(admin_user = Depends(get_admin_user)):
    """List all users (admin only)"""
    users = database_user_store.get_all_users()
    return [
        UserResponse(
            id=user.id,
            email=user.email,
            full_name=user.full_name,
            role=user.role,
            is_active=user.is_active,
            created_at=user.created_at,
            last_login=user.last_login
        )
        for user in users
    ]


@router.get("/verify-token")
async def verify_user_token(current_user = Depends(get_current_user)):
    """Verify if token is valid"""
    return {
        "valid": True,
        "user_id": current_user.id,
        "email": current_user.email,
        "role": current_user.role
    }


@router.post("/session")
async def create_session(current_user = Depends(get_current_user)):
    """Create a session for frontend use"""
    session_id = generate_session_id()
    
    session_data = SessionData(
        user_id=current_user.id,
        email=current_user.email,
        full_name=current_user.full_name,
        role=current_user.role,
        is_active=current_user.is_active,
        login_time=datetime.utcnow(),
        last_activity=datetime.utcnow()
    )
    
    return {
        "session_id": session_id,
        "session_data": session_data,
        "expires_in": SecurityConfig.SESSION_EXPIRE_HOURS * 3600
    }


# Removed default test credentials endpoint for security