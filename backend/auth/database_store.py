"""
Database-backed user store using SQLAlchemy
"""
from datetime import datetime, timedelta
from typing import Optional, List
import secrets
import hashlib
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

try:
    from backend.db import SessionLocal, engine
    from backend.auth.db_models import User, MedicalReportAnalysis
    from backend.auth.models import UserCreate, UserUpdate, UserInDB, UserRole
except ImportError:
    # Fallback for when running from backend directory
    from db import SessionLocal, engine
    from auth.db_models import User, MedicalReportAnalysis
    from auth.models import UserCreate, UserUpdate, UserInDB, UserRole


class DatabaseUserStore:
    """Database-backed user store using SQLAlchemy"""
    
    def __init__(self):
        # Create tables if they don't exist
        try:
            from backend.auth.db_models import User as UserModel, MedicalReportAnalysis as ReportModel
        except ImportError:
            from auth.db_models import User as UserModel, MedicalReportAnalysis as ReportModel
        
        try:
            UserModel.metadata.create_all(bind=engine)
            ReportModel.metadata.create_all(bind=engine)
            self._ensure_default_users()
        except Exception as e:
            print(f"Warning: Database initialization failed: {e}")
            print("Authentication service will continue with limited functionality")
    
    def _get_db(self) -> Session:
        """Get database session"""
        return SessionLocal()
    
    def _get_password_hash(self, password: str) -> str:
        """Hash password using bcrypt with fallback"""
        try:
            import bcrypt
            return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        except Exception:
            # Fallback to hashlib if bcrypt fails
            salt = secrets.token_hex(16)
            return hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 100000).hex() + ':' + salt
    
    def _verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password with fallback for different hash types"""
        try:
            import bcrypt
            return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))
        except Exception:
            # Fallback for hashlib-based passwords
            if ':' in hashed_password:
                hash_part, salt = hashed_password.split(':', 1)
                computed_hash = hashlib.pbkdf2_hmac('sha256', plain_password.encode('utf-8'), salt.encode('utf-8'), 100000).hex()
                return computed_hash == hash_part
            return False
    
    def _ensure_default_users(self):
        """Create default users if they don't exist"""
        db = None
        try:
            db = self._get_db()
            # Check if any users exist
            existing_users = db.query(User).count()
            if existing_users > 0:
                return
            
            # Create default users
            default_users = [
                {
                    "email": "admin@healthcare.com",
                    "full_name": "System Administrator",
                    "password": "Admin123!",
                    "role": UserRole.ADMIN
                },
                {
                    "email": "doctor@healthcare.com",
                    "full_name": "Dr. Jane Smith",
                    "password": "Doctor123!",
                    "role": UserRole.DOCTOR
                },
                {
                    "email": "patient@healthcare.com",
                    "full_name": "John Doe",
                    "password": "Patient123!",
                    "role": UserRole.PATIENT
                }
            ]
            
            for user_data in default_users:
                user_id = secrets.token_urlsafe(16)
                db_user = User(
                    id=user_id,
                    email=user_data["email"],
                    full_name=user_data["full_name"],
                    hashed_password=self._get_password_hash(user_data["password"]),
                    role=user_data["role"].value,
                    is_active=True,
                    created_at=datetime.utcnow()
                )
                db.add(db_user)
            
            db.commit()
            print(f"✓ Created {len(default_users)} default users in database")
            
        except Exception as e:
            if db:
                db.rollback()
            print(f"Error creating default users: {e}")
        finally:
            if db:
                db.close()
    
    def get_user_by_email(self, email: str) -> Optional[UserInDB]:
        """Get user by email"""
        db = self._get_db()
        try:
            db_user = db.query(User).filter(User.email == email).first()
            if not db_user:
                return None
            
            return UserInDB(
                id=db_user.id,
                email=db_user.email,
                full_name=db_user.full_name,
                role=UserRole(db_user.role),
                is_active=db_user.is_active,
                created_at=db_user.created_at,
                hashed_password=db_user.hashed_password,
                last_login=db_user.last_login,
                login_attempts=db_user.login_attempts,
                locked_until=db_user.locked_until
            )
        finally:
            db.close()
    
    def get_user_by_id(self, user_id: str) -> Optional[UserInDB]:
        """Get user by ID"""
        db = self._get_db()
        try:
            db_user = db.query(User).filter(User.id == user_id).first()
            if not db_user:
                return None
            
            return UserInDB(
                id=db_user.id,
                email=db_user.email,
                full_name=db_user.full_name,
                role=UserRole(db_user.role),
                is_active=db_user.is_active,
                created_at=db_user.created_at,
                hashed_password=db_user.hashed_password,
                last_login=db_user.last_login,
                login_attempts=db_user.login_attempts,
                locked_until=db_user.locked_until
            )
        finally:
            db.close()
    
    def create_user(self, user_data: UserCreate) -> UserInDB:
        """Create a new user"""
        db = self._get_db()
        try:
            # Check if user already exists
            existing_user = db.query(User).filter(User.email == user_data.email).first()
            if existing_user:
                raise ValueError("User with this email already exists")
            
            user_id = secrets.token_urlsafe(16)
            db_user = User(
                id=user_id,
                email=user_data.email,
                full_name=user_data.full_name,
                hashed_password=self._get_password_hash(user_data.password),
                role=user_data.role.value,
                is_active=True,
                created_at=datetime.utcnow()
            )
            
            db.add(db_user)
            db.commit()
            db.refresh(db_user)
            
            return UserInDB(
                id=db_user.id,
                email=db_user.email,
                full_name=db_user.full_name,
                role=UserRole(db_user.role),
                is_active=db_user.is_active,
                created_at=db_user.created_at,
                hashed_password=db_user.hashed_password,
                last_login=db_user.last_login,
                login_attempts=db_user.login_attempts,
                locked_until=db_user.locked_until
            )
            
        except IntegrityError:
            db.rollback()
            raise ValueError("User with this email already exists")
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()
    
    def update_user(self, user_id: str, user_data: UserUpdate) -> Optional[UserInDB]:
        """Update user data"""
        db = self._get_db()
        try:
            db_user = db.query(User).filter(User.id == user_id).first()
            if not db_user:
                return None
            
            update_data = user_data.model_dump(exclude_unset=True)
            for field, value in update_data.items():
                if field == "role" and isinstance(value, UserRole):
                    setattr(db_user, field, value.value)
                else:
                    setattr(db_user, field, value)
            
            db.commit()
            db.refresh(db_user)
            
            return UserInDB(
                id=db_user.id,
                email=db_user.email,
                full_name=db_user.full_name,
                role=UserRole(db_user.role),
                is_active=db_user.is_active,
                created_at=db_user.created_at,
                hashed_password=db_user.hashed_password,
                last_login=db_user.last_login,
                login_attempts=db_user.login_attempts,
                locked_until=db_user.locked_until
            )
            
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()
    
    def update_last_login(self, user_id: str):
        """Update user's last login time"""
        db = self._get_db()
        try:
            db_user = db.query(User).filter(User.id == user_id).first()
            if db_user:
                db_user.last_login = datetime.utcnow()
                db_user.login_attempts = 0
                db.commit()
        except Exception as e:
            db.rollback()
            print(f"Error updating last login: {e}")
        finally:
            db.close()
    
    def increment_login_attempts(self, email: str):
        """Increment failed login attempts"""
        db = self._get_db()
        try:
            db_user = db.query(User).filter(User.email == email).first()
            if db_user:
                db_user.login_attempts += 1
                # Lock account after 5 failed attempts for 30 minutes
                if db_user.login_attempts >= 5:
                    db_user.locked_until = datetime.utcnow() + timedelta(minutes=30)
                db.commit()
        except Exception as e:
            db.rollback()
            print(f"Error incrementing login attempts: {e}")
        finally:
            db.close()
    
    def is_account_locked(self, email: str) -> bool:
        """Check if account is locked"""
        db = self._get_db()
        try:
            db_user = db.query(User).filter(User.email == email).first()
            if db_user and db_user.locked_until:
                if datetime.utcnow() < db_user.locked_until:
                    return True
                else:
                    # Unlock account if lock period has passed
                    db_user.locked_until = None
                    db_user.login_attempts = 0
                    db.commit()
            return False
        except Exception as e:
            print(f"Error checking account lock: {e}")
            return False
        finally:
            db.close()
    
    def get_all_users(self) -> List[UserInDB]:
        """Get all users (admin only)"""
        db = self._get_db()
        try:
            db_users = db.query(User).all()
            return [
                UserInDB(
                    id=db_user.id,
                    email=db_user.email,
                    full_name=db_user.full_name,
                    role=UserRole(db_user.role),
                    is_active=db_user.is_active,
                    created_at=db_user.created_at,
                    hashed_password=db_user.hashed_password,
                    last_login=db_user.last_login,
                    login_attempts=db_user.login_attempts,
                    locked_until=db_user.locked_until
                )
                for db_user in db_users
            ]
        finally:
            db.close()

# Global database user store instance
database_user_store = DatabaseUserStore()

def get_current_user(token: str):
    """Get current user from token - for prescription routes compatibility"""
    try:
        from backend.auth.security import verify_token
    except ImportError:
        from auth.security import verify_token
    
    payload = verify_token(token)
    if not payload:
        return None
    
    email = payload.get("sub")
    if not email:
        return None
    
    return database_user_store.get_user_by_email(email)

def get_db() -> Session:
    """Dependency to get database session for FastAPI routes"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
