from enum import Enum
from fastapi import Security, HTTPException, status
from fastapi.security.api_key import APIKeyHeader
from app.config import settings

class Role(str, Enum):
    ADMIN = "admin"
    USER = "user"

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

def get_admin_key() -> str:
    return settings.admin_secret_key

def get_user_key() -> str:
    return settings.user_secret_key

def get_api_key(api_key_header: str = Security(API_KEY_HEADER)):
    """Extracts and validates the API key."""
    if not api_key_header:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API Key in X-API-Key header",
        )
    return api_key_header

def get_user_role(api_key: str) -> Role:
    """Maps an API key to a Role."""
    admin_keys = [get_admin_key()]
    user_keys = [get_user_key()]
    
    if api_key in admin_keys:
        return Role.ADMIN
    elif api_key in user_keys:
        return Role.USER
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
        )

def require_role(required_role: Role):
    """Dependency factory to enforce role-based access control."""
    def role_checker(api_key: str = Security(get_api_key)):
        user_role = get_user_role(api_key)
        
        if required_role == Role.ADMIN and user_role != Role.ADMIN:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin privileges required",
            )
        return api_key
    return role_checker
