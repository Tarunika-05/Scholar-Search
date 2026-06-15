from enum import Enum
from fastapi import Security, HTTPException, status
from fastapi.security.api_key import APIKeyHeader
import os

class Role(str, Enum):
    ADMIN = "admin"
    USER = "user"

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# In a real app, this would be in a DB or parsed from .env.
# For demo purposes, we define static roles, but default to checking os.environ
DEFAULT_ADMIN_KEY = "admin-secret-key"
DEFAULT_USER_KEY = "user-secret-key"

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
    # Allow overriding via environment variable
    admin_keys = [os.getenv("ADMIN_API_KEY", DEFAULT_ADMIN_KEY)]
    user_keys = [os.getenv("USER_API_KEY", DEFAULT_USER_KEY)]
    
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
