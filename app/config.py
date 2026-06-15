from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    # App Settings
    project_name: str = "Semantic Search API"
    environment: str = "development"
    
    # API Keys
    admin_secret_key: str = "admin-secret-key"
    user_secret_key: str = "user-secret-key"
    
    # Database
    database_url: str = "sqlite+aiosqlite:///./data/analytics.db"
    
    # External Providers
    azure_openai_endpoint: Optional[str] = None
    azure_openai_api_key: Optional[str] = None
    azure_openai_api_version: str = "2024-02-15-preview"
    azure_openai_model_name: str = "gpt-4o"
    gemini_api_key: Optional[str] = None

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8", 
        extra="ignore"
    )

settings = Settings()
