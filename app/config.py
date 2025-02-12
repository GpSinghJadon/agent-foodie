from pydantic_settings import BaseSettings
from typing import List
from functools import lru_cache


class Settings(BaseSettings):
    # Database
    database_url: str = "sqlite:///chat_agent.db"

    # Ollama Settings
    ollama_base_url: str
    ollama_model: str

    # API Settings
    restaurants_api_url: str
    audit_file: str = "audit.log"
    cors_origins: List[str] = ["*"]

    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    class Config:
        env_file = ".env"
        # case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    return Settings()
