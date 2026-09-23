"""Centralized configuration using Pydantic Settings."""

from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Router Configuration
    router_base_url: str = "http://localhost:8008"
    router_route_endpoint: str = "/route"
    router_timeout_seconds: float = 15.0

    # Memory Storage
    memory_storage_path: Path = Path("data/memory.json")

    # Application Environment
    environment: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"

    # Cloud Integration (Phase 1 Stub)
    cloud_provider_stub_mode: bool = True

    # Local Answer Generation (Phase 4)
    local_answer_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    local_generation_max_tokens: int = 128
    local_generation_temperature: float = 0.0

    # Cloud Answer Generation (Gemini - Phase 4)
    gemini_api_key: Optional[str] = None
    gemini_model: str = "gemini-1.5-flash"
    gemini_timeout_seconds: float = 15.0

    @property
    def router_url(self) -> str:
        """Construct full router route URL."""
        base = self.router_base_url.rstrip("/")
        endpoint = self.router_route_endpoint
        if not endpoint.startswith("/"):
            endpoint = f"/{endpoint}"
        return f"{base}{endpoint}"


@lru_cache()
def get_settings() -> Settings:
    """Return cached application settings instance."""
    return Settings()
