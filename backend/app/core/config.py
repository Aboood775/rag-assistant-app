from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Raspberry Pi Docs Assistant API"
    vector_store_dir: Path = BACKEND_DIR / "data" / "vector_store"
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "llama3.2"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 300
    top_k: int = 4
    fetch_k: int = 20
    lambda_mult: float = 0.9
    max_distance: float | None = None
    cors_origins: str = "http://localhost:8501"
    log_level: str = "INFO"

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
