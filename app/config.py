from __future__ import annotations

from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", protected_namespaces=())

    api_keys_json: str = Field(default="{}", validation_alias="API_KEYS_JSON")
    embedding_encryption_key: str = Field(default="", validation_alias="EMBEDDING_ENCRYPTION_KEY")
    database_url: str = "sqlite:///./data/speakers.db"
    model_cache_dir: str = "./models/speechbrain-ecapa"
    model_id: str = "speechbrain/spkrec-ecapa-voxceleb"
    model_revision: str = "eac2726"
    device: str = "cpu"
    max_concurrent_inferences: int = Field(default=2, ge=1, le=16)
    max_queued_inferences: int = Field(default=8, ge=0, le=256)
    queue_wait_seconds: int = Field(default=120, ge=1, le=900)
    torch_num_threads: int = Field(default=1, ge=1, le=64)
    max_upload_mb: int = Field(default=50, ge=1, le=500)
    max_audio_seconds: int = Field(default=180, ge=3, le=3600)
    min_usable_speech_seconds: float = Field(default=3.0, ge=1.0, le=60.0)
    recommended_enrollment_seconds: float = Field(default=20.0, ge=3.0, le=180.0)
    match_threshold: float = Field(default=0.72, ge=-1.0, le=1.0)
    match_margin: float = Field(default=0.08, ge=0.0, le=2.0)
    top_candidates: int = Field(default=5, ge=1, le=20)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
