from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "development"
    database_path: Path = Path("data/datum.db")
    checkpoint_database_path: Path = Path("data/checkpoints.db")
    upload_dir: Path = Path("data/uploads")
    prepared_dir: Path = Path("data/prepared")
    report_dir: Path = Path("data/reports")
    max_upload_size_mb: int = 50
    max_sql_rows: int = 10000
    llm_provider: str = "none"
    llm_model: str = ""
    llm_api_key: str = ""
    llm_base_url: str = ""
    llm_timeout_seconds: int = 20
    llm_max_retries: int = 2
    allowed_origins: str = "http://localhost:3000,http://localhost:5173"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    def model_post_init(self, __context: object) -> None:
        project_root = Path(__file__).resolve().parents[2]
        for field_name in (
            "database_path",
            "checkpoint_database_path",
            "upload_dir",
            "prepared_dir",
            "report_dir",
        ):
            value = getattr(self, field_name)
            if not value.is_absolute():
                setattr(self, field_name, project_root / value)

    @property
    def allowed_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
