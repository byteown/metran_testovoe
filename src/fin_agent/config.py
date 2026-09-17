from datetime import date
from pathlib import Path
from functools import lru_cache
from typing import Annotated

from pydantic import DirectoryPath, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if not Path(PROJECT_ROOT / "pyproject.toml").exists():
    raise("Project root path is missing")


class ConfigBase(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore")


class LLMSettings(ConfigBase):
    model_config = SettingsConfigDict(env_prefix="LLM_")

    url: str
    model: str
    api_key: str
    timeout_seconds: Annotated[int, Field(gt=0)]
    max_retries: Annotated[int, Field(ge=0, le=5)]
    retry_backoff_seconds: float

    @field_validator("api_key")
    @classmethod
    def _fallback(cls, v: str) -> str:
        return v.strip() or "apikey123"


class AgentSettings(ConfigBase):
    model_config = SettingsConfigDict(env_prefix="AGENT_")

    max_turns: Annotated[int, Field(ge=1, le=10)]
    token_budget: Annotated[int, Field(gt=4000)]
    max_question_length: Annotated[int, Field(ge=10, le=2000)]


class DataSettings(ConfigBase):
    data_path: DirectoryPath
    docs_path: DirectoryPath
    report_date: date

    @field_validator("data_path", "docs_path", mode="before")
    @classmethod
    def _anchor(cls, v):
        p = Path(v)
        return p if p.is_absolute() else PROJECT_ROOT / p


class Settings(BaseSettings):
    llm: LLMSettings = Field(default_factory=LLMSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    data: DataSettings = Field(default_factory=DataSettings)


@lru_cache(maxsize=1, typed=False)
def get_settings() -> Settings:
    settings = Settings()
    return settings