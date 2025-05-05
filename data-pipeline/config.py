from functools import cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = [
    "get_settings",
]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # FastAPI settings
    debug: bool = False

    # LLM settings
    llm_model: str = "openai/gpt-4o-mini"
    openai_api_key: SecretStr = SecretStr("")


@cache
def get_settings():
    return Settings()
