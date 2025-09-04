from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    OPENAI_API_KEY: str = "sk-"
    QDRANT_URL: str = "http://localhost:6333"
    COLLECTION_NAME: str = "math_arxiv_passages"
    EMBED_MODEL: str = "BAAI/bge-m3"
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_TEMPERATURE: float = 0.0

    # pydantic-settings to load .env automatically
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",         # ignore unexpected .env keys safely
    )


settings = Settings()




