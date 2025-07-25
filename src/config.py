"""Application configuration."""

from typing import Literal

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    # API Keys
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    # Vector Store Configuration
    vector_store_type: Literal["faiss", "chroma"] = "faiss"
    vector_store_path: str = "./data/vector_store"

    # Embedding Configuration
    embedding_model: str = "text-embedding-ada-002"
    embedding_provider: Literal["openai", "huggingface"] = "openai"

    # LLM Configuration
    llm_provider: Literal["openai", "anthropic"] = "openai"
    llm_model: str = "gpt-3.5-turbo"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 1000

    # Application Configuration
    log_level: str = "INFO"
    environment: str = "development"

    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
