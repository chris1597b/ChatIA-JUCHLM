"""Configuración centralizada (12-factor). Nunca hardcodear secrets en código."""
from __future__ import annotations
import os
from functools import lru_cache
from pathlib import Path

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
    _HAS_PYDANTIC_SETTINGS = True
except ImportError:  # fallback mínimo sin romper arranque
    _HAS_PYDANTIC_SETTINGS = False
    BaseSettings = object  # type: ignore

BASE_DIR = Path(__file__).resolve().parent.parent


if _HAS_PYDANTIC_SETTINGS:
    class Settings(BaseSettings):
        model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

        APP_NAME: str = "ChatJUCHLM"
        APP_ENV: str = "development"
        APP_HOST: str = "0.0.0.0"
        APP_PORT: int = 8000

        # RAG (hereda defaults de rag_juchlm.py V11, no los reemplaza)
        OLLAMA_URL: str = "http://127.0.0.1:11434"
        PDF_FOLDER: str = "pdfs"
        CHROMA_PATH: str = "./chroma_db"

        # SQL Server 2022 (solo vía .env)
        SQL_SERVER: str = ""
        SQL_DATABASE: str = ""
        SQL_USERNAME: str = ""
        SQL_PASSWORD: str = ""
        SQL_DRIVER: str = "ODBC Driver 18 for SQL Server"
        SQL_TIMEOUT: int = 5
        SQL_ENABLED: bool = False  # se activa solo si hay credenciales

        LOG_DIR: str = "logs"
        SESSION_TTL_MIN: int = 30

        def sql_configured(self) -> bool:
            return bool(self.SQL_ENABLED and self.SQL_SERVER and self.SQL_DATABASE and self.SQL_USERNAME)

    @lru_cache
    def get_settings() -> "Settings":
        return Settings()  # type: ignore
else:
    class Settings:  # fallback mínimo
        def __init__(self):
            self.APP_NAME = os.getenv("APP_NAME", "ChatJUCHLM")
            self.APP_ENV = os.getenv("APP_ENV", "development")
            self.APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
            self.APP_PORT = int(os.getenv("APP_PORT", "8000"))
            self.OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
            self.PDF_FOLDER = os.getenv("PDF_FOLDER", "pdfs")
            self.CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
            self.SQL_SERVER = os.getenv("SQL_SERVER", "")
            self.SQL_DATABASE = os.getenv("SQL_DATABASE", "")
            self.SQL_USERNAME = os.getenv("SQL_USERNAME", "")
            self.SQL_PASSWORD = os.getenv("SQL_PASSWORD", "")
            self.SQL_DRIVER = os.getenv("SQL_DRIVER", "ODBC Driver 18 for SQL Server")
            self.SQL_TIMEOUT = int(os.getenv("SQL_TIMEOUT", "5"))
            self.SQL_ENABLED = os.getenv("SQL_ENABLED", "false").lower() == "true"
            self.LOG_DIR = os.getenv("LOG_DIR", "logs")
            self.SESSION_TTL_MIN = int(os.getenv("SESSION_TTL_MIN", "30"))
        def sql_configured(self) -> bool:
            return bool(self.SQL_ENABLED and self.SQL_SERVER and self.SQL_DATABASE and self.SQL_USERNAME)

    @lru_cache
    def get_settings() -> "Settings":
        return Settings()
