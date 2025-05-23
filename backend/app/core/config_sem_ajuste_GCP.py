# app/core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Optional, Any
import os
from dotenv import load_dotenv
from pathlib import Path
import logging

# Configura um logger básico para este módulo
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - [%(levelname)s] - (CONFIG) %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DOTENV_PATH = ROOT_DIR / '.env'

if DOTENV_PATH.exists():
    load_dotenv(dotenv_path=DOTENV_PATH, override=True)
    logger.info(f"Arquivo .env encontrado e explicitamente carregado de: {DOTENV_PATH}")
else:
    logger.warning(f"Arquivo .env NÃO encontrado em: {DOTENV_PATH} para carregamento explícito. Configurações dependerão de variáveis de ambiente já definidas ou defaults da classe.")

class Settings(BaseSettings):
    PROJECT_NAME: str = "IntelliMatch AI"
    API_V1_STR: str = "/api/v1"
    
    OPENAI_API_KEY: Optional[str] = None 

    CORS_ORIGINS_STR: Optional[str] = None
    CORS_ORIGINS: List[str] = []

    CHROMA_DB_PATH: str = "./vector_store_db"
    CHROMA_COLLECTION_NAME: str = "intellimatch_collection"

    EMBEDDING_MODEL_NAME: str = "text-embedding-3-small" # Modelo de embedding mais recente e eficiente em custo
    LLM_MODEL_NAME: str = "gpt-4o-mini" # CORRIGIDO: Nome do modelo LLM desejado

    DATA_PATH_VAGAS: str = "data/vagas.json"
    DATA_PATH_APPLICANTS: str = "data/applicants.json"
    DATA_PATH_PROSPECTS: str = "data/prospects.json"

    model_config = SettingsConfigDict(
        env_file=DOTENV_PATH if DOTENV_PATH.exists() else None,
        env_file_encoding='utf-8',
        extra='ignore',
        case_sensitive=False
    )

    def __init__(self, **values: Any):
        super().__init__(**values)
        if self.CORS_ORIGINS_STR:
            self.CORS_ORIGINS = [origin.strip() for origin in self.CORS_ORIGINS_STR.split(',')]
        elif not self.CORS_ORIGINS:
            self.CORS_ORIGINS = ["*"]

        if self.OPENAI_API_KEY:
            if len(self.OPENAI_API_KEY) == 51 and self.OPENAI_API_KEY.startswith("sk-"):
                 logger.info(f"Settings __init__: OpenAI API Key carregada e parece ter formato válido.")
            else:
                 logger.warning(f"Settings __init__: OpenAI API Key carregada, mas tem formato/comprimento incomum: {len(self.OPENAI_API_KEY)} caracteres. Verifique o .env.")
        else:
            logger.error("ALERTA CRÍTICO (Settings __init__): OpenAI API Key é None ou vazia APÓS pydantic-settings tentar carregar!")
            # raise ValueError("OPENAI_API_KEY não configurada.")

settings = Settings()

if not settings.OPENAI_API_KEY:
    logger.critical("ALERTA CRÍTICO (config.py global): settings.OPENAI_API_KEY é None ou vazia após instanciação de Settings.")
else:
    logger.info(f"Verificação global em config.py: OpenAI API Key definida. Modelo LLM: {settings.LLM_MODEL_NAME}, Modelo Embedding: {settings.EMBEDDING_MODEL_NAME}")