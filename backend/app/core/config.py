# app/core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Optional, Any
import os
# A importação de load_dotenv e Path é opcional para o deploy no Cloud Run,
# pois lá as variáveis de ambiente são injetadas diretamente.
# No entanto, mantê-las não prejudica e ajuda no desenvolvimento local se você tiver um .env.
from dotenv import load_dotenv
from pathlib import Path
import logging

# Configura um logger básico para este módulo, caso o logging principal ainda não esteja configurado
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(), 
                    format='%(asctime)s - %(name)s - [%(levelname)s] - (CONFIG) %(message)s', 
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

# Tenta carregar .env se estiver rodando localmente (não encontrará no Cloud Run, o que é esperado)
# ROOT_DIR é útil para encontrar o .env no desenvolvimento local.
ROOT_DIR = Path(__file__).resolve().parent.parent.parent # Raiz do projeto (datathon-recruitment-agent/)
DOTENV_PATH = ROOT_DIR / '.env'

if DOTENV_PATH.exists() and os.getenv("GOOGLE_CLOUD_RUN_SERVICE_ID") is None: # Só carrega se não estiver no Cloud Run
    logger.info(f"Ambiente local detectado. Carregando .env de: {DOTENV_PATH}")
    load_dotenv(dotenv_path=DOTENV_PATH, override=True)
elif os.getenv("GOOGLE_CLOUD_RUN_SERVICE_ID") is not None:
    logger.info("Ambiente Google Cloud Run detectado. Variáveis de ambiente devem ser fornecidas pela plataforma.")
else:
    logger.warning(f"Arquivo .env NÃO encontrado em: {DOTENV_PATH} para carregamento explícito. Configurações dependerão de variáveis de ambiente já definidas ou defaults da classe.")


class Settings(BaseSettings):
    PROJECT_NAME: str = "IntelliMatch AI"
    API_V1_STR: str = "/api/v1"
    
    # Chave da API OpenAI:
    # No Cloud Run, esta variável de ambiente será injetada a partir do Secret Manager
    # através da configuração do serviço Cloud Run (`--set-secrets`).
    # pydantic-settings pegará automaticamente a variável de ambiente OPENAI_API_KEY.
    OPENAI_API_KEY: Optional[str] = None 

    # Configurações de CORS
    # No Cloud Run, você pode definir CORS_ORIGINS_STR como uma variável de ambiente no serviço.
    # Ex: "https://seu-frontend-url.a.run.app,http://localhost:8501"
    CORS_ORIGINS_STR: Optional[str] = None 
    CORS_ORIGINS: List[str] = []

    # Configurações do Vector Store (ChromaDB)
    # No Cloud Run, se o DB estiver embutido na imagem, o caminho será relativo ao WORKDIR.
    # Se usar um volume montado (ex: GCS FUSE), o caminho será o ponto de montagem.
    CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", "/app/vector_store_db") # Caminho dentro do container
    CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "intellimatch_collection")

    # Configurações de Modelos
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small")
    LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")

    # Caminhos para os arquivos de dados (relativos ao WORKDIR /app/ no container)
    # No Cloud Run, estes arquivos devem ser copiados para a imagem Docker durante o build.
    DATA_PATH_VAGAS: str = os.getenv("DATA_PATH_VAGAS", "data/vagas.json")
    DATA_PATH_APPLICANTS: str = os.getenv("DATA_PATH_APPLICANTS", "data/applicants.json")
    DATA_PATH_PROSPECTS: str = os.getenv("DATA_PATH_PROSPECTS", "data/prospects.json")
    
    # Variável de ambiente PORT injetada pelo Cloud Run (normalmente 8080)
    # Usada pelo Uvicorn no CMD do Dockerfile. Não precisa ser definida aqui diretamente
    # se o CMD do Dockerfile usa ${PORT}.
    # PORT: int = int(os.getenv("PORT", 8000)) # Opcional, apenas para referência

    # Configuração para pydantic-settings v2
    model_config = SettingsConfigDict(
        # Para Cloud Run, não dependemos do .env ser lido por pydantic-settings aqui,
        # pois as variáveis são injetadas no ambiente do contêiner.
        # Definir env_file=None ou omitir pode ser mais limpo para o contexto do Cloud Run.
        env_file=None, 
        env_file_encoding='utf-8',
        extra='ignore', 
        case_sensitive=False # OPENAI_API_KEY é geralmente maiúscula no ambiente
    )

    def __init__(self, **values: Any):
        super().__init__(**values)
        # Processa CORS_ORIGINS após a inicialização base
        if self.CORS_ORIGINS_STR:
            self.CORS_ORIGINS = [origin.strip() for origin in self.CORS_ORIGINS_STR.split(',')]
        elif not self.CORS_ORIGINS: # Se não veio do env_str e o default da classe (lista vazia) ainda está lá
            self.CORS_ORIGINS = ["*"] # Default para desenvolvimento, permite todas as origens

        # Log de verificação da API Key DENTRO do __init__ após super().__init__()
        # Neste ponto, pydantic-settings já tentou carregar do ambiente.
        if self.OPENAI_API_KEY:
            if len(self.OPENAI_API_KEY) == 51 and self.OPENAI_API_KEY.startswith("sk-"):
                 logger.info(f"Settings __init__: OpenAI API Key carregada e parece ter formato válido.")
            else:
                 logger.warning(f"Settings __init__: OpenAI API Key carregada, mas tem formato/comprimento incomum: {len(self.OPENAI_API_KEY)} caracteres. Verifique a configuração no Secret Manager/variáveis de ambiente do Cloud Run.")
        else:
            logger.error("ALERTA CRÍTICO (Settings __init__): OpenAI API Key é None ou vazia APÓS pydantic-settings tentar carregar!")
            logger.error(f"Verifique se a variável de ambiente OPENAI_API_KEY (ou o segredo mapeado para ela) está corretamente configurada no serviço Cloud Run e se a conta de serviço tem permissão para acessá-la.")
            # Descomente para parar a aplicação se a chave for essencial e não encontrada durante o startup.
            # raise ValueError("Configuração crítica OPENAI_API_KEY ausente. A aplicação não pode iniciar.")

# Instancia o objeto settings para ser importado e usado em toda a aplicação
settings = Settings()

# Verificação adicional após a instanciação (para logs de startup)
if not settings.OPENAI_API_KEY:
    logger.critical("ALERTA CRÍTICO (config.py global): settings.OPENAI_API_KEY é None ou vazia após instanciação de Settings. Serviços OpenAI provavelmente falharão.")
else:
    # Para não logar a chave inteira, mesmo ofuscada, nos logs de produção por padrão
    key_status = "Presente e parece válida" if (len(settings.OPENAI_API_KEY) == 51 and settings.OPENAI_API_KEY.startswith("sk-")) else "Presente, mas formato/comprimento incomum"
    logger.info(f"Verificação global em config.py: OpenAI API Key status: {key_status}. Modelo LLM: {settings.LLM_MODEL_NAME}, Modelo Embedding: {settings.EMBEDDING_MODEL_NAME}")