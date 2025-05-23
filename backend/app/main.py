# app/main.py
from fastapi import FastAPI, HTTPException, Depends, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import uvicorn
from typing import List, Dict, Any, Optional
import os
import json

# Configurações do projeto
from app.core.config import settings

# Lógica de processamento de dados e carregamento
from app.data_processing.loader import (
    load_and_process_data,
    get_all_documents_dict
)

# Lógica do agente de IA
from app.agent.agent_core import (
    create_recruitment_agent_executor, get_llm, get_embeddings,
    HumanMessage, AIMessage # 'Document' foi removido daqui
)
# Importação CORRETA para a classe Document do LangChain
from langchain_core.documents import Document # <--- ADICIONADO AQUI

# Modelos Pydantic para requisições e respostas
from app.models.pydantic_models import (
    QueryRequest, QueryResponse, JobIdRequest, JobDetailsResponse,
    ApplicantIdRequest, ApplicantDetailsResponse, SourceDocument,
    ListJobsResponse, JobSummary, ListApplicantsResponse, ApplicantSummary, StatusResponse
)

# Configuração do Logging principal da aplicação
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - [%(levelname)s] - (APP) %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

agent_executor: Optional[Any] = None
vector_store: Optional[Any] = None
raw_docs_dict: Dict[str, Any] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent_executor, vector_store, raw_docs_dict
    logger.info("LIFESPAN: Iniciando IntelliMatch AI...")
    
    if not settings.OPENAI_API_KEY:
        logger.critical("LIFESPAN CRÍTICO: OpenAI API Key não configurada em 'settings'. Serviços OpenAI não funcionarão.")
    else:
        if len(settings.OPENAI_API_KEY) == 51 and settings.OPENAI_API_KEY.startswith("sk-"):
            logger.info(f"LIFESPAN: OpenAI API Key parece válida e carregada. Modelo LLM: {settings.LLM_MODEL_NAME}")
        else:
            logger.warning(f"LIFESPAN: OpenAI API Key carregada, mas tem formato/comprimento incomum ({len(settings.OPENAI_API_KEY)} caracteres). Verifique o .env.")

    logger.info(f"ChromaDB Path (settings): {settings.CHROMA_DB_PATH}, Coleção: {settings.CHROMA_COLLECTION_NAME}")

    try:
        logger.info("LIFESPAN: Pré-inicializando embeddings...")
        embedding_function = get_embeddings()
        logger.info(f"LIFESPAN: Embeddings inicializados (tipo: {type(embedding_function)}).")

        logger.info("LIFESPAN: Pré-inicializando LLM...")
        llm_instance = get_llm()
        logger.info(f"LIFESPAN: LLM inicializado (tipo: {type(llm_instance)}).")

        logger.info("LIFESPAN: Chamando load_and_process_data...")
        processed_docs_list, vector_store_instance, all_loaded_raw_docs_dict = load_and_process_data(
            chroma_db_path=settings.CHROMA_DB_PATH,
            collection_name=settings.CHROMA_COLLECTION_NAME,
            embedding_function=embedding_function
        )
        vector_store = vector_store_instance
        raw_docs_dict = all_loaded_raw_docs_dict
        logger.info(f"LIFESPAN: load_and_process_data concluído. {len(processed_docs_list)} docs para store. {len(raw_docs_dict)} docs no raw_dict.")

        if vector_store:
            logger.info("LIFESPAN: Chamando create_recruitment_agent_executor...")
            agent_executor = create_recruitment_agent_executor(
                llm=llm_instance,
                vector_store_retriever=vector_store.as_retriever()
            )
            if agent_executor: logger.info("LIFESPAN: Agente de recrutamento inicializado com sucesso.")
            else: logger.error("LIFESPAN: Falha ao inicializar o agente (executor retornou None).")
        else:
            logger.warning("LIFESPAN: Vector store não inicializado. Criando agente sem retriever.")
            agent_executor = create_recruitment_agent_executor(llm=llm_instance, vector_store_retriever=None)
            if agent_executor: logger.info("LIFESPAN: Agente inicializado SEM ferramenta de busca.")
            else: logger.error("LIFESPAN: Falha ao inicializar agente (executor None), mesmo sem retriever.")

    except ValueError as ve:
        logger.critical(f"LIFESPAN CRÍTICO - ValueError na inicialização: {ve}", exc_info=False)
    except Exception as e:
        logger.error(f"LIFESPAN ERRO CRÍTICO GERAL na inicialização: {e}", exc_info=True)
    
    logger.info("LIFESPAN: Startup da aplicação FastAPI prestes a ser marcado como completo pelo Uvicorn.")
    yield
    logger.info("LIFESPAN: Finalizando IntelliMatch AI...")

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="API para o IntelliMatch AI, um assistente inteligente de recrutamento.",
    version="0.1.5", 
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url=f"{settings.API_V1_STR}/docs",
    redoc_url=f"{settings.API_V1_STR}/redoc",
    lifespan=lifespan
)

if settings.CORS_ORIGINS:
    logger.info(f"Configurando CORS para as seguintes origens: {settings.CORS_ORIGINS}")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin).strip() for origin in settings.CORS_ORIGINS],
        allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
    )
else: logger.warning("CORS_ORIGINS não configurado.")

API_PREFIX = settings.API_V1_STR

@app.get("/", tags=["Root"], response_model=StatusResponse)
async def read_root():
    return StatusResponse(status="ok", message=f"{settings.PROJECT_NAME} Backend is running!")

@app.post(f"{API_PREFIX}/query_agent", response_model=QueryResponse, tags=["Agent"])
async def query_agent_endpoint(request: QueryRequest = Body(...)):
    if not agent_executor:
        logger.error("Endpoint /query_agent: Agente não inicializado.")
        raise HTTPException(status_code=503, detail="Agente de IA não disponível.")
    
    logger.info(f"Query para agente: '{request.query}' (Sessão: {request.session_id})")
    agent_input = {"input": request.query}
    if request.chat_history:
        formatted_history = [HumanMessage(content=msg["content"]) if msg.get("type") == "human" else AIMessage(content=msg.get("content", "")) for msg in request.chat_history]
        if formatted_history: agent_input["chat_history"] = formatted_history
        logger.info(f"Histórico de chat com {len(formatted_history)} mensagens incluído.")

    try:
        logger.info(f"Endpoint /query_agent: Invocando agent_executor.ainvoke com input: '{agent_input.get('input')}'")
        response_dict = await agent_executor.ainvoke(agent_input)
        answer = response_dict.get("output", "Sem 'output' padrão do agente.")
        sources_data: List[SourceDocument] = []
        if "intermediate_steps" in response_dict and response_dict["intermediate_steps"]:
            for step_action, step_observation in response_dict["intermediate_steps"]:
                if isinstance(step_observation, list) and all(isinstance(doc_obj, Document) for doc_obj in step_observation):
                    for doc_obj_item in step_observation: sources_data.append(SourceDocument(page_content=doc_obj_item.page_content, metadata=doc_obj_item.metadata))
                elif isinstance(step_observation, str):
                    tool_name = step_action.tool if hasattr(step_action, 'tool') else "unknown_tool"
                    tool_input = step_action.tool_input if hasattr(step_action, 'tool_input') else {}
                    sources_data.append(SourceDocument(page_content=step_observation, metadata={"source_tool": tool_name, "tool_input": tool_input}))
        logger.info(f"Resposta do agente: {answer[:200]}... ({len(sources_data)} fontes)")
        return QueryResponse(answer=answer, sources=sources_data, session_id=request.session_id)

    except Exception as e:
        logger.error(f"Erro DURANTE agent_executor.ainvoke: {type(e).__name__} - {e}", exc_info=True)
        error_message_to_client = f"Erro interno ao processar a query com o agente."
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_details = e.response.json() 
                logger.error(f"Detalhes do erro da API OpenAI (JSON): {error_details}")
                api_error_message = error_details.get("error", {}).get("message", "")
                if api_error_message: error_message_to_client = f"Erro da API OpenAI: {api_error_message}"
            except json.JSONDecodeError: 
                try:
                    error_text = e.response.text 
                    logger.error(f"Detalhes do erro da API OpenAI (Texto): {error_text}")
                    if error_text: error_message_to_client = f"Erro da API OpenAI (texto): {error_text[:200]}"
                except Exception as e_text: logger.error(f"Não foi possível ler o texto da resposta do erro da API: {e_text}")
            except Exception as e_resp_json: logger.error(f"Erro ao processar e.response.json(): {e_resp_json}")
        elif isinstance(e, ValueError) and "OPENAI_API_KEY" in str(e):
             error_message_to_client = "Erro de configuração da API Key. Contate o administrador."
        raise HTTPException(status_code=500, detail=error_message_to_client)

@app.get(f"{API_PREFIX}/jobs", response_model=ListJobsResponse, tags=["Data Access"])
async def list_jobs_endpoint(skip: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=1000)):
    global raw_docs_dict
    if not raw_docs_dict: return ListJobsResponse(jobs=[], total=0) 
    job_summaries: List[JobSummary] = [
        JobSummary(
            job_id=doc.metadata.get("codigo_vaga", k.replace("vaga_", "")),
            title=doc.metadata.get("titulo_vaga", "N/A")
        ) for k, doc in raw_docs_dict.items() if doc.metadata.get("type") == "vaga" and doc.metadata.get("has_valid_metadata", False)
    ]
    total_jobs = len(job_summaries); paginated_jobs = job_summaries[skip : skip + limit]
    logger.info(f"Listando vagas: {len(paginated_jobs)} de {total_jobs}")
    return ListJobsResponse(jobs=paginated_jobs, total=total_jobs)

@app.post(f"{API_PREFIX}/job_details", response_model=JobDetailsResponse, tags=["Data Access"])
async def get_job_details_endpoint(request: JobIdRequest = Body(...)):
    global raw_docs_dict; doc_key = f"vaga_{str(request.job_id)}"
    if doc_key not in raw_docs_dict or not raw_docs_dict[doc_key].metadata.get("has_valid_metadata", False):
        raise HTTPException(status_code=404, detail=f"Vaga ID {request.job_id} não encontrada ou inválida.")
    job_doc = raw_docs_dict[doc_key]; metadata = job_doc.metadata; logger.info(f"Detalhes para vaga ID: {request.job_id}")
    return JobDetailsResponse(
        job_id=metadata.get("codigo_vaga", str(request.job_id)), title=metadata.get("titulo_vaga"),
        description=job_doc.page_content, cliente=metadata.get("cliente"),
        vaga_sap=metadata.get("vaga_sap"), tipo_contratacao=metadata.get("tipo_contratacao"),
        nivel_profissional=metadata.get("nivel_profissional"), nivel_academico=metadata.get("nivel_academico"),
        nivel_ingles=metadata.get("nivel_ingles"), nivel_espanhol=metadata.get("nivel_espanhol"),
        areas_atuacao_flat=metadata.get("areas_atuacao"), 
        local_trabalho=metadata.get("local_trabalho"),
        principais_atividades_flat=metadata.get("principais_atividades"), 
        competencias_tecnicas_flat=metadata.get("competencias_tecnicas")
    )

@app.get(f"{API_PREFIX}/applicants", response_model=ListApplicantsResponse, tags=["Data Access"])
async def list_applicants_endpoint(skip: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=1000)):
    global raw_docs_dict
    if not raw_docs_dict: return ListApplicantsResponse(applicants=[], total=0)
    applicant_summaries: List[ApplicantSummary] = [
        ApplicantSummary(
            applicant_id=doc.metadata.get("codigo_profissional", k.replace("candidato_", "")),
            name=doc.metadata.get("nome", "N/A")
        ) for k, doc in raw_docs_dict.items() if doc.metadata.get("type") == "candidato" and doc.metadata.get("has_valid_metadata", False)
    ]
    total_applicants = len(applicant_summaries); paginated_applicants = applicant_summaries[skip : skip + limit]
    logger.info(f"Listando candidatos: {len(paginated_applicants)} de {total_applicants}")
    return ListApplicantsResponse(applicants=paginated_applicants, total=total_applicants)

@app.post(f"{API_PREFIX}/applicant_details", response_model=ApplicantDetailsResponse, tags=["Data Access"])
async def get_applicant_details_endpoint(request: ApplicantIdRequest = Body(...)):
    global raw_docs_dict; doc_key = f"candidato_{str(request.applicant_id)}"
    if doc_key not in raw_docs_dict or not raw_docs_dict[doc_key].metadata.get("has_valid_metadata", False):
        raise HTTPException(status_code=404, detail=f"Candidato ID {request.applicant_id} não encontrado ou inválido.")
    applicant_doc = raw_docs_dict[doc_key]; metadata = applicant_doc.metadata; logger.info(f"Detalhes para candidato ID: {request.applicant_id}")
    return ApplicantDetailsResponse(
        applicant_id=metadata.get("codigo_profissional", str(request.applicant_id)), name=metadata.get("nome"),
        resume_summary=applicant_doc.page_content, email=metadata.get("email"),
        area_atuacao=metadata.get("area_atuacao"), 
        conhecimentos_tecnicos_flat=metadata.get("conhecimentos_tecnicos"), 
        nivel_profissional_candidato=metadata.get("nivel_profissional_candidato"),
        nivel_academico=metadata.get("nivel_academico"), nivel_ingles=metadata.get("nivel_ingles"),
        nivel_espanhol=metadata.get("nivel_espanhol")
    )

if __name__ == "__main__":
    logger.info("Rodando FastAPI com Uvicorn diretamente para desenvolvimento local (main.py)...")
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True, lifespan="on", log_level="info")