# app/models/pydantic_models.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union

# --- Modelos para Requisições ---

class QueryRequest(BaseModel):
    query: str = Field(..., description="A pergunta ou comando para o agente de IA.")
    session_id: Optional[str] = Field(None, description="ID de sessão opcional para manter o contexto da conversa.")
    # O histórico do chat pode ser uma lista de dicionários ou objetos Pydantic mais estruturados
    chat_history: Optional[List[Dict[str, str]]] = Field(None, description="Histórico da conversa. Ex: [{'type': 'human', 'content': 'Olá'}, {'type': 'ai', 'content': 'Oi!'}]")

class JobIdRequest(BaseModel):
    job_id: str = Field(..., description="O ID único da vaga.")

class ApplicantIdRequest(BaseModel):
    applicant_id: str = Field(..., description="O ID único do candidato.")

# --- Modelos para Respostas ---

class SourceDocument(BaseModel):
    page_content: str = Field(..., description="Conteúdo textual do documento fonte.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadados associados ao documento fonte.")
    # score: Optional[float] = None # Se o retriever fornecer um score de relevância

class QueryResponse(BaseModel):
    answer: str = Field(..., description="A resposta gerada pelo agente de IA.")
    sources: List[SourceDocument] = Field(default_factory=list, description="Lista de documentos fonte usados para gerar a resposta, se aplicável.")
    session_id: Optional[str] = Field(None, description="ID de sessão retornado, pode ser o mesmo da requisição ou um novo.")
    # error: Optional[str] = None # Para retornar mensagens de erro de forma estruturada

class JobSummary(BaseModel):
    job_id: str
    title: Optional[str] = "Título não disponível"

class ListJobsResponse(BaseModel):
    jobs: List[JobSummary]
    total: int = Field(..., description="Número total de vagas disponíveis.")
    # skip: int
    # limit: int

class ApplicantSummary(BaseModel):
    applicant_id: str
    name: Optional[str] = "Nome não disponível"

class ListApplicantsResponse(BaseModel):
    applicants: List[ApplicantSummary]
    total: int = Field(..., description="Número total de candidatos disponíveis.")

class JobDetailsResponse(BaseModel):
    job_id: str
    title: Optional[str] = "N/A"
    description: Optional[str] = Field(None, description="Conteúdo textual principal da descrição da vaga.")
    cliente: Optional[str] = "N/A"
    vaga_sap: Optional[str] = "N/A"
    tipo_contratacao: Optional[str] = "N/A"
    nivel_profissional: Optional[str] = "N/A"
    nivel_academico: Optional[str] = "N/A"
    nivel_ingles: Optional[str] = "N/A"
    nivel_espanhol: Optional[str] = "N/A"
    areas_atuacao_flat: Optional[str] = "N/A" # Campo textualizado
    local_trabalho: Optional[str] = "N/A"
    principais_atividades_flat: Optional[str] = "N/A" # Campo textualizado
    competencias_tecnicas_flat: Optional[str] = "N/A" # Campo textualizado
    # Você pode adicionar todos os metadados relevantes que textualizou em loader.py
    # ou até mesmo um campo `full_metadata: Optional[Dict[str, Any]] = None`

class ApplicantDetailsResponse(BaseModel):
    applicant_id: str
    name: Optional[str] = "N/A"
    resume_summary: Optional[str] = Field(None, description="Conteúdo textual principal do CV do candidato.")
    email: Optional[str] = "N/A"
    area_atuacao: Optional[str] = "N/A"
    conhecimentos_tecnicos_flat: Optional[str] = "N/A" # Campo textualizado
    nivel_profissional_candidato: Optional[str] = "N/A"
    nivel_academico: Optional[str] = "N/A"
    nivel_ingles: Optional[str] = "N/A"
    nivel_espanhol: Optional[str] = "N/A"
    # `full_metadata: Optional[Dict[str, Any]] = None`

# Modelo genérico para respostas de status ou erro simples
class StatusResponse(BaseModel):
    status: str
    message: Optional[str] = None

# Adicione outros modelos Pydantic conforme sua API evolui
# Exemplo: Modelo para match entre candidato e vaga
class MatchScoreRequest(BaseModel):
    job_id: str
    applicant_id: str

class MatchScoreResponse(BaseModel):
    job_id: str
    applicant_id: str
    score: int = Field(..., ge=0, le=100, description="Pontuação de match de 0 a 100.")
    justification: str = Field(..., description="Justificativa para a pontuação de match.")
    # matched_skills: Optional[List[str]] = None
    # missing_skills: Optional[List[str]] = None