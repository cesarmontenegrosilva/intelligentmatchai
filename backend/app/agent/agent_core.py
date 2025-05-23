# app/agent/agent_core.py
import logging
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools.retriever import create_retriever_tool
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
# Document é usado internamente pelo LangChain, não precisa ser exportado por este módulo
# Se alguma função aqui *retornasse* um Document para main.py, aí sim main.py precisaria saber o tipo.

from app.core.config import settings

try:
    from .prompts import (
        RECRUITMENT_AGENT_SYSTEM_PROMPT_TEMPLATE,
        RAG_CONTEXTUALIZE_PROMPT_TEMPLATE
    )
    logger.info("Módulo prompts.py carregado.")
except ImportError as e:
    logger.warning(f"Módulo prompts.py não encontrado ou erro: {e}. Usando fallbacks.")
    RECRUITMENT_AGENT_SYSTEM_PROMPT_TEMPLATE = """Você é o IntelligentMatch AI. Contexto: {context}. Responda em Português."""
    RAG_CONTEXTUALIZE_PROMPT_TEMPLATE = """Histórico: {chat_history}\nPergunta: {question}\nPergunta Independente:"""

_llm_instance: Optional[ChatOpenAI] = None
_embeddings_instance: Optional[OpenAIEmbeddings] = None

def get_llm() -> ChatOpenAI:
    global _llm_instance
    if _llm_instance is None:
        api_key_to_use = settings.OPENAI_API_KEY
        model_to_use = settings.LLM_MODEL_NAME
        logger.info(f"GET_LLM: Tentando inicializar LLM '{model_to_use}'. Chave API (de settings): {'Presente' if api_key_to_use else 'AUSENTE'}")
        if not api_key_to_use:
            logger.error("ERRO FATAL em get_llm: settings.OPENAI_API_KEY é None ou vazia!"); raise ValueError("OPENAI_API_KEY não configurada.")
        try:
            _llm_instance = ChatOpenAI(temperature=0.1, model_name=model_to_use, openai_api_key=api_key_to_use, max_tokens=1500)
            logger.info(f"LLM ({model_to_use}) instanciado.")
        except Exception as e:
            logger.error(f"Falha ao instanciar LLM ({model_to_use}): {e}", exc_info=True); raise ValueError(f"Não foi possível inicializar LLM: {e}")
    return _llm_instance

def get_embeddings() -> OpenAIEmbeddings:
    global _embeddings_instance
    if _embeddings_instance is None:
        api_key_to_use = settings.OPENAI_API_KEY
        model_to_use = settings.EMBEDDING_MODEL_NAME
        logger.info(f"GET_EMBEDDINGS: Tentando inicializar Embeddings '{model_to_use}'. Chave API (de settings): {'Presente' if api_key_to_use else 'AUSENTE'}")
        if not api_key_to_use:
            logger.error("ERRO FATAL em get_embeddings: settings.OPENAI_API_KEY é None ou vazia!"); raise ValueError("OPENAI_API_KEY não configurada.")
        try:
            _embeddings_instance = OpenAIEmbeddings(openai_api_key=api_key_to_use, model=model_to_use)
            logger.info(f"Embeddings ({model_to_use}) inicializados.")
        except Exception as e:
            logger.error(f"Falha ao instanciar Embeddings ({model_to_use}): {e}", exc_info=True); raise ValueError(f"Não foi possível inicializar Embeddings: {e}")
    return _embeddings_instance

def create_simple_rag_chain(llm: ChatOpenAI, retriever: VectorStoreRetriever):
    logger.info("Criando RAG chain simples...")
    contextualize_q_prompt = PromptTemplate.from_template(RAG_CONTEXTUALIZE_PROMPT_TEMPLATE)
    contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()
    qa_system_prompt = RECRUITMENT_AGENT_SYSTEM_PROMPT_TEMPLATE
    qa_prompt = ChatPromptTemplate.from_messages([("system", qa_system_prompt), ("human", "{question}")])
    def format_docs(docs: List[Any]) -> str: return "\n\n".join(doc.page_content for doc in docs if hasattr(doc, 'page_content'))
    def contextualized_retriever_input(input_data: Dict) -> str:
        chat_history_str = ""
        if "chat_history" in input_data and input_data["chat_history"]:
            for msg in input_data["chat_history"]:
                if isinstance(msg, HumanMessage): chat_history_str += f"Humano: {msg.content}\n"
                elif isinstance(msg, AIMessage): chat_history_str += f"IA: {msg.content}\n"
            if chat_history_str: return contextualize_q_chain.invoke({"chat_history": chat_history_str.strip(), "question": input_data["question"]})
        return input_data["question"]
    rag_chain = (RunnablePassthrough.assign(context=(lambda input_data: contextualized_retriever_input(input_data)) | retriever | format_docs) | qa_prompt | llm | StrOutputParser())
    logger.info("RAG chain simples criado.")
    return rag_chain

def create_recruitment_agent_executor(llm: ChatOpenAI, vector_store_retriever: Optional[VectorStoreRetriever] = None) -> Optional[AgentExecutor]:
    logger.info("Iniciando criação do AgentExecutor...")
    tools = []
    if vector_store_retriever:
        logger.info("Criando ferramenta retriever (knowledge_base_search)...")
        try:
            retriever_tool = create_retriever_tool(vector_store_retriever, name="knowledge_base_search", description="Busca na base de conhecimento sobre vagas, candidatos e prospecções. Use para perguntas factuais. Inclua IDs se conhecidos.")
            tools.append(retriever_tool)
            logger.info("Ferramenta 'knowledge_base_search' adicionada.")
        except Exception as e: logger.error(f"Falha ao criar retriever_tool: {e}", exc_info=True)
    else: logger.warning("Vector store retriever não fornecido. 'knowledge_base_search' não criada.")
    if not tools: logger.warning("Nenhuma ferramenta configurada. Agente limitado.")
    agent_prompt_template_str = RECRUITMENT_AGENT_SYSTEM_PROMPT_TEMPLATE.replace("\nContexto relevante de documentos recuperados (este placeholder é usado por chains RAG específicas, não diretamente pelo system prompt do agente principal se ele usa ferramentas para buscar contexto):\n{context}", "").replace("\nContexto: {context}", "")
    prompt = ChatPromptTemplate.from_messages([("system", agent_prompt_template_str + "\nResponda SEMPRE em Português do Brasil."), MessagesPlaceholder(variable_name="chat_history", optional=True), ("human", "{input}"), MessagesPlaceholder(variable_name="agent_scratchpad"),])
    try:
        logger.info(f"Criando agente com {len(tools)} ferramentas.")
        agent = create_openai_functions_agent(llm, tools, prompt)
        logger.info("Agente (OpenAI Functions) criado.")
    except Exception as e: logger.error(f"Erro ao criar OpenAI Functions Agent: {e}", exc_info=True); return None
    try:
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors="Check e repasse o erro para o usuário de forma amigável.", max_iterations=7, return_intermediate_steps=True)
        logger.info("AgentExecutor criado."); return agent_executor
    except Exception as e: logger.error(f"Erro ao criar AgentExecutor: {e}", exc_info=True); return None