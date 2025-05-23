# app/agent/tools.py
from langchain.tools import BaseTool, Tool
from langchain.pydantic_v1 import BaseModel, Field # BaseModel and Field are from pydantic_v1 for Langchain
from typing import Type, Optional, List, Dict, Any
from app.agent.agent_core import get_rag_chain, get_vector_store # RAG_PROMPT removed as it's not directly used here, assumed to be used within get_rag_chain
from app.core.config import get_llm
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from app.agent.prompts import CANDIDATE_MATCHER_LLM_PROMPT
# from app.data_processing.loader import _textualize_vacancy, _textualize_applicant # Not used directly in this version, can be removed if not needed for other parts

# Initialize the retriever globalmente ou passa como argumento para as ferramentas
_vector_store = None
_retriever = None
_rag_chain = None

try:
    _vector_store = get_vector_store() # Tenta carregar ou criar
    if _vector_store: # Ensure vector_store was loaded before creating retriever
        _retriever = _vector_store.as_retriever(search_kwargs={"k": 5})
        _rag_chain = get_rag_chain(_retriever) # Pass the actual retriever object
    else:
        print("AVISO: Vector Store não pôde ser inicializado. RAG chain não disponível.")
except Exception as e:
    print(f"AVISO: Falha ao inicializar o RAG chain para as ferramentas: {e}")
    # _rag_chain remains None, which is handled in the tools

class RecruitmentDataQueryInput(BaseModel):
    query: str = Field(description="A pergunta específica sobre candidatos, prospects ou vagas.")

class RecruitmentDataQueryTool(BaseTool):
    name: str = "recruitment_data_query_tool"
    description: str = (
        "Útil para responder perguntas factuais sobre candidatos, prospects ou vagas "
        "com base na base de conhecimento de recrutamento. "
        "Use para encontrar informações como habilidades de um candidato, detalhes de uma vaga, etc."
    )
    args_schema: Type[BaseModel] = RecruitmentDataQueryInput # Corrected: Type annotation for args_schema

    def _run(self, query: str) -> str:
        if not _rag_chain:
            return "Erro: A cadeia de consulta de dados de recrutamento não está disponível."
        try:
            response = _rag_chain.invoke({"query": query}) # 'query' is the expected input key for the RAG chain from RAG_PROMPT
            # Assuming the RAG chain output dictionary has a key like 'answer' or 'result'
            # Adjust .get("answer", ...) if your RAG chain returns a different key for the main response.
            return response.get("answer", response.get("result", "Não foi possível processar a consulta ou a chave de resposta não foi encontrada."))
        except Exception as e:
            return f"Erro ao processar a consulta com RAG: {e}"

    async def _arun(self, query: str) -> str:
        if not _rag_chain:
            return "Erro: A cadeia de consulta de dados de recrutamento não está disponível."
        try:
            # Ensure your _rag_chain supports ainvoke and the input schema is correct
            response = await _rag_chain.ainvoke({"query": query})
            return response.get("answer", response.get("result", "Não foi possível processar a consulta assíncrona ou a chave de resposta não foi encontrada."))
        except Exception as e:
            return f"Erro ao processar a consulta assíncrona com RAG: {e}"


class CandidateMatcherInput(BaseModel):
    vacancy_id_or_description: str = Field(description="O ID da vaga ou uma descrição detalhada da vaga para a qual encontrar candidatos.")

class CandidateMatcherTool(BaseTool):
    name: str = "candidate_matcher_tool"
    description: str = (
        "Encontra e avalia candidatos adequados para uma vaga específica. "
        "Recebe o ID da vaga ou uma descrição detalhada da vaga."
        "Retorna uma lista de candidatos compatíveis com uma breve justificativa e pontuação."
    )
    args_schema: Type[BaseModel] = CandidateMatcherInput # Corrected: Type annotation for args_schema

    def _get_vacancy_details(self, vacancy_id_or_description: str) -> Optional[Dict[str, Any]]: # Corrected type hint
        if not _vector_store:
            print("AVISO: Vector Store não disponível em _get_vacancy_details.")
            return None

        # Tenta buscar a vaga pelo ID primeiro
        # Assuming IDs are unique and well-defined.
        # A direct lookup might be more efficient if your vector store or another DB supports it for IDs.
        # For similarity search on ID, it's a bit of a workaround.
        try:
            # Attempt to retrieve by a specific ID if metadata allows precise filtering
            docs_by_id = _vector_store.similarity_search(
                query=f"vaga com ID {vacancy_id_or_description}", # Query can be anything if filter is effective
                filter={"type": "vacancy", "id": vacancy_id_or_description},
                k=1
            )
            if docs_by_id and docs_by_id[0].metadata.get("id") == vacancy_id_or_description:
                return {"description": docs_by_id[0].page_content, "metadata": docs_by_id[0].metadata, "id": docs_by_id[0].metadata.get("id")}
        except Exception as e:
            print(f"Info: Não foi possível buscar vaga pelo ID '{vacancy_id_or_description}' diretamente ou ocorreu um erro: {e}. Tentando busca semântica.")


        # Se não encontrou por ID exato ou se a busca por ID não é o método primário, trata a entrada como uma descrição
        docs_by_description = _vector_store.similarity_search(
            vacancy_id_or_description,
            filter={"type": "vacancy"}, # General filter for vacancies
            k=1
        )
        if docs_by_description:
            return {"description": docs_by_description[0].page_content, "metadata": docs_by_description[0].metadata, "id": docs_by_description[0].metadata.get("id")}
        return None

    def _find_candidate_profiles(self, vacancy_description: str, num_candidates: int = 5) -> List[Dict[str, Any]]: # Corrected type hint
        if not _vector_store:
            print("AVISO: Vector Store não disponível em _find_candidate_profiles.")
            return []

        candidate_docs = _vector_store.similarity_search(
            query=f"Candidatos adequados para a vaga: {vacancy_description}",
            k=num_candidates,
            filter={"type": "applicant"} # Busca apenas em candidatos
        )
        return [{"profile_text": doc.page_content, "metadata": doc.metadata, "id": doc.metadata.get("id")} for doc in candidate_docs]

    def _run(self, vacancy_id_or_description: str) -> str:
        if not _vector_store:
            return "Erro: Vector Store não disponível para correspondência de candidatos."

        vacancy_info = self._get_vacancy_details(vacancy_id_or_description)
        if not vacancy_info:
            return f"Vaga com ID ou descrição '{vacancy_id_or_description}' não encontrada."

        vacancy_description_text = vacancy_info["description"]
        vacancy_id = vacancy_info.get("id", "ID Desconhecido") # Get ID for context

        candidate_profiles_data = self._find_candidate_profiles(vacancy_description_text)
        if not candidate_profiles_data:
            return f"Nenhum candidato encontrado para a vaga: {vacancy_id} ({vacancy_description_text[:100]}...)."

        # Ensure candidate profiles are properly formatted for the LLM prompt
        candidate_profiles_for_prompt = []
        for cand in candidate_profiles_data:
            cand_id = cand.get("id", "ID Desconhecido")
            profile_text = cand.get("profile_text", "Perfil não disponível")
            candidate_profiles_for_prompt.append(f"ID Candidato: {cand_id}\nPerfil:\n{profile_text}")

        candidate_profiles_text = "\n\n---\n\n".join(candidate_profiles_for_prompt)


        llm = get_llm()
        # Ensure CANDIDATE_MATCHER_LLM_PROMPT expects "vacancy_description" and "candidate_profiles"
        prompt_template = PromptTemplate(
            template=CANDIDATE_MATCHER_LLM_PROMPT,
            input_variables=["vacancy_description", "candidate_profiles"]
        )
        chain = LLMChain(llm=llm, prompt=prompt_template) # Corrected: prompt=prompt_template

        try:
            response = chain.invoke({
                "vacancy_description": f"ID Vaga: {vacancy_id}\nDescrição da Vaga:\n{vacancy_description_text}", # Provide full context
                "candidate_profiles": candidate_profiles_text
            })
            # Assuming the LLMChain output has a 'text' key with the result
            return response.get("text", "Não foi possível gerar a avaliação de compatibilidade.")
        except Exception as e:
            return f"Erro ao avaliar candidatos com LLM: {e}"

    async def _arun(self, vacancy_id_or_description: str) -> str:
        # For a true async implementation, _get_vacancy_details, _find_candidate_profiles,
        # and LLMChain.ainvoke would need to be async.
        # This is a fallback to the synchronous version.
        # Consider using something like `return await asyncio.to_thread(self._run, vacancy_id_or_description)`
        # if you want to run the sync code in a separate thread from an async context.
        return self._run(vacancy_id_or_description)

# Lista de ferramentas para o agente
recruitment_tools: List[BaseTool] = [ # Corrected: Initialize the list with tool instances
    RecruitmentDataQueryTool(),
    CandidateMatcherTool()
]