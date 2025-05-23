# app/data_processing/loader.py
import json
import logging
from typing import List, Dict, Tuple, Any, Optional
from pathlib import Path
from collections import Counter

from langchain_core.documents import Document
# Considere mudar para: from langchain_chroma import Chroma (após instalar langchain-chroma e testar)
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.embeddings import Embeddings

from app.core.config import settings

logger = logging.getLogger(__name__)

def safe_filter_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filtra e sanitiza metadados para garantir que os valores são de tipos simples
    aceitáveis pelo ChromaDB (str, int, float, bool, None).
    Converte listas de tipos simples para strings separadas por vírgula.
    Outros tipos complexos são convertidos para sua representação string, com truncamento.
    """
    if not isinstance(metadata, dict):
        logger.warning(f"safe_filter_metadata recebeu algo que não é um dict: {type(metadata)}. Retornando dict vazio.")
        return {}
        
    safe_meta = {}
    for key, value in metadata.items():
        if value is None:
            safe_meta[key] = None
        elif isinstance(value, (str, int, float, bool)):
            safe_meta[key] = value
        elif isinstance(value, list):
            simple_list_items = [str(v) for v in value if isinstance(v, (str, int, float, bool))]
            safe_meta[key] = ", ".join(simple_list_items) if simple_list_items else "" 
        elif isinstance(value, dict):
            try:
                json_str = json.dumps(value)
                safe_meta[key] = json_str[:450] + "..." if len(json_str) > 453 else json_str # Limite um pouco maior para JSON
            except TypeError:
                str_value = str(value)
                safe_meta[key] = str_value[:450] + "..." if len(str_value) > 453 else str_value
            logger.debug(f"Metadado '{key}' era um dicionário e foi convertido para string.")
        else:
            str_value = str(value)
            safe_meta[key] = str_value[:450] + "..." if len(str_value) > 453 else str_value
            logger.debug(f"Metadado '{key}' era do tipo {type(value)} e foi convertido para string.")
    return safe_meta

def _create_error_metadata(source_type: str, doc_id: str, error_message: str, original_exception: Optional[Exception] = None) -> Dict[str, Any]:
    """Cria um dicionário de metadados padrão para casos de erro na textualização."""
    log_message = f"METADADOS DE ERRO para {source_type}, ID {doc_id}: {error_message}"
    if original_exception:
        logger.error(f"Exceção original ao criar metadados de erro para {source_type}, ID {doc_id} (mensagem: '{error_message}'):", exc_info=original_exception)
    else:
        logger.error(log_message)

    return {
        "source": source_type,
        "type": "error_document",
        "original_id": str(doc_id),
        "error_details": str(error_message)[:450],
        "has_valid_metadata": False
    }

def _load_json_file(file_path: str) -> Dict:
    path = Path(file_path)
    if not path.exists():
        logger.error(f"Arquivo JSON não encontrado em: {file_path}")
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Arquivo {file_path} carregado com sucesso ({len(data)} chaves no nível raiz).")
        return data
    except json.JSONDecodeError as e_json:
        logger.error(f"Erro ao decodificar JSON de {file_path}: {e_json}")
        return {}
    except Exception as e_gen:
        logger.error(f"Erro inesperado ao carregar {file_path}: {e_gen}", exc_info=True)
        return {}

def _textualize_vaga(vaga_id: str, vaga_data: Dict) -> Tuple[str, Dict[str, Any]]:
    doc_source_file = settings.DATA_PATH_VAGAS
    
    try:
        if not isinstance(vaga_data, dict):
            raise TypeError(f"dados de entrada não são dicionário (tipo: {type(vaga_data)})")

        ib = vaga_data.get("informacoes_basicas", {})
        pv = vaga_data.get("perfil_vaga", {})
        
        if not isinstance(ib, dict) or not isinstance(pv, dict):
            raise TypeError("estrutura interna inválida (informacoes_basicas ou perfil_vaga não são dicionários)")

        titulo = str(ib.get("titulo_vaga", "N/A"))
        cliente = str(ib.get("cliente", "N/A"))
        vaga_sap = str(ib.get("vaga_sap", "N/A"))
        tipo_contratacao = str(ib.get("tipo_contratacao", "N/A"))
        nivel_profissional = str(pv.get("nivel profissional", pv.get("nivel_profissional", "N/A")))
        nivel_academico = str(pv.get("nivel_academico", "N/A"))
        nivel_ingles = str(pv.get("nivel_ingles", "N/A"))
        nivel_espanhol = str(pv.get("nivel_espanhol", "N/A"))
        areas_atuacao_raw = pv.get("areas_atuacao", "N/A")
        principais_atividades = str(pv.get("principais_atividades", "N/A"))
        competencias_tecnicas = str(pv.get("competencia_tecnicas_e_comportamentais", "N/A"))
        local_trabalho = str(pv.get("local_trabalho", "N/A"))
        demais_obs = str(pv.get("demais_observacoes", ""))

        content_parts = [
            f"VAGA: {titulo}", f"ID da Vaga: {vaga_id}", f"Cliente: {cliente}",
            f"Tipo de Contratação: {tipo_contratacao}", f"É vaga SAP? {vaga_sap}",
            f"Nível Profissional Requerido: {nivel_profissional}",
            f"Nível Acadêmico: {nivel_academico}", f"Nível de Inglês: {nivel_ingles}"
        ]
        if nivel_espanhol.strip().lower() not in ["n/a", "nenhum", "", "none"]:
            content_parts.append(f"Nível de Espanhol: {nivel_espanhol}")
        
        areas_atuacao_str = areas_atuacao_raw if isinstance(areas_atuacao_raw, str) else ', '.join(map(str, areas_atuacao_raw)) if isinstance(areas_atuacao_raw, list) else 'N/A'
        content_parts.append(f"Áreas de Atuação: {areas_atuacao_str}")
        content_parts.append(f"Local de Trabalho: {local_trabalho}")
        content_parts.append(f"Principais Atividades:\n{principais_atividades}")
        content_parts.append(f"Competências Técnicas e Comportamentais Requeridas:\n{competencias_tecnicas}")
        if demais_obs: content_parts.append(f"Observações Adicionais: {demais_obs}")
        content = "\n".join(filter(None, content_parts))

        metadata_dict = {
            "source": doc_source_file, "type": "vaga", "codigo_vaga": str(vaga_id),
            "titulo_vaga": titulo, "cliente": cliente, "vaga_sap": vaga_sap,
            "tipo_contratacao": tipo_contratacao, "nivel_profissional": nivel_profissional,
            "nivel_academico": nivel_academico, "nivel_ingles": nivel_ingles,
            "nivel_espanhol": nivel_espanhol if nivel_espanhol.strip().lower() not in ["", "none", "n/a"] else "N/A",
            "areas_atuacao": areas_atuacao_raw, 
            "local_trabalho": local_trabalho,
            "principais_atividades": principais_atividades, 
            "competencias_tecnicas": competencias_tecnicas, 
            "has_valid_metadata": True
        }
        return content, metadata_dict

    except Exception as e:
        return f"Erro ao processar vaga ID {vaga_id}: {str(e)[:200]}", _create_error_metadata(doc_source_file, vaga_id, str(e), original_exception=e)

def _textualize_applicant(applicant_id: str, app_data: Dict) -> Tuple[str, Dict[str, Any]]:
    doc_source_file = settings.DATA_PATH_APPLICANTS
    
    try:
        if not isinstance(app_data, dict):
            raise TypeError(f"dados de entrada não são dicionário (tipo: {type(app_data)})")

        ib = app_data.get("infos_basicas", {})
        ip = app_data.get("informacoes_profissionais", {})
        fi = app_data.get("formacao_e_idiomas", {})

        if not isinstance(ib, dict) or not isinstance(ip, dict) or not isinstance(fi, dict):
            raise TypeError("estrutura interna inválida (infos_basicas, informacoes_profissionais ou formacao_e_idiomas não são dicionários)")

        nome = str(ib.get("nome", "N/A"))
        email = str(ib.get("email", "N/A"))
        objetivo = str(ib.get("objetivo_profissional", ""))
        area_atuacao_raw = ip.get("area_atuacao", "N/A")
        conhecimentos_tecnicos_raw = ip.get("conhecimentos_tecnicos", "")
        nivel_profissional_candidato = str(ip.get("nivel_profissional", "N/A"))
        nivel_academico = str(fi.get("nivel_academico", "N/A"))
        nivel_ingles = str(fi.get("nivel_ingles", "N/A"))
        nivel_espanhol = str(fi.get("nivel_espanhol", "N/A"))
        cv_pt = str(app_data.get("cv_pt", "CV não disponível."))

        content_parts = [f"CANDIDATO: {nome}", f"ID do Candidato: {applicant_id}"]
        if objetivo: content_parts.append(f"Objetivo Profissional: {objetivo}")
        
        area_atuacao_str = area_atuacao_raw if isinstance(area_atuacao_raw, str) else ', '.join(map(str, area_atuacao_raw)) if isinstance(area_atuacao_raw, list) else 'N/A'
        content_parts.append(f"Área de Atuação: {area_atuacao_str}")
        content_parts.append(f"Nível Profissional: {nivel_profissional_candidato}")
        content_parts.append(f"Nível Acadêmico: {nivel_academico}")
        content_parts.append(f"Nível de Inglês: {nivel_ingles}")
        if nivel_espanhol.strip().lower() not in ["n/a", "nenhum", "", "none"]:
            content_parts.append(f"Nível de Espanhol: {nivel_espanhol}")
        
        kt_str = conhecimentos_tecnicos_raw if isinstance(conhecimentos_tecnicos_raw, str) else ', '.join(map(str, conhecimentos_tecnicos_raw)) if isinstance(conhecimentos_tecnicos_raw, list) else 'N/A'
        if conhecimentos_tecnicos_raw and kt_str.strip().lower() not in ['n/a', '']:
            content_parts.append(f"Conhecimentos Técnicos: {kt_str}")
        content_parts.append(f"\n--- Resumo do CV ---\n{cv_pt[:3500]}...\n--- Fim do Resumo do CV ---") # Trunca CVs longos
        content = "\n".join(filter(None, content_parts))

        metadata_dict = {
            "source": doc_source_file, "type": "candidato", "codigo_profissional": str(applicant_id),
            "nome": nome, "email": email, 
            "area_atuacao": area_atuacao_raw, 
            "conhecimentos_tecnicos": conhecimentos_tecnicos_raw, 
            "nivel_profissional_candidato": nivel_profissional_candidato,
            "nivel_academico": nivel_academico, "nivel_ingles": nivel_ingles,
            "nivel_espanhol": nivel_espanhol if nivel_espanhol.strip().lower() not in ["", "none", "n/a"] else "N/A",
            "has_valid_metadata": True
        }
        return content, metadata_dict
        
    except Exception as e:
        return f"Erro ao processar candidato ID {applicant_id}. Detalhe: {str(e)[:200]}", _create_error_metadata(doc_source_file, applicant_id, str(e), original_exception=e)

def _textualize_prospect(vaga_id_prospect: str, prospect_entry_data: Dict, vaga_titulo_geral: str, prospect_index: int) -> Tuple[str, Dict[str, Any]]:
    doc_source_file = settings.DATA_PATH_PROSPECTS
    codigo_candidato_prospect = str(prospect_entry_data.get("codigo", f"unknown_{prospect_index}"))
    prospect_unique_id_part = f"vaga_{vaga_id_prospect}_cand_{codigo_candidato_prospect}_idx_{prospect_index}"
    
    try:
        if not isinstance(prospect_entry_data, dict):
            raise TypeError(f"dados de entrada para prospect não são dicionário (tipo: {type(prospect_entry_data)})")

        nome_candidato = str(prospect_entry_data.get("nome", "N/A"))
        situacao = str(prospect_entry_data.get("situacao_candidado", prospect_entry_data.get("situacao_candidato", "N/A")))
        comentario = str(prospect_entry_data.get("comentario", ""))

        content_parts = [
            f"PROSPECT para Vaga ID {vaga_id_prospect} (Título: {vaga_titulo_geral}):",
            f"Candidato: {nome_candidato} (ID Candidato: {codigo_candidato_prospect})",
            f"Situação na Vaga: {situacao}"
        ]
        if comentario: content_parts.append(f"Comentário do Recrutador: {comentario}")
        content = "\n".join(filter(None, content_parts))

        metadata_dict = {
            "source": doc_source_file, "type": "prospect",
            "vaga_id_associada": str(vaga_id_prospect),
            "codigo_candidato_associado": codigo_candidato_prospect,
            "prospect_identifier_index": prospect_index,
            "nome_candidato": nome_candidato,
            "situacao_candidato": situacao,
            "vaga_titulo_prospect": str(vaga_titulo_geral),
            "has_valid_metadata": True
        }
        return content, metadata_dict

    except Exception as e:
        return f"Erro ao processar prospect ({prospect_unique_id_part}). Detalhe: {str(e)[:200]}", _create_error_metadata(doc_source_file, prospect_unique_id_part, str(e), original_exception=e)

def _create_documents_from_data(
    vagas_data: Dict,
    applicants_data: Dict,
    prospects_data: Dict
) -> Tuple[List[Document], Dict[str, Document]]:
    documents: List[Document] = []
    raw_docs_dict: Dict[str, Document] = {} 

    logger.info("Processando dados de vagas...")
    for vaga_id, vaga_content in vagas_data.items():
        try:
            content, metadata = _textualize_vaga(str(vaga_id), vaga_content)
            
            if not isinstance(metadata, dict): # Segurança extra
                logger.error(f"PÓS-TEXTUALIZE (INESPERADO): Metadados de vaga {vaga_id} não é dict. Tipo: {type(metadata)}. Pulando.")
                continue 
            
            # Usa safe_filter_metadata apenas se os metadados originais foram considerados válidos
            # Se 'has_valid_metadata' for False, significa que _textualize_vaga já retornou metadados de erro.
            filtered_meta = safe_filter_metadata(metadata.copy()) if metadata.get("has_valid_metadata", False) else metadata
            
            doc = Document(page_content=content, metadata=filtered_meta)
            documents.append(doc)
            if filtered_meta.get("has_valid_metadata", False): # Adiciona ao raw_dict apenas se os metadados são válidos
                 raw_docs_dict[f"vaga_{vaga_id}"] = doc
        except Exception as e:
            logger.error(f"Exceção inesperada no loop de _create_documents_from_data para vaga ID {vaga_id}: {e}", exc_info=True)

    logger.info("Processando dados de candidatos...")
    for app_id, app_content in applicants_data.items():
        try:
            content, metadata = _textualize_applicant(str(app_id), app_content)
            if not isinstance(metadata, dict):
                logger.error(f"PÓS-TEXTUALIZE (INESPERADO): Metadados de candidato {app_id} não é dict. Tipo: {type(metadata)}. Pulando.")
                continue
            
            filtered_meta = safe_filter_metadata(metadata.copy()) if metadata.get("has_valid_metadata", False) else metadata
            doc = Document(page_content=content, metadata=filtered_meta)
            documents.append(doc)
            if filtered_meta.get("has_valid_metadata", False):
                raw_docs_dict[f"candidato_{app_id}"] = doc
        except Exception as e:
            logger.error(f"Exceção inesperada no loop de _create_documents_from_data para candidato ID {app_id}: {e}", exc_info=True)

    logger.info("Processando dados de prospects...")
    global_prospect_index = 0 
    for vaga_id_prospect, prospect_geral_data in prospects_data.items():
        if not isinstance(prospect_geral_data, dict):
            logger.warning(f"Entrada de prospect para vaga_id {vaga_id_prospect} não é um dicionário. Pulando.")
            continue
        vaga_titulo_geral = str(prospect_geral_data.get("titulo", "Título Desconhecido"))
        prospect_list = prospect_geral_data.get("prospects", [])
        if not isinstance(prospect_list, list):
            logger.warning(f"Lista de prospects para vaga_id {vaga_id_prospect} não é uma lista. Pulando.")
            continue
        for entry in prospect_list:
            try:
                unique_prospect_idx = global_prospect_index
                global_prospect_index +=1
                content, metadata = _textualize_prospect(str(vaga_id_prospect), entry, vaga_titulo_geral, unique_prospect_idx)
                if not isinstance(metadata, dict):
                    logger.error(f"PÓS-TEXTUALIZE (INESPERADO): Metadados de prospect (vaga {vaga_id_prospect}, cand {entry.get('codigo')}) não é dict. Pulando.")
                    continue
                
                filtered_meta = safe_filter_metadata(metadata.copy()) if metadata.get("has_valid_metadata", False) else metadata
                doc = Document(page_content=content, metadata=filtered_meta)
                documents.append(doc)
            except Exception as e:
                logger.error(f"Exceção inesperada no loop de _create_documents_from_data para prospect (Vaga {vaga_id_prospect}, Candidato {entry.get('codigo')}): {e}", exc_info=True)
    
    logger.info(f"Total de {len(documents)} documentos preparados. {len(raw_docs_dict)} docs no raw_dict (vagas/candidatos).")
    return documents, raw_docs_dict

_vector_store_cache: Optional[Chroma] = None
_raw_docs_dict_cache: Dict[str, Document] = {}

def load_and_process_data(
    chroma_db_path: str,
    collection_name: str,
    embedding_function: Embeddings 
) -> Tuple[List[Document], Optional[Chroma], Dict[str, Document]]:
    global _vector_store_cache, _raw_docs_dict_cache
    
    logger.info("Iniciando carregamento e processamento de dados...")
    vagas_data = _load_json_file(settings.DATA_PATH_VAGAS)
    applicants_data = _load_json_file(settings.DATA_PATH_APPLICANTS)
    prospects_data = _load_json_file(settings.DATA_PATH_PROSPECTS)

    if not (vagas_data or applicants_data or prospects_data):
        logger.error("Todos os arquivos de dados estão vazios ou não foram encontrados.")
        # ... (lógica de fallback como antes)
        return [], None, {}

    all_documents_prepared, raw_docs_dict_created = _create_documents_from_data(
        vagas_data, applicants_data, prospects_data
    )
    _raw_docs_dict_cache = raw_docs_dict_created # Cacheia os documentos Vaga e Candidato com metadados válidos
    
    # Filtra para ChromaDB apenas documentos que tiveram metadados válidos na origem
    documents_for_chroma = [doc for doc in all_documents_prepared if doc.metadata.get("has_valid_metadata", False)]
    
    if not documents_for_chroma:
        logger.warning("Nenhum documento com metadados válidos foi gerado para o vector store.")
        # ... (lógica de fallback como antes)
        return [], None, _raw_docs_dict_cache

    logger.info(f"Tentando criar/carregar ChromaDB em {chroma_db_path} para a coleção {collection_name}")
    try:
        vector_store = Chroma(collection_name=collection_name, embedding_function=embedding_function, persist_directory=chroma_db_path)
        logger.info(f"ChromaDB carregado/criado. Documentos atuais na coleção: {vector_store._collection.count()}")
        
        doc_ids_generated = []
        for i, doc in enumerate(documents_for_chroma):
            doc_type = doc.metadata.get("type", "unknown")
            id_str = f"error_doc_{i}" # Fallback inicial
            if doc_type == "vaga": id_str = f"vaga_{doc.metadata.get('codigo_vaga', f'v_err_{i}')}"
            elif doc_type == "candidato": id_str = f"candidato_{doc.metadata.get('codigo_profissional', f'c_err_{i}')}"
            elif doc_type == "prospect":
                idx = doc.metadata.get('prospect_identifier_index', i) 
                v_id = doc.metadata.get('vaga_id_associada', 'NA')
                c_id = doc.metadata.get('codigo_candidato_associado', 'NA')
                id_str = f"prospect_{v_id}_{c_id}_{idx}"
            else: # Documentos de erro (type="error_document") não devem chegar aqui devido ao filtro anterior
                  # mas se chegarem, terão um ID baseado no original_id
                original_id = doc.metadata.get('original_id', f'untyped_idx_{i}')
                id_str = f"{doc_type}_{original_id}"
            doc_ids_generated.append(id_str)
        
        if len(doc_ids_generated) != len(set(doc_ids_generated)):
            counts = Counter(doc_ids_generated)
            duplicates = {id_val: count for id_val, count in counts.items() if count > 1}
            logger.error(f"ERRO CRÍTICO: IDs duplicados gerados para ChromaDB! Duplicatas: {duplicates}")
            raise ValueError(f"IDs duplicados detectados: {duplicates}.")
        
        batch_size = 200 # << REDUZA ESTE VALOR SE O ERRO DE MAX_TOKENS PERSISTIR (ex: 100, 50)
        num_batches = (len(documents_for_chroma) + batch_size - 1) // batch_size
        logger.info(f"Adicionando {len(documents_for_chroma)} docs ao ChromaDB em {num_batches} lotes de (até) {batch_size}.")
        
        for i in range(num_batches):
            batch_start = i * batch_size; batch_end = (i + 1) * batch_size
            batch_docs_to_add = documents_for_chroma[batch_start:batch_end]
            batch_ids_to_add = doc_ids_generated[batch_start:batch_end]
            if not batch_docs_to_add: continue

            logger.info(f"Processando lote {i+1}/{num_batches} com {len(batch_docs_to_add)} documentos...")
            try:
                vector_store.add_documents(documents=batch_docs_to_add, ids=batch_ids_to_add)
                logger.info(f"Lote {i+1}/{num_batches} adicionado/atualizado com sucesso.")
            except Exception as e_batch:
                logger.error(f"Erro ao adicionar lote {i+1}/{num_batches} ao ChromaDB: {e_batch}", exc_info=True)
                if hasattr(e_batch, 'response') and e_batch.response is not None:
                    try: error_details = e_batch.response.json(); logger.error(f"Detalhes do erro API (lote): {error_details}")
                    except: logger.error(f"Texto do erro API (lote): {e_batch.response.text}")
                raise # Re-levanta para parar o lifespan se um lote falhar
        
        vector_store.persist() 
        logger.info(f"Todos os lotes processados. Docs totais na coleção: {vector_store._collection.count()}. Persistência chamada.")
        
    except ValueError as ve:
        logger.error(f"ValueError durante a adição de docs ao ChromaDB: {ve}", exc_info=False)
        _vector_store_cache = None; raise
    except Exception as e:
        logger.error(f"Erro geral ao inicializar/popular ChromaDB: {e}", exc_info=True)
        _vector_store_cache = None; raise 

    _vector_store_cache = vector_store
    return documents_for_chroma, _vector_store_cache, _raw_docs_dict_cache

def get_vector_store() -> Optional[Chroma]:
    global _vector_store_cache
    if _vector_store_cache is None: logger.warning("Vector store solicitado mas não está inicializado.")
    return _vector_store_cache

def get_all_documents_dict() -> Dict[str, Document]:
    global _raw_docs_dict_cache
    if not _raw_docs_dict_cache: logger.warning("Dicionário de docs brutos solicitado mas está vazio.")
    return _raw_docs_dict_cache