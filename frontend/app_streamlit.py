# frontend/app_streamlit.py
import streamlit as st
import requests
import os
from typing import List, Dict, Optional, Any

st.set_page_config(page_title="IntelligentMatch AI", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .stApp {}
    .stButton>button { width: 100%; border-radius: 0.5rem; }
    .thinking-placeholder { font-style: italic; color: #555; }
    div[data-testid="column"] > div > div > div > p { text-align: center; }
</style>
""", unsafe_allow_html=True)

FASTAPI_BASE_URL = os.getenv("FASTAPI_URL", "http://localhost:8000") 
API_PREFIX = "/api/v1" 
AGENT_QUERY_ENDPOINT = f"{FASTAPI_BASE_URL}{API_PREFIX}/query_agent"

def query_backend_agent(query: str, chat_history: Optional[List[Dict[str, str]]] = None) -> Optional[Dict[str, Any]]:
    payload: Dict[str, Any] = {"query": query}
    if chat_history: payload["chat_history"] = chat_history
    
    try:
        # st.write(f"DEBUG: Enviando para {AGENT_QUERY_ENDPOINT} com payload: {payload}")
        response = requests.post(AGENT_QUERY_ENDPOINT, json=payload, timeout=180) 
        response.raise_for_status() 
        return response.json()
    except requests.exceptions.ConnectionError as e:
        st.error(f"⚠️ Erro de conexão com o backend: {e}.")
    except requests.exceptions.Timeout:
        st.error("⚠️ Timeout ao conectar com o backend.")
    except requests.exceptions.HTTPError as e:
        error_message = f"⚠️ Erro HTTP {e.response.status_code} do backend."
        try:
            error_details = e.response.json()
            if "detail" in error_details: error_message += f" Detalhe: {error_details['detail']}"
            else: error_message += f" Resposta: {e.response.text[:500]}"
        except ValueError: error_message += f" Resposta: {e.response.text[:500]}"
        st.error(error_message)
    except Exception as e:
        st.error(f"⚠️ Erro inesperado: {e}")
    return None

col_icon, col_title = st.columns([1, 5], vertical_alignment="center")
with col_icon: st.markdown("<p style='font-size: 70px; text-align: center; margin: 0;'>🤖</p>", unsafe_allow_html=True)
with col_title: st.title("Intelligent HR Match AI"); st.subheader("Assistente Inteligente de Recrutamento")
st.markdown("---")

if "messages_ui" not in st.session_state: st.session_state.messages_ui = [{"role": "assistant", "content": "Olá! Como posso ajudar?"}]
if "agent_chat_history" not in st.session_state: st.session_state.agent_chat_history = []

for message in st.session_state.messages_ui:
    with st.chat_message(message["role"]): st.markdown(message["content"])

if user_prompt := st.chat_input("Faça uma pergunta ou dê um comando..."):
    st.session_state.messages_ui.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"): st.markdown(user_prompt)

    with st.chat_message("assistant"):
        thinking_message = st.empty()
        thinking_message.markdown("<p class='thinking-placeholder'>IntelliMatch AI está processando...</p>", unsafe_allow_html=True)
        history_for_agent_payload = [{"type": msg["role"], "content": msg["content"]} for msg in st.session_state.agent_chat_history[-6:]]
        backend_response_data = query_backend_agent(user_prompt, chat_history=history_for_agent_payload)
        assistant_response_text = "Desculpe, não consegui processar."; sources_markdown = ""
        if backend_response_data and "answer" in backend_response_data:
            assistant_response_text = backend_response_data["answer"]
            sources = backend_response_data.get("sources")
            if sources and isinstance(sources, list):
                sources_markdown += "\n\n---\n**Fontes Consultadas:**\n"
                for i, source_doc in enumerate(sources[:3]):
                    if isinstance(source_doc, dict):
                        metadata = source_doc.get("metadata", {}); source_type = metadata.get("type", "Doc")
                        title = f"Fonte {i+1} ({source_type})"
                        if source_type == "vaga": title = f"Vaga ID {metadata.get('codigo_vaga', 'N/A')}: {metadata.get('titulo_vaga', '')}"
                        elif source_type == "candidato": title = f"Candidato ID {metadata.get('codigo_profissional', 'N/A')}: {metadata.get('nome', '')}"
                        elif source_type == "prospect": title = f"Prospect Vaga {metadata.get('vaga_titulo_prospect', '')} (Cand: {metadata.get('nome_candidato', '')})"
                        elif metadata.get("source_tool"): title = f"Observação da Ferramenta: {metadata.get('source_tool')}"
                        sources_markdown += f"\n{i+1}. **{title.strip()}**\n"
        thinking_message.empty(); st.markdown(assistant_response_text + sources_markdown)

    st.session_state.agent_chat_history.append({"role": "user", "content": user_prompt})
    st.session_state.agent_chat_history.append({"role": "assistant", "content": assistant_response_text})
    st.session_state.messages_ui.append({"role": "assistant", "content": assistant_response_text + sources_markdown})
    if len(st.session_state.agent_chat_history) > 10: st.session_state.agent_chat_history = st.session_state.agent_chat_history[-10:]

with st.sidebar:
    st.header("🤖 Sobre o IntelligentMatch AI")
    st.markdown("Assistente de IA para recrutamento."); st.markdown("---")
    st.subheader("Como Usar"); st.markdown("- Pergunte sobre vagas.\n- Analise candidatos.\n- Faça correspondências.")
    st.markdown("---")
    if st.button("🧹 Limpar Chat", key="clear_chat_button"):
        st.session_state.messages_ui = [{"role": "assistant", "content": "Histórico limpo!"}]; st.session_state.agent_chat_history = []; st.rerun()
    st.markdown("---"); st.caption("Datathon - ML Engineering")