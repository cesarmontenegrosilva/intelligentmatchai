# IntelliMatch AI - Assistente Inteligente de Recrutamento

**Versão:** 0.1.5
**Data da Última Atualização:** 21 de Maio de 2025

## Índice

1.  Sobre o Projeto
2.  Objetivo do Datathon
3.  Funcionalidades Principais
4.  Arquitetura da Solução
5.  Tecnologias Utilizadas
6.  Estrutura de Pastas do Projeto
7.  Pré-requisitos
8.  Configuração do Ambiente
    - Variáveis de Ambiente (.env)
    - Dados de Entrada
9.  Como Executar a Aplicação (Localmente com Docker)
10. Acessando a Aplicação
11. Endpoints da API Backend
12. Deploy (Exemplo com Google Cloud Run)
13. Testes
14. Próximos Passos e Melhorias Futuras
15. Contribuições
16. Autores
17. Licença

---

## Sobre o Projeto

O IntelligentMatch AI é um protótipo de assistente de recrutamento inteligente desenvolvido como solução para o Datathon de Machine Learning Engineering. Ele utiliza Inteligência Artificial, especificamente modelos de linguagem grandes (LLMs) e técnicas de processamento de linguagem natural (NLP), para otimizar e agilizar o processo de recrutamento e seleção. A solução visa simular um "hunter" de talentos, capaz de analisar vagas e currículos, identificar os melhores "matches" entre candidatos e oportunidades, e interagir com o recrutador para fornecer insights e informações relevantes.

## Objetivo do Datathon

Este projeto visa solucionar as dores da empresa "Decision" (fictícia), especializada em bodyshop e recrutamento, que enfrenta desafios como a falta de padronização em entrevistas e a dificuldade em identificar o engajamento e o "fit" ideal dos candidatos em tempo hábil. O IntelliMatch AI propõe uma solução baseada em IA para melhorar a eficiência e a qualidade do processo de recrutamento, com foco no desenvolvimento e deploy de um modelo de Machine Learning/IA.

## Funcionalidades Principais

- **Análise de Vagas:** Extração de requisitos, habilidades e perfil desejado a partir de descrições de vagas.
- **Análise de Candidatos:** Processamento de currículos (CVs) para identificar experiências, habilidades, formação e outras qualificações.
- **Matching Inteligente:** Uso de embeddings e LLMs para avaliar a compatibilidade entre candidatos e vagas.
- **Agente Conversacional (Chatbot):** Interface de chat onde o recrutador pode:
  - Solicitar informações sobre vagas e candidatos.
  - Pedir para listar vagas ou candidatos com base em critérios.
  - Obter análises de "match" e justificativas.
  - Realizar perguntas gerais sobre os dados de recrutamento disponíveis.
- **Base de Conhecimento Vetorial:** Utilização de um vector store (ChromaDB) para armazenar e buscar eficientemente informações textualizadas de vagas, candidatos e prospecções (histórico de contratações).
- **API Backend:** Serviço FastAPI expondo os endpoints para o agente e para consulta de dados.
- **Interface Frontend:** Aplicação Streamlit para interação com o usuário.

## Arquitetura da Solução

A solução é composta por dois serviços principais orquestrados com Docker Compose:

1.  **Backend (FastAPI):**
    - Responsável pela lógica de negócio principal.
    - Contém o `Agente de IA` construído com LangChain, utilizando um LLM (ex: GPT-4o mini, GPT-3.5 Turbo) e um modelo de embeddings (ex: text-embedding-3-small).
    - Gerencia o `Vector Store` (ChromaDB) para o sistema de RAG (Retrieval Augmented Generation).
    - Processa os dados de entrada (`vagas.json`, `applicants.json`, `prospects.json`).
    - Expõe uma API RESTful para o frontend e para possíveis integrações futuras.
2.  **Frontend (Streamlit):**
    - Fornece uma interface de usuário baseada em chat para interação com o Agente de IA.
    - Comunica-se com o Backend através de chamadas HTTP para os endpoints da API.

Ambos os serviços são containerizados usando Docker para facilitar o desenvolvimento, deploy e escalabilidade.

## Tecnologias Utilizadas

- **Linguagem de Programação:** Python 3.9+
- **Backend:**
  - FastAPI: Framework web para construção de APIs.
  - Uvicorn: Servidor ASGI para FastAPI.
  - LangChain: Framework para desenvolvimento de aplicações com LLMs.
  - OpenAI API: Para acesso aos modelos de LLM e Embeddings.
  - ChromaDB (via `langchain-chroma`): Vector store para RAG.
  - Pydantic: Para validação de dados.
- **Frontend:**
  - Streamlit: Framework para construção de aplicações web de dados.
  - Requests: Para comunicação HTTP com o backend.
- **Containerização:**
  - Docker
  - Docker Compose
- **Dados:** JSON
- **Controle de Versão:** Git e GitHub (recomendado)

## Estrutura de Pastas do Projeto

datathon-recruitment-agent/
├── backend/
│ ├── app/
│ │ ├── init.py
│ │ ├── agent/
│ │ │ ├── init.py
│ │ │ ├── agent_core.py
│ │ │ └── prompts.py
│ │ ├── core/
│ │ │ ├── init.py
│ │ │ └── config.py
│ │ ├── data_processing/
│ │ │ ├── init.py
│ │ │ └── loader.py
│ │ ├── models/
│ │ │ ├── init.py
│ │ │ └── pydantic_models.py
│ │ └── main.py
│ ├── Dockerfile
│ └── requirements.txt
├── frontend/
│ ├── app_streamlit.py
│ ├── Dockerfile.frontend
│ └── requirements.txt
├── data/
│ ├── applicants.json
│ ├── vagas.json
│ └── prospects.json
├── vector_store_db/ # Criado pelo ChromaDB para persistência
├── .env # Arquivo para variáveis de ambiente
├── docker-compose.yml
└── README.md # Este arquivo

## Pré-requisitos

- Docker e Docker Compose instalados e configurados.
- Python 3.9 ou superior (para desenvolvimento local sem Docker, opcional).
- Uma chave de API válida da OpenAI.
- Os arquivos de dados (`vagas.json`, `applicants.json`, `prospects.json`) devem estar presentes na pasta `data/`. Certifique-se de que os arquivos JSON são válidos.

## Configuração do Ambiente

### Variáveis de Ambiente (.env)

Crie um arquivo chamado `.env` na raiz do projeto com o seguinte conteúdo, substituindo pelos seus valores:

``env
OPENAI_API_KEY="sk-SuaChaveRealDaAPIOpenAIAqui"

# Opcional: Sobrescrever modelos padrão definidos em config.py

# EMBEDDING_MODEL_NAME="text-embedding-3-small"

# LLM_MODEL_NAME="gpt-4o-mini"

# Opcional: Configuração de CORS para o backend

# CORS_ORIGINS_STR="http://localhost:8501,http://127.0.0.1:8501,https://seu-frontend-deployado.com"

Adicione .env ao seu arquivo .gitignore.

## Dados de Entrada

Os seguintes arquivos JSON são esperados na pasta data/:

- vagas.json: Informações sobre as vagas.
- applicants.json: Informações sobre os candidatos.
- prospects.json: Informações sobre prospecções e matches. Certifique-se de que esses arquivos JSON são válidos.

## Como Executar a Aplicação (Localmente com Docker)

1. Clone o Repositório (se aplicável).
2. Crie e Configure o Arquivo .env.
3. Construa as Imagens e Inicie os Contêineres:

Bash
docker-compose up --build
Para rodar em background: docker-compose up --build -d.

4. Aguarde a Inicialização. Monitore os logs.
5. Parar a Aplicação: Ctrl+C ou docker-compose down (ou docker-compose down -v).

## Acessando a Aplicação

- Frontend (Streamlit UI): http://localhost:8501
- Backend (API FastAPI Docs): http://localhost:8000/docs ou /redoc. O prefixo da API é /api/v1.

## Endpoints da API Backend

(Prefixados com /api/v1)

- POST /query_agent: Interagir com o agente de IA.
- GET /jobs: Lista vagas.
- POST /job_details: Detalhes de uma vaga.
- GET /applicants: Lista candidatos.
- POST /applicant_details: Detalhes de um candidato.
- GET /: Endpoint raiz. Consulte /docs para schemas.

## Deploy (Exemplo com Google Cloud Run)

- Construa e Envie as Imagens Docker para um registry (ex: Google Artifact Registry).
- Deploy no Cloud Run: Crie serviços para backend e frontend, configure variáveis de ambiente (OPENAI_API_KEY como segredo, FASTAPI_URL para o frontend) e persistência para vector_store_db se necessário. (Consulte o passo a passo detalhado fornecido anteriormente).

## Testes

- Testes Unitários: Ainda não implementados.
- Testes da API: Via Swagger (/docs) ou Postman.
- Testes da Interface: Interação com a aplicação Streamlit.

## Próximos Passos e Melhorias Futuras

- Implementar ferramentas mais sofisticadas para o agente.
- Melhorar a Interface do Usuário.
- Aprimorar o RAG.
- Adicionar um modelo clássico de Machine Learning para previsão de "match".
- Monitoramento e Logging em Produção.
- Segurança (autenticação/autorização na API).
- Testes Unitários e de Integração.
- Otimização de Custos.

## Contribuições

## Autores

Augusto César Montenegro e Silva - Grupo 13
cesarmontenegrosilva@gmail.com/https://github.com/cesarmontenegrosilva

## Licença

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
