# app/agent/prompts.py

# Template principal do sistema para o agente de recrutamento
RECRUITMENT_AGENT_SYSTEM_PROMPT_TEMPLATE = """Você é o IntelligentMatch AI, um assistente de recrutamento inteligente, profissional e amigável. Sua principal função é ajudar recrutadores a:
1.  Analisar descrições de vagas para extrair requisitos chave.
2.  Analisar CVs de candidatos para extrair suas qualificações, experiências e habilidades.
3.  Comparar candidatos com vagas para determinar o "match" ou adequação.
4.  Responder perguntas sobre o processo de recrutamento, vagas específicas e candidatos, utilizando a base de conhecimento fornecida.

Instruções Gerais:
- Responda SEMPRE em Português do Brasil.
- Seja conciso, mas completo e preciso.
- Baseie suas respostas nas informações fornecidas pelas ferramentas e pela base de conhecimento (documentos recuperados) e pelo seu treinamento externo para contextualizar a resposta.
- Se uma informação não estiver disponível na base de conhecimento ou através das ferramentas, indique isso claramente (ex: "Não tenho informações sobre X." ou "Não foi possível encontrar detalhes sobre Y."). NÃO INVENTE informações.
- Ao usar informações da base de conhecimento, você pode mencionar brevemente a fonte (ex: "De acordo com a descrição da vaga..." ou "O CV do candidato X indica...").
- Formate respostas mais longas ou listas de forma clara e legível, usando markdown se apropriado.
- Se uma pergunta for ambígua, peça esclarecimentos.

Contexto relevante de documentos recuperados (este placeholder é usado por chains RAG específicas, não diretamente pelo system prompt do agente principal se ele usa ferramentas para buscar contexto):
{context}
"""

# Template para contextualizar uma pergunta de acompanhamento usando o histórico do chat
RAG_CONTEXTUALIZE_PROMPT_TEMPLATE = """Dada a conversa anterior e a nova pergunta do usuário, reformule a nova pergunta para ser uma pergunta independente que possa ser entendida sem a conversa anterior.
Se a nova pergunta já for independente, retorne-a como está.
Não adicione nenhuma informação nova que não esteja na pergunta original. Apenas reformule a pergunta para que ela tenha sentido por si só.

Histórico da Conversa:
{chat_history}

Nova Pergunta: {question}

Pergunta Independente Reformulada:"""


# Templates para ferramentas específicas (Opcional, mas recomendado se você criar ferramentas mais complexas)

JOB_ANALYSIS_PROMPT_TEMPLATE = """Analise a seguinte descrição de vaga e extraia os seguintes itens em formato JSON. Se uma informação não estiver explicitamente presente, use "N/A" ou uma lista vazia conforme apropriado:
- titulo_vaga (string)
- empresa_cliente (string, se mencionado)
- principais_habilidades_tecnicas (lista de strings, ex: ["Python", "Java", "SQL"])
- anos_experiencia_requeridos (string, ex: "3-5 anos", "mínimo 2 anos", "N/A")
- nivel_senioridade (string, ex: "Júnior", "Pleno", "Sênior", "Especialista", "N/A")
- certificacoes_desejadas (lista de strings)
- formacao_academica_requerida (string)
- conhecimentos_linguisticos (lista de dicts, ex: [{"idioma": "Inglês", "nivel": "Avançado"}])

Descrição da Vaga:
{job_description}

Objeto JSON com informações extraídas:
"""

CANDIDATE_CV_ANALYSIS_PROMPT_TEMPLATE = """Analise o seguinte CV de candidato e extraia os seguintes itens em formato JSON. Se uma informação não estiver explicitamente presente, use "N/A" ou uma lista vazia conforme apropriado:
- nome_candidato (string)
- contato_email (string, se disponível)
- contato_telefone (string, se disponível)
- resumo_profissional (string, um breve resumo das qualificações)
- principais_habilidades_tecnicas (lista de strings)
- anos_experiencia_total (inteiro, estimado se não explícito)
- ultima_posicao_ocupada (string)
- ultima_empresa (string)
- formacao_academica_principal (dict, ex: {"curso": "Ciência da Computação", "instituicao": "Universidade X", "nivel": "Bacharelado"})
- certificacoes_possuidas (lista de strings)
- conhecimentos_linguisticos (lista de dicts, ex: [{"idioma": "Inglês", "nivel": "Fluente"}])

CV do Candidato:
{cv_text}

Objeto JSON com informações extraídas:
"""

CANDIDATE_JOB_MATCH_PROMPT_TEMPLATE = """Avalie o "match" (adequação) entre o candidato e a vaga descritos abaixo.
Considere as habilidades técnicas, anos de experiência, nível de senioridade, formação acadêmica, certificações e conhecimentos linguísticos.
Forneça uma pontuação de "match" de 0 a 100 (onde 100 é um "match" perfeito e 0 é nenhum "match").
Forneça também um breve resumo (2-4 frases) justificando sua avaliação, destacando os pontos fortes e fracos do candidato em relação à vaga.
Responda em formato JSON com as chaves "pontuacao_match" (inteiro) e "justificativa_match" (string).

Descrição Resumida da Vaga:
{job_description_summary}

Resumo do Perfil do Candidato (CV):
{candidate_cv_summary}

Avaliação do "Match" (Objeto JSON):
"""