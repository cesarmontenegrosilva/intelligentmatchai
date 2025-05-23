# backend/Dockerfile
FROM python:3.9-slim

# Define o diretório de trabalho no container
WORKDIR /app

# Define variáveis de ambiente para Python para otimizar a execução
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Copia apenas o arquivo de dependências primeiro para aproveitar o cache do Docker
COPY requirements.txt .

# Instala as dependências do Python
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copia todo o código da aplicação backend (que está na pasta 'app' localmente,
# relativa a este Dockerfile que está em 'backend/') para o WORKDIR /app no container.
# Se seu código FastAPI está em backend/app/, então copie 'app' para 'app' dentro do container.
COPY ./app /app/app

# Copia a pasta 'data' para dentro do WORKDIR /app no container.
# Os paths em settings.py (ex: "data/vagas.json") serão relativos a /app.
# É importante que a pasta 'data' seja acessível no contexto de build deste Dockerfile.
# Se 'data' está na raiz do projeto, você precisará ajustar o docker-compose.yml para
# montar o volume de dados, ou ajustar o contexto de build e os paths de COPY.
# Assumindo que 'data' será montada por volume via docker-compose, esta linha pode ser opcional aqui,
# mas pode ser útil se você quiser construir uma imagem com dados embutidos para alguns cenários.
# Para a configuração com docker-compose e volumes, a montagem no docker-compose.yml é mais flexível.
# COPY ./data /app/data # Opcional se montado por volume no docker-compose.yml

# Expõe a porta em que o Uvicorn vai rodar dentro do container
EXPOSE 8000

# Comando para iniciar a aplicação FastAPI com Uvicorn
# 'app.main:app' refere-se ao arquivo /app/app/main.py (pois WORKDIR é /app)
# e à variável 'app' (FastAPI instance) dentro de main.py.
# Use --reload para desenvolvimento. Remova ou ajuste workers para produção.
# Adicionado --log-level info para mais detalhes, pode ser trace para debug extremo.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--log-level", "info"]