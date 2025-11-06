# 1. Use a imagem oficial do Jupyter como base
FROM jupyter/minimal-notebook:latest

# 2. Mude para o utilizador root
USER root

# ---- ADICIONE ESTA SECÇÃO ----
# Instale as dependências de sistema para compilar o mysqlclient
# build-essential (para C compiler), pkg-config (o erro que viu) e libmariadb-dev (bibliotecas C)
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libmariadb-dev \
    && rm -rf /var/lib/apt/lists/*
# ---- FIM DA SECÇÃO ----

# Atualiza pip, setuptools e wheel para as versões mais recentes
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# 3. Copie o ficheiro de requisitos
COPY docker/requirements2.txt /tmp/requirements2.txt

# 4. Instale os requisitos
# (Esta linha agora vai funcionar porque as dependências de C estão instaladas)
RUN pip install --no-cache-dir -r /tmp/requirements2.txt
RUN pip install --force-reinstall --no-cache-dir urllib3 requests

# 5. Mude de volta para o utilizador 'jovyan'
USER jovyan

# 6. Defina o diretório de trabalho
WORKDIR /home/jovyan/work