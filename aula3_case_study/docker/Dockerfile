# 1. Use a imagem OFICIAL do Jupyter como base
# Esta imagem JÁ TEM o utilizador 'jovyan' e o script 'start-notebook.sh'
FROM jupyter/minimal-notebook:latest

# 2. Mude para o utilizador root para poder instalar pacotes
USER root

# 3. Copie o ficheiro de requisitos
# (A localização 'docker/requirements.txt' está correta por causa do 'context: ..' no compose)
COPY docker/requirements.txt /tmp/requirements.txt

# 4. Instale os seus pacotes
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# 5. Mude DE VOLTA para o utilizador 'jovyan'
# Isto agora funciona, porque esta imagem TEM o 'jovyan'
USER jovyan

# 6. Defina o diretório de trabalho
WORKDIR /home/jovyan/work