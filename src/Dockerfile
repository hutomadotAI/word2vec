# Base off of the Python image
FROM python:3.5.5-slim
LABEL maintainer "Pedro Teixeira <pedrotei@hutoma.com>"

RUN apt-get update \
 && apt install -y --no-install-recommends \
  build-essential \
 && rm -rf /var/lib/apt/lists/*

 RUN pip install --no-cache-dir --upgrade pip
# Copy common code
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the code
COPY word2vec/ /src/word2vec/

#---------------------------
# switch to non root user
# define user/group IDs as ARG
ARG USERID=1000
ARG GROUPID=1000
RUN addgroup --system --gid $GROUPID appuser
RUN adduser --system --uid $USERID --gid $GROUPID appuser

USER appuser
WORKDIR /home/appuser

ENV W2V_SERVER_PORT 9090


EXPOSE 9090

CMD ["python3", "/src/word2vec/server.py"]