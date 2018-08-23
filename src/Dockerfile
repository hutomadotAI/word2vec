# Base off of the Python image
FROM python:3.6.6-slim-stretch
LABEL maintainer "Pedro Teixeira <pedrotei@hutoma.com>"

RUN echo "2018-07-06" > /os_patch_date.txt

RUN apt-get update \
 && apt-get upgrade -y \
 && apt install -y --no-install-recommends \
  build-essential \
 && rm -rf /var/lib/apt/lists/*

# Install pipenv
ENV PIP_NO_CACHE_DIR=false
RUN pip install --upgrade pip pipenv

# Apply the pipenv
RUN mkdir -p /src/hu_w2v
COPY Pipfile* setup.py /src/
WORKDIR /src
RUN pipenv install --system

# Copy the code
COPY hu_w2v/ /src/hu_w2v/

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
ENV PYTHONPATH "${PYTHONPATH}:/src/hu_w2v/"

EXPOSE 9090

CMD ["python3", "/src/hu_w2v/server.py"]