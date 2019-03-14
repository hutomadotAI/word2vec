# Base off of the Python image
FROM python:3.7.2-slim-stretch AS common
LABEL maintainer "Pedro Teixeira <pedrotei@hutoma.com>"

RUN echo "2019-03-14" > /os_patch_date.txt

RUN apt-get update \
 && apt-get upgrade -y \
 && apt install -y --no-install-recommends \
  build-essential \
 && rm -rf /var/lib/apt/lists/*

# Install pipenv
ENV PIP_NO_CACHE_DIR=false
RUN pip install --upgrade pip pipenv

# Apply the pipenv
RUN mkdir -p /src/word2vec
COPY Pipfile* setup.py /src/
WORKDIR /src
RUN pipenv install --system
#---------------------------
FROM common AS release
#---------------------------
# create non root user
# define user/group IDs as ARG
ARG USERID=1000
ARG GROUPID=1000
RUN addgroup --system --gid $GROUPID appuser
RUN adduser --system --uid $USERID --gid $GROUPID appuser

# Copy the code
COPY . /src/

USER appuser
WORKDIR /home/appuser
ENV W2V_SERVER_PORT 9090
EXPOSE 9090
CMD ["python3", "/src/word2vec/server.py"]
#---------------------------
FROM common AS test
RUN pipenv install --dev --system
COPY --from=release /src/ /src/
RUN pytest --junitxml=/tmp/tests.xml --timeout=30
#---------------------------
FROM release
COPY --from=test /tmp/tests.xml /tmp/tests.xml