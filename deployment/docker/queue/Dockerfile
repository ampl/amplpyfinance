FROM python:3.9-slim-bullseye

# Install curl in order to download the modules necessary
RUN apt-get update && apt-get install -y curl
# Build arguments
ARG MODULES_URL=https://ampl.com/dl/modules
# Download ampl-module.linux64.tgz
RUN cd /opt/ && curl -OL ${MODULES_URL}/ampl-module.linux64.tgz && \
    tar xzvf ampl-module.linux64.tgz && rm ampl-module.linux64.tgz
# Download gurobi-module.linux64.tgz
RUN cd /opt/ && curl -OL ${MODULES_URL}/gurobi-module.linux64.tgz && \
    tar xzvf gurobi-module.linux64.tgz && rm gurobi-module.linux64.tgz
# Download coin-module.linux64.tgz
RUN cd /opt/ && curl -OL ${MODULES_URL}/coin-module.linux64.tgz && \
    tar xzvf coin-module.linux64.tgz && rm coin-module.linux64.tgz
# Add installation directory to the environment variable PATH
ENV PATH="/opt/ampl.linux-intel64/:${PATH}"

ENV CELERY_BROKER_URL redis://redis:6379/0
ENV CELERY_RESULT_BACKEND redis://redis:6379/0
ENV C_FORCE_ROOT true

COPY . /queue
WORKDIR /queue

RUN pip install -U setuptools pip
RUN pip install -r requirements.txt
