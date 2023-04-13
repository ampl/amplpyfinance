FROM python:3.10-slim-bullseye
# Upgrade pip and setuptools
RUN python -m pip install -U setuptools pip

# Install amplpy and all necessary amplpy.modules:
RUN python -m pip install amplpy --no-cache-dir # Install amplpy
RUN python -m amplpy.modules install highs coin gurobi --no-cache-dir # Install modules

ENV CELERY_BROKER_URL redis://redis:6379/0
ENV CELERY_RESULT_BACKEND redis://redis:6379/0
ENV C_FORCE_ROOT true

COPY . /queue
WORKDIR /queue
RUN python -m pip install -r requirements.txt
