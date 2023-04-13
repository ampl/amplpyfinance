FROM python:3.10-alpine

ENV CELERY_BROKER_URL redis://redis:6379/0
ENV CELERY_RESULT_BACKEND redis://redis:6379/0
ENV C_FORCE_ROOT true

ENV HOST 0.0.0.0
ENV PORT 80
ENV DEBUG true

COPY . /api
WORKDIR /api

RUN pip install -U setuptools pip
RUN pip install -r requirements.txt

EXPOSE 80

CMD ["gunicorn", "--bind", "0.0.0.0:80", "--workers", "3", "app:app"]