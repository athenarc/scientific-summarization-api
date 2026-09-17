FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    WORKERS=1 \
    PROMPTS_FILE=system_prompts.yaml

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 --shell /bin/bash appuser

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY summarizer_api.py system_prompts.yaml ./

RUN mkdir -p /app/logs \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=60s --timeout=60s --start-period=40s --retries=3 \
    CMD curl -f http://127.0.0.1:8000/health || exit 1

CMD ["sh", "-c", "exec gunicorn --workers ${WORKERS} --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 --access-logfile - --error-logfile - --log-level info --timeout 300 --keep-alive 2 summarizer_api:app"]
