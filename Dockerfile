# ─────────────────────────────────────────────────────────────────────────────
# Advisor-Brain-FSA — Dockerfile para Railway / Render
#
# Build:  docker build -t advisor-brain-fsa .
# Run:    docker run -e GOOGLE_API_KEY=xxx -p 8080:8080 advisor-brain-fsa
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.10-slim

LABEL maintainer="Advisor-Brain-FSA"
LABEL description="Bloomberg Terminal — Qualidade de Relatórios Financeiros B3"

# ── Variáveis de ambiente ─────────────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ── Dependências do sistema ───────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Diretório de trabalho ─────────────────────────────────────────────────────
WORKDIR /app

# ── Dependências Python ───────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# ── Código-fonte ──────────────────────────────────────────────────────────────
COPY . .

# ── Pastas de cache (Railway: Settings → Volumes → Mount Path: /app/data) ────
RUN mkdir -p /app/data/cache \
             /app/data/ai_reports \
    && chmod -R 777 /app/data

# ── Usuário não-root ──────────────────────────────────────────────────────────
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app

# Tarefa 1: USER appuser antes do CMD (permissão garantida)
USER appuser

# ── Porta ─────────────────────────────────────────────────────────────────────
EXPOSE 8080

# Tarefa 1: CMD em shell form — /bin/sh expande ${PORT:-8080} corretamente.
# NÃO usar exec form JSON array ["gunicorn",...] pois não expande variáveis.
# NÃO prefixar com "exec" — Railway não wrapa startCommand em shell e
# tentaria executar "exec" como binário ("executable not found").
# --worker-class gthread necessário para --threads funcionar (sync ignora).
CMD gunicorn --bind 0.0.0.0:${PORT:-8080} \
             --workers 2 \
             --worker-class gthread \
             --threads 4 \
             --timeout 120 \
             --log-level info \
             --access-logfile - \
             --error-logfile - \
             app_dash:server
