# ─────────────────────────────────────────────────────────────────────────────
# Advisor-Brain-FSA — Dockerfile para Railway / Render / qualquer container
#
# Build:  docker build -t advisor-brain-fsa .
# Run:    docker run -e GOOGLE_API_KEY=xxx -e PORT=8050 -p 8050:8050 advisor-brain-fsa
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.10-slim

# ── Metadados ─────────────────────────────────────────────────────────────────
LABEL maintainer="Advisor-Brain-FSA"
LABEL description="Bloomberg Terminal — Análise de Qualidade de Relatórios Financeiros B3"

# ── Variáveis de ambiente ─────────────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8050

# ── Dependências do sistema (mínimas para pandas / numpy) ────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Diretório de trabalho ─────────────────────────────────────────────────────
WORKDIR /app

# ── Dependências Python ───────────────────────────────────────────────────────
# Copia só o requirements primeiro para aproveitar cache de layer do Docker
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# ── Código-fonte ──────────────────────────────────────────────────────────────
COPY . .

# ── Pastas de cache persistente ───────────────────────────────────────────────
# Tarefa 2: criar data/cache (CVM DFP) e data/ai_reports (relatórios Gemini)
# e garantir permissão de escrita pelo usuário da aplicação.
#
# No Railway: monte um Volume em /app/data para que o cache sobreviva ao redeploy.
#   Settings → Volumes → Mount Path: /app/data
RUN mkdir -p /app/data/cache \
             /app/data/ai_reports \
    && chmod -R 777 /app/data

# ── Usuário não-root (segurança) ──────────────────────────────────────────────
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

# ── Porta exposta ─────────────────────────────────────────────────────────────
EXPOSE $PORT

# ── Healthcheck ───────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/')" \
    || exit 1

# ── Tarefa 3: CMD — Gunicorn lê $PORT injetado pelo Railway / Render ─────────
#
# Workers: 2 × CPU + 1  →  num padrão Railway (1 vCPU) = 3 workers
# Timeout: 120s  →  suficiente para o primeiro carregamento do ranking CVM
# keepalive: 5s  →  conexões longas para streaming do Gemini
CMD gunicorn \
    --bind "0.0.0.0:${PORT}" \
    --workers 3 \
    --timeout 120 \
    --keepalive 5 \
    --log-level info \
    --access-logfile - \
    --error-logfile - \
    app_dash:server
