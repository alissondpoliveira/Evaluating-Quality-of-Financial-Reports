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
    && chown -R appuser:appuser /app \
    && chmod +x /app/entrypoint.sh

USER appuser

# ── Porta ─────────────────────────────────────────────────────────────────────
EXPOSE 8080

# entrypoint.sh builds the market ranking cache in the background (so gunicorn
# starts immediately and the Railway health check passes), then execs gunicorn.
CMD ["/app/entrypoint.sh"]
