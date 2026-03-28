# ─────────────────────────────────────────────────────────────────────────────
# Advisor-Brain-FSA — Dockerfile para Railway / Render / qualquer container
#
# Build:  docker build -t advisor-brain-fsa .
# Run:    docker run -e GOOGLE_API_KEY=xxx -e PORT=8050 -p 8050:8050 advisor-brain-fsa
#
# SRE Notes:
#   - HEALTHCHECK Docker removido: Railway usa healthcheckPath no railway.json
#     (rota Flask /health) que responde em <5ms sem depender do Dash layout.
#   - CMD usa exec via sh -c para que SIGTERM chegue ao gunicorn (não ao shell).
#   - PORT é injetado pelo Railway em runtime; sh -c expande $PORT corretamente.
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.10-slim

# ── Metadados ─────────────────────────────────────────────────────────────────
LABEL maintainer="Advisor-Brain-FSA"
LABEL description="Bloomberg Terminal — Análise de Qualidade de Relatórios Financeiros B3"

# ── Variáveis de ambiente base ────────────────────────────────────────────────
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
# Copia só requirements primeiro → layer cacheada; só re-instala se mudar
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# ── Código-fonte ──────────────────────────────────────────────────────────────
COPY . .

# ── Tarefa 2: Pastas de cache persistente com permissão de escrita ────────────
# No Railway: Settings → Volumes → Mount Path: /app/data
# Isso faz cache CVM e AI reports sobreviverem ao redeploy.
RUN mkdir -p /app/data/cache \
             /app/data/ai_reports \
    && chmod -R 777 /app/data

# ── Usuário não-root (segurança) ──────────────────────────────────────────────
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

# ── Porta exposta (documentação; Railway usa $PORT dinâmico) ──────────────────
EXPOSE 8050

# ── Tarefa 2: Sem HEALTHCHECK Docker ─────────────────────────────────────────
# Railway usa healthcheckPath: "/health" (railway.json) via sua própria sonda.
# O Docker HEALTHCHECK com PORT baked-in causava falso-negativo quando
# Railway injeta uma porta diferente de 8050 em runtime.
# A rota Flask /health em app_dash.py responde em <5ms.

# ── CMD — exec via sh -c para propagação correta de SIGTERM ao gunicorn ───────
#
# Por que sh -c e não exec form?
#   exec form ["gunicorn", "--bind", "0.0.0.0:8050"] não expande $PORT.
#   sh -c "exec gunicorn ..." expande $PORT E faz exec (PID 1 = gunicorn).
#
# Workers: 2 (Railway Hobby: 512MB RAM; 3 workers causam OOM)
# Threads: 2 por worker (gthread) → total 4 threads concorrentes
# Timeout: 120s → cobre o primeiro carregamento do ranking CVM
# max-requests: 1000 → recicla workers prevenindo memory leaks
CMD ["sh", "-c", \
     "exec gunicorn \
      --bind 0.0.0.0:${PORT} \
      --workers 2 \
      --worker-class gthread \
      --threads 2 \
      --timeout 120 \
      --keepalive 5 \
      --max-requests 1000 \
      --max-requests-jitter 50 \
      --log-level info \
      --access-logfile - \
      --error-logfile - \
      app_dash:server"]
