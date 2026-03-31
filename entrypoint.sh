#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# entrypoint.sh — Railway / Docker startup script
#
# Strategy:
#   1. Start the market cache build in the BACKGROUND so gunicorn can start
#      immediately and Railway's health check passes.
#   2. Launch gunicorn in the foreground (PID 1 signal handler).
#
# The home dashboard reads from data/cache/market_ranking_current.json.
# While the cache is being built (or if it fails), the app falls back to
# scoring the DEFAULT_WATCHLIST live — the UI is fully functional throughout.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

CACHE_FILE="/app/data/cache/market_ranking_current.json"
LOCK_FILE="/app/data/cache/.build_market_cache.lock"
LOG_FILE="/app/data/cache/build_market_cache.log"

# ── Background cache build ────────────────────────────────────────────────────
(
  # Prevent concurrent builds (e.g. rolling restarts spawning two instances)
  if [ -f "$LOCK_FILE" ]; then
    echo "[entrypoint] Cache build already running (lock exists). Skipping." >&2
    exit 0
  fi

  touch "$LOCK_FILE"
  trap 'rm -f "$LOCK_FILE"' EXIT

  echo "[entrypoint] Starting market cache build (background)..." >&2
  python /app/build_market_cache.py \
    --workers 2 \
    --log-level WARNING \
    >> "$LOG_FILE" 2>&1 \
    && echo "[entrypoint] Cache build completed → $CACHE_FILE" >&2 \
    || echo "[entrypoint] Cache build FAILED — see $LOG_FILE" >&2
) &

# Small delay to let the build start before gunicorn loads modules
sleep 1

# ── Gunicorn (foreground) ─────────────────────────────────────────────────────
echo "[entrypoint] Starting gunicorn on port ${PORT:-8080}..." >&2
exec gunicorn \
  --bind "0.0.0.0:${PORT:-8080}" \
  --workers 2 \
  --worker-class gthread \
  --threads 4 \
  --timeout 120 \
  --log-level info \
  --access-logfile - \
  --error-logfile - \
  app_dash:server
