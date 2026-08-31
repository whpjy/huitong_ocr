#!/bin/sh
set -eu

cd /app/backend
python -m uvicorn api.main:app \
  --host 127.0.0.1 \
  --port 8000 \
  --workers "${BACKEND_WORKERS:-1}" &
backend_pid=$!

shutdown() {
  kill "$backend_pid" 2>/dev/null || true
  wait "$backend_pid" 2>/dev/null || true
}
trap shutdown INT TERM EXIT

cd /app/web_demo
caddy run --config /app/web_demo/Caddyfile --adapter caddyfile &
caddy_pid=$!

while kill -0 "$backend_pid" 2>/dev/null && kill -0 "$caddy_pid" 2>/dev/null; do
  sleep 2
done

if ! kill -0 "$backend_pid" 2>/dev/null; then
  echo "Backend process exited unexpectedly" >&2
  wait "$backend_pid"
fi

echo "Caddy process exited unexpectedly" >&2
wait "$caddy_pid"
