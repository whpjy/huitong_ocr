FROM caddy:2-alpine AS caddy

FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OCR_ORIENTATION_ENABLED=false \
    MOBILE_HTTP_HOST=0.0.0.0 \
    MOBILE_HTTP_PORT=5173 \
    OCR_API_UPSTREAM=127.0.0.1:8000

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=caddy /usr/bin/caddy /usr/bin/caddy

WORKDIR /app
COPY backend/requirements.txt /app/backend/requirements.txt
RUN python -m pip install --no-cache-dir -r /app/backend/requirements.txt

COPY backend /app/backend
COPY web_demo /app/web_demo
COPY container-entrypoint.sh /app/container-entrypoint.sh

RUN mkdir -p /app/backend/logs \
    && chmod +x /app/container-entrypoint.sh

EXPOSE 5173

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

ENTRYPOINT ["/app/container-entrypoint.sh"]
