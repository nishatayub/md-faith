FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MDFAITH_DB=/data/mdfaith.db
WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir ".[api]" \
    && useradd --create-home --uid 10001 mdfaith \
    && mkdir /data && chown mdfaith /data
USER mdfaith
VOLUME /data
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/api/health')" || exit 1
CMD ["mdfaith", "serve", "--host", "0.0.0.0", "--port", "8000", "--db", "/data/mdfaith.db"]
