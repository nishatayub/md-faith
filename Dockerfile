# ---- build stage: compile any native dependencies (e.g. MDAnalysis on aarch64, where no wheel is published)
FROM python:3.11-slim AS build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python -m venv /opt/venv && /opt/venv/bin/pip install --no-cache-dir ".[api]"

# ---- runtime stage: no compiler, non-root user
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 mdfaith && mkdir /data && chown mdfaith /data
COPY --from=build /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MDFAITH_DB=/data/mdfaith.db
USER mdfaith
WORKDIR /home/mdfaith
VOLUME /data
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/api/health')" || exit 1
CMD ["mdfaith", "serve", "--host", "0.0.0.0", "--port", "8000", "--db", "/data/mdfaith.db"]
