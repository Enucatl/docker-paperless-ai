FROM python:3.12-slim AS runtime

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
ENV UV_PROJECT_ENVIRONMENT=/usr/local

COPY listener/pyproject.toml listener/uv.lock ./
COPY common /common
RUN mkdir src && uv sync --frozen --no-install-project

COPY listener/fetch_paperless_token.py .
COPY ai/uvicorn_logging.json .
COPY listener/src/ src/
RUN uv sync --frozen --no-dev

ENTRYPOINT ["python", "-m", "paperless_listener.server"]
