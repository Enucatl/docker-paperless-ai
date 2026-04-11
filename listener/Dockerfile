FROM python:3.12-slim AS runtime

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --create-home --home-dir /home/app --shell /usr/sbin/nologin app

WORKDIR /app
ENV UV_PROJECT_ENVIRONMENT=/usr/local \
    HOME=/home/app \
    PYTHONDONTWRITEBYTECODE=1

COPY listener/pyproject.toml listener/uv.lock ./
COPY common /common
RUN rm -rf /common/build \
    && mkdir src \
    && uv sync --frozen --no-install-project

COPY listener/fetch_paperless_token.py .
COPY ai/uvicorn_logging.json .
COPY listener/src/ src/
RUN uv sync --frozen --no-dev

USER app
ENTRYPOINT ["python", "-m", "paperless_listener.server"]
