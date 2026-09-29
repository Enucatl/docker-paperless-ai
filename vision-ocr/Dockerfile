FROM python:3.14-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
RUN apt-get update && apt-get install --no-install-recommends --yes git \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
ENV UV_PROJECT_ENVIRONMENT=/usr/local PYTHONDONTWRITEBYTECODE=1
COPY vision-ocr/pyproject.toml vision-ocr/uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY vision-ocr/src/ src/
RUN uv sync --frozen --no-dev \
    && useradd --uid 10001 --create-home app
USER app
HEALTHCHECK --interval=30s --timeout=6s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=5)"
CMD ["uvicorn", "vision_ocr.app:app", "--host", "0.0.0.0", "--port", "8000"]
