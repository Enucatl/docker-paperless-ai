FROM python:3.12-slim AS runtime

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
ENV UV_PROJECT_ENVIRONMENT=/usr/local

# Install dependencies first (cached layer — only busts when pyproject.toml changes).
# src/ must exist so setuptools doesn't reject the build; it's empty at this point
# so no project packages are installed yet — only the declared dependencies.
COPY pyproject.toml uv.lock ./
RUN mkdir src && uv sync --frozen --no-install-project

# Copy source tree and tests; keep production image lean.
COPY cli.py .
COPY uvicorn_logging.json .
COPY src/ src/
COPY tests/ tests/
RUN uv sync --frozen --no-dev

# Liveness check for the watch-mode poll loop.
# cli.py calls _write_heartbeat() after every batch cycle, writing the current
# Unix timestamp (as a float, e.g. "1711900123.45") to /tmp/ai-healthy.
# The check passes if:
#   1. The file exists — the loop has completed at least one cycle since startup.
#   2. The timestamp inside is less than 600 s old — the loop is still running
#      and has not stalled or crashed silently between poll cycles.
# cut -d. -f1 strips the fractional part so pure-shell integer arithmetic works.
# start-period=120s gives the service time to connect to Paperless and the LLM
# before the first heartbeat is expected.
HEALTHCHECK --interval=60s --timeout=5s --retries=3 --start-period=120s \
  CMD [ -f /tmp/ai-healthy ] && [ $(( $(date +%s) - $(cat /tmp/ai-healthy | cut -d. -f1) )) -lt 600 ]

ENTRYPOINT ["python", "cli.py"]
CMD ["--watch"]

FROM runtime AS eval

RUN uv sync --frozen --extra eval
