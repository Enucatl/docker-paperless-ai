FROM python:3.12-slim

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install dependencies first for better caching
COPY pyproject.toml ./
RUN uv pip install --system .

# Copy modular source tree
COPY cli.py .
COPY core/ core/
COPY agents/ agents/
COPY eval/ eval/
COPY tests/ tests/
COPY prompt.txt metadata_prompt.txt ./

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
