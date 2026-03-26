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

ENTRYPOINT ["python", "cli.py"]
CMD ["--watch"]
