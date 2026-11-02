FROM python:3.14-slim AS runtime

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update \
    && apt-get install --no-install-recommends --yes git ghostscript default-jre-headless \
    && rm -rf /var/lib/apt/lists/*

# Validate Unicode conformance with the pinned upstream veraPDF release.
RUN <<'SH'
set -eu
python - <<'PY'
import hashlib
import io
import urllib.request
import zipfile

url = "https://software.verapdf.org/releases/1.30/verapdf-greenfield-1.30.3-installer.zip"
data = urllib.request.urlopen(url, timeout=60).read()
assert hashlib.sha256(data).hexdigest() == "b512bf24945503a33fc10738ecffc1f2dccaaf8593a448e94ee9ac86a9559a59"
zipfile.ZipFile(io.BytesIO(data)).extractall("/tmp/verapdf-installer")
PY
cat > /tmp/verapdf-install.xml <<'XML'
<AutomatedInstallation langpack="eng">
  <com.izforge.izpack.panels.htmlhello.HTMLHelloPanel id="welcome"/>
  <com.izforge.izpack.panels.target.TargetPanel id="install_dir"><installpath>/opt/verapdf</installpath></com.izforge.izpack.panels.target.TargetPanel>
  <com.izforge.izpack.panels.packs.PacksPanel id="sdk_pack_select">
    <pack index="0" name="veraPDF GUI" selected="false"/>
    <pack index="1" name="veraPDF CLI" selected="true"/>
    <pack index="2" name="veraPDF Documentation" selected="false"/>
    <pack index="3" name="veraPDF Sample Plugins" selected="false"/>
  </com.izforge.izpack.panels.packs.PacksPanel>
  <com.izforge.izpack.panels.install.InstallPanel id="install"/>
  <com.izforge.izpack.panels.finish.FinishPanel id="finish"/>
</AutomatedInstallation>
XML
java -jar /tmp/verapdf-installer/verapdf-greenfield-1.30.3/verapdf-izpack-installer-1.30.3.jar /tmp/verapdf-install.xml
ln -s /opt/verapdf/verapdf /usr/local/bin/verapdf
rm -rf /tmp/verapdf-installer /tmp/verapdf-install.xml
SH

RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --create-home --home-dir /home/app --shell /usr/sbin/nologin app

WORKDIR /app
ENV UV_PROJECT_ENVIRONMENT=/usr/local \
    HOME=/home/app \
    PYTHONDONTWRITEBYTECODE=1

# Install dependencies first (cached layer — only busts when pyproject.toml changes).
# src/ must exist so setuptools doesn't reject the build; it's empty at this point
# so no project packages are installed yet — only the declared dependencies.
COPY ai/pyproject.toml ai/uv.lock ./
COPY common /common
COPY listener /listener
RUN rm -rf /common/build \
    && mkdir src \
    && uv sync --frozen --no-dev --no-install-project

# Copy source tree and tests; keep production image lean.
COPY ai/cli.py .
COPY ai/fetch_paperless_token.py .
COPY ai/uvicorn_logging.json .
COPY ai/src/ src/
RUN uv sync --frozen --no-dev \
    && uv pip install --system --reinstall --no-deps /common
RUN mkdir -p /home/app/.cache/huggingface \
    && chown app:app /home/app/.cache /home/app/.cache/huggingface

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

USER app
ENTRYPOINT ["python", "cli.py"]
CMD ["--watch"]

FROM runtime AS eval

USER root
RUN uv sync --frozen --extra eval
USER app

FROM runtime AS test

USER root
COPY ai/tests/ tests/
RUN uv sync --frozen --extra test --extra eval
USER app
