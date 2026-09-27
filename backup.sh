#!/bin/bash
set -Eeuo pipefail

# --- Configuration ---
# Absolute path to the folder containing your docker-compose.yml
COMPOSE_DIR="/opt/docker/paperless-ai"
EXPORT_DIR="../export"
LOGGER_TAG="paperless-backup"
ZIP_NAME_PREFIX="paperless"

# The service name inside your docker-compose.yml
SERVICE="webserver"
DATE=$(date -u +%Y-%m-%dT%H%M%SZ)
EXPORT_PATH=/usr/src/paperless/export

trap 'status=$?; trap - ERR; logger -s "[$LOGGER_TAG]: Backup failed at line $LINENO (exit $status)."; exit "$status"' ERR

# --- Script Start ---
logger -s "[$LOGGER_TAG]: Starting Paperless Backup via Docker Compose..."

# Navigate to the directory so docker compose finds the .yml and .env files
cd "$COMPOSE_DIR" || { logger -s "[$LOGGER_TAG]: Could not find directory $COMPOSE_DIR"; exit 1; }

# 1. Full Backup
# Note the usage of 'exec -T'. This disables pseudo-TTY allocation required for Cron.
logger -s "[$LOGGER_TAG]: Creating Full Backup..."
docker compose exec -T "$SERVICE" document_exporter "$EXPORT_DIR" \
  --no-progress-bar \
  --delete \
  --zip \
  --zip-name "${ZIP_NAME_PREFIX}-$DATE"
docker compose exec -T "$SERVICE" python3 -c 'import os,sys,zipfile; p=sys.argv[1]; assert os.path.getsize(p) > 0, f"empty archive: {p}"; z=zipfile.ZipFile(p); bad=z.testzip(); assert bad is None, f"corrupt member {bad} in {p}"' \
  "$EXPORT_PATH/${ZIP_NAME_PREFIX}-$DATE.zip"

# 2. Data-Only Backup
logger -s "[$LOGGER_TAG]: Creating Data-Only Backup..."
docker compose exec -T "$SERVICE" document_exporter "$EXPORT_DIR" \
  --no-progress-bar \
  --data-only \
  --zip \
  --zip-name "${ZIP_NAME_PREFIX}-data-only-$DATE"
docker compose exec -T "$SERVICE" python3 -c 'import os,sys,zipfile; p=sys.argv[1]; assert os.path.getsize(p) > 0, f"empty archive: {p}"; z=zipfile.ZipFile(p); bad=z.testzip(); assert bad is None, f"corrupt member {bad} in {p}"' \
  "$EXPORT_PATH/${ZIP_NAME_PREFIX}-data-only-$DATE.zip"

logger -s "[$LOGGER_TAG]: Backup Complete."
