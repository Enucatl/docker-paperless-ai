#!/bin/bash

# --- Configuration ---
# Absolute path to the folder containing your docker-compose.yml
COMPOSE_DIR="/opt/docker/paperless-ai"
EXPORT_DIR="../export"
LOGGER_TAG="paperless-backup"
ZIP_NAME_PREFIX="paperless"

# The service name inside your docker-compose.yml
SERVICE="webserver"
RETENTION_DAYS=15
DATE=$(date --iso-8601=seconds)

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

# 2. Data-Only Backup
logger -s "[$LOGGER_TAG]: Creating Data-Only Backup..."
docker compose exec -T "$SERVICE" document_exporter "$EXPORT_DIR" \
  --no-progress-bar \
  --data-only \
  --zip \
  --zip-name "${ZIP_NAME_PREFIX}-data-only-$DATE"

# 3. Cleanup Old Backups
logger -s "[$LOGGER_TAG]: Removing backups older than $RETENTION_DAYS days..."
docker compose exec -T "$SERVICE" find "$EXPORT_DIR" -name "paperless-*.zip" -mtime +"$RETENTION_DAYS" -delete

logger -s "[$LOGGER_TAG]: Backup & Cleanup Complete."
