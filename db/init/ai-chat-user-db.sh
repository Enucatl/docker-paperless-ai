#!/usr/bin/env bash
set -Eeuo pipefail

readonly ai_chat_password_file="${CHAT_DATABASE_PASSWORD_FILE:-/run/secrets/ai_chat_postgres_password}"
readonly ai_chat_user="ai_chat"
readonly ai_chat_database="ai_chat"

if [[ ! -r "$ai_chat_password_file" ]]; then
  printf 'AI chat PostgreSQL password file is not readable: %s\n' "$ai_chat_password_file" >&2
  exit 1
fi

ai_chat_password="$(<"$ai_chat_password_file")"
if [[ -z "$ai_chat_password" ]]; then
  printf 'AI chat PostgreSQL password is empty\n' >&2
  exit 1
fi

role_exists="$({
  psql \
    --username "$POSTGRES_USER" \
    --dbname "$POSTGRES_DB" \
    --tuples-only \
    --no-align \
    --command "SELECT 1 FROM pg_roles WHERE rolname = '$ai_chat_user'"
} | tr -d '[:space:]')"

if [[ "$role_exists" == 1 ]]; then
  psql \
    --username "$POSTGRES_USER" \
    --dbname "$POSTGRES_DB" \
    --set "ai_chat_password=$ai_chat_password" <<EOSQL
ALTER ROLE $ai_chat_user PASSWORD :'ai_chat_password';
EOSQL
else
  psql \
    --username "$POSTGRES_USER" \
    --dbname "$POSTGRES_DB" \
    --set "ai_chat_password=$ai_chat_password" <<EOSQL
CREATE ROLE $ai_chat_user LOGIN PASSWORD :'ai_chat_password';
EOSQL
fi

database_exists="$({
  psql \
    --username "$POSTGRES_USER" \
    --dbname postgres \
    --tuples-only \
    --no-align \
    --command "SELECT 1 FROM pg_database WHERE datname = '$ai_chat_database'"
} | tr -d '[:space:]')"

if [[ "$database_exists" != 1 ]]; then
  psql \
    --username "$POSTGRES_USER" \
    --dbname postgres \
    --command "CREATE DATABASE $ai_chat_database OWNER $ai_chat_user"
fi
