#!/usr/bin/env bash
set -Eeuo pipefail

readonly phoenix_password_file="${PHOENIX_POSTGRES_PASSWORD_FILE:-/run/secrets/phoenix_postgres_password}"
readonly phoenix_user="phoenix"
readonly phoenix_database="phoenix"

if [[ ! -r "$phoenix_password_file" ]]; then
  printf 'Phoenix PostgreSQL password file is not readable: %s\n' "$phoenix_password_file" >&2
  exit 1
fi

phoenix_password="$(<"$phoenix_password_file")"
if [[ -z "$phoenix_password" ]]; then
  printf 'Phoenix PostgreSQL password is empty\n' >&2
  exit 1
fi

role_exists="$({
  psql \
    --username "$POSTGRES_USER" \
    --dbname "$POSTGRES_DB" \
    --tuples-only \
    --no-align \
    --command "SELECT 1 FROM pg_roles WHERE rolname = '$phoenix_user'"
} | tr -d '[:space:]')"

if [[ "$role_exists" == 1 ]]; then
  psql \
    --username "$POSTGRES_USER" \
    --dbname "$POSTGRES_DB" \
    --set "phoenix_password=$phoenix_password" <<EOSQL
ALTER ROLE $phoenix_user PASSWORD :'phoenix_password';
EOSQL
else
  psql \
    --username "$POSTGRES_USER" \
    --dbname "$POSTGRES_DB" \
    --set "phoenix_password=$phoenix_password" <<EOSQL
CREATE ROLE $phoenix_user LOGIN PASSWORD :'phoenix_password';
EOSQL
fi

database_exists="$({
  psql \
    --username "$POSTGRES_USER" \
    --dbname postgres \
    --tuples-only \
    --no-align \
    --command "SELECT 1 FROM pg_database WHERE datname = '$phoenix_database'"
} | tr -d '[:space:]')"

if [[ "$database_exists" != 1 ]]; then
  psql \
    --username "$POSTGRES_USER" \
    --dbname postgres \
    --command "CREATE DATABASE $phoenix_database OWNER $phoenix_user"
fi
