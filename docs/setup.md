# Setup

[← Overview](../README.md) · [Reference](reference.md)

Run commands from the repository root unless a section says otherwise.

## Before you start

This repository's `docker-compose.yml` is tailored to an existing host. It expects:

- Docker Compose and the sibling `../compose-security-baseline/hardening.yml` file.
- An external Docker network named `traefik_proxy`, with Traefik and Authelia configured for the routes in Compose.
- External Docker volumes `paperless-ai_data` and `paperless-ai_media`, plus the host directory `/opt/paperless-consume`.
- A host CA bundle at `/etc/ssl/certs/ca-certificates.crt`.
- Files for every secret listed in `docker-compose.yml` under `secrets:`. Put real credentials in the files you use; empty files can stand in for unused model providers. Keep `secrets/` private.

For a new installation, create the network and external volumes if they do not already exist:

```bash
docker network create traefik_proxy
docker volume create paperless-ai_data
docker volume create paperless-ai_media
mkdir -p /opt/paperless-consume
```

The sibling hardening file comes from [docker-compose-security-baseline](https://github.com/Enucatl/docker-compose-security-baseline). This deployment also assumes a reverse proxy and domain that match `DOCKER_DOMAIN`.

## 1. Configure environment

```bash
cp .env.example .env
mkdir -p secrets
touch secrets/{paperless_token,webhook_secret,openai_api_key,anthropic_api_key,google_api_key,openrouter_api_key,hf_token,typesafe_api_token,phoenix_postgres_password,ai_chat_postgres_password}.txt
chmod 700 secrets
chmod 600 secrets/*.txt
```

Set at minimum (use a generated secret key and your own domain):

```env
PAPERLESS_SECRET_KEY=<output of openssl rand -hex 32>
DOCKER_DOMAIN=example.com
```

Put your OpenRouter key in `secrets/openrouter_api_key.txt`. Compose reads it as
a Docker secret. The default inference endpoint uses OpenRouter.

Create strong values in `secrets/phoenix_postgres_password.txt` and `secrets/ai_chat_postgres_password.txt` before starting PostgreSQL. The database initialization scripts use them on a fresh volume. Also set a strong `POSTGRES_PASSWORD` and matching `PAPERLESS_DBPASS` in `.env`.

## 2. Start Paperless

On a new installation, start the database, Redis, and Paperless first so you can
create an API token in its UI:

```bash
docker compose up -d db broker webserver
```

Open `https://paperless.${DOCKER_DOMAIN}` through your configured reverse
proxy and finish Paperless's initial account setup.

## 3. Configure Paperless

In the Paperless UI go to **Settings -> API Tokens** and create an API token
for the AI services. Put that value into `secrets/paperless_token.txt` for
Compose, or `PAPERLESS_TOKEN` for a local run.

The token must be allowed to:

- read and patch documents
- create and update tags
- create and update custom fields
- create and update workflows

If the token cannot manage workflows and `MANAGE_PAPERLESS_WORKFLOWS=true`
(default), the `ai` service will fail on startup instead of running with a
partially configured Paperless instance.

By default the `ai` service creates or updates the required Paperless workflows
automatically on startup:

- `paperless-ai: document-added`
- `paperless-ai: document-updated`

The service also creates or updates the required `ai:run-ocr` tag if it does
not exist. If `WEBHOOK_SECRET` is set, the generated workflows include the
matching `X-Webhook-Token` header automatically.

If you prefer to manage workflows yourself, set:

```env
MANAGE_PAPERLESS_WORKFLOWS=false
```

and create the same two workflows manually:

- `Document Added`: assignment adds `ai:run-ocr`, then webhook posts to `http://webhook-listener:8001/webhook/document`
- `Document Updated`: filtered on `ai:run-ocr`, webhook posts to `http://webhook-listener:8001/webhook/document`

If you use a webhook secret, add its value as the `X-Webhook-Token` header on
both manually managed webhooks.

Tags (`ai:run-ocr`, `ai:run-metadata`) are created automatically
on first run if they do not exist. The AI service also creates these custom
fields automatically on first successful startup:

- `ai_processed` (Date)
- `ai_summary` (Long text)
- `ai_result` (Long text)

## 4. Start the full stack

```bash
docker compose up -d
```

This starts Phoenix, the webhook listener, and the always-on `ai` service
alongside Paperless, PostgreSQL, and Redis. The AI service hosts both the
copilot HTTP API and the long-running worker loop. The browser copilot is at
`https://paperless.${DOCKER_DOMAIN}/ai/chat` with the included Traefik labels.

## 5. Use the service

After the stack is up and the workflows exist, new documents flow automatically:

1. Paperless imports a file.
2. The auto-managed `document-added` workflow adds tag `ai:run-ocr`.
3. The same workflow sends the webhook to `webhook-listener`.
4. The thin webhook listener enqueues the document in Redis.
5. The `ai` service runs OCR -> metadata and serves `/chat`.
6. The pipeline removes the stage tags when each step completes.
7. The worker writes:
   - `ai_processed`
   - `ai_summary`
   - `ai_result`

The `ai` service exposes:

- `/chat` for the browser chat UI
- `/ws/chat` for the WebSocket copilot endpoint

### One-shot run

Process all pending documents and exit (useful for ad-hoc or scheduled runs):

```bash
docker compose run --rm --entrypoint python ai cli.py --once
```

### Dry run

Preview actions without modifying any documents:

```bash
docker compose run --rm --entrypoint python ai cli.py --once --dry-run
```

### Backfill or process all existing documents

For documents that were already in Paperless before the workflows existed:

1. Make sure the auto-managed `document-updated` workflow exists, or create it manually if workflow automation is disabled.
2. In the Paperless UI, bulk-select the documents you want to process.
3. Add the tag `ai:run-ocr`.
4. Paperless emits `Document Updated`, the webhook fires, and the queue fills.
5. Leave `ai` running, or drain the queue once with:

```bash
docker compose run --rm --entrypoint python ai cli.py --once
```

If you want to do the whole library in batches, just bulk-assign `ai:run-ocr`
to increasingly large slices of your archive.

## Models

Edit `INFERENCE_OCR_MODEL`, `INFERENCE_METADATA_MODEL`, and `INFERENCE_CHAT_MODEL` in `.env`, then recreate the service:

```bash
docker compose up -d --force-recreate ai
```

```env
# Default models from .env.example (served through OpenRouter)
INFERENCE_OCR_MODEL=google/gemini-3.1-flash-lite
INFERENCE_METADATA_MODEL=google/gemini-3.1-flash-lite
INFERENCE_CHAT_MODEL=google/gemini-3.1-flash-lite

# Evaluated metadata alternative (called once per document)
INFERENCE_METADATA_MODEL=inception/mercury-2.5
```

### Local or self-hosted models

The AI service uses the shared inference client. It calls OpenRouter by default;
set each stage's endpoint to use another OpenAI-compatible server.

**Ollama**:

```env
INFERENCE_OCR_MODEL=llava-llama3
INFERENCE_METADATA_MODEL=llama3.2
INFERENCE_CHAT_MODEL=llama3.2
INFERENCE_OCR_ENDPOINT=http://workstation:11434/v1
INFERENCE_METADATA_ENDPOINT=http://workstation:11434/v1
INFERENCE_CHAT_ENDPOINT=http://workstation:11434/v1
```

**vLLM**:

```env
INFERENCE_OCR_MODEL=openai/nanonets/Nanonets-OCR2-3B
INFERENCE_METADATA_MODEL=openai/meta-llama/Llama-3.2-3B-Instruct
INFERENCE_CHAT_MODEL=openai/meta-llama/Llama-3.2-3B-Instruct
INFERENCE_OCR_ENDPOINT=http://workstation:8100/v1
INFERENCE_METADATA_ENDPOINT=http://workstation:8101/v1
INFERENCE_CHAT_ENDPOINT=http://workstation:8101/v1
```

`INFERENCE_OCR_ENDPOINT`, `INFERENCE_METADATA_ENDPOINT`, and `INFERENCE_CHAT_ENDPOINT` are independent — each stage can run on different servers or ports.

For running the model endpoints themselves, see [Enucatl/vllm](https://github.com/Enucatl/vllm).

## Existing installations

These steps apply when upgrading an existing deployment. New PostgreSQL volumes run the initialization scripts automatically.

### Migrate a PostgreSQL 17 volume to 18

PostgreSQL 18 uses the versioned data directory under `/var/lib/postgresql/18/docker`,
so the database volume now mounts at `/var/lib/postgresql`.

If you already have data in the old PostgreSQL 17 volume, migrate it with a dump
and restore before starting the upgraded stack. Export `POSTGRES_PASSWORD` from
your `.env` into the shell before running these commands:

```bash
docker compose down
docker run --rm -d \
  --name paperless-postgres-17 \
  -e POSTGRES_PASSWORD="$POSTGRES_PASSWORD" \
  -v paperless-ai_pgdata:/var/lib/postgresql/data \
  docker.io/library/postgres:17
until docker exec paperless-postgres-17 pg_isready -U postgres >/dev/null; do
  sleep 1
done
docker exec paperless-postgres-17 pg_dumpall -U postgres > paperless.sql
docker stop paperless-postgres-17
docker volume rm paperless-ai_pgdata
docker compose up -d db
until docker exec "$(docker compose ps -q db)" pg_isready -U postgres >/dev/null; do
  sleep 1
done
docker exec -i "$(docker compose ps -q db)" psql -U postgres < paperless.sql
```

If your Docker project name is not `paperless-ai`, replace `paperless-ai_pgdata`
with the actual volume name reported by `docker volume ls`.

### Phoenix PostgreSQL backend

Phoenix stores its traces and evaluation data in the separate `phoenix` database
within the existing PostgreSQL service. The Phoenix role password is kept in
`secrets/phoenix_postgres_password.txt`; its host ACL is managed by
`puppet-control-repo/data/nodes/docker.yaml`.

The `db/init/phoenix-user-db.sh` script runs automatically when PostgreSQL is
initialized from an empty volume. Because the current PostgreSQL volume already
exists, run it once manually after Puppet has provisioned the secret:

```bash
docker compose up -d db
docker compose exec -T db /docker-entrypoint-initdb.d/20-phoenix-user-db.sh
docker compose up -d phoenix
```

The script is idempotent. It creates or updates the `phoenix` role and creates
the `phoenix` database if needed. Phoenix then applies its own schema migrations
on startup. The old SQLite database remains in the `phoenixdata` volume and is
not migrated by this setup.

### Copilot conversation database

The copilot stores history in its own `ai_chat` database, owned only by the
`ai_chat` PostgreSQL role. Create `secrets/ai_chat_postgres_password.txt` with
a strong password before starting the stack. On an existing PostgreSQL volume,
run the same idempotent bootstrap used for Phoenix once:

```bash
docker compose up -d db
docker compose exec -T db /docker-entrypoint-initdb.d/30-ai-chat-user-db.sh
docker compose up -d ai
```

The AI service applies its own conversation migrations at startup. It receives
the password through a Docker secret, while `CHAT_DATABASE_URL` identifies the
dedicated database. `CHAT_OWNER_HEADER` defaults to `Remote-User`; set it when
your reverse proxy uses another authenticated-user header. Without that header,
conversations use the nullable single-user owner.
