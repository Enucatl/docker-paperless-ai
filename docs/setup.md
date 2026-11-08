# Setup

This Compose project is tailored to Atlas. It expects Docker Compose, the sibling
`../compose-security-baseline/hardening.yml`, Traefik and Authelia, and the existing
external volumes `paperless-ai_data` and `paperless-ai_media`. The host consumption
directory is `/opt/paperless-consume`; the host CA bundle is mounted read-only.

Use the existing `.env` and private `secrets/` files. [`.env.example`](../.env.example)
lists deployment and tool settings. Keep PostgreSQL's `POSTGRES_PASSWORD` and
`PAPERLESS_DBPASS` consistent. Keep the Compose project name `paperless-ai` to
retain its database and other named volumes.

Start the normal services with the parent environment included:

```sh
docker compose --env-file ../.env --env-file .env up -d db broker webserver phoenix
```

Configure processor accounts, external parser/metadata descriptors, storage and
worker credentials as described in the [deployment](nano-deployment.md). Model
serving and the published Nano worker are deployed on Cerebrum, not built here.

## Tool databases

Phoenix stores traces and evaluation results in the `phoenix` database within
the existing PostgreSQL service. The parked copilot stores conversations in
`ai_chat` under its dedicated database role. Their passwords are private Docker
secrets. On an empty PostgreSQL volume, `db/init/` creates these roles and databases.
For an existing volume, the bootstrap scripts are idempotent:

```sh
docker compose exec -T db /docker-entrypoint-initdb.d/20-phoenix-user-db.sh
docker compose exec -T db /docker-entrypoint-initdb.d/30-ai-chat-user-db.sh
```

Phoenix applies its own schema updates at startup. The copilot applies its
conversation schema updates when explicitly started. Keep both databases and
`phoenixdata` even while chat is parked.

## Optional tools

Cleanup uses a Paperless API token with document/correspondent maintenance
permissions and the configured cleanup database access. Chat uses a Paperless
read token and its own conversation database. Model credentials and endpoints
are configured per service; unused provider secrets may remain empty files.
See [the reference](reference.md) for commands and evaluation settings.
