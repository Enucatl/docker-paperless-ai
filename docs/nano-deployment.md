# Atlas / Cerebrum Paperless Nano deployment

Atlas runs the `paperless-ai` Compose project with PostgreSQL 18, Redis 8,
Paperless Nano and Phoenix. Keep the existing project name and document volumes.
OCR and metadata run in the separate `paperless-workstation` project on Cerebrum.
The core image digest is pinned in `docker-compose.yml`; the worker image and
model configuration belong to the Cerebrum deployment.

## Processing and storage

The server uses `documents.external_workers.ExternalParser` and
`documents.external_workers.ExternalMetadata`. It owns durable operations,
permissions, document revision checks and publication. The worker claims work
through the operation API, downloads inputs and uploads proposals and artifacts.
It requires no direct PostgreSQL, Redis or shared document-volume access.

Atlas selects `paperless_services.watcher.APIConsumer`,
`paperless_services.scheduler.APIScheduler` and
`paperless_worker.bootstrap.LeanWorker` through the `PAPERLESS_*_CLASS` settings.
Normal folder consumption and mail ingestion are enabled.

| Field ID | Name | Type |
| --- | --- | --- |
| 1 | processing_date | date |
| 2 | audit | longtext |
| 4 | summary | longtext |
| 5 | ai_ocr_output | longtext |

`PAPERLESS_OCR_PROVENANCE_FIELD=5` selects the OCR provenance field. Preserve
these existing definitions and values. The deployed worker supports PDF OCR;
non-PDF documents remain stored and readable but are not supported OCR inputs.

## Authentication and TLS

The server account `server-processor` has `search ingestion` capabilities.
The worker account `workstation-processor` has `parser metadata` capabilities,
an unusable password and a private API token, with no document-edit or
administrative grants. Worker credentials stay outside Git in
`paperless-workstation/secrets/`. Preserve its private
`paperless-workstation_workstation_state` volume.

`https://paperless.atlas.home.arpa/api/operations/` has a dedicated Traefik route
without interactive Authelia authentication. Paperless validates processor tokens
and capabilities. The route strips `Remote-User` and `Remote_User` headers.
The main page and other API paths retain Authelia.

The worker mounts Cerebrum's host CA bundle read-only and sets
`REQUESTS_CA_BUNDLE` to it. TLS verification stays enabled.

## Operations

On Atlas include both environment files:

```sh
cd /opt/docker/paperless-ai
docker compose --env-file ../.env --env-file .env up -d db broker webserver phoenix
```

To pause new admission for maintenance, set `PAPERLESS_CONSUMER_DISABLE=true`
and `PAPERLESS_EMAIL_TASK_CRON=disable`, then recreate the server. Unset both
settings to restore the normal defaults. Keep routine backups with `backup.sh`.

The chat service `ai` is parked under the `copilot` profile. Cleanup and
evaluations are explicit tools; see the [reference](reference.md).

## Weekly worker lifecycle

Leave Cerebrum's resident worker stopped between weekly windows. Extraction and
worker containers have `restart: "no"`; boot alone starts neither.
`vllm-weekly-window.timer` on Proxmox runs Tuesdays at 01:05 UTC, waits for model
health, runs serial temporary `workstation --once` containers for the configured
hour and waits for the current pass before cleanup. It stops only services it
started and preserves infrastructure if worker completion is uncertain.
Pending operations remain on Atlas until the next window.

The lifecycle script and tests are maintained in the `production` branch of
`Enucatl/puppet-control-repo`.
