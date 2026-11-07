# Atlas / Cerebrum Paperless Nano deployment

Migration date: 2026-11-07. The existing `paperless-ai` project, PostgreSQL 18,
Redis 8 and document volumes are retained. OCR and metadata run in the separate
`paperless-workstation` project on Cerebrum.

## Images and placement

- Atlas published base: `ghcr.io/enucatl/paperless-nano@sha256:3a0979b6e45110841590599c64c7134ae4becffa28dd9252eb44bbd1211e0860`.
- Atlas running image: `sha256:73ed9c0b20eb9d07882a87710a1cfca2896ff5762efc2bc011037ce077d1449a`
  (`paperless-nano:20261107-thumbnail-fix`), a local layer over that published base.
- Cerebrum worker: `ghcr.io/enucatl/paperless-nano-worker@sha256:0f54a54c5e158476f47c9b828c5fd66b0535fafa95659fe90627cca5e920beef`.
- Both published as `v3.4.2-nano-97bc71617`.
- Server account `server-processor`: `search ingestion` only.
- Worker account `workstation-processor`: `parser metadata` only; unusable password,
  private API token, no document-edit or administrative grants.
- Cerebrum models remain at `http://cerebrum.home.arpa:8108`; metadata uses
  `inception/mercury-2.5` through OpenRouter.

The server uses the generic external parser and metadata descriptors. Worker
credentials stay in `paperless-workstation/secrets/`, outside Git. Preserve its
`paperless-workstation_workstation_state` volume across restarts.

Atlas selects `paperless_services.watcher.APIConsumer`,
`paperless_services.scheduler.APIScheduler` and
`paperless_worker.bootstrap.LeanWorker` through the `PAPERLESS_*_CLASS`
settings. The former `PAPERLESS_USE_API_*` and `PAPERLESS_USE_LEAN_WORKER`
flags are retired and prevent those services from starting.

The published core rejects its own local-thumbnail provenance during strict
external-result validation. The local layer fixes `documents/parser_operations.py`
by attaching internal provenance after validation and preserving the raw worker
proposal across retries. External validation stays strict. All 61 focused tests
passed in disposable containers (19.28 seconds). Source, patch, build recipe,
test logs and `core-image.tar.gz` are retained in both backups' `hotfix/`
directories with verified checksums. This layer has not been published to GHCR;
load that archive before recreating the pinned server on a fresh host.

## Authentication and TLS

`https://paperless.atlas.home.arpa/api/operations/` has a dedicated Traefik route
to Paperless without the interactive Authelia middleware. Paperless validates
API tokens and capabilities itself. The route strips `Remote-User` and
`Remote_User` request headers. The main page and other API paths retain Authelia.

The worker mounts Cerebrum's host CA bundle read-only and sets
`REQUESTS_CA_BUNDLE` to that mounted bundle. TLS verification remains enabled;
the image's public CA bundle alone does not include the internal certificate CA.

## Preserved metadata

Existing field IDs and values were retained while names changed:

| ID | Old name | Current name | Type |
| --- | --- | --- | --- |
| 1 | ai_processed | processing_date | date |
| 2 | ai_result | audit | longtext |
| 4 | ai_summary | summary | longtext |
| 5 | ai_ocr_output | ai_ocr_output | longtext |

Field 5 stores OCR provenance. Legacy webhook workflows 1 and 2 are disabled.
The old `ai` and `webhook-listener` services are stopped, have restart disabled,
and require the `legacy` Compose profile. Do not run them alongside the durable
operation pipeline. Their configuration and Redis queue state are retained.

## Backup and rollback

The stopped-writer rollback set is private at
`/opt/docker/paperless-ai/tmp/nano-migration-20261107` on Atlas and
`/opt/docker/vllm/paperless-workstation/backups/nano-migration-20261107` on
Cerebrum. It contains PostgreSQL, globals, full data/media/export/consume archive,
Redis snapshot, configuration/secrets and old image identities. Checksums passed
on the Cerebrum copy.

An isolated restore verified both the old image and candidate migrations with
matching snapshots of 2,355 active documents, custom fields/values, tags, tasks
and 4,396 source/archive SHA-256 hashes. Four trashed documents were excluded by
the default document manager; their files are in the complete archive. Temporary
verification containers, volumes and network were removed. Private evidence is
under `restore-verification/` on Atlas.

Old core image:
`ghcr.io/enucatl/paperless-ngx@sha256:84bdb5d23574800ed28dbea02dc2a8b361a9472643399c9c947767163a03275b`.
For rollback, stop all writers and restore the matching old database/files/config
and image together. Never start the old image against the migrated database.
Account for every new document/edit since this backup before restoring it.

## Operations

On Atlas use the parent environment as well as this project's environment:

```sh
cd /opt/docker/paperless-ai
docker compose --env-file ../.env --env-file .env up -d webserver
```

On Cerebrum, leave the resident worker stopped. The Proxmox weekly timer starts
extraction and temporary worker passes as described below.

Normal consumption and mail ingestion are enabled. Mail rule 1 is restored to
enabled, the watcher is running through the consumption API, and the scheduler
contains the mail-fetch schedule. Legacy workflows 1 and 2 remain disabled.
To pause admission for maintenance, set `PAPERLESS_CONSUMER_DISABLE=true` and
`PAPERLESS_EMAIL_TASK_CRON=disable` in `.env` and recreate the server; unset both
settings to restore the normal defaults after verification.

The example worker accepts PDFs; existing image documents remain stored and
readable. Non-PDF OCR is not supported by this worker.

The legacy installation had 51 unfinished task-history entries despite an empty
Celery queue. The 45 consumption receipts have filename-based completion evidence,
but ambiguous filenames do not establish identical input; six mail-fetch outcomes
are unknown. Preserve those history records rather than replaying them blindly.
Private evidence is `historical-review.json.log` in the Atlas backup directory.

The four documents tagged `ai:run-ocr` (2607, 2608, 2627, 2628) are trashed; retain
that state. Active document 2623 completed fresh OCR, metadata and search after
cutover, preserving its original source hash and unrelated tags/fields. Its sole
failed Redis queue entry was removed after that terminal success. All five source
PDFs match stored checksums. Private retry evidence is under `legacy-retry/`.

Synthetic document 2639 passed fresh upload and existing-document reprocessing.
Both root and metadata tasks reached terminal success; source/provenance hashes,
managed metadata, unrelated marker tag, ordinary search and searchable archive
text passed. Both archives passed actual veraPDF PDF/A-2u validation. Private
evidence is under `smoke/` on both hosts.

After core/worker restart and 46 seconds idle, fresh document 2640 also passed
OCR, metadata, search, source/provenance hashes, archive text and actual veraPDF
PDF/A-2u validation. Paddle represented the opaque random reference as Markdown
math; the check normalized non-alphanumeric formatting and searched for the stable
word `Verification`. Raw OCR and evidence are under `smoke-after-restart/`.

## Weekly-only worker lifecycle

The normal worker is intentionally offline between weekly windows. Cerebrum's
extraction and worker containers have `restart: "no"`; boot alone starts neither.
`vllm-weekly-window.timer` on Proxmox runs Tuesdays at 01:05 UTC. Its workflow
waits for model health, runs serial temporary `workstation --once` containers
for the configured hour and waits for the current pass before cleanup. It stops
only services it started and preserves infrastructure if worker completion is
uncertain. Pending operations remain on Atlas until the next window.

The tested script was installed on Proxmox and in the currently served Puppet
production module. Ten focused tests and a real zero-window startup/worker/cleanup
cycle passed. Its source is tracked in the `production` branch of
`Enucatl/puppet-control-repo`, alongside its focused tests.
