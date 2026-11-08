# Tool reference

[Overview](../README.md) · [Setup](setup.md) · [Deployment](nano-deployment.md)

## Parked chat/search copilot

The `ai` service is retained under the `copilot` profile with restart disabled.
Ordinary Compose startup leaves it stopped. It serves `/chat` and `/ws/chat`
behind the existing `/ai` proxy route, so the browser URL is
`https://paperless.${DOCKER_DOMAIN}/ai/chat` when explicitly enabled.

```sh
docker compose --profile copilot up -d --build ai
docker compose stop ai
```

The copilot searches Paperless's full-text API, reads matching document text and
returns cited answers. It maintains no separate embedding index. Conversations
are stored in the dedicated `ai_chat` PostgreSQL database. `CHAT_OWNER_HEADER`
defaults to the authenticated proxy header `Remote-User`; keep the proxy's
identity handling intact. This copilot is independent of Nano's built-in AI
feature, which remains disabled with `PAPERLESS_AI_ALLOWED=false`.

Set `INFERENCE_CHAT_MODEL`, optional `INFERENCE_CHAT_ENDPOINT`, provider keys,
`CHAT_DATABASE_URL`, its password secret and the Paperless API token before use.
Chat-specific temperature, reasoning effort, token limit and extra generation
kwargs use the `INFERENCE_CHAT_*` variables defined by
[`AgentConfig`](../ai/src/paperless_ai/core/config.py).

## Correspondent cleanup

The `cleanup` profile retains `ai-cleanup` and `cleanup-review`. They use the
runtime image and operate against the existing Paperless API/database.
Review artifacts live in the persistent `cleanup_review_data` volume at
`/review`; review decisions and the successful-apply watermark are stored in
Paperless's PostgreSQL database.

```sh
./scripts/correspondent-cleanup generate  # generate a reviewable merge plan
./scripts/correspondent-cleanup start     # open the existing plan's review service
./scripts/correspondent-cleanup status
./scripts/correspondent-cleanup stop
```

`./scripts/correspondent-cleanup weekly` runs the weekly cleanup pass. Schedule
it after the Nano worker window; it no longer waits on a separate Redis queue.

The review UI is at `https://correspondent-cleanup.${DOCKER_DOMAIN}`. Resolve the
displayed pairs and type `APPLY` to apply the approved plan. Starting the UI does
not regenerate the plan. A successful typed apply shuts down the review service.

For direct application or a dry run:

```sh
docker compose --profile cleanup run --no-deps --rm -T ai-cleanup \
  --cleanup-correspondents-apply /review/merge-plan.json --dry-run
```

Remove `--dry-run` to apply the approved plan. The wrapper uses `--no-deps` so
maintenance commands do not recreate shared services. `--cleanup-typesafe`
uses name similarity to propose candidates and TypeSafe/Jev to judge identities;
its API key is `secrets/typesafe_api_token.txt`.

## Evaluations and Phoenix

The input-only evaluation corpus and agents remain under
`ai/src/paperless_ai/eval/` and `ai/src/paperless_ai/agents/`. These agents are
experimental code, separate from the published production worker. Running this
harness does not validate that worker image or submit production operations.

Experiments in `ai/src/paperless_ai/eval/experiments.yaml` run against the
configured complete-PDF OCR endpoint and metadata models. Evaluation uses
TypeSafe/Jev judgment and logs comparisons to Phoenix at
`https://phoenix.${DOCKER_DOMAIN}`.

```sh
docker compose run --build --rm ai-eval --split code-test
docker compose run --build --rm ai-eval --split test
docker compose run --build --rm ai-eval --split validation
docker compose run --build --rm ai-eval --split all
```

`code-test` is a quick tagged-document smoke run. `test` and `validation` select
the dataset's deterministic split; `all` selects the full corpus. Each split has
its own named Phoenix input dataset. Keep Phoenix's PostgreSQL database and
stored results; do not delete volumes to refresh a dataset.

The [historical model comparison](phoenix-paperless-eval-test-report.md) records
results for the implementation actually measured on its stated date. Current
code and deployment changes do not revalidate those results.

## Tests

Run the retained tool suite with `./run_tests.sh`. Tests run offline in a
disposable Docker container; production services and volumes are not required.

## Configuration and privacy

Compose forwards only environment variables listed for the service. Model
credentials use private Docker secrets and supported `_FILE` settings. See
[`.env.example`](../.env.example) and
[`AgentConfig`](../ai/src/paperless_ai/core/config.py) for settings and defaults.

Chat sends questions and retrieved document text to its configured model.
Evaluations send complete PDFs to their OCR endpoint, text to the metadata
model, and text/predictions to TypeSafe/Jev. TypeSafe-assisted cleanup sends
correspondent names and sample titles to that service. Use local model endpoints
where required; TypeSafe jobs still use the configured external judge.

## Code layout

- `ai/src/paperless_ai/search/`: copilot, chat history and keyword search.
- `ai/src/paperless_ai/core/`: configuration, cleanup and review server.
- `ai/src/paperless_ai/agents/`: OCR, archives and metadata for evaluations.
- `ai/src/paperless_ai/eval/`: corpus, experiments and metrics.
- `common/`: shared API client, languages, secrets and telemetry.
- `db/init/`: Phoenix and chat database bootstrap.
- `scripts/`: Phoenix startup and cleanup commands.

The Compose hardening defaults come from the sibling
`compose-security-baseline` project. Production OCR and metadata are implemented
in the independent Paperless Nano worker repository/package.
