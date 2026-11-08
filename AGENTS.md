# Paperless Nano deployment and tools

## Scope

This project deploys Nano on Atlas and retains the parked chat/search copilot,
correspondent cleanup/review, evaluations and Phoenix. Production OCR and
metadata run through the independently published Nano worker on Cerebrum.
Do not restore webhook ingress, tag polling or a separate ingestion queue.

Use Redis itself for the broker/cache; bundled Compose uses
`docker.io/library/redis:8`. Preserve historical evaluation reports with the
implementation and broker actually measured.

## Python workflow

Python projects are in `ai/` and `common/`; the main suite and lockfile are in
`ai/`. Use `uv` and run Ruff formatting for changed Python projects.
Run tests in disposable Docker containers with `--rm --userns=host`, keeping
dependencies and artifacts off the host and removing temporary volumes.
Inference should be mocked in routine tests. Report pytest duration separately
from builds/startup and use `--durations=10` to investigate regressions.

## Evaluations

```sh
docker compose run --build --rm ai-eval --split code-test
```

Experiment definitions are in `ai/src/paperless_ai/eval/experiments.yaml`.
`test`, `validation` and `all` select larger corpus splits. Keep Phoenix data
and existing evaluation results when changing experiment inputs. Evaluation
agents are separate from the production worker and do not validate its image.

## Code map

- `ai/src/paperless_ai/search/`: parked chat copilot, history and keyword search.
- `ai/src/paperless_ai/core/`: configuration and correspondent cleanup/review.
- `ai/src/paperless_ai/agents/`: OCR and metadata helpers retained for evaluations.
- `ai/src/paperless_ai/eval/`: datasets, experiments and metrics.
- `common/src/paperless_common/`: API, language, secrets and telemetry helpers.
- `ai/tests/`: retained tool tests.
