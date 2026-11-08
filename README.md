# Paperless Nano on Atlas

This repository deploys Paperless Nano, PostgreSQL 18, Redis 8 and Phoenix on
Atlas. The independently published Nano worker on Cerebrum performs OCR and
metadata extraction through Paperless's durable operation API. There is no
webhook listener, tag-driven processing queue or ingestion worker in this project.

Three tools remain here:

- **Chat/search copilot:** a Paperless full-text search assistant with citations
  and persistent conversation history. It is parked under the `copilot` Compose
  profile and does not run during ordinary startup.
- **Correspondent cleanup:** one-shot plan generation, bulk maintenance and a
  browser review interface under the `cleanup` profile.
- **Evaluations and Phoenix:** input-only OCR/metadata experiments and model
  comparisons. Evaluation agents are retained for experiments; they do not
  process the production document queue.

See [deployment](docs/nano-deployment.md), [setup](docs/setup.md) and
[tool reference](docs/reference.md). The [historical evaluation report](docs/phoenix-paperless-eval-test-report.md)
records the implementation and models actually measured, rather than validation
of the current worker image.
