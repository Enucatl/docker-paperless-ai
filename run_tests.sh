#!/usr/bin/env bash
# Run the retained tools' tests with dependencies and artifacts inside Docker.
set -euo pipefail
cd "$(dirname "$0")"
image=paperless-ai-test
if [[ "${1:-}" != --no-build ]]; then
    docker build --target test -f ai/Dockerfile -t "$image" .
fi
docker run --rm --userns=host --entrypoint pytest "$image" tests/ -m "not live" --durations=10
