#!/usr/bin/env bash
# run_tests.sh — Build, test, and tear down the ephemeral E2E test environment.
#
# Usage:
#   ./run_tests.sh              # full E2E test run
#   ./run_tests.sh --no-build   # skip the docker build step (faster re-runs)
#
# The trap at the top guarantees that containers and anonymous volumes are wiped
# on exit regardless of whether the tests pass, fail, or the user hits Ctrl+C.
# tmpfs mounts on db/broker vanish automatically when their containers stop;
# the '-v' flag takes care of any anonymous volumes from the webserver.

set -euo pipefail

export COMPOSE_PROJECT_NAME=paperless-ai-test
COMPOSE="docker compose -f docker-compose.yml -f docker-compose.test.yml"

# ---------------------------------------------------------------------------
# Guaranteed teardown — runs on ANY exit (success, failure, signal)
# ---------------------------------------------------------------------------
teardown() {
    echo ""
    echo "=== Tearing down test environment ==="
    $COMPOSE down -v --remove-orphans 2>/dev/null || true
    echo "=== Teardown complete. All ephemeral storage wiped. ==="
}
trap teardown EXIT

# ---------------------------------------------------------------------------
# Parse flags
# ---------------------------------------------------------------------------
SKIP_BUILD=false
for arg in "$@"; do
    case "$arg" in
        --no-build) SKIP_BUILD=true ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# 1. Build
# ---------------------------------------------------------------------------
if [ "$SKIP_BUILD" = false ]; then
    echo "=== Building AI test image ==="
    $COMPOSE build ai ai-test webhook-listener
fi

# ---------------------------------------------------------------------------
# 2. Start infrastructure (background)
# ---------------------------------------------------------------------------
echo ""
echo "=== Starting test infrastructure ==="
# qdrant and webhook-listener are pre-started here so they are healthy by the
# time the ai container (pytest) runs.  The ai service also declares them as
# depends_on with condition: service_healthy as a belt-and-suspenders check.
# Phoenix is optional — don't fail if the image isn't available.
$COMPOSE up -d db broker gotenberg tika webserver qdrant webhook-listener phoenix 2>/dev/null || \
    $COMPOSE up -d db broker gotenberg tika webserver qdrant webhook-listener

# ---------------------------------------------------------------------------
# 3. Wait for Paperless webserver healthcheck
#    (Django runs migrations on boot — with tmpfs this is very fast)
# ---------------------------------------------------------------------------
echo ""
echo "=== Waiting for Paperless webserver to become healthy ==="
WEBSERVER_ID=$($COMPOSE ps -q webserver)
TIMEOUT=300
ELAPSED=0
while true; do
    STATUS=$(docker inspect --format='{{.State.Health.Status}}' "$WEBSERVER_ID" 2>/dev/null || echo "unknown")
    if [ "$STATUS" = "healthy" ]; then
        echo "Webserver is healthy!"
        break
    fi
    if [ "$ELAPSED" -ge "$TIMEOUT" ]; then
        echo "ERROR: Webserver not healthy after ${TIMEOUT}s (status: $STATUS)"
        echo "--- Last webserver logs ---"
        $COMPOSE logs --tail=50 webserver
        exit 1
    fi
    echo "  status=${STATUS}, elapsed=${ELAPSED}s..."
    sleep 5
    ELAPSED=$((ELAPSED + 5))
done

# ---------------------------------------------------------------------------
# 4. Run pytest inside the AI container
# ---------------------------------------------------------------------------
echo ""
echo "=== Running pytest ==="
$COMPOSE run --rm ai-test

# ---------------------------------------------------------------------------
# 5. Success (teardown fires automatically via trap)
# ---------------------------------------------------------------------------
echo ""
echo "=== All tests passed! ==="
