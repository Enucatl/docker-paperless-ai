"""Packaged browser routes and cleanup review identifier regressions."""

import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from urllib.parse import urljoin

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from paperless_ai.core import cleanup_review
from paperless_ai.core.correspondent_cleanup import (
    CorrespondentMergePlan,
    CorrespondentPairDecision,
)
from paperless_ai.search import webhook


@pytest.mark.parametrize(
    ("app", "prefix", "page", "name"),
    [(webhook.app, "/ai", "/chat", "chat"), (cleanup_review.app, "", "/", "cleanup")],
)
def test_packaged_browser_routes(app, prefix, page, name):
    """Relative assets stay under the application's authenticated proxy prefix."""
    outer = FastAPI()
    outer.mount(prefix or "/", app)
    client = TestClient(outer)
    response = client.get(prefix + page)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    for suffix, media_type in [("css", "text/css"), ("js", "javascript")]:
        asset = f"assets/{name}.{suffix}"
        assert asset in response.text
        result = client.get(urljoin(str(response.url), asset))
        assert result.status_code == 200
        assert media_type in result.headers["content-type"]
    assert client.get(prefix + "/assets/missing.js").status_code == 404
    assert client.get(prefix + "/assets/%2e%2e/config.py").status_code == 404
    if page == "/chat":
        assert client.get(prefix + page + "/").url.path == prefix + page


def test_cleanup_canonical_key_and_browser_rendering(monkeypatch, tmp_path):
    """Use one canonical key for loading, submitting, and showing review decisions."""
    candidate = CorrespondentPairDecision(
        left_id=2,
        left_name='<img src=x onerror="alert(1)">',
        right_id=3,
        right_name="<b>Other & Co</b>",
        decision="review",
        confidence="manual",
        source="test",
        reason="<script>alert(2)</script>",
        left_members=[10, 2],
        right_members=[20, 3],
    )
    plan = CorrespondentMergePlan(
        version=1,
        generated_at="now",
        paperless_url="http://paperless",
        total_correspondents=4,
        total_documents=0,
        judge_enabled=False,
        judged_pair_count=0,
        approved_clusters=[],
        orphan_correspondents=[],
        candidate_pairs=[candidate],
    )
    store = SimpleNamespace(
        decisions_for_plan=AsyncMock(return_value=[]),
        get_last_manually_reviewed_correspondent_id=AsyncMock(return_value=None),
        record_decision=AsyncMock(),
    )
    monkeypatch.setattr(cleanup_review, "_load_plan", lambda: plan)
    monkeypatch.setattr(cleanup_review.AgentConfig, "from_env", lambda: None)
    monkeypatch.setattr(
        cleanup_review.CleanupReviewStore, "from_config", AsyncMock(return_value=store)
    )
    client = TestClient(cleanup_review.app)
    data = client.get("/api/plan").json()
    key = data["review_candidates"][0]["pair_key"]
    assert key == "2:10|3:20"
    assert (
        client.post(
            "/api/decisions", json={"pair_key": key, "decision": "approve"}
        ).status_code
        == 200
    )
    assert store.record_decision.call_args.args[1:3] == (key, "approve")
    assert (
        client.post(
            "/api/decisions", json={"pair_key": "10:2|20:3", "decision": "approve"}
        ).status_code
        == 404
    )
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required for the browser script regression")
    payload = tmp_path / "plan.json"
    payload.write_text(json.dumps(data))
    subprocess.run(
        [
            node,
            str(Path(__file__).with_name("cleanup_browser_check.cjs")),
            str(Path(cleanup_review.__file__).with_name("assets") / "cleanup.js"),
            str(payload),
        ],
        check=True,
        timeout=10,
    )
