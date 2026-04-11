import io
import json
from pathlib import Path

import pytest

from paperless_ai.core.config import AgentConfig
from paperless_ai.core.correspondent_cleanup import (
    apply_correspondent_merge_plan,
    build_correspondent_merge_plan,
    load_merge_plan,
    write_merge_plan,
)


class _FakeClient:
    def __init__(self, correspondents, documents, counts=None):
        self._correspondents = correspondents
        self._documents = documents
        self._counts = counts or {}
        self.patched = []
        self.deleted = []

    async def get_all_correspondents(self, force: bool = False):
        return list(self._correspondents)

    async def iter_all_documents_brief(self):
        return list(self._documents)

    async def patch_document(self, doc_id: int, payload: dict):
        self.patched.append((doc_id, payload))

    async def count_documents_for_correspondent(self, correspondent_id: int) -> int:
        return self._counts.get(correspondent_id, 0)

    async def delete_correspondent(self, correspondent_id: int) -> None:
        self.deleted.append(correspondent_id)


def _config() -> AgentConfig:
    return AgentConfig(
        paperless_url="http://paperless:8000",
        paperless_token="token-123",
        metadata_model="gemini/gemini-2.5-flash",
        chat_model="gemini/gemini-2.5-flash",
    )


@pytest.mark.asyncio
async def test_build_correspondent_merge_plan_groups_normalized_aliases():
    client = _FakeClient(
        correspondents=[
            {"id": 1, "name": "Acme Corp."},
            {"id": 2, "name": "ACME Corporation"},
            {"id": 3, "name": "John Smith"},
            {"id": 4, "name": "Legacy Sender"},
        ],
        documents=[
            {"id": 101, "title": "Invoice April", "correspondent": 1},
            {"id": 102, "title": "Invoice May", "correspondent": 2},
            {"id": 103, "title": "Personal Letter", "correspondent": 3},
        ],
    )

    plan = await build_correspondent_merge_plan(
        client, _config(), judge_borderline=False
    )

    assert len(plan.approved_clusters) == 1
    cluster = plan.approved_clusters[0]
    assert cluster.canonical_id == 2
    assert cluster.merged_ids == [1]
    assert cluster.planned_document_ids == [101]
    assert cluster.status == "approved"
    assert [item.name for item in plan.orphan_correspondents] == ["Legacy Sender"]
    assert any(
        decision.reason == "normalized_exact_match" for decision in plan.candidate_pairs
    )


@pytest.mark.asyncio
async def test_build_correspondent_merge_plan_rejects_person_vs_org_collision():
    client = _FakeClient(
        correspondents=[
            {"id": 1, "name": "John Smith"},
            {"id": 2, "name": "Smith Holdings"},
        ],
        documents=[
            {"id": 101, "title": "Letter", "correspondent": 1},
            {"id": 102, "title": "Statement", "correspondent": 2},
        ],
    )

    plan = await build_correspondent_merge_plan(
        client, _config(), judge_borderline=False
    )

    assert plan.approved_clusters == []
    assert any(
        decision.reason == "entity_shape_conflict" for decision in plan.candidate_pairs
    )


def test_merge_plan_round_trips_json(tmp_path: Path):
    path = tmp_path / "correspondent-plan.json"
    data = {
        "version": 1,
        "generated_at": "2026-04-08T00:00:00+00:00",
        "paperless_url": "http://paperless:8000",
        "total_correspondents": 2,
        "total_documents": 3,
        "judge_enabled": False,
        "judged_pair_count": 0,
        "approved_clusters": [
            {
                "canonical_id": 2,
                "canonical_name": "ACME Corporation",
                "canonical_document_count": 2,
                "merged_ids": [1],
                "merged_names": ["Acme Corp."],
                "source_document_count": 1,
                "planned_document_ids": [101],
                "confidence": "high",
                "status": "approved",
                "reasons": ["normalized_exact_match"],
            }
        ],
        "orphan_correspondents": [
            {
                "id": 3,
                "name": "John Smith",
                "document_count": 0,
                "status": "approved",
                "reason": "no_documents_assigned",
            }
        ],
        "candidate_pairs": [],
    }
    path.write_text(json.dumps(data), encoding="utf-8")

    plan = load_merge_plan(str(path))
    write_merge_plan(plan, str(path))
    reloaded = json.loads(path.read_text(encoding="utf-8"))

    assert reloaded["approved_clusters"][0]["canonical_name"] == "ACME Corporation"


def test_merge_plan_uses_stdio_for_dash(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    data = {
        "version": 1,
        "generated_at": "2026-04-08T00:00:00+00:00",
        "paperless_url": "http://paperless:8000",
        "total_correspondents": 1,
        "total_documents": 0,
        "judge_enabled": False,
        "judged_pair_count": 0,
        "approved_clusters": [],
        "orphan_correspondents": [],
        "candidate_pairs": [],
    }
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(data)))

    plan = load_merge_plan("-")
    write_merge_plan(plan, "-")

    assert (
        json.loads(capsys.readouterr().out)["paperless_url"] == "http://paperless:8000"
    )


@pytest.mark.asyncio
async def test_apply_correspondent_merge_plan_reassigns_and_deletes(tmp_path: Path):
    client = _FakeClient(
        correspondents=[],
        documents=[],
        counts={1: 0},
    )
    path_plan = {
        "version": 1,
        "generated_at": "2026-04-08T00:00:00+00:00",
        "paperless_url": "http://paperless:8000",
        "total_correspondents": 2,
        "total_documents": 2,
        "judge_enabled": False,
        "judged_pair_count": 0,
        "approved_clusters": [
            {
                "canonical_id": 2,
                "canonical_name": "ACME Corporation",
                "canonical_document_count": 1,
                "merged_ids": [1],
                "merged_names": ["Acme Corp."],
                "source_document_count": 1,
                "planned_document_ids": [101, 102],
                "confidence": "high",
                "status": "approved",
                "reasons": ["normalized_exact_match"],
            }
        ],
        "orphan_correspondents": [
            {
                "id": 3,
                "name": "Legacy Sender",
                "document_count": 0,
                "status": "approved",
                "reason": "no_documents_assigned",
            }
        ],
        "candidate_pairs": [],
    }
    tmp = tmp_path / "plan.json"
    tmp.write_text(json.dumps(path_plan), encoding="utf-8")
    plan = load_merge_plan(str(tmp))

    summary = await apply_correspondent_merge_plan(client, plan, dry_run=False)

    assert client.patched == [
        (101, {"correspondent": 2}),
        (102, {"correspondent": 2}),
    ]
    assert client.deleted == [1, 3]
    assert summary == {
        "reassigned_documents": 2,
        "deleted_correspondents": 2,
        "skipped_clusters": 0,
        "skipped_orphans": 0,
        "skipped_nonempty_deletes": 0,
    }


@pytest.mark.asyncio
async def test_apply_correspondent_merge_plan_skips_unapproved_clusters(tmp_path: Path):
    client = _FakeClient(correspondents=[], documents=[])
    tmp = tmp_path / "skipped-plan.json"
    tmp.write_text(
        json.dumps(
            {
                "version": 1,
                "generated_at": "2026-04-08T00:00:00+00:00",
                "paperless_url": "http://paperless:8000",
                "total_correspondents": 1,
                "total_documents": 0,
                "judge_enabled": False,
                "judged_pair_count": 0,
                "approved_clusters": [
                    {
                        "canonical_id": 2,
                        "canonical_name": "ACME Corporation",
                        "canonical_document_count": 1,
                        "merged_ids": [1],
                        "merged_names": ["Acme Corp."],
                        "source_document_count": 1,
                        "planned_document_ids": [101],
                        "confidence": "high",
                        "status": "skipped",
                        "reasons": ["normalized_exact_match"],
                    }
                ],
                "orphan_correspondents": [
                    {
                        "id": 9,
                        "name": "Unused Sender",
                        "document_count": 0,
                        "status": "skipped",
                        "reason": "no_documents_assigned",
                    }
                ],
                "candidate_pairs": [],
            }
        ),
        encoding="utf-8",
    )
    plan = load_merge_plan(str(tmp))

    summary = await apply_correspondent_merge_plan(client, plan, dry_run=False)

    assert client.patched == []
    assert client.deleted == []
    assert summary["skipped_clusters"] == 1
    assert summary["skipped_orphans"] == 1


@pytest.mark.asyncio
async def test_apply_correspondent_merge_plan_deletes_orphan_only(tmp_path: Path):
    client = _FakeClient(correspondents=[], documents=[], counts={8: 0})
    tmp = tmp_path / "orphan-plan.json"
    tmp.write_text(
        json.dumps(
            {
                "version": 1,
                "generated_at": "2026-04-08T00:00:00+00:00",
                "paperless_url": "http://paperless:8000",
                "total_correspondents": 1,
                "total_documents": 0,
                "judge_enabled": False,
                "judged_pair_count": 0,
                "approved_clusters": [],
                "orphan_correspondents": [
                    {
                        "id": 8,
                        "name": "Unused Sender",
                        "document_count": 0,
                        "status": "approved",
                        "reason": "no_documents_assigned",
                    }
                ],
                "candidate_pairs": [],
            }
        ),
        encoding="utf-8",
    )
    plan = load_merge_plan(str(tmp))

    summary = await apply_correspondent_merge_plan(client, plan, dry_run=False)

    assert client.patched == []
    assert client.deleted == [8]
    assert summary == {
        "reassigned_documents": 0,
        "deleted_correspondents": 1,
        "skipped_clusters": 0,
        "skipped_orphans": 0,
        "skipped_nonempty_deletes": 0,
    }
