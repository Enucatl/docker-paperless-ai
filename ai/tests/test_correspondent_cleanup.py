import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from paperless_ai.core.config import AgentConfig
from paperless_ai.core.correspondent_cleanup import (
    CorrespondentPairDecision,
    CorrespondentMergePlan,
    OrphanCorrespondent,
    PlannedCorrespondentCluster,
    actionable_review_pairs,
    merge_clusters,
    _route_score,
    apply_correspondent_merge_plan,
    build_correspondent_merge_plan,
    load_merge_plan,
    propose_canonical_name,
    select_disjoint_merges,
    wait_for_cleanup_queues,
    write_merge_plan,
)
from paperless_ai.correspondent import CorrespondentCluster, CorrespondentResolver


class _FakeClient:
    def __init__(self, correspondents, documents, counts=None):
        self._correspondents = correspondents
        self._documents = documents
        self._counts = counts or {}
        self.patched = []
        self.renamed = []
        self.deleted = []

    async def get_all_correspondents(self, force: bool = False):
        return list(self._correspondents)

    async def iter_all_documents_brief(self):
        return list(self._documents)

    async def patch_document(self, doc_id: int, payload: dict):
        self.patched.append((doc_id, payload))

    async def patch_correspondent(self, correspondent_id: int, payload: dict):
        self.renamed.append((correspondent_id, payload))

    async def count_documents_for_correspondent(self, correspondent_id: int) -> int:
        return self._counts.get(correspondent_id, 0)

    async def delete_correspondent(self, correspondent_id: int) -> None:
        self.deleted.append(correspondent_id)


class _MemoryReviewStore:
    def __init__(self):
        self.watermark = None
        self.items = {}

    async def get_last_processed_correspondent_id(self):
        return self.watermark

    async def advance_watermark(self, correspondent_id):
        self.watermark = correspondent_id

    async def record_decision(self, plan_id, pair_key, decision, evidence):
        self.items[(plan_id, pair_key)] = (decision, evidence)

    async def decisions_for_plan(self, plan_id):
        return [
            {
                "pair_key": key,
                "decision": value[0],
                "evidence": value[1],
                "decided_at": "now",
            }
            for (item_plan, key), value in self.items.items()
            if item_plan == plan_id
        ]


def _config() -> AgentConfig:
    return AgentConfig(
        paperless_url="http://paperless:8000",
        paperless_token="token-123",
        metadata_model="gemini/gemini-2.5-flash",
        chat_model="gemini/gemini-2.5-flash",
    )


@pytest.mark.asyncio
async def test_canonical_name_prompt_anchors_identity_to_correspondent_names(
    monkeypatch: pytest.MonkeyPatch,
):
    calls = {}

    async def fake_complete(**kwargs):
        calls.update(kwargs)
        return SimpleNamespace(message={"content": '{"canonical_name":"Zürich ZH"}'})

    monkeypatch.setattr(
        "paperless_ai.core.correspondent_cleanup.complete", fake_complete
    )

    result = await propose_canonical_name(
        CorrespondentCluster(
            [483, 562],
            ["Zürich ZH", "Zurich ZH"],
            [125, 69, 1696],
            [
                "Swiss Residence Permit",
                "Swiss Residence Permit for Matteo Abis",
                "Swiss Passport for Matteo Abis",
            ],
        ),
        _config(),
    )

    payload = json.loads(calls["messages"][1]["content"])
    instructions = calls["messages"][0]["content"]
    assert result == "Zürich ZH"
    assert payload["correspondent_names"] == ["Zürich ZH", "Zurich ZH"]
    assert "member_names" not in payload
    assert "authoritative identity input" in instructions
    assert "must be derived from" in instructions
    assert "Do not infer a different person or organization" in instructions


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

    assert plan.approved_clusters == []
    assert [item.name for item in plan.orphan_correspondents] == ["Legacy Sender"]
    assert plan.candidate_pairs[0].candidate_score is not None


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
    assert all(
        decision.left_id != decision.right_id for decision in plan.candidate_pairs
    )


@pytest.mark.parametrize(
    ("score", "decision"),
    [(0.2, "reject"), (0.5, "review"), (1.0, "review"), (1.5, "merge"), (1.8, "merge")],
)
def test_typesafe_score_routing(score, decision):
    """Semantic Score boundaries directly route the three outcomes."""
    assert _route_score(score) == decision


def test_cluster_similarity_uses_best_cross_member_pair():
    resolver = CorrespondentResolver()
    left = CorrespondentCluster([1, 2], ["Unrelated Sender", "Acme Corp."])
    right = CorrespondentCluster([3], ["ACME Corporation"])

    candidate = resolver.score_cluster_pair(left, right)
    expected, _ = resolver.score_pair("Acme Corp.", "ACME Corporation")

    assert candidate.score == expected
    assert candidate.evidence_left_name == "Acme Corp."


def test_lower_cleanup_threshold_is_candidate_superset():
    clusters = [
        CorrespondentCluster([1], ["Acme Corp."]),
        CorrespondentCluster([2], ["Acme Corporation"]),
        CorrespondentCluster([3], ["Acme Holdings"]),
    ]
    resolver = CorrespondentResolver()
    low = resolver.cluster_candidates(clusters, threshold=0.65)
    high = resolver.cluster_candidates(clusters, threshold=0.80)

    assert {item.left_key + item.right_key for item in high} <= {
        item.left_key + item.right_key for item in low
    }


def test_disjoint_selection_skips_overlapping_same_edge():
    def decision(left, right, score):
        return CorrespondentPairDecision(
            left,
            str(left),
            right,
            str(right),
            "merge",
            0.9,
            "typesafe",
            typesafe_score=2,
            typesafe_confidence=0.9,
            candidate_score=score,
            left_members=[left],
            right_members=[right],
        )

    selected = select_disjoint_merges(
        [decision(1, 2, 0.9), decision(2, 3, 0.8), decision(4, 5, 0.7)]
    )

    assert [(item.left_members, item.right_members) for item in selected] == [
        ([1], [2]),
        ([4], [5]),
    ]


def test_merge_clusters_preserves_all_evidence():
    merged = merge_clusters(
        CorrespondentCluster([1], ["A"], [10], ["One"]),
        CorrespondentCluster([2], ["B"], [11], ["Two"]),
    )
    assert merged.key == (1, 2)
    assert merged.member_names == ["A", "B"]
    assert merged.document_ids == [10, 11]


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
    assert reloaded["version"] == 4
    assert "last_manually_reviewed_correspondent_id" in reloaded
    assert "automatic_applied_clusters" in reloaded


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
        documents=[
            {"id": 101, "correspondent": 1},
            {"id": 102, "correspondent": 1},
        ],
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
    assert client.renamed == [(2, {"name": "ACME Corporation"})]
    assert client.deleted == [1, 3]
    assert summary == {
        "reassigned_documents": 2,
        "renamed_correspondents": 1,
        "deleted_correspondents": 2,
        "skipped_clusters": 0,
        "skipped_orphans": 0,
        "skipped_nonempty_deletes": 0,
        "skipped_stale_documents": 0,
    }


@pytest.mark.asyncio
async def test_apply_does_not_rename_to_existing_exact_name_with_suffix_variant(
    tmp_path: Path,
):
    client = _FakeClient(
        correspondents=[
            {"id": 552, "name": "Connox GmbH"},
            {"id": 775, "name": "Connox"},
        ],
        documents=[],
        counts={775: 0},
    )
    tmp = tmp_path / "connox-plan.json"
    tmp.write_text(
        json.dumps(
            {
                "version": 3,
                "generated_at": "2026-04-08T00:00:00+00:00",
                "paperless_url": "http://paperless:8000",
                "total_correspondents": 2,
                "total_documents": 0,
                "judge_enabled": False,
                "judged_pair_count": 0,
                "approved_clusters": [
                    {
                        "canonical_id": 552,
                        "canonical_name": "Connox",
                        "canonical_document_count": 4,
                        "merged_ids": [775],
                        "merged_names": ["Connox"],
                        "source_document_count": 0,
                        "planned_document_ids": [],
                        "confidence": "high",
                        "status": "approved",
                        "reasons": [],
                    }
                ],
                "orphan_correspondents": [],
                "candidate_pairs": [],
            }
        ),
        encoding="utf-8",
    )

    summary = await apply_correspondent_merge_plan(client, load_merge_plan(str(tmp)))

    assert client.renamed == []
    assert client.deleted == [775]
    assert summary["renamed_correspondents"] == 0


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
        "renamed_correspondents": 0,
        "deleted_correspondents": 1,
        "skipped_clusters": 0,
        "skipped_orphans": 0,
        "skipped_nonempty_deletes": 0,
        "skipped_stale_documents": 0,
    }


@pytest.mark.asyncio
async def test_successful_apply_advances_snapshot_watermark(tmp_path: Path):
    store = _MemoryReviewStore()
    client = _FakeClient(correspondents=[], documents=[], counts={1: 0})
    plan = load_merge_plan(
        str(
            _write_plan(
                tmp_path,
                {
                    "approved_clusters": [],
                    "orphan_correspondents": [],
                    "scanned_max_correspondent_id": 887,
                },
            )
        )
    )

    await apply_correspondent_merge_plan(client, plan, review_store=store)

    assert await store.get_last_processed_correspondent_id() == 887


@pytest.mark.asyncio
async def test_failed_or_partial_apply_does_not_advance_watermark(tmp_path: Path):
    store = _MemoryReviewStore()
    await store.advance_watermark(100)
    client = _FakeClient(correspondents=[], documents=[], counts={1: 1})
    plan = load_merge_plan(
        str(
            _write_plan(
                tmp_path,
                {
                    "approved_clusters": [],
                    "orphan_correspondents": [
                        {
                            "id": 1,
                            "name": "Now nonempty",
                            "document_count": 0,
                            "status": "approved",
                            "reason": "no_documents_assigned",
                        }
                    ],
                    "scanned_max_correspondent_id": 887,
                },
            )
        )
    )

    await apply_correspondent_merge_plan(client, plan, review_store=store)

    assert await store.get_last_processed_correspondent_id() == 100


@pytest.mark.asyncio
async def test_automatic_apply_does_not_advance_manual_boundary(tmp_path: Path):
    store = _MemoryReviewStore()
    client = _FakeClient(correspondents=[], documents=[], counts={1: 0})
    plan = load_merge_plan(
        str(
            _write_plan(
                tmp_path,
                {
                    "orphan_correspondents": [
                        {
                            "id": 1,
                            "name": "Automatic orphan",
                            "document_count": 0,
                            "status": "approved",
                            "reason": "no_documents_assigned",
                        }
                    ],
                    "scanned_max_correspondent_id": 887,
                },
            )
        )
    )

    await apply_correspondent_merge_plan(
        client, plan, review_store=store, advance_manual_boundary=False
    )

    assert client.deleted == [1]
    assert await store.get_last_processed_correspondent_id() is None


@pytest.mark.asyncio
async def test_queue_wait_resets_empty_period_when_work_reappears():
    class Clock:
        now = 0

        def __call__(self):
            return self.now

        async def sleep(self, _seconds):
            self.now += 1

    class Queues:
        KEY_OCR = "ocr"
        KEY_METADATA = "metadata"

        def __init__(self):
            self.samples = iter([0, 1, 0, 0, 0])
            self.current = 0

        async def stage_work_count(self, stage):
            if stage == self.KEY_OCR:
                self.current = next(self.samples)
            return self.current if stage == self.KEY_OCR else 0

    clock = Clock()
    await wait_for_cleanup_queues(
        Queues(),
        empty_seconds=2,
        timeout_seconds=10,
        poll_seconds=1,
        clock=clock,
        sleep=clock.sleep,
    )
    assert clock.now == 4


@pytest.mark.asyncio
async def test_queue_wait_timeout_does_not_settle():
    class Clock:
        now = 0

        def __call__(self):
            return self.now

        async def sleep(self, _seconds):
            self.now += 1

    class Queues:
        KEY_OCR = "ocr"
        KEY_METADATA = "metadata"

        async def stage_work_count(self, _stage):
            return 1

    clock = Clock()
    with pytest.raises(TimeoutError):
        await wait_for_cleanup_queues(
            Queues(),
            empty_seconds=2,
            timeout_seconds=3,
            poll_seconds=1,
            clock=clock,
            sleep=clock.sleep,
        )


@pytest.mark.asyncio
async def test_apply_skips_documents_reassigned_after_analysis(tmp_path: Path):
    store = _MemoryReviewStore()
    client = _FakeClient(
        correspondents=[],
        documents=[{"id": 101, "correspondent": 99}],
        counts={1: 1},
    )
    plan = load_merge_plan(
        str(
            _write_plan(
                tmp_path,
                {
                    "approved_clusters": [
                        {
                            "canonical_id": 2,
                            "canonical_name": "Canonical",
                            "canonical_document_count": 0,
                            "merged_ids": [1],
                            "merged_names": ["Alias"],
                            "source_document_count": 1,
                            "planned_document_ids": [101],
                            "planned_document_correspondents": {"101": 1},
                            "confidence": "high",
                            "status": "approved",
                            "reasons": [],
                        }
                    ],
                    "scanned_max_correspondent_id": 10,
                },
            )
        )
    )

    summary = await apply_correspondent_merge_plan(client, plan, review_store=store)

    assert client.patched == []
    assert summary["skipped_stale_documents"] == 1
    assert await store.get_last_processed_correspondent_id() is None


@pytest.mark.asyncio
async def test_apply_uses_analysis_snapshot_not_records_created_during_apply(
    tmp_path: Path,
):
    class _CreatingClient(_FakeClient):
        async def patch_correspondent(self, correspondent_id: int, payload: dict):
            await super().patch_correspondent(correspondent_id, payload)
            self._correspondents.append({"id": 999, "name": "Created during apply"})

    store = _MemoryReviewStore()
    client = _CreatingClient(correspondents=[], documents=[], counts={1: 0})
    plan = load_merge_plan(
        str(
            _write_plan(
                tmp_path,
                {
                    "approved_clusters": [
                        {
                            "canonical_id": 2,
                            "canonical_name": "Canonical",
                            "canonical_document_count": 0,
                            "merged_ids": [1],
                            "merged_names": ["Alias"],
                            "source_document_count": 0,
                            "planned_document_ids": [],
                            "confidence": "high",
                            "status": "approved",
                            "reasons": [],
                        }
                    ],
                    "scanned_max_correspondent_id": 10,
                },
            )
        )
    )

    await apply_correspondent_merge_plan(client, plan, review_store=store)

    assert await store.get_last_processed_correspondent_id() == 10


@pytest.mark.asyncio
async def test_incremental_scope_compares_new_to_canonical_and_new_to_new(
    tmp_path: Path,
):
    store = _MemoryReviewStore()
    await store.advance_watermark(10)
    client = _FakeClient(
        correspondents=[
            {"id": 1, "name": "Acme Canonical"},
            {"id": 2, "name": "Acme Canonical"},
            {"id": 11, "name": "Acme Canonical"},
            {"id": 12, "name": "Acme Canonical"},
        ],
        documents=[],
    )

    plan = await build_correspondent_merge_plan(client, _config(), review_store=store)
    pairs = {
        frozenset(item.left_members + item.right_members)
        for item in plan.candidate_pairs
    }

    assert plan.cleanup_mode == "incremental"
    assert plan.scanned_max_correspondent_id == 12
    assert frozenset({1, 11}) in pairs
    assert frozenset({11, 12}) in pairs
    assert frozenset({1, 2}) not in pairs
    assert all(any(member > 10 for member in pair) for pair in pairs)


@pytest.mark.asyncio
async def test_incremental_scope_only_proposes_new_empty_orphans():
    store = _MemoryReviewStore()
    await store.advance_watermark(10)
    client = _FakeClient(
        correspondents=[
            {"id": 1, "name": "Historical empty"},
            {"id": 11, "name": "New empty"},
        ],
        documents=[],
    )

    plan = await build_correspondent_merge_plan(client, _config(), review_store=store)

    assert [item.id for item in plan.orphan_correspondents] == [11]


def _write_plan(tmp_path: Path, overrides: dict) -> Path:
    data = {
        "version": 3,
        "generated_at": "2026-04-08T00:00:00+00:00",
        "paperless_url": "http://paperless:8000",
        "total_correspondents": 0,
        "total_documents": 0,
        "judge_enabled": False,
        "judged_pair_count": 0,
        "approved_clusters": [],
        "orphan_correspondents": [],
        "candidate_pairs": [],
    }
    data.update(overrides)
    path = tmp_path / "watermark-plan.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.mark.asyncio
async def test_review_store_preserves_audit_and_allows_overlapping_approvals(
    tmp_path: Path,
):
    store = _MemoryReviewStore()
    first = {"left_members": [1], "right_members": [2], "round": 1}
    await store.record_decision("plan-a", "1-2", "approve", first)

    await store.record_decision(
        "plan-a",
        "2-3",
        "approve",
        {"left_members": [2], "right_members": [3], "round": 1},
    )

    await store.record_decision(
        "plan-a", "3-4", "reject", {"left_members": [3], "right_members": [4]}
    )
    assert [item["decision"] for item in await store.decisions_for_plan("plan-a")] == [
        "approve",
        "approve",
        "reject",
    ]


def test_actionable_review_pairs_excludes_earlier_round_stale_evidence():
    def pair(left, right, round_number):
        return CorrespondentPairDecision(
            left[0],
            str(left[0]),
            right[0],
            str(right[0]),
            "review",
            "pending",
            "resolver",
            round=round_number,
            left_members=left,
            right_members=right,
        )

    plan = CorrespondentMergePlan(
        3,
        "now",
        "http://paperless",
        3,
        0,
        False,
        0,
        [
            PlannedCorrespondentCluster(
                1, "A", 0, [2], ["B"], 0, [], "high", "approved", []
            )
        ],
        [OrphanCorrespondent(3, "C", 0, "approved", "empty")],
        [pair([1], [3], 1), pair([1, 2], [3], 2)],
    )

    assert [item.left_members for item in actionable_review_pairs(plan)] == [[1, 2]]
