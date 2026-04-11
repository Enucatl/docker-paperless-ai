"""
Review-first correspondent cleanup workflow.

Builds an auditable merge plan for duplicate/alias correspondents, then applies
approved document reassignments and deletes empty correspondents.
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
from itertools import combinations
from pathlib import Path
from typing import Any

import click
import litellm
import niquests
from pydantic import BaseModel

from paperless_ai.core.config import AgentConfig
from paperless_ai.core.paperless import PaperlessClient

log = logging.getLogger(__name__)

_LEGAL_SUFFIXES = re.compile(
    r"\b(inc|ltd|llc|ag|gmbh|corp|co|company|corporation|limited|incorporated|pllc|plc|sa|bv|nv)\b",
    re.IGNORECASE,
)
_GENERIC_ORG_TOKENS = {
    "company",
    "corp",
    "corporation",
    "group",
    "holding",
    "holdings",
    "services",
    "solutions",
    "systems",
    "international",
    "global",
}
_ORG_HINT_TOKENS = _GENERIC_ORG_TOKENS | {
    "bank",
    "insurance",
    "university",
    "department",
    "agency",
    "association",
    "inc",
    "llc",
    "ltd",
    "gmbh",
}


class _JudgeDecision(BaseModel):
    same_entity: bool
    confidence: str
    canonical_choice: str
    suggested_name: str | None = None
    reason_code: str


@dataclass
class CorrespondentRecord:
    id: int
    name: str
    normalized_name: str
    token_sort_name: str
    document_ids: list[int] = field(default_factory=list)
    sample_titles: list[str] = field(default_factory=list)

    @property
    def document_count(self) -> int:
        return len(self.document_ids)


@dataclass
class CorrespondentPairDecision:
    left_id: int
    left_name: str
    right_id: int
    right_name: str
    decision: str
    confidence: str
    source: str
    canonical_id: int | None
    canonical_name: str | None
    reason: str


@dataclass
class CorrespondentCluster:
    canonical_id: int
    canonical_name: str
    canonical_document_count: int
    merged_ids: list[int]
    merged_names: list[str]
    source_document_count: int
    planned_document_ids: list[int]
    confidence: str
    status: str
    reasons: list[str]


@dataclass
class OrphanCorrespondent:
    id: int
    name: str
    document_count: int
    status: str
    reason: str


@dataclass
class CorrespondentMergePlan:
    version: int
    generated_at: str
    paperless_url: str
    total_correspondents: int
    total_documents: int
    judge_enabled: bool
    judged_pair_count: int
    approved_clusters: list[CorrespondentCluster]
    orphan_correspondents: list[OrphanCorrespondent]
    candidate_pairs: list[CorrespondentPairDecision]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize_merge_plan(plan: CorrespondentMergePlan) -> dict[str, int]:
    merge_pairs = sum(1 for item in plan.candidate_pairs if item.decision == "merge")
    review_pairs = sum(1 for item in plan.candidate_pairs if item.decision == "review")
    rejected_pairs = sum(
        1 for item in plan.candidate_pairs if item.decision == "reject"
    )
    return {
        "approved_clusters": len(plan.approved_clusters),
        "orphan_correspondents": len(plan.orphan_correspondents),
        "candidate_pairs": len(plan.candidate_pairs),
        "merge_pairs": merge_pairs,
        "review_pairs": review_pairs,
        "rejected_pairs": rejected_pairs,
        "planned_document_moves": sum(
            len(item.planned_document_ids) for item in plan.approved_clusters
        ),
    }


def _normalize_correspondent_name(name: str) -> str:
    text = name.lower().strip()
    text = text.replace("&", " and ")
    text = text.replace(".", "")
    text = text.replace("/", " ")
    text = text.replace("-", " ")
    text = text.replace("'", "")
    text = _LEGAL_SUFFIXES.sub("", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _token_sort_name(normalized: str) -> str:
    return " ".join(sorted(token for token in normalized.split() if token))


def _looks_like_person_name(normalized: str) -> bool:
    tokens = normalized.split()
    if not 1 < len(tokens) <= 4:
        return False
    if any(token in _ORG_HINT_TOKENS for token in tokens):
        return False
    return all(token.isalpha() and len(token) > 1 for token in tokens)


def _token_overlap(left: str, right: str) -> float:
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(len(left_tokens), len(right_tokens))


def _has_conflicting_entity_shape(
    left: CorrespondentRecord, right: CorrespondentRecord
) -> bool:
    if _looks_like_person_name(left.normalized_name) != _looks_like_person_name(
        right.normalized_name
    ):
        return True
    left_tokens = set(left.normalized_name.split()) - _GENERIC_ORG_TOKENS
    right_tokens = set(right.normalized_name.split()) - _GENERIC_ORG_TOKENS
    if not left_tokens or not right_tokens:
        return False
    return left_tokens.isdisjoint(right_tokens)


def _pick_canonical(
    left: CorrespondentRecord, right: CorrespondentRecord
) -> CorrespondentRecord:
    return max(
        (left, right),
        key=lambda item: (
            item.document_count,
            len(item.name.strip()),
            -item.id,
        ),
    )


def _blocking_keys(record: CorrespondentRecord) -> set[str]:
    tokens = record.normalized_name.split()
    keys = {f"token_sort:{record.token_sort_name}"}
    if tokens:
        keys.add(f"first:{tokens[0]}")
        keys.add(f"prefix:{record.normalized_name[:12]}")
    for token in tokens:
        if len(token) >= 4:
            keys.add(f"token:{token}")
    return keys


def _deterministic_pair_decision(
    left: CorrespondentRecord,
    right: CorrespondentRecord,
) -> CorrespondentPairDecision | None:
    ratio = SequenceMatcher(None, left.normalized_name, right.normalized_name).ratio()
    overlap = _token_overlap(left.normalized_name, right.normalized_name)
    if _has_conflicting_entity_shape(left, right):
        return CorrespondentPairDecision(
            left_id=left.id,
            left_name=left.name,
            right_id=right.id,
            right_name=right.name,
            decision="reject",
            confidence="high",
            source="heuristic",
            canonical_id=None,
            canonical_name=None,
            reason="entity_shape_conflict",
        )

    if left.normalized_name and left.normalized_name == right.normalized_name:
        canonical = _pick_canonical(left, right)
        return CorrespondentPairDecision(
            left_id=left.id,
            left_name=left.name,
            right_id=right.id,
            right_name=right.name,
            decision="merge",
            confidence="high",
            source="heuristic",
            canonical_id=canonical.id,
            canonical_name=canonical.name,
            reason="normalized_exact_match",
        )

    if (
        left.token_sort_name
        and left.token_sort_name == right.token_sort_name
        and overlap >= 1.0
    ):
        canonical = _pick_canonical(left, right)
        return CorrespondentPairDecision(
            left_id=left.id,
            left_name=left.name,
            right_id=right.id,
            right_name=right.name,
            decision="merge",
            confidence="high",
            source="heuristic",
            canonical_id=canonical.id,
            canonical_name=canonical.name,
            reason="token_sort_exact_match",
        )

    if ratio >= 0.93 and overlap >= 0.67:
        canonical = _pick_canonical(left, right)
        return CorrespondentPairDecision(
            left_id=left.id,
            left_name=left.name,
            right_id=right.id,
            right_name=right.name,
            decision="merge",
            confidence="medium",
            source="heuristic",
            canonical_id=canonical.id,
            canonical_name=canonical.name,
            reason="high_similarity_overlap",
        )

    if ratio >= 0.86 and overlap >= 0.5:
        return CorrespondentPairDecision(
            left_id=left.id,
            left_name=left.name,
            right_id=right.id,
            right_name=right.name,
            decision="review",
            confidence="low",
            source="heuristic",
            canonical_id=None,
            canonical_name=None,
            reason="borderline_similarity",
        )
    return None


async def _judge_pair(
    config: AgentConfig,
    left: CorrespondentRecord,
    right: CorrespondentRecord,
) -> CorrespondentPairDecision:
    judge_temperature = 1.0 if "gemini-3" in config.llm_judge_model.lower() else 0
    prompt = (
        "Decide whether two document correspondents refer to the same real-world entity.\n"
        "Be conservative. If there is meaningful ambiguity, answer not same.\n\n"
        f"Left name: {left.name}\n"
        f"Left normalized: {left.normalized_name}\n"
        f"Left document count: {left.document_count}\n"
        f"Left sample titles: {json.dumps(left.sample_titles[:5])}\n\n"
        f"Right name: {right.name}\n"
        f"Right normalized: {right.normalized_name}\n"
        f"Right document count: {right.document_count}\n"
        f"Right sample titles: {json.dumps(right.sample_titles[:5])}\n"
    )
    kwargs: dict[str, Any] = {
        "model": config.llm_judge_model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a strict entity-resolution judge for document metadata cleanup. "
                    "Only say two names are the same entity when the evidence is strong."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "response_format": _JudgeDecision,
        "temperature": judge_temperature,
        "max_tokens": 120,
    }
    response = await litellm.acompletion(**kwargs)
    raw = response.choices[0].message.content or "{}"
    parsed = _JudgeDecision.model_validate_json(raw)

    if parsed.same_entity and parsed.confidence == "high":
        if parsed.canonical_choice == "right":
            canonical = right
        else:
            canonical = left
        return CorrespondentPairDecision(
            left_id=left.id,
            left_name=left.name,
            right_id=right.id,
            right_name=right.name,
            decision="merge",
            confidence="high",
            source="llm_judge",
            canonical_id=canonical.id,
            canonical_name=canonical.name,
            reason=parsed.reason_code,
        )

    return CorrespondentPairDecision(
        left_id=left.id,
        left_name=left.name,
        right_id=right.id,
        right_name=right.name,
        decision="reject",
        confidence=parsed.confidence,
        source="llm_judge",
        canonical_id=None,
        canonical_name=None,
        reason=parsed.reason_code,
    )


class _UnionFind:
    def __init__(self, ids: list[int]):
        self.parent = {item_id: item_id for item_id in ids}

    def find(self, item_id: int) -> int:
        parent = self.parent[item_id]
        if parent != item_id:
            self.parent[item_id] = self.find(parent)
        return self.parent[item_id]

    def union(self, left_id: int, right_id: int) -> None:
        left_root = self.find(left_id)
        right_root = self.find(right_id)
        if left_root != right_root:
            self.parent[right_root] = left_root


async def build_correspondent_merge_plan(
    client: PaperlessClient,
    config: AgentConfig,
    *,
    judge_borderline: bool = False,
) -> CorrespondentMergePlan:
    correspondents = await client.get_all_correspondents()
    documents = await client.iter_all_documents_brief()

    docs_by_correspondent: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for doc in documents:
        correspondent_id = doc.get("correspondent")
        if correspondent_id:
            docs_by_correspondent[int(correspondent_id)].append(doc)

    records: list[CorrespondentRecord] = []
    for item in correspondents:
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        docs = docs_by_correspondent.get(int(item["id"]), [])
        records.append(
            CorrespondentRecord(
                id=int(item["id"]),
                name=name,
                normalized_name=_normalize_correspondent_name(name),
                token_sort_name=_token_sort_name(_normalize_correspondent_name(name)),
                document_ids=[int(doc["id"]) for doc in docs],
                sample_titles=[str(doc.get("title") or "Untitled") for doc in docs[:5]],
            )
        )

    by_id = {record.id: record for record in records}
    candidate_pairs: dict[tuple[int, int], CorrespondentPairDecision] = {}
    blocks: dict[str, set[int]] = defaultdict(set)
    for record in records:
        for key in _blocking_keys(record):
            blocks[key].add(record.id)

    for block_ids in blocks.values():
        if len(block_ids) < 2:
            continue
        for left_id, right_id in combinations(sorted(block_ids), 2):
            pair_key = (left_id, right_id)
            if pair_key in candidate_pairs:
                continue
            decision = _deterministic_pair_decision(by_id[left_id], by_id[right_id])
            if decision is not None:
                candidate_pairs[pair_key] = decision

    judged_pair_count = 0
    if judge_borderline:
        for pair_key, decision in list(candidate_pairs.items()):
            if decision.decision != "review":
                continue
            judged_pair_count += 1
            try:
                candidate_pairs[pair_key] = await _judge_pair(
                    config,
                    by_id[pair_key[0]],
                    by_id[pair_key[1]],
                )
            except Exception as exc:
                log.warning(
                    "LLM judge failed for %s vs %s: %s", pair_key[0], pair_key[1], exc
                )
                candidate_pairs[pair_key] = CorrespondentPairDecision(
                    left_id=decision.left_id,
                    left_name=decision.left_name,
                    right_id=decision.right_id,
                    right_name=decision.right_name,
                    decision="reject",
                    confidence="low",
                    source="llm_judge",
                    canonical_id=None,
                    canonical_name=None,
                    reason="judge_error",
                )

    union_find = _UnionFind([record.id for record in records])
    approved_pairs = [
        decision
        for decision in candidate_pairs.values()
        if decision.decision == "merge"
    ]
    for decision in approved_pairs:
        union_find.union(decision.left_id, decision.right_id)

    grouped: dict[int, list[CorrespondentRecord]] = defaultdict(list)
    for record in records:
        grouped[union_find.find(record.id)].append(record)

    approved_clusters: list[CorrespondentCluster] = []
    for members in grouped.values():
        if len(members) < 2:
            continue
        canonical = max(
            members,
            key=lambda item: (
                item.document_count,
                len(item.name.strip()),
                -item.id,
            ),
        )
        merged_members = [item for item in members if item.id != canonical.id]
        if not merged_members:
            continue
        planned_document_ids = sorted(
            {doc_id for item in merged_members for doc_id in item.document_ids}
        )
        reasons = sorted(
            {
                decision.reason
                for decision in approved_pairs
                if decision.left_id in {item.id for item in members}
                and decision.right_id in {item.id for item in members}
            }
        )
        confidence = (
            "high"
            if all(
                decision.confidence == "high"
                for decision in approved_pairs
                if decision.left_id in {item.id for item in members}
                and decision.right_id in {item.id for item in members}
            )
            else "medium"
        )
        approved_clusters.append(
            CorrespondentCluster(
                canonical_id=canonical.id,
                canonical_name=canonical.name,
                canonical_document_count=canonical.document_count,
                merged_ids=sorted(item.id for item in merged_members),
                merged_names=sorted(item.name for item in merged_members),
                source_document_count=sum(
                    item.document_count for item in merged_members
                ),
                planned_document_ids=planned_document_ids,
                confidence=confidence,
                status="approved",
                reasons=reasons,
            )
        )

    approved_clusters.sort(
        key=lambda item: (-item.source_document_count, item.canonical_name.lower())
    )
    orphan_correspondents = sorted(
        [
            OrphanCorrespondent(
                id=record.id,
                name=record.name,
                document_count=record.document_count,
                status="approved",
                reason="no_documents_assigned",
            )
            for record in records
            if record.document_count == 0
        ],
        key=lambda item: item.name.lower(),
    )
    decisions = sorted(
        candidate_pairs.values(),
        key=lambda item: (item.left_name.lower(), item.right_name.lower()),
    )
    plan = CorrespondentMergePlan(
        version=1,
        generated_at=datetime.now(timezone.utc).isoformat(),
        paperless_url=config.paperless_url,
        total_correspondents=len(records),
        total_documents=len(documents),
        judge_enabled=judge_borderline,
        judged_pair_count=judged_pair_count,
        approved_clusters=approved_clusters,
        orphan_correspondents=orphan_correspondents,
        candidate_pairs=decisions,
    )
    summary = summarize_merge_plan(plan)
    log.info(
        "Correspondent cleanup plan: correspondents=%d documents=%d clusters=%d orphan_deletes=%d candidate_pairs=%d merges=%d review=%d reject=%d planned_moves=%d",
        plan.total_correspondents,
        plan.total_documents,
        summary["approved_clusters"],
        summary["orphan_correspondents"],
        summary["candidate_pairs"],
        summary["merge_pairs"],
        summary["review_pairs"],
        summary["rejected_pairs"],
        summary["planned_document_moves"],
    )
    for cluster in plan.approved_clusters[:5]:
        log.info(
            "Plan cluster: keep '%s' (id=%d) <- %s | move %d doc(s) | reasons=%s",
            cluster.canonical_name,
            cluster.canonical_id,
            ", ".join(f"{name}" for name in cluster.merged_names),
            len(cluster.planned_document_ids),
            ", ".join(cluster.reasons),
        )
    if len(plan.approved_clusters) > 5:
        log.info(
            "Plan cluster: ... %d additional cluster(s) omitted from log",
            len(plan.approved_clusters) - 5,
        )
    if plan.orphan_correspondents:
        preview = ", ".join(item.name for item in plan.orphan_correspondents[:10])
        log.info(
            "Plan orphan correspondents: %d candidate delete(s)%s%s",
            len(plan.orphan_correspondents),
            " | " if preview else "",
            preview,
        )
        if len(plan.orphan_correspondents) > 10:
            log.info(
                "Plan orphan correspondents: ... %d additional orphan(s) omitted from log",
                len(plan.orphan_correspondents) - 10,
            )
    return plan


def write_merge_plan(plan: CorrespondentMergePlan, path: str) -> None:
    text = json.dumps(plan.to_dict(), indent=2) + "\n"
    with click.open_file(path, "w", encoding="utf-8") as f:
        f.write(text)


def load_merge_plan(path: str) -> CorrespondentMergePlan:
    with click.open_file(path, "r", encoding="utf-8") as f:
        text = f.read()
    data = json.loads(text)
    return CorrespondentMergePlan(
        version=int(data["version"]),
        generated_at=str(data["generated_at"]),
        paperless_url=str(data["paperless_url"]),
        total_correspondents=int(data["total_correspondents"]),
        total_documents=int(data["total_documents"]),
        judge_enabled=bool(data["judge_enabled"]),
        judged_pair_count=int(data["judged_pair_count"]),
        approved_clusters=[
            CorrespondentCluster(**item) for item in data.get("approved_clusters", [])
        ],
        orphan_correspondents=[
            OrphanCorrespondent(**item)
            for item in data.get("orphan_correspondents", [])
        ],
        candidate_pairs=[
            CorrespondentPairDecision(**item)
            for item in data.get("candidate_pairs", [])
        ],
    )


async def apply_correspondent_merge_plan(
    client: PaperlessClient,
    plan: CorrespondentMergePlan,
    *,
    dry_run: bool = False,
) -> dict[str, int]:
    reassigned_documents = 0
    deleted_correspondents = 0
    skipped_clusters = 0
    skipped_orphans = 0
    skipped_nonempty_deletes = 0

    log.info(
        "Applying correspondent cleanup plan: clusters=%d orphan_deletes=%d dry_run=%s",
        len(plan.approved_clusters),
        len(plan.orphan_correspondents),
        dry_run,
    )

    for cluster in plan.approved_clusters:
        if cluster.status != "approved":
            skipped_clusters += 1
            log.info(
                "Skipping cluster for '%s' (id=%d) because status=%s",
                cluster.canonical_name,
                cluster.canonical_id,
                cluster.status,
            )
            continue
        log.info(
            "Applying cluster: keep '%s' (id=%d), merge %d correspondent(s), move %d document(s)",
            cluster.canonical_name,
            cluster.canonical_id,
            len(cluster.merged_ids),
            len(cluster.planned_document_ids),
        )
        for doc_id in cluster.planned_document_ids:
            if dry_run:
                log.info(
                    "DRY RUN: would move document %d -> correspondent %d",
                    doc_id,
                    cluster.canonical_id,
                )
            else:
                await client.patch_document(
                    doc_id, {"correspondent": cluster.canonical_id}
                )
            reassigned_documents += 1

        for source_id in cluster.merged_ids:
            remaining = await client.count_documents_for_correspondent(source_id)
            if remaining != 0:
                log.warning(
                    "Correspondent %d still has %d document(s); skipping delete",
                    source_id,
                    remaining,
                )
                skipped_nonempty_deletes += 1
                continue
            if dry_run:
                log.info("DRY RUN: would delete correspondent %d", source_id)
            else:
                try:
                    await client.delete_correspondent(source_id)
                except niquests.HTTPError as exc:
                    if getattr(exc.response, "status_code", None) == 404:
                        log.info(
                            "Correspondent %d already deleted; continuing", source_id
                        )
                        continue
                    raise
            deleted_correspondents += 1

    for orphan in plan.orphan_correspondents:
        if orphan.status != "approved":
            skipped_orphans += 1
            log.info(
                "Skipping orphan correspondent '%s' (id=%d) because status=%s",
                orphan.name,
                orphan.id,
                orphan.status,
            )
            continue
        remaining = await client.count_documents_for_correspondent(orphan.id)
        if remaining != 0:
            log.warning(
                "Correspondent %d now has %d document(s); skipping orphan delete",
                orphan.id,
                remaining,
            )
            skipped_nonempty_deletes += 1
            continue
        if dry_run:
            log.info(
                "DRY RUN: would delete orphan correspondent %d (%s)",
                orphan.id,
                orphan.name,
            )
        else:
            try:
                await client.delete_correspondent(orphan.id)
            except niquests.HTTPError as exc:
                if getattr(exc.response, "status_code", None) == 404:
                    log.info(
                        "Orphan correspondent %d (%s) already deleted; continuing",
                        orphan.id,
                        orphan.name,
                    )
                    continue
                raise
        deleted_correspondents += 1

    summary = {
        "reassigned_documents": reassigned_documents,
        "deleted_correspondents": deleted_correspondents,
        "skipped_clusters": skipped_clusters,
        "skipped_orphans": skipped_orphans,
        "skipped_nonempty_deletes": skipped_nonempty_deletes,
    }
    log.info(
        "Correspondent cleanup result: moved=%d deleted=%d skipped_clusters=%d skipped_orphans=%d skipped_nonempty_deletes=%d",
        summary["reassigned_documents"],
        summary["deleted_correspondents"],
        summary["skipped_clusters"],
        summary["skipped_orphans"],
        summary["skipped_nonempty_deletes"],
    )
    return summary
