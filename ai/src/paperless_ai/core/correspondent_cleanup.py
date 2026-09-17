"""Review-first iterative correspondent cleanup clustering."""

import asyncio
import csv
import json
import logging
import re
import sys
import unicodedata
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel
import niquests

from paperless_ai.correspondent import (
    ClusterCandidate,
    CorrespondentCluster,
    CorrespondentResolver,
    normalize_name,
)
from paperless_ai.core.config import AgentConfig
from paperless_ai.inference import complete
from paperless_common.paperless import PaperlessClient

log = logging.getLogger(__name__)


def _paperless_name_key(name: str) -> str:
    """Normalize names only for Paperless duplicate-name checks."""
    normalized = unicodedata.normalize("NFKC", name).casefold()
    return re.sub(r"\s+", " ", normalized).strip()


IDENTITY_LEVELS = [
    "Different canonical correspondents.",
    "Ambiguous; review.",
    "Same canonical correspondent.",
]
CLUSTER_QUESTION = """Should these two groups of Paperless correspondents be represented by one canonical correspondent? Different spellings, aliases, abbreviations, legal suffixes, and organizational subdivisions may belong together when they clearly represent the same intended parent correspondent. Departments or administrative units of the same parent organization may be grouped together. Do not merge distinct parent organizations: departments of Stadt Zürich may be grouped under Stadt Zürich, but Kanton Zürich must remain separate."""


@dataclass
class CorrespondentRecord:
    id: int
    name: str
    normalized_name: str
    document_ids: list[int] = field(default_factory=list)
    sample_titles: list[str] = field(default_factory=list)

    @property
    def document_count(self) -> int:
        """Return assigned document count."""
        return len(self.document_ids)


@dataclass
class CorrespondentPairDecision:
    left_id: int
    left_name: str
    right_id: int
    right_name: str
    decision: str
    confidence: float | str
    source: str
    canonical_id: int | None = None
    canonical_name: str | None = None
    reason: str = ""
    candidate_score: float | None = None
    candidate_components: dict[str, float] | None = None
    typesafe_score: float | None = None
    typesafe_confidence: float | None = None
    typesafe_probabilities: dict[str, float] | None = None
    typesafe_model: str | None = None
    left_document_count: int | None = None
    right_document_count: int | None = None
    round: int = 0
    left_members: list[int] = field(default_factory=list)
    right_members: list[int] = field(default_factory=list)


@dataclass
class MergeHistory:
    round: int
    left_members: list[int]
    right_members: list[int]
    candidate_score: float
    typesafe_score: float | None
    confidence: float | None


@dataclass
class PlannedCorrespondentCluster:
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
    members: list[dict[str, Any]] = field(default_factory=list)
    merge_history: list[MergeHistory] = field(default_factory=list)
    rounds_used: int = 0
    planned_document_correspondents: dict[int, int] = field(default_factory=dict)


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
    approved_clusters: list[PlannedCorrespondentCluster]
    orphan_correspondents: list[OrphanCorrespondent]
    candidate_pairs: list[CorrespondentPairDecision]
    rounds_used: int = 0
    max_rounds_reached: bool = False
    scanned_max_correspondent_id: int = 0
    last_processed_correspondent_id: int | None = None
    cleanup_mode: str = "full"

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible plan data."""
        return asdict(self)


class CleanupReviewStore:
    """Persistent cleanup state stored in the existing Paperless Postgres DB."""

    def __init__(
        self,
        *,
        host: str,
        database: str,
        user: str,
        password: str,
        port: int = 5432,
    ):
        self.connection_kwargs = {
            "host": host,
            "dbname": database,
            "user": user,
            "password": password,
            "port": port,
        }

    @classmethod
    async def from_config(cls, config: AgentConfig) -> "CleanupReviewStore":
        """Create the store from cleanup-specific Postgres settings."""
        store = cls(
            host=config.correspondent_cleanup_db_host,
            database=config.correspondent_cleanup_db_name,
            user=config.correspondent_cleanup_db_user,
            password=config.correspondent_cleanup_db_password,
        )
        await store._initialize()
        return store

    async def _connect(self):
        """Open an asynchronous PostgreSQL connection for one store operation."""
        from psycopg import AsyncConnection
        from psycopg.rows import dict_row

        return await AsyncConnection.connect(
            **self.connection_kwargs, row_factory=dict_row
        )

    async def _initialize(self) -> None:
        async with await self._connect() as connection:
            await connection.execute(
                """CREATE TABLE IF NOT EXISTS paperless_ai_cleanup_state (
                key TEXT PRIMARY KEY, value TEXT NOT NULL
                )"""
            )
            await connection.execute(
                """CREATE TABLE IF NOT EXISTS paperless_ai_cleanup_review_decisions (
                plan_id TEXT NOT NULL, pair_key TEXT NOT NULL,
                decision TEXT NOT NULL, evidence_json TEXT NOT NULL,
                decided_at TEXT NOT NULL,
                PRIMARY KEY (plan_id, pair_key)
                )"""
            )

    async def get_last_processed_correspondent_id(self) -> int | None:
        """Return the last fully applied correspondent snapshot boundary."""
        async with await self._connect() as connection:
            cursor = await connection.execute(
                "SELECT value FROM paperless_ai_cleanup_state WHERE key = %s",
                ("last_processed_correspondent_id",),
            )
            row = await cursor.fetchone()
        return int(row["value"]) if row else None

    async def advance_watermark(self, correspondent_id: int) -> None:
        """Record a successfully applied correspondent snapshot boundary."""
        async with await self._connect() as connection:
            await connection.execute(
                """INSERT INTO paperless_ai_cleanup_state(key, value) VALUES (%s, %s)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value""",
                ("last_processed_correspondent_id", str(correspondent_id)),
            )

    async def record_decision(
        self, plan_id: str, pair_key: str, decision: str, evidence: dict[str, Any]
    ) -> None:
        if decision not in {"approve", "reject"}:
            raise ValueError("review decision must be approve or reject")
        async with await self._connect() as connection:
            await connection.execute(
                """INSERT INTO paperless_ai_cleanup_review_decisions
                (plan_id, pair_key, decision, evidence_json, decided_at)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT(plan_id, pair_key) DO UPDATE SET
                  decision=excluded.decision, evidence_json=excluded.evidence_json,
                  decided_at=excluded.decided_at""",
                (
                    plan_id,
                    pair_key,
                    decision,
                    json.dumps(evidence, sort_keys=True),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    async def decisions_for_plan(self, plan_id: str) -> list[dict[str, Any]]:
        """Return recorded decisions for a generated cleanup plan."""
        async with await self._connect() as connection:
            cursor = await connection.execute(
                "SELECT pair_key, decision, evidence_json, decided_at "
                "FROM paperless_ai_cleanup_review_decisions "
                "WHERE plan_id = %s ORDER BY decided_at, pair_key",
                (plan_id,),
            )
            rows = await cursor.fetchall()
        return [
            {
                "pair_key": row["pair_key"],
                "decision": row["decision"],
                "evidence": json.loads(row["evidence_json"]),
                "decided_at": row["decided_at"],
            }
            for row in rows
        ]


class CanonicalNameResponse(BaseModel):
    canonical_name: str


def _route_score(score: float) -> str:
    """Route a System One score without confidence gating."""
    return "reject" if score < 0.5 else "review" if score < 1.5 else "merge"


def merge_clusters(
    left: CorrespondentCluster, right: CorrespondentCluster
) -> CorrespondentCluster:
    """Combine evidence only; canonical naming is deliberately deferred."""
    return CorrespondentCluster(
        sorted(set(left.member_ids + right.member_ids)),
        list(dict.fromkeys(left.member_names + right.member_names)),
        sorted(set(left.document_ids + right.document_ids)),
        list(dict.fromkeys(left.sample_titles + right.sample_titles))[:20],
    )


def select_disjoint_merges(
    items: list[CorrespondentPairDecision],
) -> list[CorrespondentPairDecision]:
    """Choose strongest clear decisions that do not share a member."""
    used: set[int] = set()
    chosen = []
    for item in sorted(
        items,
        key=lambda x: (
            -(x.typesafe_score or 0),
            -(x.typesafe_confidence or 0),
            -(x.candidate_score or 0),
            x.left_members,
            x.right_members,
        ),
    ):
        members = set(item.left_members + item.right_members)
        if not members & used:
            chosen.append(item)
            used |= members
    return chosen


def _fallback_name(
    cluster: CorrespondentCluster, records: dict[int, CorrespondentRecord] | None = None
) -> str:
    if records is None:
        return cluster.member_names[0]
    return max(
        (records[i] for i in cluster.member_ids),
        key=lambda x: (x.document_count, -x.id),
    ).name


async def propose_canonical_name(
    cluster: CorrespondentCluster,
    config: AgentConfig,
    records: dict[int, CorrespondentRecord] | None = None,
) -> str:
    """Make one structured final naming call, with a deterministic fallback."""
    fallback = _fallback_name(cluster, records)
    prompt = {
        "correspondent_names": cluster.member_names,
        "sample_document_titles": cluster.sample_titles[:10],
    }
    instructions = """Choose a simple, stable canonical correspondent name representing the complete cluster. The correspondent_names list is the authoritative identity input. The returned name must be derived from, and remain recognizably related to, one or more correspondent_names. Do not infer a different person or organization from sample_document_titles, and do not invent a name from any other source. Prefer a recognizable parent organization or person's normal name. Remove redundant legal suffixes, honorifics, department names, email addresses, duplicate qualifiers, and formatting noise when they do not distinguish the entity. Prefer an appropriate existing correspondent name. Return only JSON with canonical_name."""
    try:
        response = await complete(
            model=config.metadata_model,
            endpoint=config.metadata_endpoint,
            domain="correspondent_canonical_naming",
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": json.dumps(prompt)},
            ],
            response_format={"type": "json_object"},
            **config.get_metadata_kwargs(),
        )
        raw = response.message.get("content") or "{}"
        name = CanonicalNameResponse.model_validate_json(raw).canonical_name.strip()
        return name or fallback
    except Exception as exc:
        log.warning("Canonical-name generation failed for %s: %s", cluster.key, exc)
        return fallback


def _judge(
    client: Any,
    config: AgentConfig,
    left: CorrespondentCluster,
    right: CorrespondentCluster,
    candidate: ClusterCandidate,
    number: int,
) -> CorrespondentPairDecision:
    from typesafe_sdk import Score

    response = client.system_one(
        state={
            "cluster_a": {
                "names": left.member_names,
                "sample_document_titles": left.sample_titles[:10],
            },
            "cluster_b": {
                "names": right.member_names,
                "sample_document_titles": right.sample_titles[:10],
            },
        },
        questions={
            "identity": Score(instructions=CLUSTER_QUESTION, criteria=IDENTITY_LEVELS)
        },
        model=config.typesafe_model,
    )
    answer = response.scores["identity"]
    score = float(answer.score)
    return CorrespondentPairDecision(
        left.member_ids[0],
        candidate.evidence_left_name,
        right.member_ids[0],
        candidate.evidence_right_name,
        _route_score(score),
        float(answer.confidence),
        "typesafe",
        reason="typesafe_cluster_identity",
        candidate_score=candidate.score,
        candidate_components=candidate.components.to_dict(),
        typesafe_score=score,
        typesafe_confidence=float(answer.confidence),
        typesafe_probabilities={
            str(k): float(v) for k, v in answer.probabilities.items()
        },
        typesafe_model=getattr(response, "model", config.typesafe_model),
        left_document_count=len(left.document_ids),
        right_document_count=len(right.document_ids),
        round=number,
        left_members=list(left.key),
        right_members=list(right.key),
    )


async def judge_cluster_merge(
    left: CorrespondentCluster,
    right: CorrespondentCluster,
    candidate: ClusterCandidate,
    config: AgentConfig,
    client: Any,
    round_number: int,
) -> CorrespondentPairDecision:
    """Judge one cluster pair with TypeSafe System One."""
    return await asyncio.to_thread(
        _judge, client, config, left, right, candidate, round_number
    )


async def build_correspondent_merge_plan(
    client: PaperlessClient,
    config: AgentConfig,
    *,
    judge_borderline: bool = False,
    typesafe: bool = False,
    review_store: CleanupReviewStore | None = None,
) -> CorrespondentMergePlan:
    """Build a read-only disjoint-pair iterative cleanup plan."""
    correspondents, documents = (
        await client.get_all_correspondents(),
        await client.iter_all_documents_brief(),
    )
    docs: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for doc in documents:
        if doc.get("correspondent"):
            docs[int(doc["correspondent"])].append(doc)
    records = {
        int(x["id"]): CorrespondentRecord(
            int(x["id"]),
            str(x.get("name") or "").strip(),
            normalize_name(str(x.get("name") or "")).normalized_string,
            [int(d["id"]) for d in docs[int(x["id"])]],
            [str(d.get("title") or "Untitled") for d in docs[int(x["id"])][:5]],
        )
        for x in correspondents
        if str(x.get("name") or "").strip()
    }
    last_processed_id = (
        await review_store.get_last_processed_correspondent_id()
        if review_store
        else None
    )
    # This is the analysis snapshot boundary, including blank-name records. A
    # record created while applying must remain outside this boundary.
    scanned_max_id = max((int(x["id"]) for x in correspondents), default=0)
    cleanup_mode = "incremental" if last_processed_id is not None else "full"
    clusters = [
        CorrespondentCluster([x.id], [x.name], x.document_ids, x.sample_titles)
        for x in records.values()
    ]
    if judge_borderline:
        log.warning("--cleanup-judge-borderline is retired; use --cleanup-typesafe")
    if typesafe and not config.typesafe_api_key:
        raise ValueError(
            "TYPESAFE_API_KEY (or TYPESAFE_API_KEY_FILE) is required for --cleanup-typesafe"
        )
    ts = None
    if typesafe:
        from typesafe_sdk import TypeSafeClient

        ts = TypeSafeClient(
            api_key=config.typesafe_api_key,
            model=config.typesafe_model,
            base_url=config.typesafe_endpoint,
        )
    resolver = CorrespondentResolver()
    cache: dict[tuple[tuple[int, ...], tuple[int, ...]], CorrespondentPairDecision] = {}
    decisions: list[CorrespondentPairDecision] = []
    history: dict[tuple[int, ...], list[MergeHistory]] = defaultdict(list)
    rounds = 0
    maxed = False
    try:
        for number in range(1, config.correspondent_cleanup_max_rounds + 1):
            rounds = number
            by_key = {x.key: x for x in clusters}
            current = []
            candidates = resolver.cluster_candidates(
                clusters, threshold=config.correspondent_cleanup_candidate_threshold
            )
            if last_processed_id is not None:
                # At least one whole side must be new. This retains new↔canonical
                # and new↔new comparisons while preventing a mixed cluster from
                # broadening a reviewed incremental decision in a later round.
                candidates = [
                    item
                    for item in candidates
                    if all(member_id > last_processed_id for member_id in item.left_key)
                    or all(
                        member_id > last_processed_id for member_id in item.right_key
                    )
                ]
            cached_keys: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
            if typesafe:
                semaphore = asyncio.Semaphore(5)
                cached_keys = set(cache)

                async def evaluate(
                    candidate: ClusterCandidate,
                ) -> CorrespondentPairDecision:
                    left, right = (
                        by_key[candidate.left_key],
                        by_key[candidate.right_key],
                    )
                    key = tuple(sorted((left.key, right.key)))
                    if key in cache:
                        return cache[key]
                    async with semaphore:
                        try:
                            decision = await judge_cluster_merge(
                                left, right, candidate, config, ts, number
                            )
                            cache[key] = decision
                            return decision
                        except Exception as exc:
                            log.warning(
                                "TypeSafe failed for %s vs %s: %s",
                                left.key,
                                right.key,
                                exc,
                            )
                            return CorrespondentPairDecision(
                                left.member_ids[0],
                                candidate.evidence_left_name,
                                right.member_ids[0],
                                candidate.evidence_right_name,
                                "review",
                                "error",
                                "typesafe",
                                reason="typesafe_error",
                                candidate_score=candidate.score,
                                candidate_components=candidate.components.to_dict(),
                                round=number,
                                left_members=list(left.key),
                                right_members=list(right.key),
                            )

                current = list(
                    await asyncio.gather(*(evaluate(item) for item in candidates))
                )
            else:
                for candidate in candidates:
                    left, right = (
                        by_key[candidate.left_key],
                        by_key[candidate.right_key],
                    )
                    current.append(
                        CorrespondentPairDecision(
                            left.member_ids[0],
                            candidate.evidence_left_name,
                            right.member_ids[0],
                            candidate.evidence_right_name,
                            "review",
                            "pending",
                            "resolver",
                            reason="typesafe_not_requested",
                            candidate_score=candidate.score,
                            candidate_components=candidate.components.to_dict(),
                            round=number,
                            left_members=list(left.key),
                            right_members=list(right.key),
                        )
                    )
            decisions.extend(
                item
                for item in current
                if tuple(sorted((tuple(item.left_members), tuple(item.right_members))))
                not in cached_keys
            )
            selected = select_disjoint_merges(
                [x for x in current if x.decision == "merge"]
            )
            if not selected:
                break
            consumed: set[tuple[int, ...]] = set()
            merged = []
            for item in selected:
                left, right = (
                    by_key[tuple(item.left_members)],
                    by_key[tuple(item.right_members)],
                )
                cluster = merge_clusters(left, right)
                merged.append(cluster)
                consumed |= {left.key, right.key}
                history[cluster.key] = (
                    history[left.key]
                    + history[right.key]
                    + [
                        MergeHistory(
                            number,
                            item.left_members,
                            item.right_members,
                            item.candidate_score or 0,
                            item.typesafe_score,
                            item.typesafe_confidence,
                        )
                    ]
                )
            clusters = [x for x in clusters if x.key not in consumed] + merged
            maxed = number == config.correspondent_cleanup_max_rounds
    finally:
        if ts:
            ts.close()
    plans = []
    for cluster in clusters:
        if len(cluster.member_ids) < 2:
            continue
        survivor = max(
            (records[i] for i in cluster.member_ids),
            key=lambda x: (x.document_count, -x.id),
        )
        merged_ids = [i for i in cluster.member_ids if i != survivor.id]
        plans.append(
            PlannedCorrespondentCluster(
                survivor.id,
                await propose_canonical_name(cluster, config, records),
                survivor.document_count,
                merged_ids,
                [records[i].name for i in merged_ids],
                sum(records[i].document_count for i in merged_ids),
                sorted({d for i in merged_ids for d in records[i].document_ids}),
                "high",
                "approved",
                ["typesafe_cluster_identity"],
                [{"id": i, "name": records[i].name} for i in cluster.member_ids],
                history[cluster.key],
                max((x.round for x in history[cluster.key]), default=0),
                {
                    document_id: source
                    for source in merged_ids
                    for document_id in records[source].document_ids
                },
            )
        )
    planned_members = {
        member_id
        for cluster in plans
        for member_id in (cluster.canonical_id, *cluster.merged_ids)
    }
    orphans = [
        OrphanCorrespondent(x.id, x.name, 0, "approved", "no_documents_assigned")
        for x in records.values()
        if x.document_count == 0
        and x.id not in planned_members
        and (last_processed_id is None or x.id > last_processed_id)
    ]
    return CorrespondentMergePlan(
        3,
        datetime.now(timezone.utc).isoformat(),
        config.paperless_url,
        len(records),
        len(documents),
        typesafe,
        sum(x.source == "typesafe" for x in decisions),
        plans,
        orphans,
        decisions,
        rounds,
        maxed,
        scanned_max_id,
        last_processed_id,
        cleanup_mode,
    )


def summarize_merge_plan(plan: CorrespondentMergePlan) -> dict[str, int]:
    """Summarize planned operations."""
    return {
        "approved_clusters": len(plan.approved_clusters),
        "orphan_correspondents": len(plan.orphan_correspondents),
        "candidate_pairs": len(plan.candidate_pairs),
        "merge_pairs": sum(x.decision == "merge" for x in plan.candidate_pairs),
        "review_pairs": sum(x.decision == "review" for x in plan.candidate_pairs),
        "rejected_pairs": sum(x.decision == "reject" for x in plan.candidate_pairs),
        "planned_document_moves": sum(
            len(x.planned_document_ids) for x in plan.approved_clusters
        ),
    }


def actionable_review_pairs(
    plan: CorrespondentMergePlan,
) -> list[CorrespondentPairDecision]:
    """Return review pairs still valid after automatic later-round merges.

    ``candidate_pairs`` is the complete audit history. A review pair is
    actionable only when neither displayed side is a strict subset of a final
    automatically approved cluster; otherwise that earlier-round evidence is
    stale and must remain historical rather than being shown for approval.
    """
    automatic_clusters = [
        {cluster.canonical_id, *cluster.merged_ids}
        for cluster in plan.approved_clusters
    ]

    def stale(side: list[int]) -> bool:
        members = set(side)
        return any(
            members < cluster and members & cluster for cluster in automatic_clusters
        )

    seen: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    result = []
    for item in plan.candidate_pairs:
        if (
            item.decision != "review"
            or stale(item.left_members)
            or stale(item.right_members)
        ):
            continue
        key = (tuple(item.left_members), tuple(item.right_members))
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def write_merge_plan(plan: CorrespondentMergePlan, path: str) -> None:
    """Write JSON plan to a file or stdout."""
    text = json.dumps(plan.to_dict(), indent=2) + "\n"
    if path == "-":
        sys.stdout.write(text)
    else:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")


def write_cleanup_analysis(
    plan: CorrespondentMergePlan, directory: str
) -> dict[str, Any]:
    """Write per-round decisions and final cluster view without mutation."""
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    rows = [asdict(x) for x in plan.candidate_pairs]
    with (output / "scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]) if rows else ["round", "decision"]
        )
        writer.writeheader()
        writer.writerows(rows)
    (output / "scores.json").write_text(
        json.dumps(
            {
                "decisions": rows,
                "final_clusters": [asdict(x) for x in plan.approved_clusters],
                "max_rounds_reached": plan.max_rounds_reached,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "index.html").write_text(
        """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Correspondent cleanup review</title><style>
body{max-width:1100px;margin:2rem auto;padding:0 1rem;background:#f6f7fb;color:#18212f;font:16px system-ui,sans-serif}h1{margin-bottom:.2rem}.muted{color:#64748b}.cluster{background:#fff;border:1px solid #d9e0ea;border-radius:9px;margin:1.5rem 0;padding:1.4rem}.name{font-size:1.45rem;font-weight:700}.status{display:inline-block;background:#d9f4e3;color:#126238;border-radius:99px;padding:.2rem .6rem;font-weight:600}.members{width:100%;border-collapse:collapse;margin:1rem 0}.members td{padding:.42rem;border-bottom:1px solid #e8ebf0}.members td:last-child{text-align:right;font-family:ui-monospace,monospace;color:#64748b}.meta{display:flex;gap:2rem;flex-wrap:wrap}.history{font-size:.9rem;color:#475569}table.rounds{width:100%;border-collapse:collapse;background:#fff}.rounds th,.rounds td{padding:.5rem;text-align:left;border-bottom:1px solid #e8ebf0}.rounds th{background:#eef2f7}</style></head><body>
<h1>Correspondent cleanup review</h1><p id="summary" class="muted">Loading analysis…</p>
<main id="clusters"></main><h2>Per-round pair decisions</h2><table class="rounds"><thead><tr><th>Round</th><th>Clusters</th><th>Candidate</th><th>TypeSafe</th><th>Decision</th></tr></thead><tbody id="decisions"></tbody></table>
<script>const text=(tag,value)=>{const e=document.createElement(tag);e.textContent=value;return e};fetch('scores.json').then(r=>r.json()).then(data=>{const clusters=document.querySelector('#clusters');document.querySelector('#summary').textContent=`${data.final_clusters.length} proposed cluster(s) · ${data.decisions.length} pair decisions${data.max_rounds_reached?' · maximum rounds reached':''}`;for(const c of data.final_clusters){const section=document.createElement('section');section.className='cluster';const heading=document.createElement('div');heading.className='name';heading.textContent=`Proposed canonical: ${c.canonical_name}`;section.append(heading);const members=document.createElement('table');members.className='members';const body=document.createElement('tbody');for(const m of c.members){const row=document.createElement('tr');row.append(text('td',m.name),text('td',m.id));body.append(row)}members.append(body);section.append(text('h3','Members'),members);const meta=document.createElement('div');meta.className='meta';meta.append(text('strong',`Documents affected: ${c.source_document_count}`));const status=text('span','Cluster status: merge');status.className='status';meta.append(status);section.append(meta);if(c.merge_history.length){const history=text('p',`Rounds used: ${c.rounds_used} · ${c.merge_history.map(x=>`round ${x.round}: {${x.left_members.join(', ')}} + {${x.right_members.join(', ')}}`).join(' · ')}`);history.className='history';section.append(history)}clusters.append(section)}const decisions=document.querySelector('#decisions');for(const d of data.decisions){const row=document.createElement('tr');row.append(text('td',d.round),text('td',`{${d.left_members.join(', ')}} ↔ {${d.right_members.join(', ')}}`),text('td',d.candidate_score?.toFixed(3)??'—'),text('td',d.typesafe_score?.toFixed(3)??'—'),text('td',d.decision));decisions.append(row)}}).catch(error=>document.querySelector('#summary').textContent=`Unable to load analysis: ${error}`)</script></body></html>""",
        encoding="utf-8",
    )
    scores = [
        item.typesafe_score
        for item in plan.candidate_pairs
        if item.typesafe_score is not None
    ]
    counts = {
        decision: sum(item.decision == decision for item in plan.candidate_pairs)
        for decision in ("reject", "review", "merge")
    }
    return {
        "correspondents": plan.total_correspondents,
        "candidate_pairs": len(rows),
        "rounds_used": plan.rounds_used,
        "max_rounds_reached": plan.max_rounds_reached,
        "counts": counts,
        "scores": scores,
        "paths": {
            "csv": str(output / "scores.csv"),
            "json": str(output / "scores.json"),
            "html": str(output / "index.html"),
        },
    }


def load_merge_plan(path: str) -> CorrespondentMergePlan:
    """Load plan JSON, including legacy plan fields."""
    data = json.loads(
        sys.stdin.read() if path == "-" else Path(path).read_text(encoding="utf-8")
    )
    clusters = [
        PlannedCorrespondentCluster(
            **{
                **x,
                "members": x.get("members", []),
                "merge_history": [
                    MergeHistory(**item) for item in x.get("merge_history", [])
                ],
                "rounds_used": x.get("rounds_used", 0),
                "planned_document_correspondents": {
                    int(document_id): int(correspondent_id)
                    for document_id, correspondent_id in x.get(
                        "planned_document_correspondents", {}
                    ).items()
                },
            }
        )
        for x in data.get("approved_clusters", [])
    ]
    return CorrespondentMergePlan(
        int(data["version"]),
        str(data["generated_at"]),
        str(data["paperless_url"]),
        int(data["total_correspondents"]),
        int(data["total_documents"]),
        bool(data["judge_enabled"]),
        int(data["judged_pair_count"]),
        clusters,
        [OrphanCorrespondent(**x) for x in data.get("orphan_correspondents", [])],
        [CorrespondentPairDecision(**x) for x in data.get("candidate_pairs", [])],
        int(data.get("rounds_used", 0)),
        bool(data.get("max_rounds_reached", False)),
        int(data.get("scanned_max_correspondent_id", 0)),
        (
            int(data["last_processed_correspondent_id"])
            if data.get("last_processed_correspondent_id") is not None
            else None
        ),
        str(data.get("cleanup_mode", "full")),
    )


async def apply_correspondent_merge_plan(
    client: PaperlessClient,
    plan: CorrespondentMergePlan,
    *,
    dry_run: bool = False,
    review_store: CleanupReviewStore | None = None,
) -> dict[str, int]:
    """Apply approved plan operations; analysis itself never calls this."""
    moved = deleted = skipped_clusters = skipped_orphans = skipped_nonempty = (
        renamed
    ) = 0
    skipped_stale_documents = 0
    current_correspondents, current_documents = (
        await client.get_all_correspondents(),
        await client.iter_all_documents_brief(),
    )
    current_names = {
        _paperless_name_key(str(item.get("name") or "")): int(item["id"])
        for item in current_correspondents
        if str(item.get("name") or "").strip()
    }
    current_document_correspondents = {
        int(item["id"]): int(item["correspondent"])
        for item in current_documents
        if item.get("correspondent")
    }

    async def remaining_documents(correspondent_id: int) -> int:
        try:
            return await client.count_documents_for_correspondent(correspondent_id)
        except niquests.HTTPError as exc:
            if getattr(exc.response, "status_code", None) == 404:
                return 0
            raise

    async def delete_if_present(correspondent_id: int) -> None:
        try:
            await client.delete_correspondent(correspondent_id)
        except niquests.HTTPError as exc:
            if getattr(exc.response, "status_code", None) != 404:
                raise

    for cluster in plan.approved_clusters:
        if cluster.status != "approved":
            skipped_clusters += 1
            continue
        if dry_run:
            log.info(
                "DRY RUN: would rename correspondent %d to %s",
                cluster.canonical_id,
                cluster.canonical_name,
            )
        else:
            target_key = _paperless_name_key(cluster.canonical_name)
            name_owner = current_names.get(target_key)
            if name_owner is None or name_owner == cluster.canonical_id:
                await client.patch_correspondent(
                    cluster.canonical_id, {"name": cluster.canonical_name}
                )
                current_names[target_key] = cluster.canonical_id
                renamed += 1
            else:
                log.warning(
                    "Keeping correspondent %d name %r because correspondent %d already owns it",
                    cluster.canonical_id,
                    cluster.canonical_name,
                    name_owner,
                )
        expected_sources = cluster.planned_document_correspondents
        if not expected_sources and len(cluster.merged_ids) == 1:
            expected_sources = {
                document_id: cluster.merged_ids[0]
                for document_id in cluster.planned_document_ids
            }
        for doc in cluster.planned_document_ids:
            if current_document_correspondents.get(doc) != expected_sources.get(doc):
                skipped_stale_documents += 1
                continue
            if not dry_run:
                await client.patch_document(
                    doc, {"correspondent": cluster.canonical_id}
                )
            moved += 1
        for source in cluster.merged_ids:
            if await remaining_documents(source):
                skipped_nonempty += 1
                continue
            if not dry_run:
                await delete_if_present(source)
            deleted += 1
    for orphan in plan.orphan_correspondents:
        if orphan.status != "approved":
            skipped_orphans += 1
            continue
        if await remaining_documents(orphan.id):
            skipped_nonempty += 1
            continue
        if not dry_run:
            await delete_if_present(orphan.id)
        deleted += 1
    successful = not dry_run and not skipped_nonempty and not skipped_stale_documents
    if successful and review_store and plan.scanned_max_correspondent_id:
        # Update only after every requested Paperless operation has completed.
        # Exceptions propagate before this point, preserving the old watermark.
        await review_store.advance_watermark(plan.scanned_max_correspondent_id)
    return {
        "reassigned_documents": moved,
        "renamed_correspondents": renamed,
        "deleted_correspondents": deleted,
        "skipped_clusters": skipped_clusters,
        "skipped_orphans": skipped_orphans,
        "skipped_nonempty_deletes": skipped_nonempty,
        "skipped_stale_documents": skipped_stale_documents,
    }
