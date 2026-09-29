"""Authenticated browser review and apply workflow for correspondent cleanup."""

import asyncio
import hashlib
import os
import signal
from contextlib import asynccontextmanager
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from paperless_ai.core.config import AgentConfig
from paperless_ai.core.correspondent_cleanup import (
    CleanupReviewStore,
    CorrespondentMergePlan,
    PlannedCorrespondentCluster,
    apply_correspondent_merge_plan,
    actionable_review_pairs,
    load_merge_plan,
)
from paperless_common.paperless import PaperlessClient

PLAN_PATH = Path("/review/merge-plan.json")


def _plan_id(plan: CorrespondentMergePlan) -> str:
    payload = f"{plan.generated_at}:{plan.scanned_max_correspondent_id}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _pair_key(left_members: list[int], right_members: list[int]) -> str:
    return (
        ":".join(map(str, sorted(left_members)))
        + "|"
        + ":".join(map(str, sorted(right_members)))
    )


def _load_plan() -> CorrespondentMergePlan:
    if not PLAN_PATH.exists():
        raise HTTPException(404, "No merge plan has been generated yet")
    return load_merge_plan(str(PLAN_PATH))


class DecisionRequest(BaseModel):
    pair_key: str
    decision: str


class ApplyRequest(BaseModel):
    confirmation: str


async def _shutdown_after_apply() -> None:
    """Stop the one-shot review container after its response is sent."""
    await asyncio.sleep(1)
    os.kill(os.getpid(), signal.SIGTERM)


async def _reviewed_plan(
    plan: CorrespondentMergePlan, store: CleanupReviewStore
) -> CorrespondentMergePlan:
    """Add disjoint manual approvals as exact, current Paperless operations."""
    decisions = await store.decisions_for_plan(_plan_id(plan))
    active_keys = {
        _pair_key(item.left_members, item.right_members)
        for item in actionable_review_pairs(plan)
    }
    approved = [
        item["evidence"]
        for item in decisions
        if item["decision"] == "approve" and item["pair_key"] in active_keys
    ]
    if not approved:
        return plan

    # Overlapping approvals intentionally form one connected consolidation.
    # For example, A↔B plus B↔C yields the exact reviewed group A,B,C.
    groups: list[set[int]] = []
    for item in approved:
        members = set(item["left_members"] + item["right_members"])
        overlapping = [group for group in groups if group & members]
        merged = members.copy()
        for group in overlapping:
            merged |= group
            groups.remove(group)
        groups.append(merged)

    manual_members = {
        member_id
        for item in approved
        for member_id in item["left_members"] + item["right_members"]
    }
    # A manual pair is authoritative for its exact displayed members. If a
    # later automatic round also consumed one of those members, defer that
    # broader automatic cluster rather than silently widening the manual merge.

    config = AgentConfig.from_env()
    async with PaperlessClient(config.paperless_url, config.paperless_token) as client:
        correspondents, documents = (
            await client.get_all_correspondents(),
            await client.iter_all_documents_brief(),
        )
    by_id = {int(item["id"]): item for item in correspondents}
    docs_by_id: dict[int, list[int]] = {}
    for document in documents:
        correspondent = document.get("correspondent")
        if correspondent:
            docs_by_id.setdefault(int(correspondent), []).append(int(document["id"]))

    clusters = [
        cluster
        for cluster in plan.approved_clusters
        if not ({cluster.canonical_id, *cluster.merged_ids} & manual_members)
    ]
    for group in groups:
        member_ids = sorted(group)
        if any(member_id not in by_id for member_id in member_ids):
            raise HTTPException(409, "A reviewed correspondent no longer exists")
        survivor = max(
            member_ids,
            key=lambda member_id: (len(docs_by_id.get(member_id, [])), -member_id),
        )
        merged_ids = [member_id for member_id in member_ids if member_id != survivor]
        clusters.append(
            PlannedCorrespondentCluster(
                canonical_id=survivor,
                canonical_name=str(by_id[survivor].get("name") or "").strip(),
                canonical_document_count=len(docs_by_id.get(survivor, [])),
                merged_ids=merged_ids,
                merged_names=[
                    str(by_id[item].get("name") or "") for item in merged_ids
                ],
                source_document_count=sum(
                    len(docs_by_id.get(item, [])) for item in merged_ids
                ),
                planned_document_ids=sorted(
                    doc_id for item in merged_ids for doc_id in docs_by_id.get(item, [])
                ),
                confidence="manual",
                status="approved",
                reasons=["manual_review_approval"],
                members=[
                    {"id": item, "name": str(by_id[item].get("name") or "")}
                    for item in member_ids
                ],
                planned_document_correspondents={
                    document_id: item
                    for item in merged_ids
                    for document_id in docs_by_id.get(item, [])
                },
            )
        )
    # A manual merge owns its members; do not also delete an approved orphan.
    orphans = [
        orphan
        for orphan in plan.orphan_correspondents
        if orphan.id not in manual_members
    ]
    return replace(plan, approved_clusters=clusters, orphan_correspondents=orphans)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize cleanup persistence once for all request handlers."""
    store = await CleanupReviewStore.from_config(AgentConfig.from_env())
    yield {"review_store": store}


app = FastAPI(title="Correspondent cleanup review", lifespan=lifespan)
app.mount(
    "/assets", StaticFiles(directory=Path(__file__).with_name("assets")), name="assets"
)


@app.get("/")
async def index() -> FileResponse:
    """Serve the packaged cleanup browser UI."""
    return FileResponse(Path(__file__).with_name("assets") / "cleanup.html")


@app.get("/api/plan")
async def plan(http_request: Request) -> dict[str, Any]:
    item = _load_plan()
    data = item.to_dict()
    store = http_request.state.review_store
    data["plan_id"] = _plan_id(item)
    data["review_decisions"] = await store.decisions_for_plan(_plan_id(item))
    data[
        "last_manually_reviewed_correspondent_id"
    ] = await store.get_last_manually_reviewed_correspondent_id()
    data["apply_complete"] = bool(
        item.scanned_max_correspondent_id
        and data["last_manually_reviewed_correspondent_id"] is not None
        and data["last_manually_reviewed_correspondent_id"]
        >= item.scanned_max_correspondent_id
    )
    data["review_candidates"] = [
        {
            **asdict(candidate),
            "pair_key": _pair_key(candidate.left_members, candidate.right_members),
        }
        for candidate in actionable_review_pairs(item)
    ]
    return data


@app.post("/api/decisions")
async def decide(request: DecisionRequest, http_request: Request) -> dict[str, Any]:
    plan = _load_plan()
    candidate = next(
        (
            item
            for item in actionable_review_pairs(plan)
            if _pair_key(item.left_members, item.right_members) == request.pair_key
            and item.decision == "review"
        ),
        None,
    )
    if candidate is None:
        raise HTTPException(404, "Unknown review pair")
    store = http_request.state.review_store
    try:
        await store.record_decision(
            _plan_id(plan),
            request.pair_key,
            request.decision,
            {
                "left_members": candidate.left_members,
                "right_members": candidate.right_members,
                "pair": asdict(candidate),
            },
        )
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"ok": True}


@app.get("/api/reviewed-plan")
async def reviewed_plan(http_request: Request) -> dict[str, Any]:
    plan = _load_plan()
    store = http_request.state.review_store
    watermark = await store.get_last_manually_reviewed_correspondent_id()
    if (
        plan.scanned_max_correspondent_id
        and watermark is not None
        and watermark >= plan.scanned_max_correspondent_id
    ):
        data = plan.to_dict()
        data["apply_complete"] = True
        data["last_manually_reviewed_correspondent_id"] = watermark
        return data
    return (await _reviewed_plan(plan, store)).to_dict()


@app.post("/api/apply")
async def apply(
    request: ApplyRequest, background_tasks: BackgroundTasks, http_request: Request
) -> dict[str, int]:
    if request.confirmation != "APPLY":
        raise HTTPException(400, "Type APPLY exactly to confirm")
    plan = _load_plan()
    config = AgentConfig.from_env()
    store = http_request.state.review_store
    watermark = await store.get_last_manually_reviewed_correspondent_id()
    if (
        plan.scanned_max_correspondent_id
        and watermark is not None
        and watermark >= plan.scanned_max_correspondent_id
    ):
        raise HTTPException(409, "This cleanup plan has already been applied")
    decisions = {
        item["pair_key"]: item["decision"]
        for item in await store.decisions_for_plan(_plan_id(plan))
    }
    pending = [
        _pair_key(item.left_members, item.right_members)
        for item in actionable_review_pairs(plan)
        if _pair_key(item.left_members, item.right_members) not in decisions
    ]
    if pending:
        raise HTTPException(
            409,
            f"{len(pending)} final-round review pair(s) still need approve or reject",
        )
    reviewed = await _reviewed_plan(plan, store)
    async with PaperlessClient(config.paperless_url, config.paperless_token) as client:
        result = await apply_correspondent_merge_plan(
            client, reviewed, review_store=store
        )
    result["manual_review_boundary_id"] = (
        await store.get_last_manually_reviewed_correspondent_id() or 0
    )
    background_tasks.add_task(_shutdown_after_apply)
    return result
