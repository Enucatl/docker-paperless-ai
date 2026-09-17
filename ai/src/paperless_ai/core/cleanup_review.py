"""Authenticated browser review and apply workflow for correspondent cleanup."""

import asyncio
import hashlib
import os
import signal
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
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


app = FastAPI(title="Correspondent cleanup review")


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return """<!doctype html><meta charset=utf-8><title>Correspondent cleanup review</title>
<style>body{max-width:1000px;margin:2rem auto;font:16px system-ui;color:#18212f;padding:0 1rem}.card{border:1px solid #d9e0ea;border-radius:8px;padding:1rem;margin:1rem 0}button{margin-right:.5rem;padding:.4rem .7rem;border:1px solid #cbd5e1;border-radius:5px;background:#fff;cursor:pointer}button.approved{background:#dcfce7;border-color:#22c55e;color:#166534;font-weight:700}button.rejected{background:#fee2e2;border-color:#ef4444;color:#991b1b;font-weight:700}button:disabled{cursor:default;opacity:.8}pre{white-space:pre-wrap;background:#f6f7fb;padding:1rem}.muted{color:#64748b}</style>
<h1>Correspondent cleanup review</h1><p id=status class=muted>Loading…</p><p><input id=confirmation placeholder="Type APPLY to apply"/><button id=apply>Apply reviewed plan</button></p><main id=pairs></main><h2>Reviewed plan</h2><pre id=plan>Resolve review decisions to generate the plan.</pre>
<script>
let current; const status=document.querySelector('#status'), pairs=document.querySelector('#pairs'), output=document.querySelector('#plan');
const refresh=async()=>{const r=await fetch('/api/plan');if(!r.ok){status.textContent=await r.text();return}current=await r.json();const recorded=new Map(current.review_decisions.map(x=>[x.pair_key,x.decision]));const count=recorded.size;status.textContent=`${current.cleanup_mode} cleanup · snapshot max ID ${current.scanned_max_correspondent_id} · ${current.review_candidates.length} actionable review pair(s) · ${count} decision(s) recorded`;pairs.replaceChildren();for(const p of current.review_candidates){const c=document.createElement('section');c.className='card';const key=[...p.left_members].sort().join(':')+'|'+[...p.right_members].sort().join(':');const selected=recorded.get(key);c.innerHTML=`<strong>${p.left_name}</strong> (${p.left_members.join(', ')}) ↔ <strong>${p.right_name}</strong> (${p.right_members.join(', ')})<p class=muted>${p.reason||''} · candidate score ${p.candidate_score??'—'}</p>`;for(const choice of ['approve','reject']){const b=document.createElement('button');b.textContent=selected===choice?(choice==='approve'?'Approved':'Rejected'):choice;b.className=selected===choice?(choice==='approve'?'approved':'rejected'):'';b.onclick=async()=>{const response=await fetch('/api/decisions',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({pair_key:key,decision:choice})});if(!response.ok){alert((await response.json()).detail||'Unable to record decision');return}status.textContent=`Decision recorded: ${choice}`;await showPlan();await refresh()};c.append(b)}pairs.append(c)};await showPlan()};
const showPlan=async()=>{const r=await fetch('/api/reviewed-plan');output.textContent=r.ok?JSON.stringify(await r.json(),null,2):await r.text()};
document.querySelector('#apply').onclick=async()=>{const r=await fetch('/api/apply',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({confirmation:document.querySelector('#confirmation').value})});const text=await r.text();let body;try{body=JSON.parse(text)}catch{body={detail:text}}if(!r.ok){status.textContent=body.detail||'Apply failed';alert(body.detail||text);return}status.textContent=`Apply completed: ${body.reassigned_documents} documents reassigned, ${body.deleted_correspondents} correspondents deleted · watermark ${body.watermark_id}`;output.textContent=JSON.stringify(body,null,2);document.querySelector('#apply').disabled=true};refresh();
</script>"""


@app.get("/api/plan")
async def plan() -> dict[str, Any]:
    item = _load_plan()
    data = item.to_dict()
    store = await CleanupReviewStore.from_config(AgentConfig.from_env())
    data["plan_id"] = _plan_id(item)
    data["review_decisions"] = await store.decisions_for_plan(_plan_id(item))
    data[
        "last_processed_correspondent_id"
    ] = await store.get_last_processed_correspondent_id()
    data["apply_complete"] = bool(
        item.scanned_max_correspondent_id
        and data["last_processed_correspondent_id"] is not None
        and data["last_processed_correspondent_id"] >= item.scanned_max_correspondent_id
    )
    data["review_candidates"] = [
        asdict(candidate) for candidate in actionable_review_pairs(item)
    ]
    return data


@app.post("/api/decisions")
async def decide(request: DecisionRequest) -> dict[str, Any]:
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
    store = await CleanupReviewStore.from_config(AgentConfig.from_env())
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
async def reviewed_plan() -> dict[str, Any]:
    plan = _load_plan()
    store = await CleanupReviewStore.from_config(AgentConfig.from_env())
    watermark = await store.get_last_processed_correspondent_id()
    if (
        plan.scanned_max_correspondent_id
        and watermark is not None
        and watermark >= plan.scanned_max_correspondent_id
    ):
        data = plan.to_dict()
        data["apply_complete"] = True
        data["last_processed_correspondent_id"] = watermark
        return data
    return (await _reviewed_plan(plan, store)).to_dict()


@app.post("/api/apply")
async def apply(
    request: ApplyRequest, background_tasks: BackgroundTasks
) -> dict[str, int]:
    if request.confirmation != "APPLY":
        raise HTTPException(400, "Type APPLY exactly to confirm")
    plan = _load_plan()
    config = AgentConfig.from_env()
    store = await CleanupReviewStore.from_config(config)
    watermark = await store.get_last_processed_correspondent_id()
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
    result["watermark_id"] = await store.get_last_processed_correspondent_id() or 0
    background_tasks.add_task(_shutdown_after_apply)
    return result
