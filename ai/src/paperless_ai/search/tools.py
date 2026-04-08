"""Tool wrappers for the Paperless search copilot."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import litellm
from qdrant_client import AsyncQdrantClient
from qdrant_client import models

from paperless_ai.core.config import AgentConfig
from paperless_ai.core.paperless import PaperlessClient
from paperless_ai.core.telemetry import set_span_attributes, start_span
from paperless_ai.search.embedder_types import SearchEmbedder
from paperless_ai.search.qdrant_store import COLLECTION
from paperless_ai.search.retriever import (
    RETRIEVAL_MODE_DENSE_K,
    RetrievalMode,
    SearchFilters,
    _extract_qdrant_hits,
    hybrid_retrieve,
)

JUDGE_DOC_MAX_CHARS = 12000
JUDGE_BATCH_SIZE = 5
VALID_RETRIEVAL_MODES = tuple(RETRIEVAL_MODE_DENSE_K)


@dataclass
class ToolSourceRef:
    doc_id: int
    source_type: str


@dataclass
class ToolExecutionResult:
    content: str
    summary: str
    preview: str
    source_refs: list[ToolSourceRef] = field(default_factory=list)


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_available_metadata",
            "description": (
                "Return the exact correspondent, document type, storage path, and tag names "
                "available in Paperless. Use this before applying metadata filters."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": (
                "Search Paperless documents using semantic search with optional exact metadata "
                "filters for correspondent, document type, storage path, tags, and year. "
                "Use mode=precision for singular lookups and mode=recall for exhaustive lists. "
                "When mode=recall, always provide an explicit limit."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "correspondent": {"type": "string"},
                    "document_type": {"type": "string"},
                    "storage_path": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "year": {"type": "string", "description": "4-digit year"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                    "mode": {"type": "string", "enum": ["precision", "recall"]},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_full_document",
            "description": "Read the OCR text of a specific Paperless document by ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_id": {"type": "integer"},
                    "max_chars": {"type": "integer", "minimum": 500, "maximum": 20000},
                },
                "required": ["doc_id"],
            },
        },
    },
]


def _snippet(text: str, limit: int = 280) -> str:
    clean = " ".join((text or "").split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 3].rstrip() + "..."


def _chat_completion_kwargs(config: AgentConfig, messages: list[dict[str, Any]]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": config.chat_model,
        "messages": messages,
        **config.get_chat_litellm_kwargs(),
    }
    if "temperature" not in kwargs:
        kwargs["temperature"] = 0.0
    kwargs.pop("tools", None)
    kwargs.pop("tool_choice", None)
    if config.chat_api_base:
        kwargs["api_base"] = config.chat_api_base
    return kwargs


def _format_hit(payload: dict[str, Any]) -> str:
    doc_id = payload.get("doc_id", "?")
    parts = [f"Doc {doc_id}"]
    if payload.get("title"):
        parts.append(str(payload["title"]))
    if payload.get("correspondent"):
        parts.append(str(payload["correspondent"]))
    if payload.get("date"):
        parts.append(str(payload["date"]))
    if payload.get("document_type"):
        parts.append(f"Type: {payload['document_type']}")
    if payload.get("storage_path"):
        parts.append(f"Path: {payload['storage_path']}")
    tags = payload.get("tags") or []
    if tags:
        parts.append(f"Tags: {', '.join(tags)}")
    return f"[{' | '.join(parts)}]: {_snippet(str(payload.get('text') or ''))}"


async def _lookup_payloads(qdrant_url: str, doc_ids: list[int], limit: int) -> dict[int, dict[str, Any]]:
    with start_span(
        "paperless_ai.search.lookup_payloads",
        **{
            "paperless_ai.search.lookup_doc_count": len(doc_ids),
        },
    ) as span:
        qdrant = AsyncQdrantClient(url=qdrant_url)
        try:
            hits = ([], None)
            if doc_ids:
                hits = await qdrant.scroll(
                    collection_name=COLLECTION,
                    scroll_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="doc_id",
                                match=models.MatchAny(any=doc_ids),
                            )
                        ]
                    ),
                    with_payload=True,
                    limit=max(limit * 8, 20),
                )
        finally:
            await qdrant.close()
        payload_by_doc_id: dict[int, dict[str, Any]] = {}
        for point in _extract_qdrant_hits(hits):
            if not point.payload or point.payload.get("doc_id") is None:
                continue
            payload_by_doc_id.setdefault(int(point.payload["doc_id"]), point.payload)
        set_span_attributes(span, **{"paperless_ai.search.lookup_payload_count": len(payload_by_doc_id)})
        return payload_by_doc_id


async def _judge_precision_documents(
    *,
    query: str,
    doc_ids: list[int],
    client: PaperlessClient,
    config: AgentConfig,
) -> list[int]:
    with start_span(
        "paperless_ai.search.precision_judge",
        **{
            "paperless_ai.search.query": query,
            "paperless_ai.search.judge_doc_count": len(doc_ids),
            "paperless_ai.search.judge_batch_size": JUDGE_BATCH_SIZE,
        },
    ) as span:
        kept_doc_ids: list[int] = []
        failed_batches = 0
        for start in range(0, len(doc_ids), JUDGE_BATCH_SIZE):
            batch = doc_ids[start : start + JUDGE_BATCH_SIZE]
            docs = []
            for doc_id in batch:
                doc = await client.get_document_with_content(int(doc_id))
                if doc is None:
                    continue
                docs.append(
                    {
                        "id": int(doc_id),
                        "title": doc.get("title") or "Untitled",
                        "content": str(doc.get("content") or "").strip()[:JUDGE_DOC_MAX_CHARS],
                    }
                )
            if not docs:
                continue

            prompt = (
                "You are filtering candidate documents for a search query. "
                "Return strict JSON of the form {\"keep_doc_ids\":[...]} containing only document IDs "
                "that are genuinely relevant to the query. Preserve the most relevant order and do not "
                "include explanations.\n\n"
                f"Query: {query}\n\nCandidates:\n{json.dumps(docs, ensure_ascii=True)}"
            )
            try:
                response = await litellm.acompletion(
                    **_chat_completion_kwargs(
                        config,
                        [
                            {"role": "system", "content": "Return only strict JSON."},
                            {"role": "user", "content": prompt},
                        ],
                    )
                )
                raw = str(response.choices[0].message.content or "").strip()
                parsed = json.loads(raw)
                keep_doc_ids = [int(doc_id) for doc_id in parsed.get("keep_doc_ids", [])]
                keep_set = set(batch)
                kept_doc_ids.extend([doc_id for doc_id in keep_doc_ids if doc_id in keep_set])
            except Exception:
                failed_batches += 1
                kept_doc_ids.extend(batch)
        set_span_attributes(
            span,
            **{
                "paperless_ai.search.judge_kept_doc_count": len(kept_doc_ids),
                "paperless_ai.search.judge_failed_batches": failed_batches,
            },
        )
        return kept_doc_ids


async def search_documents(
    query: str,
    *,
    embedder: SearchEmbedder,
    qdrant_url: str,
    config: AgentConfig,
    correspondent: str | None = None,
    document_type: str | None = None,
    storage_path: str | None = None,
    tags: list[str] | None = None,
    year: str | None = None,
    limit: int | None = None,
    mode: RetrievalMode = "precision",
    client: PaperlessClient | None = None,
) -> ToolExecutionResult:
    """Run hybrid retrieval against Qdrant and Paperless and return formatted snippets."""
    with start_span(
        "paperless_ai.tool.search_documents",
        **{
            "paperless_ai.tool.mode": mode,
            "paperless_ai.tool.limit": limit,
            "paperless_ai.search.query": query,
        },
    ) as span:
        if mode not in VALID_RETRIEVAL_MODES:
            content = (
                f"Invalid search mode {mode!r}. "
                f"Allowed values: {', '.join(VALID_RETRIEVAL_MODES)}."
            )
            set_span_attributes(span, **{"paperless_ai.tool.validation_error": content})
            return ToolExecutionResult(content=content, summary=content, preview=content)
        if mode == "recall" and limit is None:
            content = "Recall searches require an explicit limit."
            set_span_attributes(span, **{"paperless_ai.tool.validation_error": content})
            return ToolExecutionResult(content=content, summary=content, preview=content)
        resolved_limit = 20 if limit is None else limit
        filters = SearchFilters(
            correspondent=correspondent,
            document_type=document_type,
            storage_path=storage_path,
            tags=tags,
            year=year,
        )
        fused_ids, chunk_map = await hybrid_retrieve(
            embedder=embedder,
            qdrant_url=qdrant_url,
            query=query,
            client=client,
            filters=filters,
            dense_k=RETRIEVAL_MODE_DENSE_K[mode],
            rerank_candidates=max(resolved_limit, 50),
            mode=mode,
        )
        set_span_attributes(span, **{"paperless_ai.tool.prejudge_doc_count": len(fused_ids)})
        if mode == "precision" and fused_ids and client is not None:
            fused_ids = await _judge_precision_documents(
                query=query,
                doc_ids=fused_ids,
                client=client,
                config=config,
            )
        if not fused_ids:
            return ToolExecutionResult(
                content="No matching documents found.",
                summary="No documents matched the search.",
                preview="No matching documents found.",
            )

        payload_by_doc_id = await _lookup_payloads(qdrant_url, fused_ids[:resolved_limit], resolved_limit)
        formatted: list[str] = []
        source_refs: list[ToolSourceRef] = []
        for doc_id in fused_ids[:resolved_limit]:
            payload = payload_by_doc_id.get(int(doc_id), {"doc_id": int(doc_id), "text": chunk_map.get(int(doc_id), "")})
            formatted.append(_format_hit(payload))
            source_refs.append(ToolSourceRef(doc_id=int(doc_id), source_type="search"))

        content = "\n".join(formatted) if formatted else "No matching documents found."
        set_span_attributes(
            span,
            **{
                "paperless_ai.tool.final_doc_count": len(fused_ids),
                "paperless_ai.tool.returned_doc_count": len(source_refs),
            },
        )
        if not formatted:
            return ToolExecutionResult(
                content=content,
                summary="No documents matched the search.",
                preview=content,
            )
        return ToolExecutionResult(
            content=content,
            summary=f"Found {len(source_refs)} matching document(s).",
            preview=_snippet(content, limit=420),
            source_refs=source_refs,
        )


async def read_full_document(
    doc_id: int,
    *,
    client: PaperlessClient,
    max_chars: int = 8000,
) -> ToolExecutionResult:
    """Read the OCR text for a single Paperless document."""
    with start_span(
        "paperless_ai.tool.read_full_document",
        **{
            "paperless_ai.tool.doc_id": doc_id,
            "paperless_ai.tool.max_chars": max_chars,
        },
    ) as span:
        doc = await client.get_document_with_content(int(doc_id))
        if doc is None:
            content = f"Document {doc_id} was not found."
            return ToolExecutionResult(content=content, summary=content, preview=content)

        content = str(doc.get("content") or "").strip()
        set_span_attributes(span, **{"paperless_ai.tool.document_char_count": len(content)})
        if not content:
            empty = f"Document {doc_id} has no OCR content."
            return ToolExecutionResult(content=empty, summary=empty, preview=empty)

        title = doc.get("title") or "Untitled"
        if len(content) > max_chars:
            content = content[:max_chars].rstrip() + "\n\n[truncated]"
        result = f"[Doc {doc_id} | {title}]\n{content}"
        return ToolExecutionResult(
            content=result,
            summary=f"Read OCR text for document {doc_id}.",
            preview=_snippet(result, limit=420),
            source_refs=[ToolSourceRef(doc_id=int(doc_id), source_type="read")],
        )


async def get_available_metadata(*, client: PaperlessClient) -> ToolExecutionResult:
    """Return exact Paperless metadata names for agent-side discovery."""
    with start_span("paperless_ai.tool.get_available_metadata") as span:
        metadata = await client.get_available_metadata()
        content = "\n".join(
            [
                f"Available Correspondents: {', '.join(metadata['correspondents']) or '(none)'}",
                f"Available Document Types: {', '.join(metadata['document_types']) or '(none)'}",
                f"Available Storage Paths: {', '.join(metadata['storage_paths']) or '(none)'}",
                f"Available Tags: {', '.join(metadata['tags']) or '(none)'}",
            ]
        )
        summary = (
            "Loaded metadata names "
            f"({len(metadata['correspondents'])} correspondents, "
            f"{len(metadata['document_types'])} document types, "
            f"{len(metadata['storage_paths'])} storage paths, "
            f"{len(metadata['tags'])} tags)."
        )
        set_span_attributes(
            span,
            **{
                "paperless_ai.tool.correspondent_count": len(metadata["correspondents"]),
                "paperless_ai.tool.document_type_count": len(metadata["document_types"]),
                "paperless_ai.tool.storage_path_count": len(metadata["storage_paths"]),
                "paperless_ai.tool.tag_count": len(metadata["tags"]),
            },
        )
        return ToolExecutionResult(content=content, summary=summary, preview=_snippet(content, limit=420))


async def execute_tool_call(
    name: str,
    arguments: dict[str, Any],
    *,
    client: PaperlessClient,
    embedder: SearchEmbedder,
    qdrant_url: str,
    config: AgentConfig,
) -> str:
    """Compatibility wrapper returning only the tool content."""
    result = await execute_tool_call_detailed(
        name,
        arguments,
        client=client,
        embedder=embedder,
        qdrant_url=qdrant_url,
        config=config,
    )
    return result.content


async def execute_tool_call_detailed(
    name: str,
    arguments: dict[str, Any],
    *,
    client: PaperlessClient,
    embedder: SearchEmbedder,
    qdrant_url: str,
    config: AgentConfig,
) -> ToolExecutionResult:
    """Dispatch a tool call with UI-friendly metadata for the chat frontend."""
    if name == "get_available_metadata":
        return await get_available_metadata(client=client)
    if name == "search_documents":
        return await search_documents(
            arguments.get("query", ""),
            embedder=embedder,
            qdrant_url=qdrant_url,
            config=config,
            client=client,
            correspondent=arguments.get("correspondent"),
            document_type=arguments.get("document_type"),
            storage_path=arguments.get("storage_path"),
            tags=arguments.get("tags"),
            year=arguments.get("year"),
            limit=(None if "limit" not in arguments else int(arguments["limit"])),
            mode=str(arguments.get("mode", "precision")),
        )
    if name == "read_full_document":
        return await read_full_document(
            int(arguments["doc_id"]),
            client=client,
            max_chars=int(arguments.get("max_chars", 8000)),
        )
    raise ValueError(f"Unknown tool: {name}")


def parse_tool_arguments(raw_arguments: Any) -> dict[str, Any]:
    """Parse the function arguments returned by the LLM."""
    if raw_arguments is None:
        return {}
    if isinstance(raw_arguments, dict):
        return raw_arguments
    if isinstance(raw_arguments, str):
        return json.loads(raw_arguments) if raw_arguments.strip() else {}
    raise TypeError(f"Unsupported tool argument type: {type(raw_arguments).__name__}")
