"""Tool wrappers for the Paperless search copilot."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client import models

from paperless_ai.core.paperless import PaperlessClient
from paperless_ai.search.embedder import LocalLazySearchEmbedder
from paperless_ai.search.qdrant_store import COLLECTION
from paperless_ai.search.retriever import (
    SearchFilters,
    _extract_qdrant_hits,
    hybrid_retrieve,
)


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
                "filters for correspondent, document type, storage path, tags, and year."
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


async def search_documents(
    query: str,
    *,
    embedder: LocalLazySearchEmbedder,
    qdrant_url: str,
    correspondent: str | None = None,
    document_type: str | None = None,
    storage_path: str | None = None,
    tags: list[str] | None = None,
    year: str | None = None,
    limit: int = 5,
    client: PaperlessClient | None = None,
    rerank_model: str | None = None,
    rerank_api_base: str | None = None,
) -> str:
    """Run hybrid retrieval against Qdrant and Paperless and return formatted snippets."""
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
        rerank_model=rerank_model,
        rerank_api_base=rerank_api_base,
    )
    if not fused_ids:
        return "No matching documents found."

    qdrant = AsyncQdrantClient(url=qdrant_url)
    try:
        seen_doc_ids: set[int] = set()
        formatted: list[str] = []
        doc_ids = fused_ids[:limit]
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
        else:
            hits = ([], None)
    finally:
        await qdrant.close()

    points = _extract_qdrant_hits(hits)
    payload_by_doc_id = {
        int(point.payload["doc_id"]): point.payload
        for point in points
        if point.payload and point.payload.get("doc_id") is not None
    }

    for doc_id in fused_ids:
        if doc_id in seen_doc_ids or len(formatted) >= limit:
            continue
        seen_doc_ids.add(int(doc_id))
        payload = payload_by_doc_id.get(int(doc_id), {"doc_id": int(doc_id), "text": chunk_map.get(int(doc_id), "")})
        formatted.append(_format_hit(payload))

    if not formatted:
        return "No matching documents found."
    return "\n".join(formatted)


async def search_documents_detailed(
    query: str,
    *,
    embedder: LocalLazySearchEmbedder,
    qdrant_url: str,
    correspondent: str | None = None,
    document_type: str | None = None,
    storage_path: str | None = None,
    tags: list[str] | None = None,
    year: str | None = None,
    limit: int = 5,
    client: PaperlessClient | None = None,
    rerank_model: str | None = None,
    rerank_api_base: str | None = None,
) -> ToolExecutionResult:
    """Run hybrid search and return both prompt text and UI metadata."""
    content = await search_documents(
        query,
        embedder=embedder,
        qdrant_url=qdrant_url,
        correspondent=correspondent,
        document_type=document_type,
        storage_path=storage_path,
        tags=tags,
        year=year,
        limit=limit,
        client=client,
        rerank_model=rerank_model,
        rerank_api_base=rerank_api_base,
    )
    if content == "No matching documents found.":
        return ToolExecutionResult(
            content=content,
            summary="No documents matched the search.",
            preview="No matching documents found.",
        )
    source_refs = []
    for line in content.splitlines():
        if line.startswith("[Doc "):
            try:
                doc_id = int(line.split()[1])
            except (ValueError, IndexError):
                continue
            source_refs.append(ToolSourceRef(doc_id=doc_id, source_type="search"))
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
) -> str:
    """Read the OCR text for a single Paperless document."""
    doc = await client.get_document_with_content(int(doc_id))
    if doc is None:
        return f"Document {doc_id} was not found."

    content = str(doc.get("content") or "").strip()
    if not content:
        return f"Document {doc_id} has no OCR content."

    title = doc.get("title") or "Untitled"
    if len(content) > max_chars:
        content = content[:max_chars].rstrip() + "\n\n[truncated]"
    return f"[Doc {doc_id} | {title}]\n{content}"


async def read_full_document_detailed(
    doc_id: int,
    *,
    client: PaperlessClient,
    max_chars: int = 8000,
) -> ToolExecutionResult:
    """Read OCR text and return both prompt text and UI metadata."""
    content = await read_full_document(doc_id, client=client, max_chars=max_chars)
    if content.startswith("Document ") and (
        content.endswith(" was not found.") or content.endswith(" has no OCR content.")
    ):
        return ToolExecutionResult(content=content, summary=content, preview=content)
    return ToolExecutionResult(
        content=content,
        summary=f"Read OCR text for document {doc_id}.",
        preview=_snippet(content, limit=420),
        source_refs=[ToolSourceRef(doc_id=int(doc_id), source_type="read")],
    )


async def get_available_metadata(*, client: PaperlessClient) -> str:
    """Return exact Paperless metadata names for agent-side discovery."""
    metadata = await client.get_available_metadata()
    return "\n".join(
        [
            f"Available Correspondents: {', '.join(metadata['correspondents']) or '(none)'}",
            f"Available Document Types: {', '.join(metadata['document_types']) or '(none)'}",
            f"Available Storage Paths: {', '.join(metadata['storage_paths']) or '(none)'}",
            f"Available Tags: {', '.join(metadata['tags']) or '(none)'}",
        ]
    )


async def get_available_metadata_detailed(*, client: PaperlessClient) -> ToolExecutionResult:
    """Return exact Paperless metadata names with a concise summary for the UI."""
    content = await get_available_metadata(client=client)
    metadata = await client.get_available_metadata()
    summary = (
        "Loaded metadata names "
        f"({len(metadata['correspondents'])} correspondents, "
        f"{len(metadata['document_types'])} document types, "
        f"{len(metadata['storage_paths'])} storage paths, "
        f"{len(metadata['tags'])} tags)."
    )
    return ToolExecutionResult(content=content, summary=summary, preview=_snippet(content, limit=420))


async def execute_tool_call(
    name: str,
    arguments: dict[str, Any],
    *,
    client: PaperlessClient,
    embedder: LocalLazySearchEmbedder,
    qdrant_url: str,
) -> str:
    """Dispatch a tool call from the chat agent."""
    if name == "get_available_metadata":
        return await get_available_metadata(client=client)
    if name == "search_documents":
        return await search_documents(
            arguments.get("query", ""),
            embedder=embedder,
            qdrant_url=qdrant_url,
            client=client,
            correspondent=arguments.get("correspondent"),
            document_type=arguments.get("document_type"),
            storage_path=arguments.get("storage_path"),
            tags=arguments.get("tags"),
            year=arguments.get("year"),
            limit=int(arguments.get("limit", 5)),
        )
    if name == "read_full_document":
        return await read_full_document(
            int(arguments["doc_id"]),
            client=client,
            max_chars=int(arguments.get("max_chars", 8000)),
        )
    raise ValueError(f"Unknown tool: {name}")


async def execute_tool_call_detailed(
    name: str,
    arguments: dict[str, Any],
    *,
    client: PaperlessClient,
    embedder: LocalLazySearchEmbedder,
    qdrant_url: str,
    rerank_model: str | None = None,
    rerank_api_base: str | None = None,
) -> ToolExecutionResult:
    """Dispatch a tool call with UI-friendly metadata for the chat frontend."""
    if name == "get_available_metadata":
        return await get_available_metadata_detailed(client=client)
    if name == "search_documents":
        return await search_documents_detailed(
            arguments.get("query", ""),
            embedder=embedder,
            qdrant_url=qdrant_url,
            client=client,
            correspondent=arguments.get("correspondent"),
            document_type=arguments.get("document_type"),
            storage_path=arguments.get("storage_path"),
            tags=arguments.get("tags"),
            year=arguments.get("year"),
            limit=int(arguments.get("limit", 5)),
            rerank_model=rerank_model,
            rerank_api_base=rerank_api_base,
        )
    if name == "read_full_document":
        return await read_full_document_detailed(
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
