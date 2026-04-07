"""Tool wrappers for the Paperless search copilot."""

from __future__ import annotations

import json
from typing import Any

from qdrant_client import AsyncQdrantClient

from paperless_ai.core.paperless import PaperlessClient
from paperless_ai.search.embedder import LocalLazySearchEmbedder
from paperless_ai.search.qdrant_store import COLLECTION
from paperless_ai.search.retriever import (
    SearchFilters,
    _extract_qdrant_hits,
    build_qdrant_filter,
)

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
                    "limit": {"type": "integer", "minimum": 1, "maximum": 10},
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
) -> str:
    """Run filtered semantic search against Qdrant and return formatted snippets."""
    result = await embedder.embed_query(query)
    filters = SearchFilters(
        correspondent=correspondent,
        document_type=document_type,
        storage_path=storage_path,
        tags=tags,
        year=year,
    )

    qdrant = AsyncQdrantClient(url=qdrant_url)
    try:
        hits = await qdrant.query_points(
            collection_name=COLLECTION,
            query=result.dense,
            using="dense",
            limit=max(1, min(limit * 4, 40)),
            with_payload=True,
            query_filter=build_qdrant_filter(filters),
        )
    finally:
        await qdrant.close()
    hits = _extract_qdrant_hits(hits)

    seen_doc_ids: set[int] = set()
    formatted: list[str] = []
    for hit in hits:
        payload = hit.payload or {}
        doc_id = payload.get("doc_id")
        if doc_id is None or doc_id in seen_doc_ids:
            continue
        seen_doc_ids.add(int(doc_id))
        formatted.append(_format_hit(payload))
        if len(formatted) >= limit:
            break

    if not formatted:
        return "No matching documents found."
    return "\n".join(formatted)


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


def parse_tool_arguments(raw_arguments: Any) -> dict[str, Any]:
    """Parse the function arguments returned by the LLM."""
    if raw_arguments is None:
        return {}
    if isinstance(raw_arguments, dict):
        return raw_arguments
    if isinstance(raw_arguments, str):
        return json.loads(raw_arguments) if raw_arguments.strip() else {}
    raise TypeError(f"Unsupported tool argument type: {type(raw_arguments).__name__}")
