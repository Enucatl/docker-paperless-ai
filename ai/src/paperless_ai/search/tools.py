"""Tool wrappers for the Paperless search copilot."""

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

from paperless_common.paperless import PaperlessClient
from paperless_common.telemetry import set_span_attributes, start_span


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
                "Search Paperless full text using a short, distinctive keyword query. "
                "Paperless supports uppercase AND, OR, NOT, and parentheses; terms without "
                "an operator imply AND. Use OR for alternate terms and quotes for exact phrases. "
                "Avoid natural language sentences. If no results are found, retry with "
                "shorter keywords or a different distinctive term. Optional exact metadata "
                "filters are available for correspondent, document type, storage path, tags, and year. "
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


def _format_hit(doc: dict[str, Any]) -> str:
    return (
        f"[Doc {doc.get('id', '?')} | {doc.get('title') or 'Untitled'}]: "
        f"{_snippet(str(doc.get('content') or ''))}"
    )


async def search_documents(
    query: str,
    *,
    client: PaperlessClient,
    correspondent: str | None = None,
    document_type: str | None = None,
    storage_path: str | None = None,
    tags: list[str] | None = None,
    year: str | None = None,
    limit: int | None = None,
    mode: str = "precision",
) -> ToolExecutionResult:
    """Search Paperless full text and return matching document snippets."""
    with start_span(
        "paperless_ai.tool.search_documents",
        **{
            "paperless_ai.tool.mode": mode,
            "paperless_ai.tool.limit": limit,
            "paperless_ai.search.query": query,
        },
    ) as span:
        if not query.strip():
            content = "Provide one or more search keywords."
            set_span_attributes(span, **{"paperless_ai.tool.validation_error": content})
            return ToolExecutionResult(
                content=content, summary=content, preview=content
            )
        if mode not in ("precision", "recall"):
            content = (
                f"Invalid search mode {mode!r}. Allowed values: precision, recall."
            )
            set_span_attributes(span, **{"paperless_ai.tool.validation_error": content})
            return ToolExecutionResult(
                content=content, summary=content, preview=content
            )
        if mode == "recall" and limit is None:
            content = "Recall searches require an explicit limit."
            set_span_attributes(span, **{"paperless_ai.tool.validation_error": content})
            return ToolExecutionResult(
                content=content, summary=content, preview=content
            )
        if limit is not None and not 1 <= limit <= 100:
            content = "Search limits must be between 1 and 100."
            set_span_attributes(span, **{"paperless_ai.tool.validation_error": content})
            return ToolExecutionResult(
                content=content, summary=content, preview=content
            )

        result_limit = 20 if limit is None else limit
        doc_ids = await client.search_documents_all(
            query,
            limit=result_limit,
            correspondent=correspondent,
            document_type=document_type,
            storage_path=storage_path,
            tags=tags,
            year=year,
        )
        set_span_attributes(
            span, **{"paperless_ai.tool.keyword_doc_count": len(doc_ids)}
        )
        docs = await asyncio.gather(
            *(
                client.get_document_with_content(doc_id)
                for doc_id in doc_ids[:result_limit]
            )
        )
        results = [doc for doc in docs if doc is not None]
        formatted = [_format_hit(doc) for doc in results]
        source_refs = [
            ToolSourceRef(doc_id=int(doc["id"]), source_type="search")
            for doc in results
        ]
        content = "\n".join(formatted) if formatted else "No matching documents found."
        set_span_attributes(
            span,
            **{
                "paperless_ai.tool.final_doc_count": len(results),
                "paperless_ai.tool.returned_doc_count": len(source_refs),
            },
        )
        return ToolExecutionResult(
            content=content,
            summary=f"Found {len(results)} matching document(s)."
            if results
            else "No documents matched the search.",
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
            return ToolExecutionResult(
                content=content, summary=content, preview=content
            )

        content = str(doc.get("content") or "").strip()
        set_span_attributes(
            span, **{"paperless_ai.tool.document_char_count": len(content)}
        )
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
                "paperless_ai.tool.correspondent_count": len(
                    metadata["correspondents"]
                ),
                "paperless_ai.tool.document_type_count": len(
                    metadata["document_types"]
                ),
                "paperless_ai.tool.storage_path_count": len(metadata["storage_paths"]),
                "paperless_ai.tool.tag_count": len(metadata["tags"]),
            },
        )
        return ToolExecutionResult(
            content=content, summary=summary, preview=_snippet(content, limit=420)
        )


async def execute_tool_call(
    name: str,
    arguments: dict[str, Any],
    *,
    client: PaperlessClient,
) -> str:
    """Compatibility wrapper returning only the tool content."""
    return (await execute_tool_call_detailed(name, arguments, client=client)).content


async def execute_tool_call_detailed(
    name: str,
    arguments: dict[str, Any],
    *,
    client: PaperlessClient,
) -> ToolExecutionResult:
    """Dispatch a tool call with UI-friendly metadata for the chat frontend."""
    if name == "get_available_metadata":
        return await get_available_metadata(client=client)
    if name == "search_documents":
        return await search_documents(
            arguments.get("query", ""),
            client=client,
            correspondent=arguments.get("correspondent"),
            document_type=arguments.get("document_type"),
            storage_path=arguments.get("storage_path"),
            tags=arguments.get("tags"),
            year=arguments.get("year"),
            limit=None if "limit" not in arguments else int(arguments["limit"]),
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
