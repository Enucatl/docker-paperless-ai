"""
Paperless-ngx REST API client.

Handles polling, downloading, and patching documents via the API.
"""

import logging
from difflib import SequenceMatcher
from typing import Any

import niquests

from paperless_common.telemetry import set_span_attributes, start_span

log = logging.getLogger(__name__)


def _raise_for_status(r: niquests.Response) -> None:
    """Like raise_for_status() but includes the response body in the exception message."""
    try:
        r.raise_for_status()
    except niquests.HTTPError as e:
        raise niquests.HTTPError(
            f"{e} — {e.response.text}",
            response=e.response,
        ) from None


class PaperlessClient:
    def __init__(self, base_url: str, token: str):
        self._base_url = base_url.rstrip("/")
        self._client = niquests.AsyncSession(
            base_url=self._base_url,
            headers={"Authorization": f"Token {token}"},
            timeout=60,
        )
        self.paperless_version: str | None = None
        self._correspondents_cache: list[dict] | None = None
        self._tags_cache: list[dict] | None = None
        self._tag_id_cache: dict[str, int] = {}
        self._document_types_cache: list[dict] | None = None
        self._storage_paths_cache: list[dict] | None = None
        self._workflows_cache: list[dict] | None = None
        self._custom_field_id_cache: dict[str, int] = {}

    async def aclose(self):
        await self._client.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        await self.aclose()

    async def get_tag_id(self, name: str, create: bool = True) -> int:
        """Return tag ID by name, optionally creating it if missing."""
        cached_id = self._tag_id_cache.get(name)
        if cached_id is not None:
            return cached_id
        r = await self._client.get("/api/tags/", params={"name": name})
        _raise_for_status(r)
        self.paperless_version = r.headers.get("x-version", self.paperless_version)
        results = r.json()["results"]
        for tag in results:
            if tag["name"] == name:
                tag_id = int(tag["id"])
                self._tag_id_cache[name] = tag_id
                return tag_id
        if not create:
            raise ValueError(f"Tag '{name}' not found")
        r = await self._client.post("/api/tags/", json={"name": name})
        _raise_for_status(r)
        tag_id = r.json()["id"]
        self._tag_id_cache[name] = int(tag_id)
        log.info("Created tag '%s' (id=%d)", name, tag_id)
        return tag_id

    async def count_pending_documents(self, tag_id: int) -> int:
        """Return the number of documents tagged with the pending tag."""
        r = await self._client.get(
            "/api/documents/",
            params={"tags__id__in": tag_id, "page_size": 1},
        )
        _raise_for_status(r)
        return r.json().get("count", 0)

    async def get_document(self, doc_id: int) -> dict | None:
        """Fetch a single document by ID.  Returns None if not found."""
        r = await self._client.get(
            f"/api/documents/{doc_id}/",
            params={
                "fields": "id,title,correspondent,document_type,storage_path,created,custom_fields,tags,language"
            },
        )
        if r.status_code == 404:
            return None
        _raise_for_status(r)
        return r.json()

    async def get_document_with_content(self, doc_id: int) -> dict | None:
        """Fetch a single document including the content (OCR text) field."""
        r = await self._client.get(
            f"/api/documents/{doc_id}/",
            params={
                "fields": "id,title,correspondent,document_type,storage_path,created,custom_fields,tags,language,content"
            },
        )
        if r.status_code == 404:
            return None
        _raise_for_status(r)
        return r.json()

    async def get_document_for_chat(self, doc_id: int) -> dict | None:
        """Fetch a document with the fields needed for chat source cards."""
        r = await self._client.get(
            f"/api/documents/{doc_id}/",
            params={
                "fields": (
                    "id,title,correspondent,document_type,storage_path,created,tags,"
                    "archive_serial_number,original_filename"
                )
            },
        )
        if r.status_code == 404:
            return None
        _raise_for_status(r)
        return r.json()

    async def download_original(self, doc_id: int) -> bytes:
        """Download the original (pre-OCR) file for a document."""
        r = await self._client.get(
            f"/api/documents/{doc_id}/download/",
            params={"original": "true"},
            timeout=120,
        )
        _raise_for_status(r)
        return r.content

    async def _get_all_objects(
        self,
        endpoint: str,
        cache_attr: str,
        force: bool = False,
    ) -> list[dict]:
        """Return all paginated objects from a Paperless endpoint, using a cache."""
        cached = getattr(self, cache_attr)
        if cached is not None and not force:
            return cached

        items: list[dict] = []
        page = 1
        while True:
            r = await self._client.get(
                endpoint, params={"page": page, "page_size": 250}
            )
            _raise_for_status(r)
            data = r.json()
            items.extend(data["results"])
            if not data.get("next"):
                break
            page += 1

        setattr(self, cache_attr, items)
        log.info("Loaded %d object(s) from %s", len(items), endpoint)
        return items

    @staticmethod
    def _resource_label(item: dict[str, Any]) -> str | None:
        """Extract the human-readable label for a Paperless metadata object."""
        for key in ("name", "path", "title"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    async def _get_all_correspondents(self, force: bool = False) -> list[dict]:
        """Return all correspondents, using a cache within the batch run."""
        return await self._get_all_objects(
            "/api/correspondents/",
            "_correspondents_cache",
            force=force,
        )

    async def get_all_correspondents(self, force: bool = False) -> list[dict]:
        """Return all correspondents with cache reuse."""
        return await self._get_all_correspondents(force=force)

    async def _get_all_tags(self, force: bool = False) -> list[dict]:
        return await self._get_all_objects("/api/tags/", "_tags_cache", force=force)

    async def _get_all_document_types(self, force: bool = False) -> list[dict]:
        return await self._get_all_objects(
            "/api/document_types/",
            "_document_types_cache",
            force=force,
        )

    async def _get_all_storage_paths(self, force: bool = False) -> list[dict]:
        return await self._get_all_objects(
            "/api/storage_paths/",
            "_storage_paths_cache",
            force=force,
        )

    async def find_or_create_correspondent(self, name: str) -> int:
        """Match against correspondents (exact → fuzzy via API filter), or create a new one."""
        log.info("Correspondent lookup: '%s'", name)

        # Try exact match against cache first (fast path for repeated lookups in a batch)
        if self._correspondents_cache is not None:
            for c in self._correspondents_cache:
                if c["name"].lower() == name.lower():
                    log.info("Exact match '%s' → id=%d (from cache)", name, c["id"])
                    return c["id"]

        # Use API filter to fetch only candidates similar to the lookup name.
        # This prevents O(n) memory load for accounts with thousands of correspondents.
        # Fetch first 100 candidates with matching substring (case-insensitive icontains).
        r = await self._client.get(
            "/api/correspondents/",
            params={"name__icontains": name, "page_size": 100},
        )
        _raise_for_status(r)
        candidates = r.json()["results"]

        # Exact match (case-insensitive) in filtered results
        for c in candidates:
            if c["name"].lower() == name.lower():
                log.info("Exact match '%s' → id=%d", name, c["id"])
                # Cache if we have a cache
                if self._correspondents_cache is not None:
                    self._correspondents_cache.append(c)
                return c["id"]

        # Fuzzy match in filtered results
        best, best_ratio = None, 0.0
        for c in candidates:
            ratio = SequenceMatcher(None, name.lower(), c["name"].lower()).ratio()
            if ratio > best_ratio:
                best, best_ratio = c, ratio
        if best_ratio >= 0.80:
            log.info(
                "Fuzzy match '%s' → '%s' id=%d (ratio=%.2f)",
                name,
                best["name"],
                best["id"],
                best_ratio,
            )
            # Cache if we have a cache
            if self._correspondents_cache is not None:
                self._correspondents_cache.append(best)
            return best["id"]

        # No match found — create a new correspondent
        log.info("No match for '%s' (best ratio=%.2f) — creating", name, best_ratio)
        r = await self._client.post("/api/correspondents/", json={"name": name})
        _raise_for_status(r)
        new_corr = r.json()
        new_id = new_corr["id"]
        log.info("Created correspondent '%s' (id=%d)", name, new_id)
        # Add to cache so subsequent lookups in this batch find it immediately
        if self._correspondents_cache is not None:
            self._correspondents_cache.append(new_corr)
        return new_id

    async def patch_document(self, doc_id: int, payload: dict) -> None:
        r = await self._client.patch(f"/api/documents/{doc_id}/", json=payload)
        _raise_for_status(r)

    async def delete_correspondent(self, correspondent_id: int) -> None:
        r = await self._client.delete(f"/api/correspondents/{correspondent_id}/")
        _raise_for_status(r)
        if self._correspondents_cache is not None:
            self._correspondents_cache = [
                item
                for item in self._correspondents_cache
                if int(item.get("id", -1)) != correspondent_id
            ]

    async def add_note(self, doc_id: int, note: str) -> None:
        r = await self._client.post(
            f"/api/documents/{doc_id}/notes/", json={"note": note}
        )
        _raise_for_status(r)

    async def list_notes(self, doc_id: int) -> list[dict]:
        r = await self._client.get(f"/api/documents/{doc_id}/notes/")
        _raise_for_status(r)
        return r.json()

    async def delete_note(self, doc_id: int, note_id: int) -> None:
        r = await self._client.delete(f"/api/documents/{doc_id}/notes/{note_id}/")
        _raise_for_status(r)

    async def iter_all_documents(self) -> list[dict]:
        """Page through all documents and return them."""
        docs, page = [], 1
        while True:
            r = await self._client.get(
                "/api/documents/", params={"page": page, "page_size": 100}
            )
            _raise_for_status(r)
            data = r.json()
            docs.extend(data["results"])
            if not data.get("next"):
                break
            page += 1
        return docs

    async def iter_all_documents_brief(self) -> list[dict]:
        """Page through all documents with only the fields needed for cleanup tasks."""
        docs, page = [], 1
        while True:
            r = await self._client.get(
                "/api/documents/",
                params={
                    "page": page,
                    "page_size": 250,
                    "fields": "id,title,correspondent",
                },
            )
            _raise_for_status(r)
            data = r.json()
            docs.extend(data["results"])
            if not data.get("next"):
                break
            page += 1
        return docs

    async def count_documents_for_correspondent(self, correspondent_id: int) -> int:
        r = await self._client.get(
            "/api/documents/",
            params={
                "correspondent__id": correspondent_id,
                "page_size": 1,
                "fields": "id",
            },
        )
        _raise_for_status(r)
        return int(r.json().get("count", 0))

    async def get_or_create_custom_field(
        self, name: str, data_type: str = "date"
    ) -> int:
        """Return custom field ID by name, creating it if missing."""
        cached_id = self._custom_field_id_cache.get(name)
        if cached_id is not None:
            return cached_id
        # Page through all fields — the ?name= filter is not reliable in all paperless versions
        page = 1
        while True:
            r = await self._client.get(
                "/api/custom_fields/", params={"page": page, "page_size": 250}
            )
            _raise_for_status(r)
            data = r.json()
            for field in data["results"]:
                if field["name"] == name:
                    existing_type = field.get("data_type")
                    if existing_type != data_type:
                        r = await self._client.patch(
                            f"/api/custom_fields/{field['id']}/",
                            json={"data_type": data_type},
                        )
                        _raise_for_status(r)
                        log.info(
                            "Updated custom field '%s' (id=%d) type %s -> %s",
                            name,
                            field["id"],
                            existing_type,
                            data_type,
                        )
                    self._custom_field_id_cache[name] = int(field["id"])
                    log.info("Found custom field '%s' (id=%d)", name, field["id"])
                    return field["id"]
            if not data.get("next"):
                break
            page += 1
        r = await self._client.post(
            "/api/custom_fields/", json={"name": name, "data_type": data_type}
        )
        if not r.ok:
            log.warning("Custom field create failed (%d): %s", r.status_code, r.text)
        _raise_for_status(r)
        field_id = r.json()["id"]
        self._custom_field_id_cache[name] = int(field_id)
        log.info(
            "Created custom field '%s' (id=%d, type=%s)", name, field_id, data_type
        )
        return field_id

    async def get_correspondent_name(self, correspondent_id: int) -> str | None:
        """Return the name of a correspondent by ID, or None on failure."""
        try:
            correspondents = await self._get_all_correspondents()
            for correspondent in correspondents:
                if correspondent.get("id") == correspondent_id:
                    return self._resource_label(correspondent)
        except Exception:
            return None
        return None

    async def get_document_type_name(self, document_type_id: int | None) -> str | None:
        """Return the name of a document type by ID, or None on failure."""
        if not document_type_id:
            return None
        try:
            for doc_type in await self._get_all_document_types():
                if doc_type.get("id") == document_type_id:
                    return self._resource_label(doc_type)
        except Exception:
            return None
        return None

    async def get_storage_path_name(self, storage_path_id: int | None) -> str | None:
        """Return the display path/name of a storage path by ID, or None on failure."""
        if not storage_path_id:
            return None
        try:
            for storage_path in await self._get_all_storage_paths():
                if storage_path.get("id") == storage_path_id:
                    return self._resource_label(storage_path)
        except Exception:
            return None
        return None

    async def get_tag_names(self, tag_ids: list[int]) -> list[str]:
        """Resolve a list of tag IDs to human-readable tag names."""
        if not tag_ids:
            return []
        try:
            by_id = {
                tag["id"]: label
                for tag in await self._get_all_tags()
                if (label := self._resource_label(tag)) is not None
            }
            return [by_id[tag_id] for tag_id in tag_ids if tag_id in by_id]
        except Exception:
            return []

    async def get_available_metadata(self) -> dict[str, list[str]]:
        """Return the exact metadata names that exist in Paperless for agent filtering."""
        correspondents = await self._get_all_correspondents()
        document_types = await self._get_all_document_types()
        storage_paths = await self._get_all_storage_paths()
        tags = await self._get_all_tags()
        return {
            "correspondents": sorted(
                label
                for item in correspondents
                if (label := self._resource_label(item))
            ),
            "document_types": sorted(
                label
                for item in document_types
                if (label := self._resource_label(item))
            ),
            "storage_paths": sorted(
                label for item in storage_paths if (label := self._resource_label(item))
            ),
            "tags": sorted(
                label for item in tags if (label := self._resource_label(item))
            ),
        }

    async def get_document_chat_metadata(self, doc_id: int) -> dict[str, Any] | None:
        """Resolve display metadata for a document source card."""
        doc = await self.get_document_for_chat(doc_id)
        if doc is None:
            return None

        correspondent_id = doc.get("correspondent")
        document_type_id = doc.get("document_type")
        storage_path_id = doc.get("storage_path")
        tag_ids = doc.get("tags") or []

        return {
            "id": int(doc["id"]),
            "title": doc.get("title") or "Untitled",
            "created": doc.get("created"),
            "correspondent_name": (
                await self.get_correspondent_name(correspondent_id)
                if correspondent_id
                else None
            ),
            "document_type_name": (
                await self.get_document_type_name(document_type_id)
                if document_type_id
                else None
            ),
            "storage_path_name": (
                await self.get_storage_path_name(storage_path_id)
                if storage_path_id
                else None
            ),
            "tag_names": await self.get_tag_names(tag_ids),
            "archive_serial_number": doc.get("archive_serial_number"),
            "original_filename": doc.get("original_filename"),
        }

    async def _resolve_correspondent_id(self, name: str) -> int | None:
        for correspondent in await self._get_all_correspondents():
            if self._resource_label(correspondent) == name:
                return int(correspondent["id"])
        return None

    async def _resolve_document_type_id(self, name: str) -> int | None:
        for document_type in await self._get_all_document_types():
            if self._resource_label(document_type) == name:
                return int(document_type["id"])
        return None

    async def _resolve_storage_path_id(self, path: str) -> int | None:
        for storage_path in await self._get_all_storage_paths():
            if self._resource_label(storage_path) == path:
                return int(storage_path["id"])
        return None

    async def _resolve_tag_ids(self, names: list[str]) -> list[int] | None:
        if not names:
            return []
        by_name = {
            label: int(tag["id"])
            for tag in await self._get_all_tags()
            if (label := self._resource_label(tag)) is not None
        }
        resolved = [by_name[name] for name in names if name in by_name]
        if len(resolved) != len(names):
            return None
        return resolved

    async def _build_search_params(
        self,
        query: str,
        *,
        correspondent: str | None = None,
        document_type: str | None = None,
        storage_path: str | None = None,
        tags: list[str] | None = None,
        year: str | None = None,
        page_size: int = 250,
        page: int = 1,
    ) -> dict[str, Any] | None:
        params: dict[str, Any] = {
            "query": query,
            "fields": "id",
            "page_size": page_size,
            "page": page,
        }
        if correspondent:
            correspondent_id = await self._resolve_correspondent_id(correspondent)
            if correspondent_id is None:
                return None
            params["correspondent__id"] = correspondent_id
        if document_type:
            document_type_id = await self._resolve_document_type_id(document_type)
            if document_type_id is None:
                return None
            params["document_type__id"] = document_type_id
        if storage_path:
            storage_path_id = await self._resolve_storage_path_id(storage_path)
            if storage_path_id is None:
                return None
            params["storage_path__id"] = storage_path_id
        if tags:
            tag_ids = await self._resolve_tag_ids(tags)
            if tag_ids is None:
                return None
            if tag_ids:
                params["tags__id__in"] = ",".join(str(tag_id) for tag_id in tag_ids)
        if year:
            params["created__year"] = str(year)
        return params

    async def search_documents(
        self,
        query: str,
        page_size: int = 50,
    ) -> list[int]:
        """Search via Paperless full-text API and return doc IDs in relevance order."""
        r = await self._client.get(
            "/api/documents/",
            params={"query": query, "page_size": page_size, "fields": "id"},
        )
        _raise_for_status(r)
        return [doc["id"] for doc in r.json().get("results", [])]

    async def search_documents_all(
        self,
        query: str,
        *,
        correspondent: str | None = None,
        document_type: str | None = None,
        storage_path: str | None = None,
        tags: list[str] | None = None,
        year: str | None = None,
    ) -> list[int]:
        """Search via Paperless and return all matching doc IDs in relevance order."""
        with start_span(
            "paperless_ai.search.keyword_search",
            **{
                "paperless_ai.search.query": query,
                "paperless_ai.search.filter.correspondent": correspondent,
                "paperless_ai.search.filter.document_type": document_type,
                "paperless_ai.search.filter.storage_path": storage_path,
                "paperless_ai.search.filter.year": str(year)
                if year is not None
                else None,
                "paperless_ai.search.filter.tag_count": len(tags or []),
            },
        ) as span:
            page = 1
            pages_fetched = 0
            doc_ids: list[int] = []
            seen: set[int] = set()
            while True:
                params = await self._build_search_params(
                    query,
                    correspondent=correspondent,
                    document_type=document_type,
                    storage_path=storage_path,
                    tags=tags,
                    year=year,
                    page_size=250,
                    page=page,
                )
                if params is None:
                    set_span_attributes(
                        span,
                        **{
                            "paperless_ai.search.filter_resolution_failed": True,
                            "paperless_ai.search.keyword_result_count": 0,
                        },
                    )
                    return []
                r = await self._client.get("/api/documents/", params=params)
                _raise_for_status(r)
                data = r.json()
                pages_fetched += 1
                for doc in data.get("results", []):
                    doc_id = int(doc["id"])
                    if doc_id not in seen:
                        seen.add(doc_id)
                        doc_ids.append(doc_id)
                if not data.get("next"):
                    break
                page += 1
            set_span_attributes(
                span,
                **{
                    "paperless_ai.search.keyword_pages_fetched": pages_fetched,
                    "paperless_ai.search.keyword_result_count": len(doc_ids),
                },
            )
            return doc_ids

    async def _get_all_workflows(self) -> list[dict]:
        return await self._get_all_objects(
            "/api/workflows/", "_workflows_cache", force=False
        )

    async def ensure_workflow(self, name: str, payload: dict) -> int:
        """Create or update a Paperless workflow by name."""
        workflows = await self._get_all_workflows()
        existing = next((wf for wf in workflows if wf.get("name") == name), None)
        if existing is None:
            r = await self._client.post("/api/workflows/", json=payload)
            _raise_for_status(r)
            workflow = r.json()
            workflows.append(workflow)
            log.info("Created workflow '%s' (id=%d)", name, workflow["id"])
            return workflow["id"]

        r = await self._client.patch(f"/api/workflows/{existing['id']}/", json=payload)
        _raise_for_status(r)
        log.info("Updated workflow '%s' (id=%d)", name, existing["id"])
        return existing["id"]

    async def ensure_ai_workflows(
        self,
        *,
        tag_ocr: str,
        webhook_url: str,
        webhook_secret: str | None = None,
    ) -> tuple[int, int]:
        """Ensure the Paperless AI workflows exist and are configured correctly."""
        tag_ocr_id = await self.get_tag_id(tag_ocr, create=True)
        webhook_headers = {"X-Webhook-Token": webhook_secret} if webhook_secret else {}

        added_payload = {
            "name": "paperless-ai: document-added",
            "enabled": True,
            "order": 0,
            "triggers": [
                {
                    "type": 2,  # DOCUMENT_ADDED
                    "sources": [],
                    "filter_has_tags": [],
                }
            ],
            "actions": [
                {
                    "type": 1,  # ASSIGNMENT
                    "assign_tags": [tag_ocr_id],
                },
                {
                    "type": 4,  # WEBHOOK
                    "webhook": {
                        "url": webhook_url,
                        "use_params": True,
                        "as_json": True,
                        "params": {"doc_url": "{{doc_url}}"},
                        "headers": webhook_headers,
                    },
                },
            ],
        }
        updated_payload = {
            "name": "paperless-ai: document-updated",
            "enabled": True,
            "order": 1,
            "triggers": [
                {
                    "type": 3,  # DOCUMENT_UPDATED
                    "sources": [],
                    "filter_has_tags": [tag_ocr_id],
                }
            ],
            "actions": [
                {
                    "type": 4,  # WEBHOOK
                    "webhook": {
                        "url": webhook_url,
                        "use_params": True,
                        "as_json": True,
                        "params": {"doc_url": "{{doc_url}}"},
                        "headers": webhook_headers,
                    },
                }
            ],
        }

        added_id = await self.ensure_workflow(added_payload["name"], added_payload)
        updated_id = await self.ensure_workflow(
            updated_payload["name"], updated_payload
        )
        return added_id, updated_id

    async def update_tags(self, doc: dict, remove_id: int, add_id: int | None) -> None:
        """Remove pending tag from document, optionally adding another."""
        current_tags = [t for t in doc["tags"] if t != remove_id]
        if add_id is not None and add_id not in current_tags:
            current_tags.append(add_id)
        r = await self._client.patch(
            f"/api/documents/{doc['id']}/", json={"tags": current_tags}
        )
        _raise_for_status(r)
