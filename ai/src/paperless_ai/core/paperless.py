"""
Paperless-ngx REST API client.

Handles polling, downloading, and patching documents via the API.
This module is intentionally kept unchanged from the original implementation.
"""

import logging
from difflib import SequenceMatcher

import httpx

log = logging.getLogger(__name__)


def _raise_for_status(r: httpx.Response) -> None:
    """Like raise_for_status() but includes the response body in the exception message."""
    try:
        r.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise httpx.HTTPStatusError(
            f"{e} — {e.response.text}",
            request=e.request,
            response=e.response,
        ) from None


class PaperlessClient:
    def __init__(self, base_url: str, token: str):
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Token {token}"},
            timeout=60,
        )
        self.paperless_version: str | None = None
        self._correspondents_cache: list[dict] | None = None

    async def aclose(self):
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        await self.aclose()

    async def get_tag_id(self, name: str, create: bool = True) -> int:
        """Return tag ID by name, optionally creating it if missing."""
        r = await self._client.get("/api/tags/", params={"name": name})
        _raise_for_status(r)
        self.paperless_version = r.headers.get("x-version", self.paperless_version)
        results = r.json()["results"]
        for tag in results:
            if tag["name"] == name:
                return tag["id"]
        if not create:
            raise ValueError(f"Tag '{name}' not found")
        r = await self._client.post("/api/tags/", json={"name": name})
        _raise_for_status(r)
        tag_id = r.json()["id"]
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
                "fields": "id,title,correspondent,created_date,custom_fields,tags,language"
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
                "fields": "id,title,correspondent,created_date,custom_fields,tags,language,content"
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

    async def _get_all_correspondents(self, force: bool = False) -> list[dict]:
        """Return all correspondents, using a cache within the batch run."""
        if self._correspondents_cache is not None and not force:
            return self._correspondents_cache
        all_corr: list[dict] = []
        page = 1
        while True:
            r = await self._client.get(
                "/api/correspondents/", params={"page": page, "page_size": 250}
            )
            _raise_for_status(r)
            data = r.json()
            all_corr.extend(data["results"])
            if not data.get("next"):
                break
            page += 1
        self._correspondents_cache = all_corr
        log.info("Loaded %d correspondent(s)", len(all_corr))
        return all_corr

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

    async def add_note(self, doc_id: int, note: str) -> None:
        r = await self._client.post(f"/api/documents/{doc_id}/notes/", json={"note": note})
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

    async def get_or_create_custom_field(self, name: str, data_type: str = "date") -> int:
        """Return custom field ID by name, creating it if missing."""
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
                    log.info("Found custom field '%s' (id=%d)", name, field["id"])
                    return field["id"]
            if not data.get("next"):
                break
            page += 1
        r = await self._client.post(
            "/api/custom_fields/", json={"name": name, "data_type": data_type}
        )
        if not r.is_success:
            log.warning("Custom field create failed (%d): %s", r.status_code, r.text)
        _raise_for_status(r)
        field_id = r.json()["id"]
        log.info(
            "Created custom field '%s' (id=%d, type=%s)", name, field_id, data_type
        )
        return field_id

    async def get_correspondent_name(self, correspondent_id: int) -> str | None:
        """Return the name of a correspondent by ID, or None on failure."""
        try:
            r = await self._client.get(f"/api/correspondents/{correspondent_id}/")
            _raise_for_status(r)
            return r.json().get("name")
        except Exception:
            return None

    async def update_tags(self, doc: dict, remove_id: int, add_id: int | None) -> None:
        """Remove pending tag from document, optionally adding another."""
        current_tags = [t for t in doc["tags"] if t != remove_id]
        if add_id is not None and add_id not in current_tags:
            current_tags.append(add_id)
        r = await self._client.patch(
            f"/api/documents/{doc['id']}/", json={"tags": current_tags}
        )
        _raise_for_status(r)
