#!/usr/bin/env python3
"""
Batch AI post-processing for paperless-ngx.

Polls the paperless API for documents tagged "ai-review-pending",
re-OCRs them with a vision LLM, extracts metadata (title, date,
correspondent) with a text LLM, and patches the document via the API.

No paperless source patches required — everything goes through the REST API.

Usage:
    python batch.py --once      # process all pending documents and exit
    python batch.py --watch     # poll continuously (default when run via Docker)
    python batch.py --dry-run   # log what would happen without modifying anything

Environment variables (see README.md for full list):
    PAPERLESS_URL        Paperless-ngx base URL (required)
    PAPERLESS_TOKEN      API authentication token (required)
    OCR_MODEL            LiteLLM model string for OCR vision model
    METADATA_MODEL       LiteLLM model string for metadata text model
    OCR_API_BASE         Base URL for local OCR server (Ollama, vLLM, etc.)
    METADATA_API_BASE    Base URL for local metadata server
"""

import argparse
import asyncio
import base64
import io
import json
import logging
import os
import re
import signal
import sys
import time
from datetime import date, datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

import fitz  # PyMuPDF
import httpx
import litellm
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def _read_secret(env_var: str) -> str | None:
    """Read env var, or if FOO_FILE is set, read its content from that file."""
    file_path = os.environ.get(f"{env_var}_FILE")
    if file_path:
        p = Path(file_path)
        if p.is_file():
            return p.read_text().strip()
        log.warning(
            "Secret file %s not found (referenced by %s_FILE)", file_path, env_var
        )
    return os.environ.get(env_var)


def _load_prompt(path: str) -> str | None:
    p = Path(path)
    if p.exists():
        return p.read_text(encoding="utf-8").strip()
    return None


PAPERLESS_URL = os.environ.get("PAPERLESS_URL", "").rstrip("/")
PAPERLESS_TOKEN = _read_secret("PAPERLESS_TOKEN") or ""

OCR_MODEL = os.environ.get("OCR_MODEL", "gemini/gemini-2.5-flash")
METADATA_MODEL = os.environ.get("METADATA_MODEL") or OCR_MODEL
OCR_API_BASE = os.environ.get("OCR_API_BASE")
METADATA_API_BASE = os.environ.get("METADATA_API_BASE")
OCR_REASONING_EFFORT = os.environ.get("OCR_REASONING_EFFORT", "minimal") or None

POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL", "300"))
LLM_RETRIES = int(os.environ.get("LLM_RETRIES", "3"))
OCR_CONCURRENCY = int(os.environ.get("OCR_CONCURRENCY", "4"))
TAG_PENDING = os.environ.get("TAG_PENDING", "ai-review-pending")
DRY_RUN = os.environ.get("DRY_RUN", "false").lower() in ("1", "true", "yes")

_OCR_PROMPT_DEFAULT = (
    "Extract the text from the above document as if you were reading it naturally. "
    "Return the tables in HTML format. Return the equations in LaTeX representation. "
    "If there is an image in the document and image caption is not present, add a small "
    "description of the image inside the <img></img> tag; otherwise, add the image caption "
    "inside <img></img>. Watermarks should be wrapped in brackets. "
    "Ex: <watermark>OFFICIAL COPY</watermark>. Page numbers should be wrapped in brackets. "
    "Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. "
    "Prefer using ☐ and ☑ for check boxes."
)

_METADATA_PROMPT_DEFAULT = """\
Extract the following metadata from the document text below.
Respond with a JSON object and no other text:

{
  "title": "<concise descriptive title, max 100 chars, no file extension>",
  "date": "<primary document date as YYYY-MM-DD, e.g. invoice date, letter date>",
  "correspondent": "<name of the sender or issuing organisation — not the recipient>"
}

Use null for any field you cannot determine with confidence.
The correspondent is typically a company, institution or person who sent or issued the document.
Do not invent information; if uncertain, use null.
"""

OCR_PROMPT = _load_prompt("/app/prompt.txt") or _OCR_PROMPT_DEFAULT
METADATA_PROMPT = _load_prompt("/app/metadata_prompt.txt") or _METADATA_PROMPT_DEFAULT

# Inject API keys into environment so LiteLLM picks them up automatically.
for _key in ("GOOGLE_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
    _val = _read_secret(_key)
    if _val:
        os.environ[_key] = _val

# Drop unsupported params (e.g. reasoning_effort) silently for local endpoints.
litellm.drop_params = True


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


# Lazily initialised inside the event loop (asyncio.Semaphore must be created inside a running loop).
_ocr_semaphore: asyncio.Semaphore | None = None

# Set by SIGTERM/SIGINT handler; checked between documents and poll cycles.
_shutdown_requested = False

HEALTHCHECK_FILE = "/tmp/ai-healthy"


def _write_heartbeat() -> None:
    """Write a timestamp to the healthcheck file after each poll cycle."""
    try:
        Path(HEALTHCHECK_FILE).write_text(str(time.time()))
    except OSError:
        pass  # non-critical


def _get_ocr_semaphore() -> asyncio.Semaphore:
    global _ocr_semaphore
    if _ocr_semaphore is None:
        _ocr_semaphore = asyncio.Semaphore(OCR_CONCURRENCY)
    return _ocr_semaphore


# ---------------------------------------------------------------------------
# Paperless API client
# ---------------------------------------------------------------------------


class PaperlessClient:
    def __init__(self, base_url: str, token: str):
        self._client = httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Token {token}"},
            timeout=60,
        )
        self.paperless_version: str | None = None
        self._correspondents_cache: list[dict] | None = None

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def get_tag_id(self, name: str, create: bool = True) -> int:
        """Return tag ID by name, optionally creating it if missing."""
        r = self._client.get("/api/tags/", params={"name": name})
        _raise_for_status(r)
        self.paperless_version = r.headers.get("x-version", self.paperless_version)
        results = r.json()["results"]
        for tag in results:
            if tag["name"] == name:
                return tag["id"]
        if not create:
            raise ValueError(f"Tag '{name}' not found")
        r = self._client.post("/api/tags/", json={"name": name})
        _raise_for_status(r)
        tag_id = r.json()["id"]
        log.info("Created tag '%s' (id=%d)", name, tag_id)
        return tag_id

    def count_pending_documents(self, tag_id: int) -> int:
        """Return the number of documents tagged with the pending tag."""
        r = self._client.get(
            "/api/documents/",
            params={"tags__id__in": tag_id, "page_size": 1},
        )
        _raise_for_status(r)
        return r.json().get("count", 0)

    def iter_pending_documents(self, tag_id: int, page_size: int = 20):
        """Yield pending documents one page at a time, keeping memory flat."""
        params = {
            "tags__id__in": tag_id,
            "page_size": page_size,
            "ordering": "created",
            "fields": "id,title,correspondent,created_date,custom_fields,tags,language",
        }
        url = "/api/documents/"
        while url:
            r = self._client.get(url, params=params)
            _raise_for_status(r)
            data = r.json()
            yield from data["results"]
            url = data.get("next") or None
            params = {}  # next URL already has params baked in

    def download_original(self, doc_id: int) -> bytes:
        """Download the original (pre-OCR) file for a document."""
        r = self._client.get(
            f"/api/documents/{doc_id}/download/",
            params={"original": "true"},
            timeout=120,
        )
        _raise_for_status(r)
        return r.content

    def _get_all_correspondents(self, force: bool = False) -> list[dict]:
        """Return all correspondents, using a cache within the batch run."""
        if self._correspondents_cache is not None and not force:
            return self._correspondents_cache
        all_corr: list[dict] = []
        page = 1
        while True:
            r = self._client.get(
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

    def find_or_create_correspondent(self, name: str) -> int:
        """Match against all correspondents (exact → fuzzy), or create a new one."""
        log.info("Correspondent lookup: '%s'", name)
        candidates = self._get_all_correspondents()

        # Exact match (case-insensitive)
        for c in candidates:
            if c["name"].lower() == name.lower():
                log.info("Exact match '%s' → id=%d", name, c["id"])
                return c["id"]

        # Fuzzy match
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
            return best["id"]

        log.info("No match for '%s' (best ratio=%.2f) — creating", name, best_ratio)
        r = self._client.post("/api/correspondents/", json={"name": name})
        _raise_for_status(r)
        new_corr = r.json()
        new_id = new_corr["id"]
        log.info("Created correspondent '%s' (id=%d)", name, new_id)
        # Add to cache so subsequent lookups in this batch find it immediately
        if self._correspondents_cache is not None:
            self._correspondents_cache.append(new_corr)
        return new_id

    def patch_document(self, doc_id: int, payload: dict) -> None:
        r = self._client.patch(f"/api/documents/{doc_id}/", json=payload)
        _raise_for_status(r)

    def add_note(self, doc_id: int, note: str) -> None:
        r = self._client.post(f"/api/documents/{doc_id}/notes/", json={"note": note})
        _raise_for_status(r)

    def list_notes(self, doc_id: int) -> list[dict]:
        r = self._client.get(f"/api/documents/{doc_id}/notes/")
        _raise_for_status(r)
        return r.json()

    def delete_note(self, doc_id: int, note_id: int) -> None:
        r = self._client.delete(f"/api/documents/{doc_id}/notes/{note_id}/")
        _raise_for_status(r)

    def iter_all_documents(self) -> list[dict]:
        """Page through all documents and return them."""
        docs, page = [], 1
        while True:
            r = self._client.get(
                "/api/documents/", params={"page": page, "page_size": 100}
            )
            _raise_for_status(r)
            data = r.json()
            docs.extend(data["results"])
            if not data.get("next"):
                break
            page += 1
        return docs

    def get_or_create_custom_field(self, name: str, data_type: str = "date") -> int:
        """Return custom field ID by name, creating it if missing."""
        # Page through all fields — the ?name= filter is not reliable in all paperless versions
        page = 1
        while True:
            r = self._client.get(
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
        r = self._client.post(
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

    def get_correspondent_name(self, correspondent_id: int) -> str | None:
        """Return the name of a correspondent by ID, or None on failure."""
        try:
            r = self._client.get(f"/api/correspondents/{correspondent_id}/")
            _raise_for_status(r)
            return r.json().get("name")
        except Exception:
            return None

    def update_tags(self, doc: dict, remove_id: int, add_id: int | None) -> None:
        """Remove pending tag from document, optionally adding another."""
        current_tags = [t for t in doc["tags"] if t != remove_id]
        if add_id is not None and add_id not in current_tags:
            current_tags.append(add_id)
        r = self._client.patch(
            f"/api/documents/{doc['id']}/", json={"tags": current_tags}
        )
        _raise_for_status(r)


# ---------------------------------------------------------------------------
# OCR helpers
# ---------------------------------------------------------------------------


def document_to_pages(data: bytes) -> list[Image.Image]:
    """Convert a document to a list of page images.

    Tries PDF first (rendered at 300 DPI), then falls back to PIL for
    images (JPEG, PNG, multi-frame TIFF, etc.).
    Raises ValueError for unsupported formats.
    """
    # Try PDF
    try:
        doc = fitz.open(stream=data, filetype="pdf")
        images = []
        for page in doc:
            mat = fitz.Matrix(300 / 72, 300 / 72)  # 300 DPI
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
        doc.close()
        if images:
            return images
    except Exception:
        pass  # Not a valid PDF — try as image

    # Try as image (JPEG, PNG, TIFF, etc.)
    try:
        img = Image.open(io.BytesIO(data))
        pages = []
        frame = 0
        while True:
            try:
                img.seek(frame)
                pages.append(img.copy().convert("RGB"))
                frame += 1
            except EOFError:
                break
        return pages if pages else [img.convert("RGB")]
    except Exception:
        pass

    raise ValueError(
        "Unsupported document format — only PDF and common image formats (JPEG, PNG, TIFF) are supported"
    )


def _image_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


async def ocr_page(image: Image.Image, language: str | None = None) -> str:
    """OCR a single page image using the configured vision LLM."""
    prompt = OCR_PROMPT
    if language:
        prompt = f"The document language is primarily '{language}'. " + prompt

    b64 = _image_to_b64(image)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]
    kwargs: dict = {
        "model": OCR_MODEL,
        "messages": messages,
        "num_retries": LLM_RETRIES,
    }
    if OCR_REASONING_EFFORT:
        kwargs["reasoning_effort"] = OCR_REASONING_EFFORT
    if OCR_API_BASE:
        kwargs["api_base"] = OCR_API_BASE

    response = await litellm.acompletion(**kwargs)
    return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------


async def extract_metadata(text: str, existing: dict | None = None) -> dict:
    """Extract title, date, correspondent from document text using a text LLM."""
    # Take first 4000 + last 2000 chars so footer dates/signatures are included
    if len(text) > 6000:
        snippet = text[:4000] + "\n...\n" + text[-2000:]
    else:
        snippet = text

    # NuExtract-style: template passed via chat_template_kwargs, document text as user message.
    # Temperature 0 for deterministic structured extraction.
    template = json.dumps(json.loads(METADATA_PROMPT), indent=4)
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": snippet}],
        }
    ]
    examples = [
        {
            "input": (
                "Rechnung Nr. 2024-1234\nDatum: 15. März 2024\n\nSiemens AG\n"
                "Werner-von-Siemens-Str. 1, 80333 München\n\nAn: Mustermann GmbH\n\n"
                "Leistungszeitraum: Februar 2024\nBetrag: EUR 4.250,00"
            ),
            "output": json.dumps({
                "oneline_short_summary": "Rechnung Siemens AG 2024-1234",
                "document_date": "2024-03-15",
                "correspondent_institution_or_individual": "Siemens AG",
            }),
        },
        {
            "input": (
                "Kontoauszug\nDeutsche Bank AG\nIBAN: DE12 3456 7890 1234 5678 90\n\n"
                "Zeitraum: 01.02.2025 – 28.02.2025\nKontoinhaber: Max Mustermann\n\n"
                "Abschlusssaldo: 3.421,00 EUR"
            ),
            "output": json.dumps({
                "oneline_short_summary": "Kontoauszug Deutsche Bank Februar 2025",
                "document_date": "2025-02-28",
                "correspondent_institution_or_individual": "Deutsche Bank AG",
            }),
        },
    ]
    kwargs: dict = {
        "model": METADATA_MODEL,
        "messages": messages,
        "temperature": 0,
        "extra_body": {
            "chat_template_kwargs": {"template": template, "examples": examples}
        },
        "num_retries": LLM_RETRIES,
    }
    if METADATA_API_BASE:
        kwargs["api_base"] = METADATA_API_BASE

    response = await litellm.acompletion(**kwargs)
    raw = response.choices[0].message.content or "{}"
    log.info("Metadata raw response: %s", raw)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Some models don't honour json_object mode; strip markdown fences and retry
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        data = json.loads(match.group()) if match else {}
    return {
        "title": data.get("oneline_short_summary"),
        "date": data.get("document_date"),
        "correspondent": data.get("correspondent_institution_or_individual"),
    }


# ---------------------------------------------------------------------------
# Document processing
# ---------------------------------------------------------------------------


async def process_document(
    doc: dict,
    client: PaperlessClient,
    pending_id: int,
    custom_field_id: int,
    ai_result_field_id: int,
    dry_run: bool,
) -> bool:
    doc_id = doc["id"]
    log.info("Processing document %d: %s", doc_id, doc.get("title", "(no title)"))

    # Download original file
    try:
        data = client.download_original(doc_id)
    except Exception as e:
        log.error("Document %d: download failed: %s", doc_id, e)
        return False

    # Convert document to page images and release the raw bytes
    try:
        images = document_to_pages(data)
    except ValueError as e:
        log.warning("Document %d: %s — skipping", doc_id, e)
        return False
    except Exception as e:
        log.error("Document %d: page conversion failed: %s", doc_id, e)
        return False

    num_pages = len(images)
    if num_pages > 100:
        log.warning(
            "Document %d: %d pages — this may use significant memory and API quota",
            doc_id,
            num_pages,
        )
    log.info(
        "Document %d: %d page(s), OCR via %s (concurrency=%d)",
        doc_id,
        num_pages,
        OCR_MODEL,
        OCR_CONCURRENCY,
    )

    # OCR all pages in parallel, bounded by semaphore
    t_start = time.time()
    language = doc.get("language")
    sem = _get_ocr_semaphore()

    async def _ocr_one(idx: int, img: Image.Image) -> tuple[int, str]:
        async with sem:
            text = await ocr_page(img, language=language)
            log.debug(
                "Document %d: page %d/%d — %d chars", doc_id, idx, num_pages, len(text)
            )
            return idx, text

    try:
        results = await asyncio.gather(*[
            _ocr_one(i, img) for i, img in enumerate(images, 1)
        ])
    except Exception as e:
        log.error("Document %d: OCR failed: %s", doc_id, e)
        return False
    page_texts = [text for _, text in sorted(results)]
    num_pages = len(images)

    full_text = "\n\n".join(page_texts)
    log.info("Document %d: OCR complete — %d chars total", doc_id, len(full_text))

    # Build existing metadata hints for the LLM
    existing_hints: dict = {}
    if doc.get("title"):
        existing_hints["title"] = doc["title"]
    if doc.get("created_date"):
        existing_hints["date"] = doc["created_date"]
    if doc.get("correspondent"):
        existing_hints["correspondent"] = client.get_correspondent_name(
            doc["correspondent"]
        )

    # Extract metadata
    meta: dict = {}
    try:
        meta = await extract_metadata(full_text, existing=existing_hints or None)
        log.info("Document %d: metadata — %s", doc_id, meta)
    except Exception as e:
        log.warning(
            "Document %d: metadata extraction failed: %s — skipping metadata", doc_id, e
        )

    # Build PATCH payload
    today = datetime.now(timezone.utc).date().isoformat()
    managed_fields = {custom_field_id, ai_result_field_id}
    existing_cf = [
        cf for cf in doc.get("custom_fields", []) if cf["field"] not in managed_fields
    ]
    payload: dict = {
        "content": full_text,
        "custom_fields": existing_cf + [{"field": custom_field_id, "value": today}],
    }

    ai_title = meta.get("title")
    if ai_title:
        payload["title"] = str(ai_title)[:128]  # paperless title field max length

    ai_date = meta.get("date")
    if ai_date:
        try:
            parsed = datetime.fromisoformat(str(ai_date)).date()
            if date(1900, 1, 1) <= parsed <= date.today():
                payload["created_date"] = parsed.isoformat()
            else:
                log.warning(
                    "Document %d: AI date '%s' out of range, skipping", doc_id, ai_date
                )
        except ValueError:
            log.warning(
                "Document %d: invalid AI date format '%s', skipping", doc_id, ai_date
            )

    ai_correspondent = meta.get("correspondent")
    if ai_correspondent:
        try:
            log.info(
                "Document %d: looking up correspondent '%s'", doc_id, ai_correspondent
            )
            correspondent_id = client.find_or_create_correspondent(
                str(ai_correspondent).strip()
            )
            payload["correspondent"] = correspondent_id
            log.info("Document %d: correspondent id=%d", doc_id, correspondent_id)
        except Exception as e:
            log.warning("Document %d: correspondent lookup failed: %s", doc_id, e)
    else:
        log.info(
            "Document %d: skipping correspondent (ai=%r, existing=%r)",
            doc_id,
            ai_correspondent,
            doc.get("correspondent"),
        )

    ai_result = json.dumps(
        {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "elapsed_s": round(time.time() - t_start, 1),
            "OCR_MODEL": OCR_MODEL,
            "OCR_API_BASE": OCR_API_BASE,
            "METADATA_MODEL": METADATA_MODEL,
            "METADATA_API_BASE": METADATA_API_BASE,
            "pages": num_pages,
            "chars": len(full_text),
            "paperless_version": client.paperless_version,
            "ai_metadata": meta,
        },
        ensure_ascii=False,
    )
    payload["custom_fields"].append({"field": ai_result_field_id, "value": ai_result})

    if dry_run:
        log.info(
            "Document %d: [dry-run] would PATCH fields: %s",
            doc_id,
            sorted(payload.keys()),
        )
        log.info("Document %d: [dry-run] would remove tag %d", doc_id, pending_id)
        return True

    # Apply updates and swap tags
    log.info("Document %d: PATCHing fields: %s", doc_id, sorted(payload.keys()))
    try:
        client.patch_document(doc_id, payload)
        log.info("Document %d: PATCH OK, removing pending tag", doc_id)
        client.update_tags(doc, remove_id=pending_id, add_id=None)
        log.info("Document %d: done", doc_id)
        return True
    except Exception as e:
        log.error("Document %d: PATCH failed: %s", doc_id, e)
        return False


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------


async def run_batch(
    client: PaperlessClient,
    pending_id: int,
    custom_field_id: int,
    ai_result_field_id: int,
    dry_run: bool,
) -> tuple[int, int]:
    """Process all pending documents. Returns (success_count, failure_count)."""
    total = client.count_pending_documents(pending_id)
    if total == 0:
        log.info("No documents tagged '%s'", TAG_PENDING)
        return 0, 0

    log.info("Found %d document(s) to process", total)
    success, failure = 0, 0
    for doc in client.iter_pending_documents(pending_id):
        if _shutdown_requested:
            log.info(
                "Shutdown requested — stopping batch after %d/%d documents",
                success + failure,
                total,
            )
            break
        ok = await process_document(
            doc, client, pending_id, custom_field_id, ai_result_field_id, dry_run
        )
        if ok:
            success += 1
        else:
            failure += 1

    return success, failure


def purge_ai_notes(client: PaperlessClient, dry_run: bool) -> None:
    """Delete all notes that were written by previous AI processing runs."""
    docs = client.iter_all_documents()
    log.info("Scanning %d document(s) for AI-generated notes", len(docs))
    deleted = 0
    for doc in docs:
        doc_id = doc["id"]
        try:
            notes = client.list_notes(doc_id)
        except Exception as e:
            log.warning("Document %d: could not fetch notes: %s", doc_id, e)
            continue
        for note in notes:
            text = note.get("note", "")
            try:
                parsed = json.loads(text)
            except (json.JSONDecodeError, TypeError):
                continue
            if "OCR_MODEL" not in parsed:
                continue
            note_id = note["id"]
            if dry_run:
                log.info("Document %d: [dry-run] would delete note %d", doc_id, note_id)
            else:
                try:
                    client.delete_note(doc_id, note_id)
                    log.info("Document %d: deleted note %d", doc_id, note_id)
                    deleted += 1
                except Exception as e:
                    log.warning(
                        "Document %d: could not delete note %d: %s", doc_id, note_id, e
                    )
    log.info("Done. %d note(s) deleted.", deleted)


async def main_async(args: argparse.Namespace) -> None:
    if not PAPERLESS_URL:
        log.error("PAPERLESS_URL is not set")
        sys.exit(1)
    if not PAPERLESS_TOKEN:
        log.error("PAPERLESS_TOKEN (or PAPERLESS_TOKEN_FILE) is not set")
        sys.exit(1)

    dry_run = args.dry_run or DRY_RUN
    if dry_run:
        log.info("DRY RUN mode — no documents will be modified")

    log.info("Paperless URL: %s", PAPERLESS_URL)
    log.info(
        "OCR model: %s%s",
        OCR_MODEL,
        f" (api_base={OCR_API_BASE})" if OCR_API_BASE else "",
    )
    log.info(
        "Metadata model: %s%s",
        METADATA_MODEL,
        f" (api_base={METADATA_API_BASE})" if METADATA_API_BASE else "",
    )

    with PaperlessClient(PAPERLESS_URL, PAPERLESS_TOKEN) as client:
        # Verify Paperless connectivity — fatal if unreachable.
        log.info("Checking Paperless API connectivity...")
        try:
            r = client._client.get("/api/", follow_redirects=True)
            _raise_for_status(r)
            log.info(
                "Paperless API reachable (version: %s)",
                r.headers.get("x-version", "unknown"),
            )
        except Exception as e:
            log.error("Cannot reach Paperless API at %s: %s", PAPERLESS_URL, e)
            sys.exit(1)

        # Verify LLM connectivity — warning only (GPU workstation may be off; documents queue up).
        log.info("Checking LLM connectivity (model: %s)...", METADATA_MODEL)
        try:
            _kwargs: dict = {
                "model": METADATA_MODEL,
                "messages": [{"role": "user", "content": "Reply with OK"}],
                "max_tokens": 5,
            }
            if METADATA_API_BASE:
                _kwargs["api_base"] = METADATA_API_BASE
            await litellm.acompletion(**_kwargs)
            log.info("LLM connectivity OK")
        except Exception as e:
            log.warning(
                "LLM connectivity check failed: %s — will retry during processing", e
            )

        if args.purge_notes:
            purge_ai_notes(client, dry_run)
            return

        log.info("Resolving tag: '%s'", TAG_PENDING)
        try:
            pending_id = client.get_tag_id(TAG_PENDING, create=True)
        except Exception as e:
            log.error("Failed to resolve tag: %s", e)
            sys.exit(1)

        try:
            custom_field_id = client.get_or_create_custom_field(
                "ai_processed", data_type="date"
            )
            ai_result_field_id = client.get_or_create_custom_field(
                "ai_result", data_type="longtext"
            )
        except Exception as e:
            log.error("Failed to resolve custom field: %s", e)
            sys.exit(1)

        log.info(
            "Tag ID: pending=%d | custom fields: ai_processed=%d ai_result=%d",
            pending_id,
            custom_field_id,
            ai_result_field_id,
        )

        if args.once:
            success, failure = await run_batch(
                client, pending_id, custom_field_id, ai_result_field_id, dry_run
            )
            _write_heartbeat()
            log.info("Done. Success: %d, Failed: %d", success, failure)
        else:
            global _shutdown_requested

            def _request_shutdown(signum: int, frame: object) -> None:
                global _shutdown_requested
                log.info(
                    "Received %s — will stop after current document completes",
                    signal.Signals(signum).name,
                )
                _shutdown_requested = True

            signal.signal(signal.SIGTERM, _request_shutdown)
            signal.signal(signal.SIGINT, _request_shutdown)

            log.info(
                "Watch mode: polling every %ds (SIGTERM/Ctrl+C to stop gracefully)",
                POLL_INTERVAL,
            )
            while not _shutdown_requested:
                try:
                    success, failure = await run_batch(
                        client, pending_id, custom_field_id, ai_result_field_id, dry_run
                    )
                    if success or failure:
                        log.info(
                            "Batch done. Success: %d, Failed: %d", success, failure
                        )
                except Exception as e:
                    log.error("Batch error: %s", e)
                _write_heartbeat()
                if _shutdown_requested:
                    break
                log.info("Sleeping %ds...", POLL_INTERVAL)
                try:
                    await asyncio.sleep(POLL_INTERVAL)
                except asyncio.CancelledError:
                    break
            log.info("Shutdown complete.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch AI post-processing for paperless-ngx documents"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--once",
        action="store_true",
        help="Process all pending documents once and exit",
    )
    mode.add_argument(
        "--watch",
        action="store_true",
        help="Poll continuously (default when run via Docker)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log what would happen without modifying any documents",
    )
    parser.add_argument(
        "--purge-notes",
        action="store_true",
        help="Delete all AI-generated notes from previous runs and exit",
    )
    args = parser.parse_args()

    # Default to watch mode if neither flag is set
    if not args.once:
        args.once = False

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
