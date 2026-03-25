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
        log.warning("Secret file %s not found (referenced by %s_FILE)", file_path, env_var)
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

Document text:
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

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def get_tag_id(self, name: str, create: bool = True) -> int:
        """Return tag ID by name, optionally creating it if missing."""
        r = self._client.get("/api/tags/", params={"name": name})
        r.raise_for_status()
        self.paperless_version = r.headers.get("x-version", self.paperless_version)
        results = r.json()["results"]
        for tag in results:
            if tag["name"] == name:
                return tag["id"]
        if not create:
            raise ValueError(f"Tag '{name}' not found")
        r = self._client.post("/api/tags/", json={"name": name})
        r.raise_for_status()
        tag_id = r.json()["id"]
        log.info("Created tag '%s' (id=%d)", name, tag_id)
        return tag_id

    def list_pending_documents(self, tag_id: int) -> list[dict]:
        """Return all documents tagged with the pending tag."""
        docs = []
        params = {"tags__id__in": tag_id, "page_size": 50, "ordering": "created"}
        url = "/api/documents/"
        while url:
            r = self._client.get(url, params=params)
            r.raise_for_status()
            data = r.json()
            docs.extend(data["results"])
            url = data.get("next")
            params = {}  # next URL already has params baked in
        return docs

    def download_original(self, doc_id: int) -> bytes:
        """Download the original (pre-OCR) file for a document."""
        r = self._client.get(
            f"/api/documents/{doc_id}/download/",
            params={"original": "true"},
            timeout=120,
        )
        r.raise_for_status()
        return r.content

    def find_or_create_correspondent(self, name: str) -> int:
        """Search for a correspondent by name (exact then fuzzy), or create a new one."""
        log.info("Correspondent lookup: '%s'", name)
        # Search by name — returns candidates (partial match from API)
        r = self._client.get("/api/correspondents/", params={"name": name, "page_size": 25})
        r.raise_for_status()
        candidates = r.json()["results"]
        log.debug("Correspondent search for '%s': %d candidate(s)", name, len(candidates))
        # Exact match first (case-insensitive)
        for c in candidates:
            if c["name"].lower() == name.lower():
                log.info("Exact match '%s' → id=%d", name, c["id"])
                return c["id"]
        # Fuzzy match among candidates
        best_id, best_ratio = None, 0.0
        for c in candidates:
            ratio = SequenceMatcher(None, name.lower(), c["name"].lower()).ratio()
            if ratio > best_ratio:
                best_id, best_ratio = c["id"], ratio
        if best_ratio >= 0.80:
            log.info("Fuzzy match '%s' → id=%d (ratio=%.2f)", name, best_id, best_ratio)
            return best_id
        log.info("No match for '%s' (best ratio=%.2f) — creating", name, best_ratio)
        try:
            r = self._client.post("/api/correspondents/", json={"name": name})
            r.raise_for_status()
        except Exception:
            # Race condition: created by a concurrent request or earlier in batch
            r = self._client.get("/api/correspondents/", params={"name": name, "page_size": 25})
            r.raise_for_status()
            for c in r.json()["results"]:
                if c["name"].lower() == name.lower():
                    log.info("Correspondent '%s' found after creation conflict → id=%d", name, c["id"])
                    return c["id"]
            raise
        new_id = r.json()["id"]
        log.info("Created correspondent '%s' (id=%d)", name, new_id)
        return new_id

    def patch_document(self, doc_id: int, payload: dict) -> None:
        r = self._client.patch(f"/api/documents/{doc_id}/", json=payload)
        r.raise_for_status()

    def add_note(self, doc_id: int, note: str) -> None:
        r = self._client.post(f"/api/documents/{doc_id}/notes/", json={"note": note})
        r.raise_for_status()

    def get_or_create_custom_field(self, name: str, data_type: str = "date") -> int:
        """Return custom field ID by name, creating it if missing."""
        r = self._client.get("/api/custom_fields/", params={"name": name})
        r.raise_for_status()
        for field in r.json()["results"]:
            if field["name"] == name:
                return field["id"]
        r = self._client.post("/api/custom_fields/", json={"name": name, "data_type": data_type})
        r.raise_for_status()
        field_id = r.json()["id"]
        log.info("Created custom field '%s' (id=%d, type=%s)", name, field_id, data_type)
        return field_id

    def update_tags(self, doc: dict, remove_id: int, add_id: int | None) -> None:
        """Remove pending tag from document, optionally adding another."""
        current_tags = [t for t in doc["tags"] if t != remove_id]
        if add_id is not None and add_id not in current_tags:
            current_tags.append(add_id)
        r = self._client.patch(f"/api/documents/{doc['id']}/", json={"tags": current_tags})
        r.raise_for_status()


# ---------------------------------------------------------------------------
# OCR helpers
# ---------------------------------------------------------------------------

def pdf_to_images(data: bytes) -> list[Image.Image]:
    """Convert PDF bytes to a list of page images at 300 DPI."""
    doc = fitz.open(stream=data, filetype="pdf")
    images = []
    for page in doc:
        mat = fitz.Matrix(300 / 72, 300 / 72)  # 300 DPI
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    return images


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
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    kwargs: dict = {"model": OCR_MODEL, "messages": messages}
    if OCR_REASONING_EFFORT:
        kwargs["reasoning_effort"] = OCR_REASONING_EFFORT
    if OCR_API_BASE:
        kwargs["api_base"] = OCR_API_BASE

    response = await litellm.acompletion(**kwargs)
    return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------

async def extract_metadata(text: str) -> dict:
    """Extract title, date, correspondent from document text using a text LLM."""
    # Take first 4000 + last 2000 chars so footer dates/signatures are included
    if len(text) > 6000:
        snippet = text[:4000] + "\n...\n" + text[-2000:]
    else:
        snippet = text

    messages = [{"role": "user", "content": METADATA_PROMPT + snippet}]
    kwargs: dict = {
        "model": METADATA_MODEL,
        "messages": messages,
        "response_format": {"type": "json_object"},
    }
    if METADATA_API_BASE:
        kwargs["api_base"] = METADATA_API_BASE

    response = await litellm.acompletion(**kwargs)
    raw = response.choices[0].message.content or "{}"
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Some models don't honour json_object mode; strip markdown fences and retry
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        return json.loads(match.group()) if match else {}


# ---------------------------------------------------------------------------
# Document processing
# ---------------------------------------------------------------------------

async def process_document(
    doc: dict,
    client: PaperlessClient,
    pending_id: int,
    custom_field_id: int,
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

    # Convert PDF to images
    try:
        images = pdf_to_images(data)
    except Exception as e:
        log.error("Document %d: PDF conversion failed: %s", doc_id, e)
        return False

    log.info("Document %d: %d page(s), OCR via %s", doc_id, len(images), OCR_MODEL)

    # OCR each page
    t_start = time.time()
    language = doc.get("language")
    page_texts: list[str] = []
    for i, img in enumerate(images, 1):
        try:
            text = await ocr_page(img, language=language)
            page_texts.append(text)
            log.debug("Document %d: page %d/%d — %d chars", doc_id, i, len(images), len(text))
        except Exception as e:
            log.error("Document %d: page %d OCR failed: %s", doc_id, i, e)
            return False

    full_text = "\n\n".join(page_texts)
    log.info("Document %d: OCR complete — %d chars total", doc_id, len(full_text))

    # Extract metadata
    meta: dict = {}
    try:
        meta = await extract_metadata(full_text)
        log.info("Document %d: metadata — %s", doc_id, meta)
    except Exception as e:
        log.warning("Document %d: metadata extraction failed: %s — skipping metadata", doc_id, e)

    # Build PATCH payload
    today = datetime.now(timezone.utc).date().isoformat()
    existing_cf = [cf for cf in doc.get("custom_fields", []) if cf["field"] != custom_field_id]
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
            parsed = datetime.strptime(str(ai_date), "%Y-%m-%d").date()
            if date(1900, 1, 1) <= parsed <= date.today():
                payload["created_date"] = ai_date
            else:
                log.warning("Document %d: AI date '%s' out of range, skipping", doc_id, ai_date)
        except ValueError:
            log.warning("Document %d: invalid AI date format '%s', skipping", doc_id, ai_date)

    ai_correspondent = meta.get("correspondent")
    if ai_correspondent:
        try:
            log.info("Document %d: looking up correspondent '%s'", doc_id, ai_correspondent)
            correspondent_id = client.find_or_create_correspondent(str(ai_correspondent).strip())
            payload["correspondent"] = correspondent_id
            log.info("Document %d: correspondent id=%d", doc_id, correspondent_id)
        except Exception as e:
            log.warning("Document %d: correspondent lookup failed: %s", doc_id, e)
    else:
        log.info("Document %d: skipping correspondent (ai=%r, existing=%r)", doc_id, ai_correspondent, doc.get("correspondent"))

    if dry_run:
        log.info("Document %d: [dry-run] would PATCH fields: %s", doc_id, sorted(payload.keys()))
        log.info("Document %d: [dry-run] would set custom field %d = %s", doc_id, custom_field_id, today)
        log.info("Document %d: [dry-run] would remove tag %d", doc_id, pending_id)
        return True

    # Apply updates and swap tags
    log.info("Document %d: PATCHing fields: %s", doc_id, sorted(payload.keys()))
    try:
        client.patch_document(doc_id, payload)
        log.info("Document %d: PATCH OK, removing pending tag", doc_id)
        client.update_tags(doc, remove_id=pending_id, add_id=None)
        note = json.dumps({
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "elapsed_s": round(time.time() - t_start, 1),
            "OCR_MODEL": OCR_MODEL,
            "OCR_API_BASE": OCR_API_BASE,
            "METADATA_MODEL": METADATA_MODEL,
            "METADATA_API_BASE": METADATA_API_BASE,
            "pages": len(images),
            "chars": len(full_text),
            "paperless_version": client.paperless_version,
            "ai_metadata": meta,
        }, ensure_ascii=False)
        try:
            client.add_note(doc_id, note)
        except Exception as e:
            log.warning("Document %d: could not add note: %s", doc_id, e)
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
    dry_run: bool,
) -> tuple[int, int]:
    """Process all pending documents. Returns (success_count, failure_count)."""
    docs = client.list_pending_documents(pending_id)
    if not docs:
        log.info("No documents tagged '%s'", TAG_PENDING)
        return 0, 0

    log.info("Found %d document(s) to process", len(docs))
    success, failure = 0, 0
    for doc in docs:
        ok = await process_document(doc, client, pending_id, custom_field_id, dry_run)
        if ok:
            success += 1
        else:
            failure += 1

    return success, failure


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
    log.info("OCR model: %s%s", OCR_MODEL, f" (api_base={OCR_API_BASE})" if OCR_API_BASE else "")
    log.info(
        "Metadata model: %s%s",
        METADATA_MODEL,
        f" (api_base={METADATA_API_BASE})" if METADATA_API_BASE else "",
    )

    with PaperlessClient(PAPERLESS_URL, PAPERLESS_TOKEN) as client:
        log.info("Resolving tag: '%s'", TAG_PENDING)
        try:
            pending_id = client.get_tag_id(TAG_PENDING, create=True)
        except Exception as e:
            log.error("Failed to resolve tag: %s", e)
            sys.exit(1)

        try:
            custom_field_id = client.get_or_create_custom_field("ai_processed", data_type="date")
        except Exception as e:
            log.error("Failed to resolve custom field: %s", e)
            sys.exit(1)

        log.info("Tag ID: pending=%d | custom field: ai_processed=%d", pending_id, custom_field_id)

        if args.once:
            success, failure = await run_batch(client, pending_id, custom_field_id, dry_run)
            log.info("Done. Success: %d, Failed: %d", success, failure)
        else:
            log.info("Watch mode: polling every %ds (Ctrl+C to stop)", POLL_INTERVAL)
            while True:
                try:
                    success, failure = await run_batch(client, pending_id, custom_field_id, dry_run)
                    if success or failure:
                        log.info("Batch done. Success: %d, Failed: %d", success, failure)
                except Exception as e:
                    log.error("Batch error: %s", e)
                log.info("Sleeping %ds...", POLL_INTERVAL)
                time.sleep(POLL_INTERVAL)


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
    args = parser.parse_args()

    # Default to watch mode if neither flag is set
    if not args.once:
        args.once = False

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
