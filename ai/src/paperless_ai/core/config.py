"""
Centralised configuration for the AI processing service.

Reads all settings from environment variables, injects Docker secrets,
and exposes a validated AgentConfig Pydantic model.
"""

import os
from importlib.resources import files as _pkg_files
from typing import List, Optional

import litellm
from pydantic import BaseModel

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


def _read_secret(env_var: str) -> str | None:
    """Read env var, or if FOO_FILE is set, read its content from that file.

    Gracefully handles missing or inaccessible secret files (e.g., not in Docker Swarm).
    """
    file_path = os.environ.get(f"{env_var}_FILE")
    if file_path:
        p = Path(file_path)
        try:
            if p.is_file():
                return p.read_text().strip()
        except (PermissionError, FileNotFoundError):
            pass  # Secret file not available (not in Swarm mode)
    return os.environ.get(env_var)


def _inject_secrets() -> None:
    """Read Docker secrets (_FILE variants) and inject them into os.environ."""
    for key in ("GOOGLE_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "PAPERLESS_TOKEN"):
        val = _read_secret(key)
        if val:
            os.environ[key] = val


def _load_prompt(name: str) -> str:
    """Load a prompt file bundled as package data."""
    return _pkg_files("paperless_ai").joinpath(name).read_text(encoding="utf-8").strip()


class JuryMemberConfig(BaseModel):
    """Configuration for a single judge in the LLM-as-a-jury panel."""
    model: str
    # Passed to LiteLLMModel.model_kwargs — covers api_base, reasoning_effort, etc.
    api_base: Optional[str] = None
    reasoning_effort: Optional[str] = None
    temperature: Optional[float] = None

    def to_litellm_model_kwargs(self) -> dict:
        """Extra kwargs forwarded to litellm.completion via LiteLLMModel.model_kwargs."""
        kwargs = {}
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        return kwargs


class AgentConfig(BaseModel):
    name: Optional[str] = None  # Friendly name for experiments in Phoenix
    paperless_url: str
    paperless_token: str

    ocr_model: str = "gemini/gemini-2.5-flash"
    metadata_model: Optional[str] = None
    ocr_api_base: Optional[str] = None
    metadata_api_base: Optional[str] = None
    ocr_reasoning_effort: Optional[str] = "minimal"
    metadata_reasoning_effort: Optional[str] = None

    poll_interval: int = 300
    llm_retries: int = 3
    ocr_concurrency: int = 4
    tag_ocr: str = "ai:run-ocr"
    tag_metadata: str = "ai:run-metadata"
    tag_embed: str = "ai:run-embed"
    dry_run: bool = False

    @property
    def tag_pending(self) -> str:
        """Backward-compat alias — the OCR tag is the pipeline entry point."""
        return self.tag_ocr
    temperature: Optional[float] = None

    ocr_prompt: str = _OCR_PROMPT_DEFAULT
    metadata_prompt: str = _METADATA_PROMPT_DEFAULT

    # Maximum number of retries when NuExtract returns invalid JSON.
    nuextract_json_retries: int = 5

    # Redis queue (DB 1, isolated from Paperless DB 0)
    redis_url: str = "redis://broker:6379/1"

    # Qdrant vector store
    qdrant_url: str = "http://qdrant:6333"

    # Infinity embedding server (bge-m3, dense + sparse)
    infinity_url: str = "http://complex.home.arpa:8102"
    embedding_model: str = "BAAI/bge-m3"

    # Text chunking for embedding
    chunk_max_chars: int = 2048   # ≈ 512 tokens
    chunk_overlap: int = 256

    # Optional path to a Python file that exports format_chunk_for_embedding.
    # When set, this hook replaces the default situated-embedding header logic.
    embed_hook_file: Optional[str] = None

    # Smart agent batch size for memory-safe vision OCR loops
    vision_batch_size: int = 5
    # Cap the longest image dimension (px) before encoding for vision OCR.
    # Prevents context-length errors on models with small token budgets.
    # None = no cap (use full 300 DPI render).
    ocr_max_image_dimension: Optional[int] = None

    # Page-sampling strategy for long documents.
    # When a PDF has more than ocr_page_limit_threshold pages, only the first
    # ocr_first_pages and last ocr_last_pages are sent through vision OCR.
    # Paperless-ngx Tesseract already covers the full document for keyword
    # search; the vision pass is only needed for metadata extraction and
    # semantic embedding, where the cover/header and final summary pages carry
    # the vast majority of signal.
    # Set ocr_page_limit_threshold=0 to always apply the limit, or a large
    # number (e.g. 9999) to effectively disable it.
    ocr_page_limit_threshold: int = 40
    ocr_first_pages: int = 20
    ocr_last_pages: int = 20
    metadata_max_tokens: int = 1000

    # Dotted import path to the agent class to use in eval experiments.
    agent_class: str = "paperless_ai.agents.smart_graph_agent.SmartDocumentAgent"

    # Model used as LLM judge for title quality evaluation.
    # Should be a strong, fixed model independent of the experiment being
    # evaluated to avoid self-grading bias.
    llm_judge_model: str = "gemini/gemini-2.5-flash"

    # Optional jury of LLM judges for title quality evaluation.
    # When set, each member votes independently and the final score is
    # determined by majority vote, which improves alignment with human
    # judgment compared to a single judge.
    # If None, falls back to a single judge using llm_judge_model.
    jury: Optional[List[JuryMemberConfig]] = None

    @property
    def effective_metadata_model(self) -> str:
        return self.metadata_model or self.ocr_model

    def get_ocr_litellm_kwargs(self) -> dict:
        """Hyperparameter kwargs for the OCR (vision) LiteLLM call."""
        kwargs = {}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.ocr_reasoning_effort:
            kwargs["reasoning_effort"] = self.ocr_reasoning_effort
        return kwargs

    def get_metadata_litellm_kwargs(self) -> dict:
        """Hyperparameter kwargs for the metadata extraction LiteLLM call."""
        kwargs: dict = {"max_tokens": self.metadata_max_tokens}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        effort = self.metadata_reasoning_effort or self.ocr_reasoning_effort
        if effort:
            kwargs["reasoning_effort"] = effort
        return kwargs

    @classmethod
    def from_env(cls) -> "AgentConfig":
        _inject_secrets()
        litellm.drop_params = True

        paperless_token = _read_secret("PAPERLESS_TOKEN") or ""
        ocr_model = os.environ.get("OCR_MODEL", "gemini/gemini-2.5-flash")
        temp_str = os.environ.get("TEMPERATURE")
        temperature = float(temp_str) if temp_str else None

        return cls(
            name=os.environ.get("AGENT_NAME", "default"),
            paperless_url=os.environ.get("PAPERLESS_URL", "").rstrip("/"),
            paperless_token=paperless_token,
            ocr_model=ocr_model,
            metadata_model=os.environ.get("METADATA_MODEL") or None,
            ocr_api_base=os.environ.get("OCR_API_BASE") or None,
            metadata_api_base=os.environ.get("METADATA_API_BASE") or None,
            ocr_reasoning_effort=os.environ.get("OCR_REASONING_EFFORT", "minimal") or None,
            poll_interval=int(os.environ.get("POLL_INTERVAL", "300")),
            llm_retries=int(os.environ.get("LLM_RETRIES", "3")),
            ocr_concurrency=int(os.environ.get("OCR_CONCURRENCY", "4")),
            tag_ocr=os.environ.get("TAG_OCR", os.environ.get("TAG_PENDING", "ai:run-ocr")),
            tag_metadata=os.environ.get("TAG_METADATA", "ai:run-metadata"),
            tag_embed=os.environ.get("TAG_EMBED", "ai:run-embed"),
            dry_run=os.environ.get("DRY_RUN", "false").lower() in ("1", "true", "yes"),
            temperature=temperature,
            ocr_prompt=_load_prompt("prompt.txt"),
            metadata_prompt=_load_prompt("metadata_prompt.txt"),
            metadata_max_tokens=int(os.environ.get("METADATA_MAX_TOKENS", "1000")),
            nuextract_json_retries=int(os.environ.get("NUEXTRACT_JSON_RETRIES", "5")),
            redis_url=os.environ.get("REDIS_URL", "redis://broker:6379/1"),
            qdrant_url=os.environ.get("QDRANT_URL", "http://qdrant:6333"),
            infinity_url=os.environ.get("INFINITY_URL", "http://complex.home.arpa:8102"),
            embedding_model=os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3"),
            chunk_max_chars=int(os.environ.get("CHUNK_MAX_CHARS", "2048")),
            chunk_overlap=int(os.environ.get("CHUNK_OVERLAP", "256")),
            embed_hook_file=os.environ.get("EMBED_HOOK_FILE") or None,
            ocr_page_limit_threshold=int(os.environ.get("OCR_PAGE_LIMIT_THRESHOLD", "40")),
            ocr_first_pages=int(os.environ.get("OCR_FIRST_PAGES", "20")),
            ocr_last_pages=int(os.environ.get("OCR_LAST_PAGES", "20")),
        )
