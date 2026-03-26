"""
Centralised configuration for the AI processing service.

Reads all settings from environment variables, injects Docker secrets,
and exposes a validated AgentConfig Pydantic model.
"""

import os
from pathlib import Path
from typing import Optional

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
    """Read env var, or if FOO_FILE is set, read its content from that file."""
    file_path = os.environ.get(f"{env_var}_FILE")
    if file_path:
        p = Path(file_path)
        if p.is_file():
            return p.read_text().strip()
    return os.environ.get(env_var)


def _inject_secrets() -> None:
    """Read Docker secrets (_FILE variants) and inject them into os.environ."""
    for key in ("GOOGLE_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "PAPERLESS_TOKEN"):
        val = _read_secret(key)
        if val:
            os.environ[key] = val


def _load_prompt(path: str) -> str | None:
    p = Path(path)
    if p.exists():
        return p.read_text(encoding="utf-8").strip()
    return None


class AgentConfig(BaseModel):
    name: Optional[str] = None  # Friendly name for experiments in Phoenix
    paperless_url: str
    paperless_token: str

    ocr_model: str = "gemini/gemini-2.5-flash"
    metadata_model: Optional[str] = None
    ocr_api_base: Optional[str] = None
    metadata_api_base: Optional[str] = None
    ocr_reasoning_effort: Optional[str] = "minimal"

    poll_interval: int = 300
    llm_retries: int = 3
    ocr_concurrency: int = 4
    tag_pending: str = "ai-review-pending"
    dry_run: bool = False
    temperature: Optional[float] = None

    ocr_prompt: str = _OCR_PROMPT_DEFAULT
    metadata_prompt: str = _METADATA_PROMPT_DEFAULT

    # Smart agent batch size for memory-safe vision OCR loops
    vision_batch_size: int = 5

    @property
    def effective_metadata_model(self) -> str:
        return self.metadata_model or self.ocr_model

    def get_litellm_kwargs(self) -> dict:
        """Helper to safely construct hyperparameter kwargs for LiteLLM."""
        kwargs = {}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.ocr_reasoning_effort:
            kwargs["reasoning_effort"] = self.ocr_reasoning_effort
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
            tag_pending=os.environ.get("TAG_PENDING", "ai-review-pending"),
            dry_run=os.environ.get("DRY_RUN", "false").lower() in ("1", "true", "yes"),
            temperature=temperature,
            ocr_prompt=_load_prompt("/app/prompt.txt") or _OCR_PROMPT_DEFAULT,
            metadata_prompt=_load_prompt("/app/metadata_prompt.txt") or _METADATA_PROMPT_DEFAULT,
        )
