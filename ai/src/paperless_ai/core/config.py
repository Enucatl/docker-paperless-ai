"""
Centralised configuration for the AI processing service.

Reads all settings from environment variables, injects Docker secrets,
and exposes a validated AgentConfig Pydantic model.
"""

import os
from importlib.resources import files as _pkg_files
from pathlib import Path
from typing import Any, Dict, List, Optional

import litellm
from pydantic import AliasChoices, BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    for key in (
        "GOOGLE_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "HF_TOKEN",
        "PAPERLESS_TOKEN",
        "WEBHOOK_SECRET",
    ):
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


class AgentConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="",
        case_sensitive=False,
        populate_by_name=True,
    )

    name: str = Field(
        default="default",
        validation_alias=AliasChoices("name", "AGENT_NAME"),
    )
    paperless_url: str = ""
    paperless_token: str = ""

    @field_validator("paperless_url", mode="after")
    @classmethod
    def strip_trailing_slash(cls, v: str) -> str:
        return v.rstrip("/")

    ocr_model: str = "gemini/gemini-2.5-flash"
    metadata_model: str
    chat_model: str = Field(
        validation_alias=AliasChoices("chat_model", "CHAT_MODEL"),
    )
    ocr_api_base: Optional[str] = None
    metadata_api_base: Optional[str] = None
    chat_api_base: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("chat_api_base", "CHAT_API_BASE"),
    )
    ocr_reasoning_effort: Optional[str] = "minimal"
    metadata_reasoning_effort: Optional[str] = None
    chat_reasoning_effort: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("chat_reasoning_effort", "CHAT_REASONING_EFFORT"),
    )

    poll_interval: int = 300
    llm_retries: int = 3
    ocr_concurrency: int = 4
    # TAG_PENDING is the legacy name — keep for backward compat
    tag_ocr: str = Field(
        default="ai:run-ocr",
        validation_alias=AliasChoices("tag_ocr", "TAG_OCR", "TAG_PENDING"),
    )
    tag_metadata: str = "ai:run-metadata"
    tag_embed: str = "ai:run-embed"
    dry_run: bool = False

    @property
    def tag_pending(self) -> str:
        """Backward-compat alias — the OCR tag is the pipeline entry point."""
        return self.tag_ocr
    # TEMPERATURE is a generic fallback when specific ones are not set
    ocr_temperature: Optional[float] = Field(
        default=None,
        validation_alias=AliasChoices("ocr_temperature", "OCR_TEMPERATURE", "TEMPERATURE"),
    )
    metadata_temperature: Optional[float] = Field(
        default=None,
        validation_alias=AliasChoices("metadata_temperature", "METADATA_TEMPERATURE", "TEMPERATURE"),
    )
    chat_temperature: Optional[float] = Field(
        default=None,
        validation_alias=AliasChoices("chat_temperature", "CHAT_TEMPERATURE"),
    )

    ocr_prompt: str = Field(default_factory=lambda: _load_prompt("prompt.txt"))
    metadata_prompt: str = Field(default_factory=lambda: _load_prompt("metadata_prompt.txt"))

    # Maximum number of retries when NuExtract returns invalid JSON.
    nuextract_json_retries: int = 5

    # Redis queue (DB 1, isolated from Paperless DB 0)
    redis_url: str = "redis://broker:6379/1"

    # Qdrant vector store
    qdrant_url: str = "http://qdrant:6333"

    # Paperless workflow automation
    manage_paperless_workflows: bool = True
    paperless_webhook_url: str = "http://webhook-listener:8001/webhook/document"
    webhook_secret: Optional[str] = None

    # Embedding API server (vLLM OpenAI-compatible embeddings endpoint).
    embedding_api_base: str = Field(
        default="http://complex.home.arpa:8102",
        validation_alias=AliasChoices("embedding_api_base", "EMBEDDING_API_BASE"),
    )
    embedding_model: str = "BAAI/bge-m3"

    # Text chunking for embedding
    chunk_size: int = Field(
        default=512,
        validation_alias=AliasChoices("chunk_size", "CHUNK_SIZE"),
    )
    chunk_overlap: int = 50

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
    # Per-page output cap for vision OCR calls.  A single page of text rarely
    # needs more than ~2000 tokens; a hard limit prevents runaway generation
    # when a model transcribes embedded binary data (e.g. base64 images in
    # web-archive documents) instead of summarising it.
    ocr_max_tokens: int = 4096
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

    # Extra kwargs to forward to LiteLLM for OCR calls.
    # Supports any parameter the downstream API accepts (e.g., top_p, top_k,
    # presence_penalty, or vLLM-specific fields via extra_body).
    ocr_extra_kwargs: Optional[Dict[str, Any]] = None

    # Extra kwargs to forward to LiteLLM for metadata extraction calls.
    metadata_extra_kwargs: Optional[Dict[str, Any]] = None

    # Extra kwargs to forward to LiteLLM for chat calls.
    chat_extra_kwargs: Optional[Dict[str, Any]] = None

    # LLM-based chunk situating for embeddings (contextual retrieval).
    # When situation_model is set, each chunk is prefixed with a short LLM-
    # generated context before being sent to the embedding model.
    # situation_context_chars caps how much of the full document text is passed
    # to the situation model; 0 = disabled (pass the full text).
    situation_model: Optional[str] = None
    situation_api_base: Optional[str] = None
    situation_temperature: float = 0.0
    situation_max_tokens: int = 200
    situation_context_chars: int = 0

    def get_ocr_litellm_kwargs(self) -> dict:
        """Hyperparameter kwargs for the OCR (vision) LiteLLM call."""
        kwargs: dict = {"max_tokens": self.ocr_max_tokens}
        if self.ocr_temperature is not None:
            kwargs["temperature"] = self.ocr_temperature
        if self.ocr_reasoning_effort:
            kwargs["reasoning_effort"] = self.ocr_reasoning_effort
        if self.ocr_extra_kwargs:
            kwargs.update(self.ocr_extra_kwargs)
        return kwargs

    def get_metadata_litellm_kwargs(self) -> dict:
        """Hyperparameter kwargs for the metadata extraction LiteLLM call."""
        kwargs: dict = {"max_tokens": self.metadata_max_tokens}
        if self.metadata_temperature is not None:
            kwargs["temperature"] = self.metadata_temperature
        effort = self.metadata_reasoning_effort or self.ocr_reasoning_effort
        if effort:
            kwargs["reasoning_effort"] = effort
        if self.metadata_extra_kwargs:
            kwargs.update(self.metadata_extra_kwargs)
        return kwargs

    def get_chat_litellm_kwargs(self) -> dict:
        """Hyperparameter kwargs for the chat LiteLLM call."""
        kwargs: dict = {"max_tokens": self.metadata_max_tokens}
        if self.chat_temperature is not None:
            kwargs["temperature"] = self.chat_temperature
        elif self.metadata_temperature is not None:
            kwargs["temperature"] = self.metadata_temperature

        effort = self.chat_reasoning_effort or self.metadata_reasoning_effort or self.ocr_reasoning_effort
        if effort:
            kwargs["reasoning_effort"] = effort
        if self.chat_extra_kwargs:
            kwargs.update(self.chat_extra_kwargs)
        return kwargs

    def get_situation_litellm_kwargs(self) -> dict:
        """Hyperparameter kwargs for the chunk situation LiteLLM call."""
        return {
            "max_tokens": self.situation_max_tokens,
            "temperature": self.situation_temperature,
        }

    @classmethod
    def from_env(cls) -> "AgentConfig":
        """Load configuration from environment variables and Docker secrets."""
        _inject_secrets()       # read *_FILE env vars and inject into os.environ
        litellm.drop_params = True
        return cls()            # pydantic-settings reads all env vars automatically
