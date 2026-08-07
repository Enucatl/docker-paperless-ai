"""
Centralised configuration for the AI processing service.

Reads all settings from environment variables, injects Docker secrets,
and exposes a validated AgentConfig Pydantic model.
"""

import os
from importlib.resources import files as _pkg_files
from typing import Any, Dict, List, Literal, Optional

from paperless_common.secrets import read_secret
from pydantic import AliasChoices, BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _inject_secrets() -> None:
    """Read Docker secrets (_FILE variants) and inject them into os.environ."""
    for key in (
        "GOOGLE_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "HF_TOKEN",
        "PAPERLESS_TOKEN",
        "WEBHOOK_SECRET",
    ):
        val = read_secret(key)
        if val:
            os.environ[key] = val


def _load_prompt(name: str) -> str:
    """Load a prompt file bundled as package data."""
    return _pkg_files("paperless_ai").joinpath(name).read_text(encoding="utf-8").strip()


class JuryMemberConfig(BaseModel):
    """Configuration for a single judge in the LLM-as-a-jury panel."""

    model: str
    endpoint: Optional[str] = Field(
        default=None,
        validation_alias="INFERENCE_EVALUATION_ENDPOINT",
    )
    reasoning_effort: Optional[str] = None
    temperature: Optional[float] = None


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

    @field_validator(
        "ocr_endpoint",
        "metadata_endpoint",
        "chat_endpoint",
        "embedding_endpoint",
        "situation_endpoint",
        mode="before",
    )
    @classmethod
    def empty_endpoint_is_none(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = str(value).strip()
        return stripped or None

    ocr_model: str = Field(
        default="gemini-2.5-flash",
        validation_alias="INFERENCE_OCR_MODEL",
    )
    metadata_model: str = Field(validation_alias="INFERENCE_METADATA_MODEL")
    chat_model: str = Field(validation_alias="INFERENCE_CHAT_MODEL")
    ocr_endpoint: Optional[str] = Field(
        default=None,
        validation_alias="INFERENCE_OCR_ENDPOINT",
    )
    metadata_endpoint: Optional[str] = Field(
        default=None,
        validation_alias="INFERENCE_METADATA_ENDPOINT",
    )
    chat_endpoint: Optional[str] = Field(
        default=None,
        validation_alias="INFERENCE_CHAT_ENDPOINT",
    )
    ocr_reasoning_effort: Optional[str] = Field(
        default="minimal",
        validation_alias=AliasChoices(
            "INFERENCE_OCR_REASONING_EFFORT",
            "OCR_REASONING_EFFORT",
            "ocr_reasoning_effort",
        ),
    )
    metadata_reasoning_effort: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices(
            "INFERENCE_METADATA_REASONING_EFFORT",
            "METADATA_REASONING_EFFORT",
            "metadata_reasoning_effort",
        ),
    )
    metadata_response_format: Literal["auto", "json_schema", "json_object", "none"] = (
        Field(
            default="auto",
            validation_alias=AliasChoices(
                "metadata_response_format",
                "INFERENCE_METADATA_RESPONSE_FORMAT",
                "METADATA_RESPONSE_FORMAT",
            ),
        )
    )
    chat_reasoning_effort: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices(
            "INFERENCE_CHAT_REASONING_EFFORT",
            "CHAT_REASONING_EFFORT",
            "chat_reasoning_effort",
        ),
    )

    poll_interval: int = 300
    llm_retries: int = 3
    stage_max_attempts: int = 3
    ocr_concurrency: int = 4
    # TAG_PENDING is the legacy name — keep for backward compat
    tag_ocr: str = Field(
        default="ai:run-ocr",
        validation_alias=AliasChoices("tag_ocr", "TAG_OCR", "TAG_PENDING"),
    )
    tag_metadata: str = "ai:run-metadata"
    tag_embed: str = "ai:run-embed"
    dry_run: bool = False

    # TEMPERATURE is a generic fallback when specific ones are not set
    ocr_temperature: Optional[float] = Field(
        default=None,
        validation_alias=AliasChoices(
            "INFERENCE_OCR_TEMPERATURE",
            "OCR_TEMPERATURE",
            "TEMPERATURE",
            "ocr_temperature",
        ),
    )
    metadata_temperature: Optional[float] = Field(
        default=None,
        validation_alias=AliasChoices(
            "INFERENCE_METADATA_TEMPERATURE",
            "METADATA_TEMPERATURE",
            "TEMPERATURE",
            "metadata_temperature",
        ),
    )
    chat_temperature: Optional[float] = Field(
        default=None,
        validation_alias=AliasChoices(
            "INFERENCE_CHAT_TEMPERATURE", "CHAT_TEMPERATURE", "chat_temperature"
        ),
    )

    ocr_prompt: str = Field(default_factory=lambda: _load_prompt("prompt.txt"))
    metadata_prompt: str = Field(
        default_factory=lambda: _load_prompt("metadata_prompt.txt")
    )

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
    embedding_endpoint: str = Field(
        default="http://complex.home.arpa:8102",
        validation_alias=AliasChoices("INFERENCE_EMBEDDING_ENDPOINT"),
    )
    embedding_model: str = Field(
        default="BAAI/bge-m3",
        validation_alias=AliasChoices("INFERENCE_EMBEDDING_MODEL"),
    )

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
    ocr_max_tokens: int = Field(
        default=4096,
        validation_alias=AliasChoices(
            "INFERENCE_OCR_MAX_TOKENS", "OCR_MAX_TOKENS", "ocr_max_tokens"
        ),
    )
    metadata_max_tokens: int = Field(
        default=1000,
        validation_alias=AliasChoices(
            "INFERENCE_METADATA_MAX_TOKENS",
            "METADATA_MAX_TOKENS",
            "metadata_max_tokens",
        ),
    )
    chat_max_tokens: int = Field(
        default=1000,
        validation_alias=AliasChoices(
            "INFERENCE_CHAT_MAX_TOKENS", "CHAT_MAX_TOKENS", "chat_max_tokens"
        ),
    )

    # Dotted import path to the agent class to use in eval experiments.
    agent_class: str = "paperless_ai.agents.smart_graph_agent.SmartDocumentAgent"

    # Model used as LLM judge for title quality evaluation.
    # Should be a strong, fixed model independent of the experiment being
    # evaluated to avoid self-grading bias.
    llm_judge_model: str = Field(
        default="gemini-2.5-flash",
        validation_alias=AliasChoices("INFERENCE_EVALUATION_MODEL"),
    )
    evaluation_endpoint: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("INFERENCE_EVALUATION_ENDPOINT"),
    )
    evaluation_temperature: Optional[float] = Field(
        default=None, alias="INFERENCE_EVALUATION_TEMPERATURE"
    )
    evaluation_reasoning_effort: Optional[str] = Field(
        default=None, alias="INFERENCE_EVALUATION_REASONING_EFFORT"
    )
    evaluation_max_tokens: Optional[int] = Field(
        default=None, alias="INFERENCE_EVALUATION_MAX_TOKENS"
    )
    evaluation_extra_kwargs: Optional[Dict[str, Any]] = Field(
        default=None, alias="INFERENCE_EVALUATION_EXTRA_KWARGS"
    )

    # Optional jury of LLM judges for title quality evaluation.
    # When set, each member votes independently and the final score is
    # determined by majority vote, which improves alignment with human
    # judgment compared to a single judge.
    # If None, falls back to a single judge using llm_judge_model.
    jury: Optional[List[JuryMemberConfig]] = None

    # Extra kwargs to forward to OpenAI-compatible inference for OCR calls.
    # Supports any parameter the downstream API accepts (e.g., top_p, top_k,
    # presence_penalty, or vLLM-specific fields via extra_body).
    ocr_extra_kwargs: Optional[Dict[str, Any]] = Field(
        default=None,
        validation_alias=AliasChoices(
            "INFERENCE_OCR_EXTRA_KWARGS", "OCR_EXTRA_KWARGS", "ocr_extra_kwargs"
        ),
    )

    # Extra kwargs to forward to OpenAI-compatible inference for metadata extraction calls.
    metadata_extra_kwargs: Optional[Dict[str, Any]] = Field(
        default=None,
        validation_alias=AliasChoices(
            "INFERENCE_METADATA_EXTRA_KWARGS",
            "METADATA_EXTRA_KWARGS",
            "metadata_extra_kwargs",
        ),
    )

    # Extra kwargs to forward to OpenAI-compatible inference for chat calls.
    chat_extra_kwargs: Optional[Dict[str, Any]] = Field(
        default=None,
        validation_alias=AliasChoices(
            "INFERENCE_CHAT_EXTRA_KWARGS", "CHAT_EXTRA_KWARGS", "chat_extra_kwargs"
        ),
    )

    # LLM-based chunk situating for embeddings (contextual retrieval).
    # When situation_model is set, each chunk is prefixed with a short LLM-
    # generated context before being sent to the embedding model.
    # situation_context_chars caps how much of the full document text is passed
    # to the situation model; 0 = disabled (pass the full text).
    situation_model: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("INFERENCE_SITUATION_MODEL"),
    )
    situation_endpoint: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("INFERENCE_SITUATION_ENDPOINT"),
    )
    situation_temperature: float = Field(
        default=0.0,
        validation_alias=AliasChoices(
            "INFERENCE_SITUATION_TEMPERATURE",
            "SITUATION_TEMPERATURE",
            "situation_temperature",
        ),
    )
    situation_reasoning_effort: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices(
            "INFERENCE_SITUATION_REASONING_EFFORT",
            "SITUATION_REASONING_EFFORT",
            "situation_reasoning_effort",
        ),
    )
    situation_max_tokens: int = Field(
        default=200,
        validation_alias=AliasChoices(
            "INFERENCE_SITUATION_MAX_TOKENS",
            "SITUATION_MAX_TOKENS",
            "situation_max_tokens",
        ),
    )
    situation_extra_kwargs: Optional[Dict[str, Any]] = Field(
        default=None,
        validation_alias=AliasChoices(
            "INFERENCE_SITUATION_EXTRA_KWARGS",
            "SITUATION_EXTRA_KWARGS",
            "situation_extra_kwargs",
        ),
    )
    situation_context_chars: int = Field(
        default=0, alias="INFERENCE_SITUATION_CONTEXT_CHARS"
    )
    situation_concurrency: int = 8

    @field_validator(
        "ocr_model",
        "metadata_model",
        "chat_model",
        "embedding_model",
        "situation_model",
        "llm_judge_model",
        mode="before",
    )
    @classmethod
    def normalize_model_prefix(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return str(value).removeprefix("openrouter/").removeprefix("openai/")

    def get_ocr_kwargs(self) -> dict:
        """Hyperparameter kwargs for OCR (vision) requests."""
        kwargs: dict = {"max_tokens": self.ocr_max_tokens}
        if self.ocr_temperature is not None:
            kwargs["temperature"] = self.ocr_temperature
        if self.ocr_reasoning_effort:
            kwargs["reasoning_effort"] = self.ocr_reasoning_effort
        if self.ocr_extra_kwargs:
            kwargs.update(self.ocr_extra_kwargs)
        return kwargs

    def get_metadata_kwargs(self) -> dict:
        """Hyperparameter kwargs for metadata extraction requests."""
        kwargs: dict = {"max_tokens": self.metadata_max_tokens}
        if self.metadata_temperature is not None:
            kwargs["temperature"] = self.metadata_temperature
        effort = self.metadata_reasoning_effort or self.ocr_reasoning_effort
        if effort:
            kwargs["reasoning_effort"] = effort
        if self.metadata_extra_kwargs:
            kwargs.update(self.metadata_extra_kwargs)
        return kwargs

    def get_chat_kwargs(self) -> dict:
        """Hyperparameter kwargs for chat requests."""
        kwargs: dict = {"max_tokens": self.chat_max_tokens}
        if self.chat_temperature is not None:
            kwargs["temperature"] = self.chat_temperature

        if self.chat_reasoning_effort:
            kwargs["reasoning_effort"] = self.chat_reasoning_effort
        if self.chat_extra_kwargs:
            kwargs.update(self.chat_extra_kwargs)
        return kwargs

    def get_situation_kwargs(self) -> dict:
        """Hyperparameter kwargs for chunk-situation requests."""
        kwargs = {
            "max_tokens": self.situation_max_tokens,
            "temperature": self.situation_temperature,
        }
        if self.situation_reasoning_effort:
            kwargs["reasoning_effort"] = self.situation_reasoning_effort
        if self.situation_extra_kwargs:
            kwargs.update(self.situation_extra_kwargs)
        return kwargs

    @classmethod
    def from_env(cls) -> "AgentConfig":
        """Load configuration from environment variables and Docker secrets."""
        _inject_secrets()  # read *_FILE env vars and inject into os.environ
        return cls()  # pydantic-settings reads all env vars automatically
