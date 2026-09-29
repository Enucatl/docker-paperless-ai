"""
Centralised configuration for the AI processing service.

Reads all settings from environment variables, injects Docker secrets,
and exposes a validated AgentConfig Pydantic model.
"""

import os
from importlib.resources import files as _pkg_files
from typing import Any, Dict, Literal, Optional

from paperless_common.secrets import read_secret
from pydantic import AliasChoices, Field, field_validator
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
        "TYPESAFE_API_KEY",
    ):
        val = read_secret(key)
        if val:
            os.environ[key] = val


def _load_prompt(name: str) -> str:
    """Load a prompt file bundled as package data."""
    return _pkg_files("paperless_ai").joinpath(name).read_text(encoding="utf-8").strip()


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
        mode="before",
    )
    @classmethod
    def empty_endpoint_is_none(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = str(value).strip()
        return stripped or None

    ocr_timeout: float = Field(
        default=600,
        gt=0,
        validation_alias=AliasChoices(
            "INFERENCE_OCR_TIMEOUT", "INFERENCE_PADDLE_TIMEOUT"
        ),
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
    metadata_reasoning_effort: Optional[str] = Field(
        default="minimal",
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
    correspondent_match_threshold: float = Field(
        default=0.80,
        validation_alias="CORRESPONDENT_MATCH_THRESHOLD",
    )
    correspondent_cleanup_candidate_threshold: float = Field(
        default=0.65,
        validation_alias="CORRESPONDENT_CLEANUP_CANDIDATE_THRESHOLD",
        ge=0,
        le=1,
    )
    correspondent_cleanup_max_rounds: int = Field(
        default=6,
        validation_alias="CORRESPONDENT_CLEANUP_MAX_ROUNDS",
        ge=1,
    )
    correspondent_cleanup_db_host: str = Field(
        default="db", validation_alias="CORRESPONDENT_CLEANUP_DB_HOST"
    )
    correspondent_cleanup_db_name: str = Field(
        default="paperless", validation_alias="CORRESPONDENT_CLEANUP_DB_NAME"
    )
    correspondent_cleanup_db_user: str = Field(
        default="paperless", validation_alias="CORRESPONDENT_CLEANUP_DB_USER"
    )
    correspondent_cleanup_db_password: str = Field(
        default="", validation_alias="CORRESPONDENT_CLEANUP_DB_PASSWORD"
    )
    typesafe_api_key: str | None = Field(
        default=None, validation_alias="TYPESAFE_API_KEY"
    )
    typesafe_model: str = Field(default="jev-latest", validation_alias="TYPESAFE_MODEL")
    typesafe_endpoint: str | None = Field(
        default=None, validation_alias="TYPESAFE_ENDPOINT"
    )
    # TAG_PENDING is the legacy name — keep for backward compat
    tag_ocr: str = Field(
        default="ai:run-ocr",
        validation_alias=AliasChoices("tag_ocr", "TAG_OCR", "TAG_PENDING"),
    )
    tag_metadata: str = "ai:run-metadata"
    dry_run: bool = False

    # TEMPERATURE is a generic fallback when specific ones are not set
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

    metadata_prompt: str = Field(
        default_factory=lambda: _load_prompt("metadata_prompt.txt")
    )

    # Redis queue (DB 1, isolated from Paperless DB 0)
    redis_url: str = "redis://broker:6379/1"

    # Paperless workflow automation
    manage_paperless_workflows: bool = True
    paperless_webhook_url: str = "http://webhook-listener:8001/webhook/document"
    webhook_secret: Optional[str] = None

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

    @field_validator(
        "metadata_model",
        "chat_model",
        mode="before",
    )
    @classmethod
    def normalize_model_prefix(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return str(value).removeprefix("openrouter/").removeprefix("openai/")

    def get_metadata_kwargs(self) -> dict:
        """Hyperparameter kwargs for metadata extraction requests."""
        kwargs: dict = {"max_tokens": self.metadata_max_tokens}
        if self.metadata_temperature is not None:
            kwargs["temperature"] = self.metadata_temperature
        effort = self.metadata_reasoning_effort
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

    @classmethod
    def from_env(cls) -> "AgentConfig":
        """Load configuration from environment variables and Docker secrets."""
        _inject_secrets()  # read *_FILE env vars and inject into os.environ
        return cls()  # pydantic-settings reads all env vars automatically
