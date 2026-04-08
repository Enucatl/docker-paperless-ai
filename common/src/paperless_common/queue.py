"""
Redis-backed deduplicating queues for the document processing pipeline.
"""

from __future__ import annotations

import logging

import redis.asyncio as aioredis

log = logging.getLogger(__name__)

_WEBHOOK_SUPPRESS_PREFIX = "paperless-ai:webhook-suppress:"
_WEBHOOK_SUPPRESS_TTL_SECONDS = 300


def _webhook_suppress_key(doc_id: int) -> str:
    return f"{_WEBHOOK_SUPPRESS_PREFIX}{doc_id}"


class TaskQueues:
    """Stage-specific Redis queues for the decoupled tag-driven pipeline."""

    KEY_OCR = "paperless-ai:queue:ocr"
    KEY_METADATA = "paperless-ai:queue:metadata"
    KEY_EMBED = "paperless-ai:queue:embed"
    KEY_REFRESH = "paperless-ai:queue:refresh"

    def __init__(self, redis_url: str = "redis://broker:6379/1"):
        self._redis: aioredis.Redis = aioredis.from_url(
            redis_url, decode_responses=False
        )

    async def enqueue(self, doc_id: int, stage: str) -> bool:
        result = await self._redis.sadd(stage, doc_id)
        added = bool(result)
        if added:
            log.debug("Queued document %d → %s", doc_id, stage)
        return added

    async def enqueue_ocr(self, doc_id: int) -> bool:
        return await self.enqueue(doc_id, self.KEY_OCR)

    async def enqueue_metadata(self, doc_id: int) -> bool:
        return await self.enqueue(doc_id, self.KEY_METADATA)

    async def enqueue_embed(self, doc_id: int) -> bool:
        return await self.enqueue(doc_id, self.KEY_EMBED)

    async def enqueue_refresh(self, doc_id: int) -> bool:
        return await self.enqueue(doc_id, self.KEY_REFRESH)

    async def peek_stage(self, stage: str) -> set[int]:
        members = await self._redis.smembers(stage)
        return {int(member) for member in members}

    async def remove(self, doc_id: int, stage: str) -> None:
        await self._redis.srem(stage, doc_id)
        log.debug("Dequeued document %d from %s", doc_id, stage)

    async def pending_count(self) -> dict[str, int]:
        async with self._redis.pipeline() as pipe:
            pipe.scard(self.KEY_OCR)
            pipe.scard(self.KEY_METADATA)
            pipe.scard(self.KEY_EMBED)
            pipe.scard(self.KEY_REFRESH)
            ocr, metadata, embed, refresh = await pipe.execute()
        return {
            "ocr": int(ocr),
            "metadata": int(metadata),
            "embed": int(embed),
            "refresh": int(refresh),
        }

    async def suppress_webhook(
        self, doc_id: int, ttl_seconds: int = _WEBHOOK_SUPPRESS_TTL_SECONDS
    ) -> None:
        await self._redis.set(_webhook_suppress_key(doc_id), b"1", ex=ttl_seconds)

    async def clear_webhook_suppression(self, doc_id: int) -> None:
        await self._redis.delete(_webhook_suppress_key(doc_id))

    async def is_webhook_suppressed(self, doc_id: int) -> bool:
        return bool(await self._redis.exists(_webhook_suppress_key(doc_id)))

    async def close(self) -> None:
        await self._redis.aclose()
