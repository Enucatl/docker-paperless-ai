"""
Redis-backed deduplicating queues for the document processing pipeline.

DocumentQueue — legacy single-queue (kept for eval/test compat).
TaskQueues    — three-stage queue for the decoupled tag-driven pipeline:
    paperless-ai:queue:ocr      → vision OCR worker
    paperless-ai:queue:metadata → metadata extraction worker
    paperless-ai:queue:embed    → embedding worker

All queues use Redis Sets so duplicate webhooks for the same document
collapse into a single entry.  Redis DB 1 is used, isolated from Paperless DB 0.
"""

import logging

import redis.asyncio as aioredis

log = logging.getLogger(__name__)

_QUEUE_KEY = "paperless-ai:pending"


class DocumentQueue:
    def __init__(self, redis_url: str = "redis://broker:6379/1"):
        self._redis: aioredis.Redis = aioredis.from_url(
            redis_url, decode_responses=False
        )

    async def enqueue(self, doc_id: int) -> bool:
        """Add doc_id to the pending set.  Returns True if it was newly added."""
        result = await self._redis.sadd(_QUEUE_KEY, doc_id)
        added = bool(result)
        if added:
            log.debug("Queued document %d", doc_id)
        return added

    async def peek_all(self) -> set[int]:
        """Return all pending doc IDs without removing them."""
        members = await self._redis.smembers(_QUEUE_KEY)
        return {int(m) for m in members}

    async def remove(self, doc_id: int) -> None:
        """Remove doc_id from the queue after successful processing."""
        await self._redis.srem(_QUEUE_KEY, doc_id)
        log.debug("Dequeued document %d", doc_id)

    async def pending_count(self) -> int:
        return await self._redis.scard(_QUEUE_KEY)

    async def close(self) -> None:
        await self._redis.aclose()


class TaskQueues:
    """Three-stage Redis queues for the decoupled tag-driven pipeline."""

    KEY_OCR = "paperless-ai:queue:ocr"
    KEY_METADATA = "paperless-ai:queue:metadata"
    KEY_EMBED = "paperless-ai:queue:embed"

    def __init__(self, redis_url: str = "redis://broker:6379/1"):
        self._redis: aioredis.Redis = aioredis.from_url(
            redis_url, decode_responses=False
        )

    async def enqueue(self, doc_id: int, stage: str) -> bool:
        """Add doc_id to the given stage queue. Returns True if newly added."""
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

    async def peek_stage(self, stage: str) -> set[int]:
        """Return all doc IDs in the given stage queue without removing them."""
        members = await self._redis.smembers(stage)
        return {int(m) for m in members}

    async def remove(self, doc_id: int, stage: str) -> None:
        """Remove doc_id from the given stage queue after successful processing."""
        await self._redis.srem(stage, doc_id)
        log.debug("Dequeued document %d from %s", doc_id, stage)

    async def pending_count(self) -> dict[str, int]:
        """Return pending counts for all three stages."""
        ocr = await self._redis.scard(self.KEY_OCR)
        metadata = await self._redis.scard(self.KEY_METADATA)
        embed = await self._redis.scard(self.KEY_EMBED)
        return {"ocr": int(ocr), "metadata": int(metadata), "embed": int(embed)}

    async def close(self) -> None:
        await self._redis.aclose()
