"""
Redis-backed deduplicating queue for documents pending processing.

Uses a Redis Set so that multiple webhooks for the same document collapse
into a single entry.  All AI processing (OCR, metadata, embedding) is
triggered by the worker draining this queue.

Key: paperless-ai:pending   (Redis DB 1, isolated from Paperless DB 0)
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
