"""
Redis-backed deduplicating queues for the document processing pipeline.
"""

import logging
import time

import redis.asyncio as aioredis

log = logging.getLogger(__name__)

_WEBHOOK_SUPPRESS_PREFIX = "paperless-ai:webhook-suppress:"
_WEBHOOK_SUPPRESS_TTL_SECONDS = 300
_RETRY_PREFIX = "paperless-ai:retry:"
_DELAYED_PREFIX = "paperless-ai:queue:delayed:"
_DEFAULT_RETRY_BASE_DELAY_SECONDS = 60
_DEFAULT_RETRY_MAX_DELAY_SECONDS = 3600


def _webhook_suppress_key(doc_id: int) -> str:
    return f"{_WEBHOOK_SUPPRESS_PREFIX}{doc_id}"


def _retry_key(stage: str, doc_id: int) -> str:
    return f"{_RETRY_PREFIX}{stage}:{doc_id}"


def _delayed_key(stage: str) -> str:
    return f"{_DELAYED_PREFIX}{stage}"


class TaskQueues:
    """Stage-specific Redis queues for the decoupled tag-driven pipeline."""

    KEY_OCR = "paperless-ai:queue:ocr"
    KEY_METADATA = "paperless-ai:queue:metadata"
    KEY_EMBED = "paperless-ai:queue:embed"
    KEY_REFRESH = "paperless-ai:queue:refresh"
    KEY_FAILED = "paperless-ai:queue:failed"

    def __init__(self, redis_url: str = "redis://broker:6379/1"):
        self._redis: aioredis.Redis = aioredis.from_url(
            redis_url, decode_responses=False
        )

    async def enqueue(self, doc_id: int, stage: str) -> bool:
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.zrem(_delayed_key(stage), doc_id)
            pipe.sadd(stage, doc_id)
            _removed, result = await pipe.execute()
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
        await self.release_due(stage)
        members = await self._redis.smembers(stage)
        return {int(member) for member in members}

    async def stage_size(self, stage: str) -> int:
        await self.release_due(stage)
        return int(await self._redis.scard(stage))

    async def remove(self, doc_id: int, stage: str) -> None:
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.srem(stage, doc_id)
            pipe.zrem(_delayed_key(stage), doc_id)
            pipe.delete(_retry_key(stage, doc_id))
            await pipe.execute()
        log.debug("Dequeued document %d from %s", doc_id, stage)

    async def mark_failure(
        self,
        doc_id: int,
        stage: str,
        *,
        max_attempts: int,
        base_delay_seconds: int = _DEFAULT_RETRY_BASE_DELAY_SECONDS,
        max_delay_seconds: int = _DEFAULT_RETRY_MAX_DELAY_SECONDS,
    ) -> tuple[int, bool]:
        """Increment retry state and atomically delay or dead-letter the item."""
        retry_count, moved, delay = await self._redis.eval(
            """
            local retry_count = redis.call("INCR", KEYS[1])
            redis.call("SREM", KEYS[3], ARGV[1])

            if retry_count >= tonumber(ARGV[2]) then
                redis.call("ZREM", KEYS[2], ARGV[1])
                redis.call("SADD", KEYS[4], ARGV[1])
                redis.call("DEL", KEYS[1])
                return {retry_count, 1, 0}
            end

            local delay = tonumber(ARGV[3]) * (2 ^ math.max(0, retry_count - 1))
            delay = math.min(delay, tonumber(ARGV[4]))
            redis.call("ZADD", KEYS[2], tonumber(ARGV[5]) + delay, ARGV[1])
            return {retry_count, 0, delay}
            """,
            4,
            _retry_key(stage, doc_id),
            _delayed_key(stage),
            stage,
            self.KEY_FAILED,
            doc_id,
            max_attempts,
            base_delay_seconds,
            max_delay_seconds,
            time.time(),
        )
        retry_count = int(retry_count)
        moved_to_failed = bool(moved)
        if moved_to_failed:
            log.debug("Document %d moved from %s to failed queue", doc_id, stage)
        else:
            log.debug(
                "Document %d delayed for %.0fs before retrying %s",
                doc_id,
                delay,
                stage,
            )
        return retry_count, moved_to_failed

    async def release_due(self, stage: str, *, now: float | None = None) -> int:
        """Move due delayed retries back into the ready stage set."""
        delayed_key = _delayed_key(stage)
        due = await self._redis.zrangebyscore(
            delayed_key, min=0, max=now if now is not None else time.time()
        )
        if not due:
            return 0

        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.zrem(delayed_key, *due)
            pipe.sadd(stage, *due)
            removed, added = await pipe.execute()
        log.debug("Released %d delayed retry item(s) into %s", int(removed), stage)
        return int(added)

    async def pending_count(self) -> dict[str, int]:
        await self.release_due(self.KEY_OCR)
        await self.release_due(self.KEY_METADATA)
        await self.release_due(self.KEY_EMBED)
        await self.release_due(self.KEY_REFRESH)
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
