"""PostgreSQL persistence for browser copilot conversations."""

from __future__ import annotations

import uuid
from importlib.resources import files
from typing import Any

from psycopg import AsyncConnection
from psycopg.types.json import Jsonb


def _timestamp(value: Any) -> str:
    """Return a JSON-safe timestamp string."""
    return value.isoformat()


def _conversation(row: tuple) -> dict[str, str]:
    """Convert a conversation database row to its API representation."""
    return {
        "id": str(row[0]),
        "title": row[2],
        "created_at": _timestamp(row[3]),
        "updated_at": _timestamp(row[4]),
    }


class ChatStore:
    """Own the chat-history schema and user-scoped conversation operations."""

    def __init__(self, database_url: str, password: str | None = None):
        """Create a store backed by a dedicated PostgreSQL database."""
        self._database_url = database_url
        self._password = password

    async def _connect(self) -> AsyncConnection:
        """Open a short-lived PostgreSQL connection."""
        return await AsyncConnection.connect(
            self._database_url, password=self._password
        )

    async def migrate(self) -> None:
        """Apply this service's SQL migrations exactly once."""
        async with await self._connect() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chat_schema_migrations (
                        name TEXT PRIMARY KEY,
                        applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
                    )
                    """
                )
                await cursor.execute("SELECT name FROM chat_schema_migrations")
                applied = {row[0] for row in await cursor.fetchall()}
                migration_dir = files("paperless_ai.search").joinpath("migrations")
                for migration in sorted(
                    migration_dir.iterdir(), key=lambda item: item.name
                ):
                    if migration.name in applied:
                        continue
                    async with connection.transaction():
                        await cursor.execute(migration.read_text())
                        await cursor.execute(
                            "INSERT INTO chat_schema_migrations (name) VALUES (%s)",
                            (migration.name,),
                        )

    async def create_conversation(
        self, owner_id: str | None, title: str | None = None
    ) -> dict[str, str]:
        """Create and return a conversation owned by the current user."""
        conversation_id = uuid.uuid4()
        async with await self._connect() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    """
                    INSERT INTO conversations (id, owner_id, title)
                    VALUES (%s, %s, %s)
                    RETURNING id, owner_id, title, created_at, updated_at
                    """,
                    (conversation_id, owner_id, title or "New conversation"),
                )
                return _conversation(await cursor.fetchone())

    async def list_conversations(self, owner_id: str | None) -> list[dict[str, str]]:
        """List only conversations belonging to the current user."""
        async with await self._connect() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    """
                    SELECT id, owner_id, title, created_at, updated_at
                    FROM conversations
                    WHERE owner_id IS NOT DISTINCT FROM %s
                    ORDER BY updated_at DESC
                    """,
                    (owner_id,),
                )
                return [_conversation(row) for row in await cursor.fetchall()]

    async def load_conversation(
        self, conversation_id: str, owner_id: str | None
    ) -> dict[str, Any] | None:
        """Load one owned conversation, its messages, and source snapshots."""
        async with await self._connect() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    """
                    SELECT id, owner_id, title, created_at, updated_at
                    FROM conversations
                    WHERE id = %s AND owner_id IS NOT DISTINCT FROM %s
                    """,
                    (conversation_id, owner_id),
                )
                row = await cursor.fetchone()
                if row is None:
                    return None
                conversation = _conversation(row)
                await cursor.execute(
                    """
                    SELECT id, role, content, tool_activity_json, model, usage_json, created_at
                    FROM messages WHERE conversation_id = %s ORDER BY position
                    """,
                    (conversation_id,),
                )
                messages = [
                    {
                        "id": str(message[0]),
                        "role": message[1],
                        "content": message[2],
                        "tool_activity": message[3] or [],
                        "model": message[4],
                        "usage": message[5],
                        "created_at": _timestamp(message[6]),
                        "sources": [],
                    }
                    for message in await cursor.fetchall()
                ]
                by_id = {message["id"]: message for message in messages}
                await cursor.execute(
                    """
                    SELECT message_id, paperless_document_id, matched, inspected,
                           relevance, display_metadata_json
                    FROM conversation_documents
                    WHERE message_id IN (
                        SELECT id FROM messages WHERE conversation_id = %s
                    )
                    ORDER BY paperless_document_id
                    """,
                    (conversation_id,),
                )
                for document in await cursor.fetchall():
                    message = by_id.get(str(document[0]))
                    if message is not None:
                        message["sources"].append(
                            {
                                **(document[5] or {}),
                                "id": document[1],
                                "matched": document[2],
                                "inspected": document[3],
                                "relevance": document[4],
                            }
                        )
                conversation["messages"] = messages
                return conversation

    async def rename_conversation(
        self, conversation_id: str, owner_id: str | None, title: str
    ) -> dict[str, str] | None:
        """Rename one owned conversation and return its updated summary."""
        async with await self._connect() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    """
                    UPDATE conversations SET title = %s, updated_at = now()
                    WHERE id = %s AND owner_id IS NOT DISTINCT FROM %s
                    RETURNING id, owner_id, title, created_at, updated_at
                    """,
                    (title, conversation_id, owner_id),
                )
                row = await cursor.fetchone()
                return _conversation(row) if row else None

    async def delete_conversation(
        self, conversation_id: str, owner_id: str | None
    ) -> bool:
        """Delete one owned conversation and its cascading child records."""
        async with await self._connect() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    """
                    DELETE FROM conversations
                    WHERE id = %s AND owner_id IS NOT DISTINCT FROM %s
                    """,
                    (conversation_id, owner_id),
                )
                return cursor.rowcount == 1

    async def append_turn(
        self,
        conversation_id: str,
        owner_id: str | None,
        user_content: str,
        assistant_content: str,
        tool_activity: list[dict[str, Any]],
        model: str,
        usage: dict[str, int] | None,
        sources: list[dict[str, Any]],
    ) -> bool:
        """Persist one completed user/assistant turn transactionally."""
        user_message_id = uuid.uuid4()
        assistant_message_id = uuid.uuid4()
        async with await self._connect() as connection:
            async with connection.transaction():
                async with connection.cursor() as cursor:
                    await cursor.execute(
                        """
                        SELECT title FROM conversations
                        WHERE id = %s AND owner_id IS NOT DISTINCT FROM %s FOR UPDATE
                        """,
                        (conversation_id, owner_id),
                    )
                    row = await cursor.fetchone()
                    if row is None:
                        return False
                    await cursor.execute(
                        """
                        INSERT INTO messages (id, conversation_id, role, content)
                        VALUES (%s, %s, 'user', %s)
                        """,
                        (user_message_id, conversation_id, user_content),
                    )
                    await cursor.execute(
                        """
                        INSERT INTO messages
                            (id, conversation_id, role, content, tool_activity_json, model, usage_json)
                        VALUES (%s, %s, 'assistant', %s, %s, %s, %s)
                        """,
                        (
                            assistant_message_id,
                            conversation_id,
                            assistant_content,
                            Jsonb(tool_activity),
                            model,
                            Jsonb(usage) if usage is not None else None,
                        ),
                    )
                    for source in sources:
                        await cursor.execute(
                            """
                            INSERT INTO conversation_documents
                                (message_id, paperless_document_id, matched, inspected,
                                 relevance, display_metadata_json)
                            VALUES (%s, %s, %s, %s, %s, %s)
                            """,
                            (
                                assistant_message_id,
                                source["id"],
                                bool(source.get("matched")),
                                bool(source.get("inspected")),
                                Jsonb(source["relevance"])
                                if source.get("relevance") is not None
                                else None,
                                Jsonb(source),
                            ),
                        )
                    title = row[0]
                    if title == "New conversation":
                        await cursor.execute(
                            """
                            UPDATE conversations
                            SET title = %s, updated_at = now() WHERE id = %s
                            """,
                            (user_content.strip()[:120] or title, conversation_id),
                        )
                    else:
                        await cursor.execute(
                            "UPDATE conversations SET updated_at = now() WHERE id = %s",
                            (conversation_id,),
                        )
        return True
