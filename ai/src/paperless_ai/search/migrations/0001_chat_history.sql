CREATE TABLE conversations (
    id UUID PRIMARY KEY,
    owner_id TEXT,
    title TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX conversations_owner_updated_idx
    ON conversations (owner_id, updated_at DESC);

CREATE TABLE messages (
    id UUID PRIMARY KEY,
    position BIGINT GENERATED ALWAYS AS IDENTITY,
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    tool_activity_json JSONB,
    model TEXT,
    usage_json JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX messages_conversation_position_idx
    ON messages (conversation_id, position);

CREATE TABLE conversation_documents (
    message_id UUID NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    paperless_document_id INTEGER NOT NULL,
    matched BOOLEAN NOT NULL DEFAULT FALSE,
    inspected BOOLEAN NOT NULL DEFAULT FALSE,
    relevance JSONB,
    display_metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (message_id, paperless_document_id)
);
