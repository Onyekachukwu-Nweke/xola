-- L2-01: memories table
-- Stores semantic facts that persist across conversations.
-- Embeddings are stored as pgvector vectors and retrieved by cosine similarity.

-- Enable pgvector extension (idempotent)
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS memories (
    id         UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    content    TEXT         NOT NULL,
    -- 1536 dimensions = text-embedding-3-small default; adjust via config if model changes
    embedding  vector(1536) NOT NULL,
    metadata   JSONB        NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ  NOT NULL DEFAULT now()
);

-- IVFFlat approximate nearest-neighbour index using cosine distance.
-- lists = 100 is appropriate for up to ~1 M rows; tune as data grows.
CREATE INDEX IF NOT EXISTS memories_embedding_idx
    ON memories
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
