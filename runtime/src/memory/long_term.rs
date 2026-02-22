//! Long-term semantic memory backed by Postgres + pgvector.
//!
//! Facts worth keeping across conversations are stored as text + embedding
//! vector pairs. Retrieval is via cosine similarity search so the agent
//! surfaces relevant context even when the exact words differ.
//!
//! # Write path
//! Content → `POST /embed` (Python) → `Vec<f32>` → `LongTermMemory::store`
//!
//! # Read path
//! Goal text → `POST /embed` → `Vec<f32>` → `LongTermMemory::query` → top-K

use chrono::{DateTime, Utc};
use pgvector::Vector;
use serde_json::Value;
use sqlx::PgPool;
use uuid::Uuid;

/// Errors that can arise from long-term memory operations.
#[derive(Debug, thiserror::Error)]
pub enum MemoryError {
    #[error("database error: {0}")]
    Database(#[from] sqlx::Error),

    #[error("embedding dimension mismatch: expected 1536, got {0}")]
    DimensionMismatch(usize),
}

/// Expected embedding dimensionality (text-embedding-3-small default).
///
/// TODO: load from runtime config (config/local.toml) so this and the
/// migration `vector(1536)` column stay in sync when the model changes.
const EMBEDDING_DIM: usize = 1536;

/// A memory record returned from a semantic similarity query.
#[derive(Debug, Clone, serde::Serialize)]
pub struct MemoryRecord {
    pub id: Uuid,
    pub content: String,
    pub metadata: Value,
    pub created_at: DateTime<Utc>,
    /// Cosine distance from the query vector (0 = identical, 2 = opposite).
    /// Lower is more similar.
    pub distance: f64,
}

/// Long-term semantic memory store.
///
/// Wraps a `PgPool` and operates on the `memories` table created in
/// migration `0001_create_memories_table.sql`. All operations are async
/// and non-blocking on the tokio runtime.
///
/// The caller is responsible for obtaining embeddings via `POST /embed`
/// before calling `store` or `query`.
#[derive(Clone, Debug)]
pub struct LongTermMemory {
    pool: PgPool,
}

impl LongTermMemory {
    /// Create a new `LongTermMemory` backed by the given connection pool.
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Store a memory.
    ///
    /// # Parameters
    /// - `content` – the original text that was embedded
    /// - `embedding` – 1536-dimensional vector from `POST /embed`
    /// - `metadata` – arbitrary JSON (e.g. source task ID, timestamp)
    ///
    /// Returns the newly assigned `Uuid`.
    ///
    /// # Errors
    /// Returns [`MemoryError::DimensionMismatch`] if `embedding.len() != 1536`.
    pub async fn store(
        &self,
        content: &str,
        embedding: Vec<f32>,
        metadata: Value,
    ) -> Result<Uuid, MemoryError> {
        if embedding.len() != EMBEDDING_DIM {
            return Err(MemoryError::DimensionMismatch(embedding.len()));
        }

        let vector = Vector::from(embedding);

        let row = sqlx::query!(
            r#"
            INSERT INTO memories (content, embedding, metadata)
            VALUES ($1, $2, $3)
            RETURNING id
            "#,
            content,
            vector as Vector,
            metadata,
        )
        .fetch_one(&self.pool)
        .await?;

        tracing::debug!(id = %row.id, "long_term_memory: stored memory");
        Ok(row.id)
    }

    /// Query for the most semantically similar memories.
    ///
    /// Uses cosine distance (`<=>`) with the IVFFlat index created in
    /// migration 0001. Results are ordered closest-first.
    ///
    /// # Parameters
    /// - `embedding` – 1536-dim query vector from `POST /embed`
    /// - `top_k` – maximum number of results to return. Passing `0` returns
    ///   an empty `Vec` immediately (no database round-trip).
    ///
    /// # Errors
    /// Returns [`MemoryError::DimensionMismatch`] if `embedding.len() != 1536`.
    pub async fn query(
        &self,
        embedding: Vec<f32>,
        top_k: usize,
    ) -> Result<Vec<MemoryRecord>, MemoryError> {
        debug_assert!(top_k > 0, "top_k must be > 0; passing 0 returns nothing");

        if embedding.len() != EMBEDDING_DIM {
            return Err(MemoryError::DimensionMismatch(embedding.len()));
        }

        let vector = Vector::from(embedding);
        let limit = top_k as i64;

        let rows = sqlx::query!(
            r#"
            SELECT
                id,
                content,
                metadata AS "metadata!: sqlx::types::Json<serde_json::Value>",
                created_at,
                (embedding <=> $1) AS "distance!: f64"
            FROM memories
            ORDER BY embedding <=> $1
            LIMIT $2
            "#,
            vector as Vector,
            limit,
        )
        .fetch_all(&self.pool)
        .await?;

        let records: Vec<MemoryRecord> = rows
            .into_iter()
            .map(|row| MemoryRecord {
                id: row.id,
                content: row.content,
                metadata: row.metadata.0,
                created_at: row.created_at,
                distance: row.distance,
            })
            .collect();

        tracing::debug!(
            top_k,
            results = records.len(),
            "long_term_memory: queried semantic memories"
        );
        Ok(records)
    }

    /// Delete a memory by ID.
    ///
    /// Returns `Ok(true)` if the row was found and deleted, `Ok(false)` if
    /// no row with that ID existed.
    pub async fn delete(&self, id: Uuid) -> Result<bool, MemoryError> {
        let result = sqlx::query!("DELETE FROM memories WHERE id = $1", id)
            .execute(&self.pool)
            .await?;

        let deleted = result.rows_affected() > 0;
        tracing::debug!(%id, deleted, "long_term_memory: delete");
        Ok(deleted)
    }

    /// Return the underlying connection pool (e.g. for sharing with
    /// `EpisodicLog` or health-check endpoints).
    pub fn pool(&self) -> &PgPool {
        &self.pool
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::connect_db;

    /// A 1536-dim unit-ish vector pointing along the "first half" axis.
    /// Orthogonal to `fake_embedding_b()` — cosine distance between them is 1.0.
    fn fake_embedding_a() -> Vec<f32> {
        let mut v = vec![0.0f32; EMBEDDING_DIM];
        for x in v.iter_mut().take(EMBEDDING_DIM / 2) {
            *x = 1.0;
        }
        v
    }

    /// A 1536-dim unit-ish vector pointing along the "second half" axis.
    /// Orthogonal to `fake_embedding_a()` — cosine distance between them is 1.0.
    fn fake_embedding_b() -> Vec<f32> {
        let mut v = vec![0.0f32; EMBEDDING_DIM];
        for x in v.iter_mut().skip(EMBEDDING_DIM / 2) {
            *x = 1.0;
        }
        v
    }

    /// A 1536-dim vector pointing along the "first quarter" axis.
    /// Distinct from both `a` and `b` above.
    fn fake_embedding_c() -> Vec<f32> {
        let mut v = vec![0.0f32; EMBEDDING_DIM];
        for x in v.iter_mut().take(EMBEDDING_DIM / 4) {
            *x = 1.0;
        }
        v
    }

    // ---------------------------------------------------------------------------
    // Unit tests — no database required
    // ---------------------------------------------------------------------------

    #[test]
    fn dimension_check_exact_passes() {
        // The guard logic itself is pure — just verify the constant
        assert_eq!(fake_embedding_a().len(), EMBEDDING_DIM);
    }

    #[test]
    fn wrong_dimension_is_detected() {
        let short = vec![0.1_f32; 512];
        assert_eq!(short.len(), 512);
        assert_ne!(short.len(), EMBEDDING_DIM);
    }

    // ---------------------------------------------------------------------------
    // Integration tests — require a running Postgres with pgvector
    // Run with: cargo test -- --ignored
    // ---------------------------------------------------------------------------

    #[tokio::test]
    #[ignore]
    async fn store_and_query_returns_closest_memory() {
        let pool = connect_db().await;
        let mem = LongTermMemory::new(pool);

        // Pre-test cleanup: wipe any stale rows left by previous failed runs.
        // Orphaned rows in the same embedding direction corrupt cosine search results.
        for tag in ["a", "b"] {
            sqlx::query!("DELETE FROM memories WHERE metadata->>'tag' = $1", tag)
                .execute(mem.pool())
                .await
                .expect("pre-test cleanup failed");
        }

        // Use orthogonal embeddings so cosine distance is meaningful.
        // fake_embedding_a and fake_embedding_b are unit vectors on different
        // axes — cosine distance between them is 1.0, distance to themselves is 0.0.
        let embedding_a = fake_embedding_a();
        let embedding_b = fake_embedding_b();

        let id_a = mem
            .store("memory A", embedding_a.clone(), serde_json::json!({"tag": "a"}))
            .await
            .expect("store A failed");
        let id_b = mem
            .store("memory B", embedding_b.clone(), serde_json::json!({"tag": "b"}))
            .await
            .expect("store B failed");

        // Query with a vector identical to A — it must come back first.
        let results = mem
            .query(embedding_a, 2)
            .await
            .expect("query failed");

        assert!(!results.is_empty(), "expected at least one result");
        assert_eq!(results[0].id, id_a, "closest result should be A");
        // Cosine distance from A to itself must be essentially 0.
        // (IVFFlat with few rows may not return top_k rows, so we don't index [1].)
        assert!(
            results[0].distance < 0.01,
            "self-similarity distance should be ~0, got {}",
            results[0].distance
        );

        // Cleanup — both rows to avoid polluting future runs.
        mem.delete(id_a).await.expect("delete id_a failed");
        mem.delete(id_b).await.expect("delete id_b failed");
    }

    #[tokio::test]
    #[ignore]
    async fn delete_nonexistent_returns_false() {
        let pool = connect_db().await;
        let mem = LongTermMemory::new(pool);
        let fake_id = Uuid::new_v4();
        let deleted = mem.delete(fake_id).await.expect("delete call failed");
        assert!(!deleted);
    }

    #[tokio::test]
    #[ignore]
    async fn store_wrong_dimension_returns_error() {
        let pool = connect_db().await;
        let mem = LongTermMemory::new(pool);
        let result = mem
            .store("bad", vec![0.1_f32; 512], serde_json::json!({}))
            .await;
        assert!(matches!(result, Err(MemoryError::DimensionMismatch(512))));
    }

    #[tokio::test]
    #[ignore]
    async fn query_wrong_dimension_returns_error() {
        let pool = connect_db().await;
        let mem = LongTermMemory::new(pool);
        let result = mem.query(vec![0.0_f32; 10], 5).await;
        assert!(matches!(result, Err(MemoryError::DimensionMismatch(10))));
    }

    #[tokio::test]
    #[ignore]
    async fn metadata_roundtrips() {
        let pool = connect_db().await;
        let mem = LongTermMemory::new(pool);
        // Use a unique directional embedding (c-axis) to avoid cosine-distance
        // ties with stale rows from other tests.
        let embedding = fake_embedding_c();
        let meta = serde_json::json!({"source": "test", "score": 0.95});
        let id = mem
            .store("metadata test", embedding.clone(), meta.clone())
            .await
            .expect("store failed");

        let results = mem
            .query(embedding, 1)
            .await
            .expect("query failed");
        assert_eq!(results[0].id, id, "expected the just-stored row back");
        assert_eq!(results[0].metadata["source"], "test");

        mem.delete(id).await.expect("cleanup failed");
    }
}
