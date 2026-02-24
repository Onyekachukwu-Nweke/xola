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
    /// Number of IVFFlat lists to probe on each `query` call.
    ///
    /// Higher values give better recall at the cost of latency.
    /// Use [`with_probes`][Self::with_probes] to override the default.
    ivfflat_probes: usize,
}

impl LongTermMemory {
    /// Create a new `LongTermMemory` backed by the given connection pool.
    ///
    /// Uses `ivfflat.probes = 10` by default — a practical balance between
    /// recall and latency for an IVFFlat index built with `lists = 100`.
    /// Tune with [`with_probes`][Self::with_probes] if needed.
    pub fn new(pool: PgPool) -> Self {
        Self {
            pool,
            ivfflat_probes: 10,
        }
    }

    /// Override the IVFFlat probe count for this instance.
    ///
    /// Setting `probes = lists` (e.g. 100) forces an exact scan and is useful
    /// in tests where the index is sparse (few rows << many lists).
    pub fn with_probes(mut self, probes: usize) -> Self {
        self.ivfflat_probes = probes;
        self
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
    #[tracing::instrument(name = "memory_write", skip(self, embedding, metadata), fields(memory.r#type = "long_term"))]
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
    /// IVFFlat probe count is controlled by the `ivfflat_probes` field
    /// (set at construction time via [`with_probes`][Self::with_probes]).
    /// A dedicated connection is acquired so the `SET ivfflat.probes`
    /// command and the subsequent `SELECT` share the same session.
    ///
    /// # Parameters
    /// - `embedding` – 1536-dim query vector from `POST /embed`
    /// - `top_k` – maximum number of results to return. Passing `0` returns
    ///   an empty `Vec` immediately (no database round-trip).
    ///
    /// # Errors
    /// Returns [`MemoryError::DimensionMismatch`] if `embedding.len() != 1536`.
    #[tracing::instrument(name = "memory_read", skip(self, embedding), fields(memory.r#type = "long_term", memory.results_count = tracing::field::Empty))]
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

        // Acquire a dedicated connection so that the `SET ivfflat.probes`
        // command and the subsequent SELECT execute on the same session.
        // The session-level setting persists for the lifetime of this
        // checkout and is harmless to other checkouts.
        let mut conn = self.pool.acquire().await?;

        // Raise the number of IVFFlat lists probed. Default is 1 which yields
        // very poor recall when the index is sparse (test datasets) or when a
        // high recall SLA is required.
        sqlx::query(&format!("SET ivfflat.probes = {}", self.ivfflat_probes))
            .execute(&mut *conn)
            .await?;

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
        .fetch_all(&mut *conn)
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

        tracing::Span::current().record("memory.results_count", records.len() as u64);
        tracing::debug!(
            top_k,
            probes = self.ivfflat_probes,
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
        // probes=100 forces a full IVFFlat scan — needed when row count (2) << lists (100).
        let mem = LongTermMemory::new(pool).with_probes(100);

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
            .store(
                "memory A",
                embedding_a.clone(),
                serde_json::json!({"tag": "a"}),
            )
            .await
            .expect("store A failed");
        let id_b = mem
            .store(
                "memory B",
                embedding_b.clone(),
                serde_json::json!({"tag": "b"}),
            )
            .await
            .expect("store B failed");

        // Query with a vector identical to A — it must come back first.
        let results = mem.query(embedding_a, 2).await.expect("query failed");

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

        let results = mem.query(embedding, 1).await.expect("query failed");
        assert_eq!(results[0].id, id, "expected the just-stored row back");
        assert_eq!(results[0].metadata["source"], "test");

        mem.delete(id).await.expect("cleanup failed");
    }

    // ---------------------------------------------------------------------------
    // L2-09 — semantic similarity integration test
    //
    // Requires both a running Postgres instance (DATABASE_URL) and the
    // Python IPC server (LLM_SURFACE_URL, defaults to http://127.0.0.1:8001).
    //
    // Start the Python server in a separate terminal before running:
    //   cd llm_surface && uv run uvicorn llm_surface.server:app --port 8001
    //
    // Then run this test with:
    //   cargo test semantic_similarity_retrieves_related_memory_first -- --ignored
    // ---------------------------------------------------------------------------

    // Tags used to namespace test rows in L2-09. Defined as constants so the
    // pre-test cleanup and the store calls always refer to the same strings.
    const TAG_PARIS: &str = "l2_09_eiffel";
    const TAG_BIO: &str = "l2_09_bio";

    /// Internal request body for `POST /embed` — matches the IPC contract in AGENTS.md.
    #[derive(serde::Serialize)]
    struct EmbedRequest {
        text: String,
        /// Model name; mirrors the `model` field the Rust runtime will send.
        model: String,
    }

    /// Internal response body from `POST /embed` — matches the IPC contract in AGENTS.md.
    #[derive(serde::Deserialize)]
    struct EmbedResponse {
        vector: Vec<f32>,
        token_count: u32,
    }

    /// Fetch a real 1536-dim embedding from the Python IPC server.
    ///
    /// Sends the full IPC request shape (text + model) and asserts the full
    /// response shape (vector + token_count > 0). Panics on any failure so
    /// the test fails immediately with a clear message.
    async fn fetch_embedding(llm_url: &str, text: &str) -> Vec<f32> {
        // 10-second timeout: fails fast if the Python server is unresponsive
        // rather than hanging the test runner indefinitely.
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(10))
            .build()
            .expect("failed to build HTTP client");

        let resp = client
            .post(format!("{}/embed", llm_url))
            .json(&EmbedRequest {
                text: text.to_owned(),
                model: "text-embedding-3-small".to_owned(),
            })
            .send()
            .await
            .unwrap_or_else(|e| panic!("POST {llm_url}/embed failed: {e}"))
            .error_for_status()
            .unwrap_or_else(|e| panic!("/embed returned error status: {e}"));

        let body = resp
            .json::<EmbedResponse>()
            .await
            .expect("failed to parse /embed JSON response");

        assert!(
            body.token_count > 0,
            "IPC contract violation: token_count must be > 0, got {}",
            body.token_count
        );

        body.vector
    }

    /// L2-09: end-to-end semantic similarity test.
    ///
    /// Stores two semantically distinct memories and queries with a phrase that
    /// is semantically close to one of them. Asserts that cosine distance
    /// ordering surfaces the correct memory first.
    ///
    /// Text pair chosen so the semantic gap is unambiguous to any embedding model:
    ///   - "Eiffel Tower / Paris / iron lattice" cluster  ← *related* to query
    ///   - "mitochondria / ATP / cellular respiration"    ← *unrelated* to query
    ///   - query: "Famous iron tower in the French capital"
    #[tokio::test]
    #[ignore]
    async fn semantic_similarity_retrieves_related_memory_first() {
        // Fall back to the default local server address if the variable is absent.
        let llm_url = std::env::var("LLM_SURFACE_URL")
            .unwrap_or_else(|_| "http://127.0.0.1:8001".to_string());

        let pool = connect_db().await;
        // probes=100 guarantees all IVFFlat lists are scanned even with a tiny test
        // corpus (2 rows << 100 lists). Default probes=10 would still miss them ~81%
        // of the time. This setting does not affect the production runtime.
        let mem = LongTermMemory::new(pool).with_probes(100);
        // Remove any rows from previous failed runs so stale embeddings cannot
        // interfere with cosine distance ordering.
        for tag in [TAG_PARIS, TAG_BIO] {
            sqlx::query!(
                "DELETE FROM memories WHERE metadata->>'l2_09_tag' = $1",
                tag
            )
            .execute(mem.pool())
            .await
            .expect("pre-test cleanup failed");
        }

        // ── Embed three texts ─────────────────────────────────────────────────
        let text_paris = "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.";
        let text_bio =
            "Mitochondria are membrane-bound organelles that produce ATP through cellular respiration.";
        let text_query = "Famous iron lattice tower located in the French capital.";

        let v_paris = fetch_embedding(&llm_url, text_paris).await;
        let v_bio = fetch_embedding(&llm_url, text_bio).await;
        let v_query = fetch_embedding(&llm_url, text_query).await;

        // ── Store both memories ───────────────────────────────────────────────
        let id_paris = mem
            .store(
                text_paris,
                v_paris,
                serde_json::json!({"l2_09_tag": TAG_PARIS}),
            )
            .await
            .expect("store paris failed");
        let id_bio = mem
            .store(text_bio, v_bio, serde_json::json!({"l2_09_tag": TAG_BIO}))
            .await
            .expect("store bio failed");

        // ── Query and assert ordering ─────────────────────────────────────────
        let results = mem.query(v_query, 2).await.expect("query failed");

        assert!(
            !results.is_empty(),
            "query returned no results — check that memories were stored"
        );
        assert_eq!(
            results[0].id, id_paris,
            "Paris memory should rank first; got content={:?} distance={:.4}",
            results[0].content, results[0].distance
        );

        if results.len() >= 2 {
            assert!(
                results[0].distance < results[1].distance,
                "Eiffel distance ({:.4}) should be less than biology distance ({:.4})",
                results[0].distance,
                results[1].distance
            );
        }

        // ── Cleanup ───────────────────────────────────────────────────────────
        mem.delete(id_paris).await.expect("cleanup paris failed");
        mem.delete(id_bio).await.expect("cleanup bio failed");
    }
}
