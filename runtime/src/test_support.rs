//! Shared helpers for integration tests across the `xola-runtime` crate.
//!
//! Centralises environment loading and DB connection setup so no test module
//! needs its own `dotenvy` call or `connect()` helper.
//!
//! Usage:
//! ```ignore
//! #[cfg(test)]
//! mod tests {
//!     use crate::test_support::{connect_db, setup_env};
//!     // ...
//! }
//! ```

use sqlx::PgPool;
use std::sync::OnceLock;

/// Load `.env` exactly once per test process.
///
/// Uses [`OnceLock`] so concurrent test threads don't race on
/// `dotenvy::dotenv()`. Safe to call from every test — only the first
/// invocation does any work.
///
/// `.ok()` silences the "file not found" error so tests work in CI where
/// `DATABASE_URL` (and other vars) are injected directly into the environment.
pub fn setup_env() {
    static INIT: OnceLock<()> = OnceLock::new();
    INIT.get_or_init(|| {
        dotenvy::dotenv().ok();
    });
}

/// Open a `PgPool` from `DATABASE_URL`, loading `.env` first if needed.
///
/// Panics with a clear message if `DATABASE_URL` is absent even after
/// loading the env file — this surfaces misconfiguration early.
pub async fn connect_db() -> PgPool {
    setup_env();
    let url = std::env::var("DATABASE_URL").expect(
        "DATABASE_URL must be set or present in .env for integration tests. \
         Copy .env.example → .env and start Postgres with: \
         docker compose -f docker/docker-compose.yml up -d postgres",
    );
    PgPool::connect(&url).await.expect("failed to connect to Postgres")
}
