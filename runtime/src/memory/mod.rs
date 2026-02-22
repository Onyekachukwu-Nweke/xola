//! Memory subsystem for the Xola agent runtime.
//!
//! Three memory types work together to give the agent context across
//! turns and across tasks. See `docs/layer-2-memory.md` for design rationale.
//!
//! | Type | Scope | Storage |
//! |------|-------|---------|
//! | [`ShortTermMemory`] | Current conversation | In-process `VecDeque` |
//! | [`LongTermMemory`]  | All time             | Postgres + pgvector   |
//! | `EpisodicLog` (L2-05) | All tasks          | Postgres JSONB        |

pub mod long_term;
pub mod short_term;

pub use long_term::{LongTermMemory, MemoryError, MemoryRecord};
pub use short_term::{Message, Role, ShortTermMemory};
