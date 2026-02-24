//! Short-term memory for the Xola agent runtime.
//!
//! A circular message buffer capped by a token budget. It holds the recent
//! conversation context that gets serialised into each call to `POST /reason`.
//!
//! # Design
//!
//! - Messages are stored as a `VecDeque<Message>` so the oldest entry can be
//!   dropped efficiently from the front when the budget fills.
//! - **Token counting is the caller's responsibility.** Rust does not have
//!   access to tiktoken; the token count for each message must be computed in
//!   Python and passed in alongside the message content. This keeps the
//!   language boundary clean: Python owns all LLM/tokenisation logic.
//! - `needs_summarization` fires when the buffer crosses 80 % of its budget,
//!   giving the planner (L3) a chance to compress old messages before eviction
//!   starts throwing context away.

use std::collections::VecDeque;

/// A role in a conversation turn.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
}

impl std::fmt::Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Role::System => write!(f, "system"),
            Role::User => write!(f, "user"),
            Role::Assistant => write!(f, "assistant"),
        }
    }
}

/// A single message in the conversation window.
///
/// `token_count` must be supplied by the caller and must have been computed
/// by Python's tiktoken helper (`POST /embed` returns the count alongside the
/// vector; the `/reason` handler also tracks per-message counts).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
    /// Pre-computed token count from Python's tiktoken. Must be > 0.
    pub token_count: usize,
}

impl Message {
    /// # Panics (debug builds only)
    ///
    /// Debug-asserts that `token_count > 0`. A zero-token message would be
    /// silently stored and never evicted, corrupting the budget accounting.
    pub fn new(role: Role, content: impl Into<String>, token_count: usize) -> Self {
        debug_assert!(
            token_count > 0,
            "token_count must be > 0; got 0 for a message — did tiktoken return 0 tokens?"
        );
        Self {
            role,
            content: content.into(),
            token_count,
        }
    }
}

/// Fraction of `max_tokens` at which `needs_summarization` returns `true`.
/// Chosen to give the planner headroom before hard eviction begins.
const SUMMARIZATION_THRESHOLD: f64 = 0.80;

/// Token-budgeted circular buffer of recent conversation messages.
///
/// See [`crate::memory`] module-level docs and `docs/layer-2-memory.md` for
/// design context.
///
/// # Example
///
/// ```rust
/// use xola_runtime::memory::short_term::{Message, Role, ShortTermMemory};
///
/// let mut stm = ShortTermMemory::new(1000);
/// stm.push(Message::new(Role::User, "Hello!", 3));
/// assert_eq!(stm.token_count(), 3);
/// assert_eq!(stm.messages().len(), 1);
/// ```
#[derive(Debug)]
pub struct ShortTermMemory {
    messages: VecDeque<Message>,
    token_count: usize,
    max_tokens: usize,
}

impl ShortTermMemory {
    /// Create a new buffer capped at `max_tokens`.
    ///
    /// # Panics
    ///
    /// Panics if `max_tokens` is 0.
    pub fn new(max_tokens: usize) -> Self {
        assert!(max_tokens > 0, "max_tokens must be greater than 0");
        Self {
            messages: VecDeque::new(),
            token_count: 0,
            max_tokens,
        }
    }

    /// Push a message into the buffer.
    ///
    /// If the message alone exceeds `max_tokens`, it is still accepted (the
    /// buffer will hold it as the sole entry, evicting everything else). This
    /// prevents a single large message from deadlocking the agent.
    ///
    /// Eviction removes the **oldest** message(s) until the new message fits.
    #[tracing::instrument(
        name = "stm_push",
        skip(self, msg),
        fields(
            memory.type_ = "short_term",
            memory.role = %msg.role,
            memory.tokens = msg.token_count,
            memory.buffer_tokens = self.token_count,
        )
    )]
    pub fn push(&mut self, msg: Message) {
        let incoming_tokens = msg.token_count;

        // Evict oldest messages until the new one fits (or buffer is empty)
        while !self.messages.is_empty() && self.token_count + incoming_tokens > self.max_tokens {
            if let Some(evicted) = self.messages.pop_front() {
                self.token_count = self.token_count.saturating_sub(evicted.token_count);
                tracing::debug!(
                    role = %evicted.role,
                    tokens = evicted.token_count,
                    "short_term_memory: evicted oldest message"
                );
            }
        }

        self.token_count += incoming_tokens;
        self.messages.push_back(msg);

        tracing::debug!(
            token_count = self.token_count,
            max_tokens = self.max_tokens,
            messages = self.messages.len(),
            "short_term_memory: pushed message"
        );
    }

    /// Returns all messages in chronological order (oldest first).
    ///
    /// Collects via `iter()` to guarantee all messages are returned regardless
    /// of the `VecDeque`'s internal ring-buffer layout. Prefer `iter()` in
    /// hot paths to avoid the allocation.
    pub fn messages(&self) -> Vec<&Message> {
        self.messages.iter().collect()
    }

    /// Returns an iterator over all messages in chronological order.
    pub fn iter(&self) -> impl Iterator<Item = &Message> {
        self.messages.iter()
    }

    /// Current number of tokens stored across all buffered messages.
    pub fn token_count(&self) -> usize {
        self.token_count
    }

    /// Maximum token capacity of this buffer.
    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    /// Number of messages currently in the buffer.
    pub fn len(&self) -> usize {
        self.messages.len()
    }

    /// Returns `true` if the buffer contains no messages.
    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }

    /// Returns `true` when the buffer has crossed the summarization threshold
    /// (`SUMMARIZATION_THRESHOLD` × `max_tokens`).
    ///
    /// When this returns `true` the planner should trigger a Python
    /// summarization call before the next `push` starts evicting context.
    pub fn needs_summarization(&self) -> bool {
        self.token_count as f64 >= self.max_tokens as f64 * SUMMARIZATION_THRESHOLD
    }

    /// Replace the current messages with a single summarized message.
    ///
    /// Called by the planner (L3) after receiving a compressed summary from
    /// Python's `POST /reason` summarization path. The summary message's
    /// token count must be accurately supplied by the caller.
    pub fn replace_with_summary(&mut self, summary: Message) {
        tracing::info!(
            previous_messages = self.messages.len(),
            previous_tokens = self.token_count,
            summary_tokens = summary.token_count,
            "short_term_memory: compressing buffer to summary"
        );
        let summary_tokens = summary.token_count;
        self.messages.clear();
        self.messages.push_back(summary);
        self.token_count = summary_tokens;
    }

    /// Clear all messages and reset the token count.
    pub fn clear(&mut self) {
        self.messages.clear();
        self.token_count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(role: Role, content: &str, tokens: usize) -> Message {
        Message::new(role, content, tokens)
    }

    // --- Construction ---

    #[test]
    fn new_buffer_is_empty() {
        let stm = ShortTermMemory::new(1000);
        assert!(stm.is_empty());
        assert_eq!(stm.token_count(), 0);
        assert_eq!(stm.len(), 0);
    }

    #[test]
    #[should_panic(expected = "max_tokens must be greater than 0")]
    fn zero_max_tokens_panics() {
        ShortTermMemory::new(0);
    }

    // --- Push & basic accounting ---

    #[test]
    fn push_single_message_updates_count() {
        let mut stm = ShortTermMemory::new(1000);
        stm.push(msg(Role::User, "Hello!", 5));
        assert_eq!(stm.token_count(), 5);
        assert_eq!(stm.len(), 1);
    }

    #[test]
    fn push_multiple_messages_accumulates_tokens() {
        let mut stm = ShortTermMemory::new(1000);
        stm.push(msg(Role::User, "Hello!", 5));
        stm.push(msg(Role::Assistant, "Hi there!", 8));
        stm.push(msg(Role::User, "How are you?", 4));
        assert_eq!(stm.token_count(), 17);
        assert_eq!(stm.len(), 3);
    }

    #[test]
    fn messages_are_in_chronological_order() {
        let mut stm = ShortTermMemory::new(1000);
        stm.push(msg(Role::User, "first", 1));
        stm.push(msg(Role::Assistant, "second", 1));
        stm.push(msg(Role::User, "third", 1));
        let msgs: Vec<&Message> = stm.iter().collect();
        assert_eq!(msgs[0].content, "first");
        assert_eq!(msgs[1].content, "second");
        assert_eq!(msgs[2].content, "third");
    }

    // --- Eviction ---

    #[test]
    fn evicts_oldest_when_budget_fills() {
        let mut stm = ShortTermMemory::new(10);
        stm.push(msg(Role::User, "A", 4)); // total: 4
        stm.push(msg(Role::Assistant, "B", 4)); // total: 8
        stm.push(msg(Role::User, "C", 4)); // total: 12 → evict A → 8
        assert_eq!(stm.len(), 2);
        assert_eq!(stm.token_count(), 8);
        assert_eq!(stm.iter().next().unwrap().content, "B");
    }

    #[test]
    fn evicts_multiple_messages_to_make_room() {
        let mut stm = ShortTermMemory::new(10);
        stm.push(msg(Role::User, "A", 3));
        stm.push(msg(Role::User, "B", 3));
        stm.push(msg(Role::User, "C", 3)); // total: 9
        stm.push(msg(Role::User, "big", 9)); // needs 9, must evict A+B+C
        assert_eq!(stm.len(), 1);
        assert_eq!(stm.token_count(), 9);
        assert_eq!(stm.iter().next().unwrap().content, "big");
    }

    #[test]
    fn oversized_message_evicts_all_and_still_accepted() {
        let mut stm = ShortTermMemory::new(5);
        stm.push(msg(Role::User, "small", 3));
        // Push a message larger than max_tokens — must still be accepted
        stm.push(msg(Role::User, "very long message", 10));
        assert_eq!(stm.len(), 1);
        assert_eq!(stm.token_count(), 10);
    }

    #[test]
    fn token_count_stays_accurate_after_evictions() {
        let mut stm = ShortTermMemory::new(15);
        for i in 0..10 {
            stm.push(msg(Role::User, &format!("msg {i}"), 3));
        }
        // Expected: last 5 messages remain (5 × 3 = 15)
        assert_eq!(stm.token_count(), 15);
        let expected_tokens: usize = stm.iter().map(|m| m.token_count).sum();
        assert_eq!(stm.token_count(), expected_tokens);
    }

    // --- needs_summarization ---

    #[test]
    fn needs_summarization_below_threshold_is_false() {
        let mut stm = ShortTermMemory::new(1000);
        stm.push(msg(Role::User, "short", 100)); // 10 % of budget
        assert!(!stm.needs_summarization());
    }

    #[test]
    fn needs_summarization_at_threshold_is_true() {
        let mut stm = ShortTermMemory::new(1000);
        stm.push(msg(Role::User, "content", 800)); // exactly 80 %
        assert!(stm.needs_summarization());
    }

    #[test]
    fn needs_summarization_above_threshold_is_true() {
        let mut stm = ShortTermMemory::new(1000);
        stm.push(msg(Role::User, "content", 900)); // 90 %
        assert!(stm.needs_summarization());
    }

    // --- replace_with_summary ---

    #[test]
    fn replace_with_summary_clears_and_sets_summary() {
        let mut stm = ShortTermMemory::new(1000);
        stm.push(msg(Role::User, "A", 100));
        stm.push(msg(Role::Assistant, "B", 200));
        stm.replace_with_summary(msg(Role::System, "summary of conversation", 50));
        assert_eq!(stm.len(), 1);
        assert_eq!(stm.token_count(), 50);
        assert_eq!(stm.iter().next().unwrap().role, Role::System);
    }

    // --- clear ---

    #[test]
    fn clear_empties_buffer_and_resets_count() {
        let mut stm = ShortTermMemory::new(1000);
        stm.push(msg(Role::User, "a", 10));
        stm.push(msg(Role::Assistant, "b", 20));
        stm.clear();
        assert!(stm.is_empty());
        assert_eq!(stm.token_count(), 0);
    }

    // --- iter vs messages() ---

    #[test]
    fn iter_covers_all_messages() {
        let mut stm = ShortTermMemory::new(1000);
        stm.push(msg(Role::User, "x", 1));
        stm.push(msg(Role::User, "y", 1));
        stm.push(msg(Role::User, "z", 1));
        assert_eq!(stm.iter().count(), 3);
    }
}
