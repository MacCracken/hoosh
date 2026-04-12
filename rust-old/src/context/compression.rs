//! Prompt compression — mechanical reduction of message content.
//!
//! Applies low-cost text transformations to reduce token counts without
//! changing semantics: whitespace collapsing, duplicate instruction removal,
//! and stale tool-call pair pruning.

use crate::inference::{Message, Role};

/// Compress messages by applying mechanical text reductions.
///
/// Returns a new Vec of compressed messages. Non-destructive: the original
/// messages are not modified.
#[must_use]
pub fn compress_messages(messages: &[Message]) -> Vec<Message> {
    let mut result = Vec::with_capacity(messages.len());

    for msg in messages {
        let mut compressed = msg.clone();
        compressed.content =
            crate::inference::MessageContent::Text(collapse_whitespace(&compressed.content.text()));
        result.push(compressed);
    }

    prune_stale_tool_pairs(&mut result);
    result
}

/// Collapse runs of whitespace into single spaces, trim leading/trailing.
#[must_use]
fn collapse_whitespace(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut last_was_space = true; // trim leading
    for ch in text.chars() {
        if ch.is_whitespace() {
            if !last_was_space {
                result.push(' ');
                last_was_space = true;
            }
        } else {
            result.push(ch);
            last_was_space = false;
        }
    }
    // Trim trailing space
    if result.ends_with(' ') {
        result.pop();
    }
    result
}

/// Remove completed tool-call / tool-result pairs that are older than the
/// last N assistant messages.
///
/// When a conversation grows long, old tool interactions are no longer relevant
/// to the current context. This removes assistant messages with tool calls and
/// their corresponding tool-result messages if they appear before the last
/// `KEEP_RECENT_TOOLS` assistant turns.
fn prune_stale_tool_pairs(messages: &mut Vec<Message>) {
    const KEEP_RECENT_TOOLS: usize = 3;

    // Find indices of assistant messages with tool calls
    let tool_call_indices: Vec<usize> = messages
        .iter()
        .enumerate()
        .filter(|(_, m)| m.role == Role::Assistant && !m.tool_calls.is_empty())
        .map(|(i, _)| i)
        .collect();

    if tool_call_indices.len() <= KEEP_RECENT_TOOLS {
        return;
    }

    // Indices of tool-call messages to remove (all but the last KEEP_RECENT_TOOLS)
    let stale_count = tool_call_indices.len() - KEEP_RECENT_TOOLS;
    let stale_indices: Vec<usize> = tool_call_indices[..stale_count].to_vec();

    // Collect tool call IDs from stale assistant messages
    let stale_tool_ids: std::collections::HashSet<String> = stale_indices
        .iter()
        .flat_map(|&i| messages[i].tool_calls.iter().map(|tc| tc.id.clone()))
        .collect();

    // Mark indices to remove: stale tool-call messages + their tool-result messages
    let mut remove_set: Vec<bool> = vec![false; messages.len()];
    for &idx in &stale_indices {
        remove_set[idx] = true;
    }
    for (i, msg) in messages.iter().enumerate() {
        if msg.role == Role::Tool
            && let Some(ref id) = msg.tool_call_id
            && stale_tool_ids.contains(id)
        {
            remove_set[i] = true;
        }
    }

    // Remove marked messages in O(n) using retain
    let mut idx = 0;
    messages.retain(|_| {
        let keep = !remove_set[idx];
        idx += 1;
        keep
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collapse_whitespace_basic() {
        assert_eq!(collapse_whitespace("hello  world"), "hello world");
        assert_eq!(collapse_whitespace("  foo  bar  "), "foo bar");
        assert_eq!(
            collapse_whitespace("no\nnewlines\there"),
            "no newlines here"
        );
    }

    #[test]
    fn collapse_whitespace_empty() {
        assert_eq!(collapse_whitespace(""), "");
        assert_eq!(collapse_whitespace("   "), "");
    }

    #[test]
    fn collapse_whitespace_already_clean() {
        assert_eq!(collapse_whitespace("already clean"), "already clean");
    }

    #[test]
    fn compress_messages_basic() {
        let messages = vec![
            Message::new(Role::System, "You  are   helpful."),
            Message::new(Role::User, "Hello   world"),
        ];
        let compressed = compress_messages(&messages);
        assert_eq!(compressed[0].content, "You are helpful.");
        assert_eq!(compressed[1].content, "Hello world");
    }

    #[test]
    fn compress_preserves_message_count_without_tools() {
        let messages = vec![
            Message::new(Role::User, "Hi"),
            Message::new(Role::Assistant, "Hello"),
        ];
        let compressed = compress_messages(&messages);
        assert_eq!(compressed.len(), 2);
    }

    #[test]
    fn prune_stale_tool_pairs_keeps_recent() {
        let mut messages = vec![Message::new(Role::User, "query 1")];

        // Add 5 tool-call/result pairs
        for i in 0..5 {
            let call_id = format!("call_{i}");
            messages.push(Message {
                role: Role::Assistant,
                content: format!("Using tool {i}").into(),
                tool_call_id: None,
                tool_calls: vec![crate::tools::ToolCall {
                    id: call_id.clone(),
                    name: "test_tool".into(),
                    arguments: serde_json::json!({}),
                }],
            });
            messages.push(Message {
                role: Role::Tool,
                content: format!("Result {i}").into(),
                tool_call_id: Some(call_id),
                tool_calls: vec![],
            });
        }
        messages.push(Message::new(Role::User, "final question"));

        let original_len = messages.len();
        prune_stale_tool_pairs(&mut messages);

        // Should have removed 2 stale pairs (5 - 3 = 2 pairs = 4 messages)
        assert_eq!(messages.len(), original_len - 4);

        // User messages should be preserved
        assert_eq!(messages.first().unwrap().content, "query 1");
        assert_eq!(messages.last().unwrap().content, "final question");
    }

    #[test]
    fn prune_stale_tool_pairs_few_tools_no_change() {
        let mut messages = vec![
            Message::new(Role::User, "hi"),
            Message {
                role: Role::Assistant,
                content: "using tool".into(),
                tool_call_id: None,
                tool_calls: vec![crate::tools::ToolCall {
                    id: "c1".into(),
                    name: "t".into(),
                    arguments: serde_json::json!({}),
                }],
            },
            Message {
                role: Role::Tool,
                content: "result".into(),
                tool_call_id: Some("c1".into()),
                tool_calls: vec![],
            },
        ];
        let len_before = messages.len();
        prune_stale_tool_pairs(&mut messages);
        assert_eq!(messages.len(), len_before); // no change, only 1 tool pair
    }
}
