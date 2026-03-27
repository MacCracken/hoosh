//! Built-in PII and sensitive data patterns.

/// Built-in pattern definitions for common PII types.
pub struct BuiltinPatterns;

impl BuiltinPatterns {
    /// Returns (name, regex_pattern, classification_level) tuples for built-in PII patterns.
    #[must_use]
    pub fn all() -> Vec<(
        &'static str,
        &'static str,
        super::scanner::ClassificationLevel,
    )> {
        use super::scanner::ClassificationLevel;
        vec![
            // Email addresses
            (
                "email",
                r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
                ClassificationLevel::Internal,
            ),
            // US phone numbers
            (
                "phone_us",
                r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
                ClassificationLevel::Confidential,
            ),
            // Social Security Numbers
            (
                "ssn",
                r"\b\d{3}-\d{2}-\d{4}\b",
                ClassificationLevel::Restricted,
            ),
            // Credit card numbers (basic)
            (
                "credit_card",
                r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b",
                ClassificationLevel::Restricted,
            ),
            // IPv4 addresses
            (
                "ipv4",
                r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
                ClassificationLevel::Internal,
            ),
            // API keys / secrets (common prefixes)
            (
                "api_key",
                r"\b(?:sk-|pk-|api[_\-]?key[=: ]+)[a-zA-Z0-9]{20,}\b",
                ClassificationLevel::Restricted,
            ),
            // AWS access key IDs
            (
                "aws_key",
                r"\bAKIA[0-9A-Z]{16}\b",
                ClassificationLevel::Restricted,
            ),
            // GitHub tokens
            (
                "github_token",
                r"\b(?:ghp_|gho_|ghu_|ghs_|ghr_)[a-zA-Z0-9]{36,}\b",
                ClassificationLevel::Restricted,
            ),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_patterns_not_empty() {
        assert!(!BuiltinPatterns::all().is_empty());
    }

    #[test]
    fn all_patterns_have_names() {
        for (name, pattern, _) in BuiltinPatterns::all() {
            assert!(!name.is_empty());
            assert!(!pattern.is_empty());
        }
    }
}
