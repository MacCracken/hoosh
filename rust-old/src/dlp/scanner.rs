//! DLP scanner engine — regex-based content scanning with classification.

use regex::{Regex, RegexSet};
use serde::{Deserialize, Serialize};

use super::patterns::BuiltinPatterns;

/// Privacy classification level (higher = more restricted).
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[non_exhaustive]
pub enum ClassificationLevel {
    /// No restrictions — safe for any provider.
    #[default]
    Public = 0,
    /// Prefer local, allow remote if necessary.
    Internal = 1,
    /// Local models only — must not leave the machine.
    Confidential = 2,
    /// Block entirely or heavily redact before processing.
    Restricted = 3,
}

/// A single pattern match found during scanning.
#[derive(Debug, Clone)]
pub struct PatternMatch {
    /// Name of the pattern that matched (e.g. "email", "ssn").
    pub pattern_name: String,
    /// Classification level of this pattern.
    pub level: ClassificationLevel,
    /// Byte offset in the scanned text.
    pub offset: usize,
    /// Length of the match in bytes.
    pub length: usize,
}

/// Result of scanning content.
#[derive(Debug)]
pub struct ScanResult {
    /// Highest classification level found across all matches.
    pub level: ClassificationLevel,
    /// Individual pattern matches.
    pub matches: Vec<PatternMatch>,
}

/// DLP scanner configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DlpConfig {
    /// Whether DLP scanning is enabled.
    #[serde(default)]
    pub enabled: bool,
    /// Default classification when no patterns match.
    #[serde(default)]
    pub default_level: ClassificationLevel,
    /// Custom patterns (in addition to built-ins).
    #[serde(default)]
    pub custom_patterns: Vec<CustomPattern>,
}

impl Default for DlpConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            default_level: ClassificationLevel::Public,
            custom_patterns: Vec::new(),
        }
    }
}

/// A user-defined scanning pattern.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomPattern {
    /// Pattern name for reporting.
    pub name: String,
    /// Regex pattern string.
    pub pattern: String,
    /// Classification level when matched.
    pub level: ClassificationLevel,
}

/// Compiled DLP scanner with pre-built regex set for single-pass matching.
pub struct DlpScanner {
    /// Fast regex set for initial match detection.
    regex_set: RegexSet,
    /// Individual compiled regexes for match details (parallel to regex_set).
    regexes: Vec<Regex>,
    /// Pattern metadata (name, level) — parallel to regexes.
    pattern_info: Vec<(String, ClassificationLevel)>,
    /// Default classification level.
    default_level: ClassificationLevel,
    /// Whether scanning is enabled.
    enabled: bool,
}

impl DlpScanner {
    /// Build a scanner from config. Compiles all patterns at construction time.
    ///
    /// Returns `Err` if any custom pattern has invalid regex syntax.
    pub fn new(config: &DlpConfig) -> anyhow::Result<Self> {
        let mut pattern_strings = Vec::new();
        let mut pattern_info = Vec::new();

        // Add built-in patterns
        for (name, pattern, level) in BuiltinPatterns::all() {
            pattern_strings.push(pattern.to_string());
            pattern_info.push((name.to_string(), level));
        }

        // Add custom patterns
        for custom in &config.custom_patterns {
            // Validate regex syntax
            let _ = Regex::new(&custom.pattern)
                .map_err(|e| anyhow::anyhow!("invalid DLP pattern '{}': {}", custom.name, e))?;
            pattern_strings.push(custom.pattern.clone());
            pattern_info.push((custom.name.clone(), custom.level));
        }

        let regex_set = RegexSet::new(&pattern_strings)?;
        let regexes: Vec<Regex> = pattern_strings
            .iter()
            .map(|p| Regex::new(p).map_err(|e| anyhow::anyhow!("regex compile failed: {e}")))
            .collect::<anyhow::Result<Vec<_>>>()?;

        Ok(Self {
            regex_set,
            regexes,
            pattern_info,
            default_level: config.default_level,
            enabled: config.enabled,
        })
    }

    /// Scan text content for PII and sensitive patterns.
    #[must_use]
    pub fn scan(&self, text: &str) -> ScanResult {
        if !self.enabled || text.is_empty() {
            return ScanResult {
                level: self.default_level,
                matches: Vec::new(),
            };
        }

        let matching_indices: Vec<usize> = self.regex_set.matches(text).into_iter().collect();

        if matching_indices.is_empty() {
            return ScanResult {
                level: self.default_level,
                matches: Vec::new(),
            };
        }

        let mut all_matches = Vec::new();
        let mut highest_level = self.default_level;

        for &idx in &matching_indices {
            let (ref name, level) = self.pattern_info[idx];
            if level > highest_level {
                highest_level = level;
            }
            for m in self.regexes[idx].find_iter(text) {
                all_matches.push(PatternMatch {
                    pattern_name: name.clone(),
                    level,
                    offset: m.start(),
                    length: m.len(),
                });
            }
        }

        ScanResult {
            level: highest_level,
            matches: all_matches,
        }
    }

    /// Scan all messages in a request, returning the highest classification.
    #[must_use]
    pub fn scan_messages(&self, messages: &[crate::inference::Message]) -> ScanResult {
        if !self.enabled {
            return ScanResult {
                level: self.default_level,
                matches: Vec::new(),
            };
        }

        let mut highest_level = self.default_level;
        let mut all_matches = Vec::new();

        for msg in messages {
            let result = self.scan(&msg.content.text());
            if result.level > highest_level {
                highest_level = result.level;
            }
            all_matches.extend(result.matches);
        }

        ScanResult {
            level: highest_level,
            matches: all_matches,
        }
    }

    /// Whether scanning is enabled.
    #[must_use]
    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn enabled_config() -> DlpConfig {
        DlpConfig {
            enabled: true,
            default_level: ClassificationLevel::Public,
            custom_patterns: Vec::new(),
        }
    }

    #[test]
    fn scanner_disabled_returns_default() {
        let config = DlpConfig::default();
        let scanner = DlpScanner::new(&config).unwrap();
        let result = scanner.scan("user@example.com has SSN 123-45-6789");
        assert_eq!(result.level, ClassificationLevel::Public);
        assert!(result.matches.is_empty());
    }

    #[test]
    fn scanner_detects_email() {
        let scanner = DlpScanner::new(&enabled_config()).unwrap();
        let result = scanner.scan("Contact me at user@example.com please");
        assert!(result.level >= ClassificationLevel::Internal);
        assert!(result.matches.iter().any(|m| m.pattern_name == "email"));
    }

    #[test]
    fn scanner_detects_ssn() {
        let scanner = DlpScanner::new(&enabled_config()).unwrap();
        let result = scanner.scan("SSN: 123-45-6789");
        assert_eq!(result.level, ClassificationLevel::Restricted);
        assert!(result.matches.iter().any(|m| m.pattern_name == "ssn"));
    }

    #[test]
    fn scanner_detects_credit_card() {
        let scanner = DlpScanner::new(&enabled_config()).unwrap();
        let result = scanner.scan("Card: 4111 1111 1111 1111");
        assert_eq!(result.level, ClassificationLevel::Restricted);
        assert!(
            result
                .matches
                .iter()
                .any(|m| m.pattern_name == "credit_card")
        );
    }

    #[test]
    fn scanner_detects_aws_key() {
        let scanner = DlpScanner::new(&enabled_config()).unwrap();
        let result = scanner.scan("Key: AKIAIOSFODNN7EXAMPLE");
        assert_eq!(result.level, ClassificationLevel::Restricted);
        assert!(result.matches.iter().any(|m| m.pattern_name == "aws_key"));
    }

    #[test]
    fn scanner_detects_github_token() {
        let scanner = DlpScanner::new(&enabled_config()).unwrap();
        let result = scanner.scan("Token: ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij");
        assert_eq!(result.level, ClassificationLevel::Restricted);
        assert!(
            result
                .matches
                .iter()
                .any(|m| m.pattern_name == "github_token")
        );
    }

    #[test]
    fn scanner_clean_text() {
        let scanner = DlpScanner::new(&enabled_config()).unwrap();
        let result = scanner.scan("Explain Rust ownership in one sentence.");
        assert_eq!(result.level, ClassificationLevel::Public);
        assert!(result.matches.is_empty());
    }

    #[test]
    fn scanner_highest_level_wins() {
        let scanner = DlpScanner::new(&enabled_config()).unwrap();
        // Has both email (Internal) and SSN (Restricted)
        let result = scanner.scan("user@test.com SSN: 123-45-6789");
        assert_eq!(result.level, ClassificationLevel::Restricted);
        assert!(result.matches.len() >= 2);
    }

    #[test]
    fn custom_pattern() {
        let config = DlpConfig {
            enabled: true,
            default_level: ClassificationLevel::Public,
            custom_patterns: vec![CustomPattern {
                name: "project_code".into(),
                pattern: r"\bPROJECT-X\b".into(),
                level: ClassificationLevel::Confidential,
            }],
        };
        let scanner = DlpScanner::new(&config).unwrap();
        let result = scanner.scan("Working on PROJECT-X deliverables");
        assert_eq!(result.level, ClassificationLevel::Confidential);
        assert!(
            result
                .matches
                .iter()
                .any(|m| m.pattern_name == "project_code")
        );
    }

    #[test]
    fn invalid_custom_pattern_errors() {
        let config = DlpConfig {
            enabled: true,
            default_level: ClassificationLevel::Public,
            custom_patterns: vec![CustomPattern {
                name: "bad".into(),
                pattern: r"[invalid".into(),
                level: ClassificationLevel::Internal,
            }],
        };
        assert!(DlpScanner::new(&config).is_err());
    }

    #[test]
    fn scan_messages() {
        use crate::inference::{Message, Role};
        let scanner = DlpScanner::new(&enabled_config()).unwrap();
        let messages = vec![
            Message::new(Role::User, "My email is test@example.com"),
            Message::new(Role::User, "My SSN is 123-45-6789"),
        ];
        let result = scanner.scan_messages(&messages);
        assert_eq!(result.level, ClassificationLevel::Restricted);
        assert!(result.matches.len() >= 2);
    }

    #[test]
    fn empty_text_returns_default() {
        let scanner = DlpScanner::new(&enabled_config()).unwrap();
        let result = scanner.scan("");
        assert_eq!(result.level, ClassificationLevel::Public);
        assert!(result.matches.is_empty());
    }

    #[test]
    fn classification_ordering() {
        assert!(ClassificationLevel::Public < ClassificationLevel::Internal);
        assert!(ClassificationLevel::Internal < ClassificationLevel::Confidential);
        assert!(ClassificationLevel::Confidential < ClassificationLevel::Restricted);
    }
}
