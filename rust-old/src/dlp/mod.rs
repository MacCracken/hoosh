//! Data Loss Prevention — PII scanning, content classification, privacy-aware routing.
//!
//! Feature-gated behind `dlp`. When enabled, scans inference request content
//! for PII and sensitive patterns, classifying requests into privacy levels
//! that drive routing decisions (e.g., confidential data → local models only).

mod patterns;
mod scanner;

pub use patterns::BuiltinPatterns;
pub use scanner::{ClassificationLevel, DlpConfig, DlpScanner, PatternMatch, ScanResult};
