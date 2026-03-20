//! Configuration file loading for hoosh.
//!
//! Loads `hoosh.toml` from the current directory or a specified path.
//! Environment variables in API keys are resolved at load time.

use std::path::Path;

use serde::Deserialize;

use crate::budget::TokenPool;
use crate::cache::CacheConfig;
use crate::provider::ProviderType;
use crate::router::{ProviderRoute, RoutingStrategy};
use crate::server::ServerConfig;

/// Top-level configuration file structure.
#[derive(Debug, Deserialize)]
pub struct HooshConfig {
    #[serde(default)]
    pub server: ServerSection,
    #[serde(default)]
    pub cache: CacheSection,
    #[serde(default)]
    pub providers: Vec<ProviderSection>,
    #[serde(default)]
    pub budgets: Vec<BudgetPoolSection>,
    #[serde(default)]
    pub whisper: WhisperSection,
    #[serde(default)]
    pub tts: TtsSection,
}

#[derive(Debug, Default, Deserialize)]
pub struct WhisperSection {
    /// Path to whisper model file (e.g. "models/ggml-base.en.bin").
    pub model: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
pub struct TtsSection {
    /// URL of the TTS backend (e.g. "http://localhost:5500" for openedai-speech).
    pub url: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct BudgetPoolSection {
    /// Pool name (e.g. "default", "agent-1").
    pub name: String,
    /// Maximum tokens allowed in this pool.
    pub capacity: u64,
}

#[derive(Debug, Deserialize)]
pub struct ServerSection {
    #[serde(default = "default_bind")]
    pub bind: String,
    #[serde(default = "default_port")]
    pub port: u16,
    #[serde(default)]
    pub strategy: StrategyValue,
}

impl Default for ServerSection {
    fn default() -> Self {
        Self {
            bind: default_bind(),
            port: default_port(),
            strategy: StrategyValue::default(),
        }
    }
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StrategyValue {
    #[default]
    Priority,
    RoundRobin,
    LowestLatency,
    Direct,
}

#[derive(Debug, Deserialize)]
pub struct CacheSection {
    #[serde(default = "default_cache_max")]
    pub max_entries: usize,
    #[serde(default = "default_cache_ttl")]
    pub ttl_secs: u64,
    #[serde(default = "default_true")]
    pub enabled: bool,
}

#[derive(Debug, Deserialize)]
pub struct ProviderSection {
    /// Provider type name (e.g. "ollama", "openai", "anthropic").
    #[serde(rename = "type")]
    pub provider_type: ProviderType,
    /// Base URL. Uses provider-specific default if omitted.
    pub base_url: Option<String>,
    /// API key — literal string or `"$ENV_VAR"` to read from environment.
    pub api_key: Option<String>,
    /// Priority (lower = preferred). Defaults to 10.
    #[serde(default = "default_priority")]
    pub priority: u32,
    /// Model patterns this provider handles.
    #[serde(default)]
    pub models: Vec<String>,
    /// Whether this provider is enabled. Defaults to true.
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Maximum tokens per request for this provider.
    #[serde(default)]
    pub max_tokens_limit: Option<u32>,
}

fn default_bind() -> String {
    "127.0.0.1".into()
}
fn default_port() -> u16 {
    8088
}
fn default_cache_max() -> usize {
    1000
}
fn default_cache_ttl() -> u64 {
    300
}
fn default_true() -> bool {
    true
}
fn default_priority() -> u32 {
    10
}

impl Default for CacheSection {
    fn default() -> Self {
        Self {
            max_entries: default_cache_max(),
            ttl_secs: default_cache_ttl(),
            enabled: true,
        }
    }
}

/// Resolve an API key value. If it starts with `$`, read from environment.
fn resolve_api_key(raw: &Option<String>) -> Option<String> {
    let raw = raw.as_ref()?;
    if let Some(var_name) = raw.strip_prefix('$') {
        match std::env::var(var_name) {
            Ok(val) => Some(val),
            Err(_) => {
                tracing::warn!(
                    "API key env var ${var_name} is not set — provider will have no API key"
                );
                None
            }
        }
    } else {
        Some(raw.clone())
    }
}

/// Default base URL for a provider type.
fn default_base_url(provider_type: ProviderType) -> &'static str {
    match provider_type {
        ProviderType::Ollama => "http://localhost:11434",
        ProviderType::LlamaCpp => "http://localhost:8080",
        ProviderType::Synapse => "http://localhost:5000",
        ProviderType::LmStudio => "http://localhost:1234",
        ProviderType::LocalAi => "http://localhost:8080",
        ProviderType::OpenAi => "https://api.openai.com",
        ProviderType::Anthropic => "https://api.anthropic.com",
        ProviderType::DeepSeek => "https://api.deepseek.com",
        ProviderType::Mistral => "https://api.mistral.ai",
        ProviderType::Groq => "https://api.groq.com/openai",
        ProviderType::OpenRouter => "https://openrouter.ai/api",
        ProviderType::Google => "https://generativelanguage.googleapis.com",
        ProviderType::Grok => "https://api.x.ai",
        ProviderType::Whisper => "http://localhost:8080",
    }
}

impl HooshConfig {
    /// Load configuration from a TOML file.
    pub fn load(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let contents = std::fs::read_to_string(path.as_ref())?;
        let config: HooshConfig = toml::from_str(&contents).map_err(|e| {
            // Redact error details that might contain API keys
            let msg = e.to_string();
            if msg.contains("api_key") {
                anyhow::anyhow!("failed to parse config: TOML syntax error near api_key field")
            } else {
                anyhow::anyhow!("failed to parse config: {e}")
            }
        })?;
        Ok(config)
    }

    /// Try loading from `hoosh.toml` in the current directory, or return defaults.
    pub fn load_or_default() -> Self {
        if Path::new("hoosh.toml").exists() {
            match Self::load("hoosh.toml") {
                Ok(config) => {
                    tracing::info!("loaded config from hoosh.toml");
                    config
                }
                Err(e) => {
                    tracing::error!("failed to load hoosh.toml: {e}");
                    std::process::exit(1);
                }
            }
        } else {
            Self {
                server: ServerSection::default(),
                cache: CacheSection::default(),
                providers: Vec::new(),
                budgets: Vec::new(),
                whisper: WhisperSection::default(),
                tts: TtsSection::default(),
            }
        }
    }

    /// Convert to provider routes.
    pub fn routes(&self) -> Vec<ProviderRoute> {
        self.providers
            .iter()
            .map(|p| {
                let base_url = p
                    .base_url
                    .clone()
                    .unwrap_or_else(|| default_base_url(p.provider_type).into());
                ProviderRoute {
                    provider: p.provider_type,
                    priority: p.priority,
                    model_patterns: p.models.clone(),
                    enabled: p.enabled,
                    base_url,
                    api_key: resolve_api_key(&p.api_key),
                    max_tokens_limit: p.max_tokens_limit,
                }
            })
            .collect()
    }

    /// Convert to ServerConfig, merging CLI overrides.
    pub fn into_server_config(
        self,
        bind_override: Option<&str>,
        port_override: Option<u16>,
    ) -> ServerConfig {
        let routes = self.routes();
        let strategy = match self.server.strategy {
            StrategyValue::Priority => RoutingStrategy::Priority,
            StrategyValue::RoundRobin => RoutingStrategy::RoundRobin,
            StrategyValue::LowestLatency => RoutingStrategy::LowestLatency,
            StrategyValue::Direct => RoutingStrategy::Direct,
        };
        let budget_pools = self
            .budgets
            .iter()
            .map(|b| TokenPool::new(&b.name, b.capacity))
            .collect();

        ServerConfig {
            bind: bind_override.map(String::from).unwrap_or(self.server.bind),
            port: port_override.unwrap_or(self.server.port),
            routes,
            strategy,
            cache_config: CacheConfig {
                max_entries: self.cache.max_entries,
                ttl_secs: self.cache.ttl_secs,
                enabled: self.cache.enabled,
            },
            budget_pools,
            whisper_model: self.whisper.model,
            tts_model: self.tts.url,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_minimal_config() {
        let toml = "";
        let config: HooshConfig = toml::from_str(toml).unwrap();
        assert_eq!(config.server.port, 8088);
        assert_eq!(config.server.bind, "127.0.0.1");
        assert!(config.providers.is_empty());
    }

    #[test]
    fn parse_full_config() {
        let toml = r#"
[server]
bind = "0.0.0.0"
port = 9000
strategy = "round_robin"

[cache]
max_entries = 500
ttl_secs = 600
enabled = false

[[providers]]
type = "Ollama"
base_url = "http://gpu-box:11434"
priority = 1
models = ["llama*", "mistral*"]

[[providers]]
type = "OpenAi"
api_key = "$OPENAI_API_KEY"
priority = 10
models = ["gpt-*"]
"#;
        let config: HooshConfig = toml::from_str(toml).unwrap();
        assert_eq!(config.server.port, 9000);
        assert_eq!(config.server.bind, "0.0.0.0");
        assert_eq!(config.cache.max_entries, 500);
        assert!(!config.cache.enabled);
        assert_eq!(config.providers.len(), 2);
        assert_eq!(config.providers[0].provider_type, ProviderType::Ollama);
        assert_eq!(config.providers[1].provider_type, ProviderType::OpenAi);
        assert_eq!(
            config.providers[1].api_key.as_deref(),
            Some("$OPENAI_API_KEY")
        );
    }

    #[test]
    fn routes_from_config() {
        let toml = r#"
[[providers]]
type = "Ollama"
priority = 1
models = ["llama*"]

[[providers]]
type = "OpenAi"
api_key = "sk-test-key"
priority = 5
models = ["gpt-*"]
"#;
        let config: HooshConfig = toml::from_str(toml).unwrap();
        let routes = config.routes();
        assert_eq!(routes.len(), 2);
        assert_eq!(routes[0].base_url, "http://localhost:11434");
        assert!(routes[0].api_key.is_none());
        assert_eq!(routes[1].base_url, "https://api.openai.com");
        assert_eq!(routes[1].api_key.as_deref(), Some("sk-test-key"));
    }

    #[test]
    fn resolve_api_key_literal() {
        let key = Some("sk-literal".into());
        assert_eq!(resolve_api_key(&key).as_deref(), Some("sk-literal"));
    }

    #[test]
    fn resolve_api_key_env_var() {
        // SAFETY: test is single-threaded, no concurrent env access
        unsafe { std::env::set_var("HOOSH_TEST_KEY_1234", "sk-from-env") };
        let key = Some("$HOOSH_TEST_KEY_1234".into());
        assert_eq!(resolve_api_key(&key).as_deref(), Some("sk-from-env"));
        unsafe { std::env::remove_var("HOOSH_TEST_KEY_1234") };
    }

    #[test]
    fn resolve_api_key_missing_env() {
        let key = Some("$HOOSH_NONEXISTENT_KEY_999".into());
        assert!(resolve_api_key(&key).is_none());
    }

    #[test]
    fn resolve_api_key_none() {
        assert!(resolve_api_key(&None).is_none());
    }

    #[test]
    fn default_base_urls() {
        assert_eq!(
            default_base_url(ProviderType::Ollama),
            "http://localhost:11434"
        );
        assert_eq!(
            default_base_url(ProviderType::OpenAi),
            "https://api.openai.com"
        );
        assert_eq!(
            default_base_url(ProviderType::Anthropic),
            "https://api.anthropic.com"
        );
        assert_eq!(
            default_base_url(ProviderType::Groq),
            "https://api.groq.com/openai"
        );
    }

    #[test]
    fn into_server_config_with_overrides() {
        let toml = r#"
[server]
port = 9000
bind = "0.0.0.0"
"#;
        let config: HooshConfig = toml::from_str(toml).unwrap();
        let sc = config.into_server_config(Some("127.0.0.1"), Some(8080));
        assert_eq!(sc.bind, "127.0.0.1");
        assert_eq!(sc.port, 8080);
    }

    #[test]
    fn into_server_config_no_overrides() {
        let toml = r#"
[server]
port = 9000
bind = "0.0.0.0"
"#;
        let config: HooshConfig = toml::from_str(toml).unwrap();
        let sc = config.into_server_config(None, None);
        assert_eq!(sc.bind, "0.0.0.0");
        assert_eq!(sc.port, 9000);
    }

    #[test]
    fn provider_defaults() {
        let toml = r#"
[[providers]]
type = "Ollama"
"#;
        let config: HooshConfig = toml::from_str(toml).unwrap();
        let p = &config.providers[0];
        assert_eq!(p.priority, 10);
        assert!(p.enabled);
        assert!(p.models.is_empty());
        assert!(p.base_url.is_none());
    }
}
