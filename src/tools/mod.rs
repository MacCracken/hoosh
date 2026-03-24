//! Tool use & function calling — unified abstraction across LLM providers,
//! with optional MCP integration via bote + szal.

#[cfg(feature = "tools")]
mod bridge;
mod convert;
mod types;

#[cfg(feature = "tools")]
pub use bridge::McpBridge;
pub use convert::*;
pub use types::*;
