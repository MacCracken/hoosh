//! Core inference types: requests, responses, model metadata.

pub mod batch;

use serde::{Deserialize, Serialize};

/// An inference request.
///
/// # Example
///
/// ```
/// use hoosh::InferenceRequest;
///
/// let req = InferenceRequest {
///     model: "llama3".into(),
///     prompt: "Hello".into(),
///     max_tokens: Some(100),
///     temperature: Some(0.7),
///     ..Default::default()
/// };
/// assert_eq!(req.model, "llama3");
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InferenceRequest {
    /// Model identifier (e.g. "llama3", "gpt-4o", "claude-sonnet-4-20250514").
    pub model: String,
    /// The prompt or last user message.
    pub prompt: String,
    /// System prompt (optional).
    pub system: Option<String>,
    /// Conversation history (for multi-turn).
    pub messages: Vec<Message>,
    /// Maximum tokens to generate.
    pub max_tokens: Option<u32>,
    /// Temperature (0.0–2.0).
    pub temperature: Option<f64>,
    /// Top-p nucleus sampling.
    pub top_p: Option<f64>,
    /// Whether to stream the response.
    pub stream: bool,
    /// Tool definitions the model may call.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<crate::tools::ToolDefinition>,
    /// How the model should choose tools.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<crate::tools::ToolChoice>,
}

/// Message content — plain text or multi-part (text + images).
///
/// Deserializes from either a JSON string (`"hello"`) or an array of content
/// parts (`[{"type":"text","text":"hello"}, {"type":"image_url",...}]`),
/// matching the OpenAI API format.
///
/// # Example
///
/// ```
/// use hoosh::inference::MessageContent;
///
/// // Plain text
/// let text = MessageContent::Text("Hello".into());
/// assert_eq!(text.text(), "Hello");
/// assert!(!text.has_images());
///
/// // Deserializes from JSON string
/// let mc: MessageContent = serde_json::from_str(r#""hello""#).unwrap();
/// assert_eq!(mc, "hello");
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    /// Plain text content.
    Text(String),
    /// Multi-part content (text, images, etc.).
    Parts(Vec<ContentPart>),
}

impl MessageContent {
    /// Extract the text content, concatenating text parts if multi-part.
    #[must_use]
    pub fn text(&self) -> std::borrow::Cow<'_, str> {
        match self {
            Self::Text(s) => std::borrow::Cow::Borrowed(s),
            Self::Parts(parts) => {
                let mut buf = String::new();
                for p in parts {
                    if let ContentPart::Text { text } = p {
                        if !buf.is_empty() {
                            buf.push(' ');
                        }
                        buf.push_str(text);
                    }
                }
                std::borrow::Cow::Owned(buf)
            }
        }
    }

    /// Whether the content contains any image parts.
    #[must_use]
    pub fn has_images(&self) -> bool {
        match self {
            Self::Text(_) => false,
            Self::Parts(parts) => parts
                .iter()
                .any(|p| matches!(p, ContentPart::ImageUrl { .. })),
        }
    }
}

impl Default for MessageContent {
    fn default() -> Self {
        Self::Text(String::new())
    }
}

impl From<String> for MessageContent {
    fn from(s: String) -> Self {
        Self::Text(s)
    }
}

impl From<&str> for MessageContent {
    fn from(s: &str) -> Self {
        Self::Text(s.to_string())
    }
}

impl PartialEq<&str> for MessageContent {
    fn eq(&self, other: &&str) -> bool {
        self.text() == *other
    }
}

impl PartialEq<str> for MessageContent {
    fn eq(&self, other: &str) -> bool {
        self.text() == other
    }
}

/// A single content part in a multi-modal message.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
#[non_exhaustive]
pub enum ContentPart {
    /// Text content.
    #[serde(rename = "text")]
    Text { text: String },
    /// Image URL content.
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrl },
}

/// Image URL with optional detail level.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageUrl {
    /// URL of the image (https:// or data:image/...).
    pub url: String,
    /// Detail level: "low", "high", or "auto".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// A single message in a conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: MessageContent,
    /// For tool-result messages: the ID of the tool call this responds to.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// For assistant messages: tool calls the model made.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<crate::tools::ToolCall>,
}

impl Message {
    /// Create a new text message with no tool fields.
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: MessageContent::Text(content.into()),
            tool_call_id: None,
            tool_calls: Vec::new(),
        }
    }
}

/// Message role.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

/// An inference response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResponse {
    /// Generated text.
    pub text: String,
    /// Model that produced the response.
    pub model: String,
    /// Token usage.
    pub usage: TokenUsage,
    /// Tool calls made by the model (empty if none).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<crate::tools::ToolCall>,
    /// Provider that handled the request.
    pub provider: String,
    /// Latency in milliseconds.
    pub latency_ms: u64,
}

/// Detected emotion with intensity.
#[cfg(feature = "sentiment")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionScore {
    pub emotion: String,
    pub intensity: f32,
}

/// Sentiment analysis result for an inference response.
#[cfg(feature = "sentiment")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentimentAnalysis {
    /// Overall valence: -1.0 (very negative) to 1.0 (very positive).
    pub valence: f32,
    /// Confidence in the classification.
    pub confidence: f32,
    /// Whether the overall sentiment is positive.
    pub is_positive: bool,
    /// Whether the overall sentiment is negative.
    pub is_negative: bool,
    /// Emotion breakdown with intensities.
    pub emotions: Vec<EmotionScore>,
    /// Keywords that contributed to the classification.
    pub matched_keywords: Vec<String>,
}

/// Per-sentence sentiment breakdown.
#[cfg(feature = "sentiment")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentenceSentiment {
    pub text: String,
    pub valence: f32,
    pub confidence: f32,
}

/// Document-level sentiment with per-sentence breakdown.
#[cfg(feature = "sentiment")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentSentiment {
    /// Aggregate sentiment across all sentences.
    pub aggregate: SentimentAnalysis,
    /// Per-sentence breakdown.
    pub sentences: Vec<SentenceSentiment>,
}

/// Custom sentiment configuration for domain-specific analysis.
#[cfg(feature = "sentiment")]
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SentimentConfig {
    /// Additional positive keywords for the domain.
    #[serde(default)]
    pub extra_positive: Vec<String>,
    /// Additional negative keywords for the domain.
    #[serde(default)]
    pub extra_negative: Vec<String>,
    /// Additional trust keywords.
    #[serde(default)]
    pub extra_trust: Vec<String>,
    /// Additional curiosity keywords.
    #[serde(default)]
    pub extra_curiosity: Vec<String>,
    /// Additional frustration keywords.
    #[serde(default)]
    pub extra_frustration: Vec<String>,
}

#[cfg(feature = "sentiment")]
impl SentimentConfig {
    fn to_bhava_config(&self) -> bhava::sentiment::SentimentConfig {
        let mut cfg = bhava::sentiment::SentimentConfig::new();
        cfg.extra_positive = self.extra_positive.clone();
        cfg.extra_negative = self.extra_negative.clone();
        cfg.extra_trust = self.extra_trust.clone();
        cfg.extra_curiosity = self.extra_curiosity.clone();
        cfg.extra_frustration = self.extra_frustration.clone();
        cfg
    }
}

/// Convert a bhava SentimentResult to hoosh's SentimentAnalysis.
#[cfg(feature = "sentiment")]
fn from_bhava_result(result: &bhava::sentiment::SentimentResult) -> SentimentAnalysis {
    SentimentAnalysis {
        valence: result.valence,
        confidence: result.confidence,
        is_positive: result.is_positive(),
        is_negative: result.is_negative(),
        emotions: result
            .emotions
            .iter()
            .map(|(emotion, intensity)| EmotionScore {
                emotion: format!("{emotion:?}"),
                intensity: *intensity,
            })
            .collect(),
        matched_keywords: result.matched_keywords.clone(),
    }
}

/// Analyze response text for sentiment (requires `sentiment` feature).
#[cfg(feature = "sentiment")]
#[must_use]
pub fn analyze_response_sentiment(text: &str) -> SentimentAnalysis {
    let result = bhava::sentiment::analyze(text);
    from_bhava_result(&result)
}

/// Analyze response text with a custom configuration.
#[cfg(feature = "sentiment")]
#[must_use]
pub fn analyze_response_sentiment_with_config(
    text: &str,
    config: &SentimentConfig,
) -> SentimentAnalysis {
    let bhava_cfg = config.to_bhava_config();
    let result = bhava::sentiment::analyze_with_config(text, &bhava_cfg);
    from_bhava_result(&result)
}

/// Analyze response text at the document level with per-sentence breakdown.
#[cfg(feature = "sentiment")]
#[must_use]
pub fn analyze_response_document(text: &str) -> DocumentSentiment {
    let doc = bhava::sentiment::analyze_sentences(text);
    DocumentSentiment {
        aggregate: from_bhava_result(&doc.aggregate),
        sentences: doc
            .sentences
            .iter()
            .map(|s| SentenceSentiment {
                text: s.text.clone(),
                valence: s.sentiment.valence,
                confidence: s.sentiment.confidence,
            })
            .collect(),
    }
}

/// Analyze response text at the document level with custom config.
#[cfg(feature = "sentiment")]
#[must_use]
pub fn analyze_response_document_with_config(
    text: &str,
    config: &SentimentConfig,
) -> DocumentSentiment {
    let bhava_cfg = config.to_bhava_config();
    let doc = bhava::sentiment::analyze_sentences_with_config(text, &bhava_cfg);
    DocumentSentiment {
        aggregate: from_bhava_result(&doc.aggregate),
        sentences: doc
            .sentences
            .iter()
            .map(|s| SentenceSentiment {
                text: s.text.clone(),
                valence: s.sentiment.valence,
                confidence: s.sentiment.confidence,
            })
            .collect(),
    }
}

/// Extension trait for analyzing sentiment on inference responses.
#[cfg(feature = "sentiment")]
pub trait ResponseSentiment {
    /// Analyze the sentiment of this response's text.
    fn sentiment(&self) -> SentimentAnalysis;

    /// Analyze with per-sentence breakdown.
    fn document_sentiment(&self) -> DocumentSentiment;
}

#[cfg(feature = "sentiment")]
impl ResponseSentiment for InferenceResponse {
    fn sentiment(&self) -> SentimentAnalysis {
        analyze_response_sentiment(&self.text)
    }

    fn document_sentiment(&self) -> DocumentSentiment {
        analyze_response_document(&self.text)
    }
}

/// Token usage breakdown.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TokenUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

/// Model metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Model identifier.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Provider that serves this model.
    pub provider: String,
    /// Parameter count (if known).
    pub parameters: Option<u64>,
    /// Context window size in tokens.
    pub context_length: Option<u32>,
    /// Whether the model is currently loaded/available.
    pub available: bool,
}

/// Speech-to-text request.
#[derive(Debug, Clone)]
pub struct TranscriptionRequest {
    /// Audio data (WAV, MP3, OGG, FLAC).
    pub audio: Vec<u8>,
    /// Language hint (ISO 639-1, e.g. "en").
    pub language: Option<String>,
    /// Whether to include word-level timestamps.
    pub word_timestamps: bool,
}

/// Speech-to-text response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionResponse {
    /// Transcribed text.
    pub text: String,
    /// Detected language.
    pub language: String,
    /// Duration of audio in seconds.
    pub duration_secs: f64,
    /// Word-level segments (if requested).
    pub segments: Vec<TranscriptionSegment>,
}

/// A timed segment of a transcription.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionSegment {
    pub text: String,
    pub start_secs: f64,
    pub end_secs: f64,
}

/// Text-to-speech request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeechRequest {
    /// Text to synthesize.
    pub input: String,
    /// Voice name or ID.
    #[serde(default = "default_voice")]
    pub voice: String,
    /// Speech speed multiplier (0.25–4.0).
    #[serde(default = "default_speed")]
    pub speed: f32,
    /// Output format: "wav", "pcm".
    #[serde(default = "default_audio_format")]
    pub response_format: String,
}

fn default_voice() -> String {
    "default".into()
}
fn default_speed() -> f32 {
    1.0
}
fn default_audio_format() -> String {
    "wav".into()
}

/// Text-to-speech response (audio bytes).
#[derive(Debug, Clone)]
pub struct SpeechResponse {
    /// Raw audio data.
    pub audio: Vec<u8>,
    /// Audio format (e.g. "wav").
    pub format: String,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Duration in seconds.
    pub duration_secs: f64,
}

/// Embeddings request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingsRequest {
    /// Model to use for embeddings.
    pub model: String,
    /// Input text(s) to embed.
    pub input: EmbeddingsInput,
}

/// Input for embeddings — single string or array of strings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
#[non_exhaustive]
pub enum EmbeddingsInput {
    Single(String),
    Multiple(Vec<String>),
}

/// Embeddings response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingsResponse {
    pub object: String,
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: EmbeddingsUsage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingData {
    pub object: String,
    pub embedding: Vec<f32>,
    pub index: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingsUsage {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_default() {
        let req = InferenceRequest::default();
        assert!(req.model.is_empty());
        assert!(!req.stream);
        assert!(req.messages.is_empty());
    }

    #[test]
    fn token_usage_default() {
        let usage = TokenUsage::default();
        assert_eq!(usage.total_tokens, 0);
    }

    #[test]
    fn serde_roundtrip_request() {
        let req = InferenceRequest {
            model: "llama3".into(),
            prompt: "hello".into(),
            temperature: Some(0.7),
            ..Default::default()
        };
        let json = serde_json::to_string(&req).unwrap();
        let back: InferenceRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.model, "llama3");
        assert!((back.temperature.unwrap() - 0.7).abs() < f64::EPSILON);
    }

    #[test]
    fn serde_roundtrip_response() {
        let resp = InferenceResponse {
            text: "Rust uses ownership.".into(),
            model: "llama3".into(),
            usage: TokenUsage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
            },
            provider: "ollama".into(),
            latency_ms: 42,
            tool_calls: Vec::new(),
        };
        let json = serde_json::to_string(&resp).unwrap();
        let back: InferenceResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(back.usage.total_tokens, 15);
    }

    #[test]
    fn embeddings_input_single_serde() {
        let input = EmbeddingsInput::Single("hello world".into());
        let json = serde_json::to_string(&input).unwrap();
        assert_eq!(json, "\"hello world\"");
        let back: EmbeddingsInput = serde_json::from_str(&json).unwrap();
        match back {
            EmbeddingsInput::Single(s) => assert_eq!(s, "hello world"),
            _ => panic!("expected Single variant"),
        }
    }

    #[test]
    fn embeddings_input_multiple_serde() {
        let input = EmbeddingsInput::Multiple(vec!["a".into(), "b".into()]);
        let json = serde_json::to_string(&input).unwrap();
        let back: EmbeddingsInput = serde_json::from_str(&json).unwrap();
        match back {
            EmbeddingsInput::Multiple(v) => {
                assert_eq!(v.len(), 2);
                assert_eq!(v[0], "a");
                assert_eq!(v[1], "b");
            }
            _ => panic!("expected Multiple variant"),
        }
    }

    #[test]
    fn embeddings_request_roundtrip() {
        let req = EmbeddingsRequest {
            model: "text-embedding-ada-002".into(),
            input: EmbeddingsInput::Single("test input".into()),
        };
        let json = serde_json::to_string(&req).unwrap();
        let back: EmbeddingsRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.model, "text-embedding-ada-002");
        match back.input {
            EmbeddingsInput::Single(s) => assert_eq!(s, "test input"),
            _ => panic!("expected Single variant"),
        }
    }

    #[test]
    fn role_serde() {
        let json = serde_json::to_string(&Role::Assistant).unwrap();
        assert_eq!(json, "\"assistant\"");
        let back: Role = serde_json::from_str(&json).unwrap();
        assert_eq!(back, Role::Assistant);
    }

    #[test]
    fn role_serde_all_variants() {
        for (role, expected) in [
            (Role::System, "\"system\""),
            (Role::User, "\"user\""),
            (Role::Assistant, "\"assistant\""),
        ] {
            let json = serde_json::to_string(&role).unwrap();
            assert_eq!(json, expected);
            let back: Role = serde_json::from_str(&json).unwrap();
            assert_eq!(back, role);
        }
    }

    #[test]
    fn message_serde_roundtrip() {
        let msg = Message::new(Role::User, "What is Rust?");
        let json = serde_json::to_string(&msg).unwrap();
        let back: Message = serde_json::from_str(&json).unwrap();
        assert_eq!(back.role, Role::User);
        assert_eq!(back.content, "What is Rust?");
    }

    #[test]
    fn speech_request_defaults() {
        let json = r#"{"input":"hello"}"#;
        let req: SpeechRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.input, "hello");
        assert_eq!(req.voice, "default");
        assert!((req.speed - 1.0).abs() < f32::EPSILON);
        assert_eq!(req.response_format, "wav");
    }

    #[test]
    fn speech_request_custom() {
        let req = SpeechRequest {
            input: "hi".into(),
            voice: "nova".into(),
            speed: 1.5,
            response_format: "pcm".into(),
        };
        let json = serde_json::to_string(&req).unwrap();
        let back: SpeechRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.voice, "nova");
        assert!((back.speed - 1.5).abs() < f32::EPSILON);
        assert_eq!(back.response_format, "pcm");
    }

    #[test]
    fn model_info_serde() {
        let info = ModelInfo {
            id: "llama3:8b".into(),
            name: "LLaMA 3 8B".into(),
            provider: "ollama".into(),
            parameters: Some(8_000_000_000),
            context_length: Some(8192),
            available: true,
        };
        let json = serde_json::to_string(&info).unwrap();
        let back: ModelInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(back.id, "llama3:8b");
        assert_eq!(back.parameters, Some(8_000_000_000));
        assert!(back.available);
    }

    #[test]
    fn transcription_response_serde() {
        let resp = TranscriptionResponse {
            text: "Hello world".into(),
            language: "en".into(),
            duration_secs: 2.5,
            segments: vec![TranscriptionSegment {
                text: "Hello".into(),
                start_secs: 0.0,
                end_secs: 1.0,
            }],
        };
        let json = serde_json::to_string(&resp).unwrap();
        let back: TranscriptionResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(back.text, "Hello world");
        assert_eq!(back.segments.len(), 1);
        assert!((back.duration_secs - 2.5).abs() < f64::EPSILON);
    }

    #[test]
    fn embeddings_response_serde() {
        let resp = EmbeddingsResponse {
            object: "list".into(),
            data: vec![EmbeddingData {
                object: "embedding".into(),
                embedding: vec![0.1, 0.2, 0.3],
                index: 0,
            }],
            model: "text-embedding-ada-002".into(),
            usage: EmbeddingsUsage {
                prompt_tokens: 5,
                total_tokens: 5,
            },
        };
        let json = serde_json::to_string(&resp).unwrap();
        let back: EmbeddingsResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(back.data.len(), 1);
        assert_eq!(back.data[0].embedding.len(), 3);
        assert_eq!(back.usage.prompt_tokens, 5);
    }

    #[test]
    fn inference_request_with_messages() {
        let req = InferenceRequest {
            model: "gpt-4o".into(),
            prompt: String::new(),
            system: Some("You are helpful.".into()),
            messages: vec![
                Message::new(Role::System, "You are helpful."),
                Message::new(Role::User, "Hi"),
            ],
            max_tokens: Some(1000),
            temperature: Some(0.5),
            top_p: Some(0.9),
            stream: true,
            ..Default::default()
        };
        let json = serde_json::to_string(&req).unwrap();
        let back: InferenceRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.messages.len(), 2);
        assert!(back.stream);
        assert_eq!(back.system.as_deref(), Some("You are helpful."));
        assert_eq!(back.top_p, Some(0.9));
    }

    #[cfg(feature = "sentiment")]
    #[test]
    fn sentiment_analysis_positive() {
        let result = analyze_response_sentiment("This is great and wonderful!");
        assert!(result.valence > 0.0);
        assert!(result.is_positive);
        assert!(!result.is_negative);
        assert!(result.confidence > 0.0);
    }

    #[cfg(feature = "sentiment")]
    #[test]
    fn sentiment_analysis_negative() {
        let result = analyze_response_sentiment("This is terrible and horrible!");
        assert!(result.valence < 0.0);
        assert!(!result.is_positive);
        assert!(result.is_negative);
    }

    #[cfg(feature = "sentiment")]
    #[test]
    fn sentiment_analysis_neutral() {
        let result = analyze_response_sentiment("The function returns an integer.");
        // Neutral text should have low absolute valence
        assert!(result.valence.abs() < 0.5);
    }

    #[cfg(feature = "sentiment")]
    #[test]
    fn response_sentiment_trait() {
        let resp = InferenceResponse {
            text: "I love this answer, it's fantastic!".into(),
            model: "test".into(),
            usage: TokenUsage::default(),
            provider: "test".into(),
            latency_ms: 0,
            tool_calls: Vec::new(),
        };
        let s = resp.sentiment();
        assert!(s.valence > 0.0);
        assert!(s.is_positive);
    }

    #[cfg(feature = "sentiment")]
    #[test]
    fn sentiment_analysis_serde() {
        let sa = SentimentAnalysis {
            valence: 0.8,
            confidence: 0.9,
            is_positive: true,
            is_negative: false,
            emotions: vec![EmotionScore {
                emotion: "Joy".into(),
                intensity: 0.8,
            }],
            matched_keywords: vec!["great".into()],
        };
        let json = serde_json::to_string(&sa).unwrap();
        let back: SentimentAnalysis = serde_json::from_str(&json).unwrap();
        assert!((back.valence - 0.8).abs() < f32::EPSILON);
        assert!(back.is_positive);
    }

    #[cfg(feature = "sentiment")]
    #[test]
    fn sentiment_empty_text() {
        let result = analyze_response_sentiment("");
        assert!(result.valence.abs() < 0.5);
        assert!(result.emotions.is_empty() || result.confidence < 0.1);
    }

    #[cfg(feature = "sentiment")]
    #[test]
    fn sentiment_whitespace_only() {
        let result = analyze_response_sentiment("   \n\t  ");
        assert!(result.valence.abs() < 0.5);
    }

    #[cfg(feature = "sentiment")]
    #[test]
    fn sentiment_single_char() {
        let result = analyze_response_sentiment("x");
        // Should not panic; valence near neutral
        let _ = result.valence;
    }

    #[cfg(feature = "sentiment")]
    #[test]
    fn document_sentiment_basic() {
        let doc = analyze_response_document("I love this. I hate that.");
        assert_eq!(doc.sentences.len(), 2);
        assert!(doc.sentences[0].valence > 0.0);
        assert!(doc.sentences[1].valence < 0.0);
    }

    #[cfg(feature = "sentiment")]
    #[test]
    fn document_sentiment_empty() {
        let doc = analyze_response_document("");
        // Should not panic
        let _ = doc.aggregate.valence;
    }

    #[cfg(feature = "sentiment")]
    #[test]
    fn sentiment_with_custom_config() {
        let config = SentimentConfig {
            extra_positive: vec!["blazing".into()],
            ..Default::default()
        };
        let result = analyze_response_sentiment_with_config("blazing fast", &config);
        assert!(result.is_positive);
    }

    #[cfg(feature = "sentiment")]
    #[test]
    fn sentiment_emotions_present() {
        let result = analyze_response_sentiment("I am very happy and excited!");
        // Should detect at least one emotion
        assert!(!result.emotions.is_empty());
    }

    #[cfg(feature = "sentiment")]
    #[test]
    fn sentiment_config_serde() {
        let config = SentimentConfig {
            extra_positive: vec!["blazing".into()],
            extra_negative: vec!["glacial".into()],
            ..Default::default()
        };
        let json = serde_json::to_string(&config).unwrap();
        let back: SentimentConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.extra_positive, vec!["blazing"]);
        assert_eq!(back.extra_negative, vec!["glacial"]);
    }

    #[test]
    fn message_content_text_serde() {
        let msg = Message::new(Role::User, "hello");
        let json = serde_json::to_string(&msg).unwrap();
        let back: Message = serde_json::from_str(&json).unwrap();
        assert_eq!(back.content, "hello");
    }

    #[test]
    fn message_content_parts_serde() {
        let msg = Message {
            role: Role::User,
            content: MessageContent::Parts(vec![
                ContentPart::Text {
                    text: "What is this?".into(),
                },
                ContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url: "https://example.com/img.png".into(),
                        detail: Some("high".into()),
                    },
                },
            ]),
            tool_call_id: None,
            tool_calls: vec![],
        };
        let json = serde_json::to_string(&msg).unwrap();
        let back: Message = serde_json::from_str(&json).unwrap();
        assert!(back.content.has_images());
        assert_eq!(back.content.text(), "What is this?");
    }

    #[test]
    fn message_content_text_from_plain_string_json() {
        // OpenAI format: "content": "hello"
        let json = r#"{"role":"user","content":"hello"}"#;
        let msg: Message = serde_json::from_str(json).unwrap();
        assert_eq!(msg.content, "hello");
        assert!(!msg.content.has_images());
    }

    #[test]
    fn message_content_parts_from_array_json() {
        // OpenAI format: "content": [{"type":"text","text":"hi"},{"type":"image_url",...}]
        let json = r#"{"role":"user","content":[{"type":"text","text":"describe this"},{"type":"image_url","image_url":{"url":"data:image/png;base64,abc"}}]}"#;
        let msg: Message = serde_json::from_str(json).unwrap();
        assert!(msg.content.has_images());
        assert_eq!(msg.content.text(), "describe this");
    }

    #[test]
    fn message_content_no_images_in_text() {
        let content = MessageContent::Text("just text".into());
        assert!(!content.has_images());
    }

    #[test]
    fn message_content_partial_eq_str() {
        let content = MessageContent::Text("hello".into());
        assert_eq!(content, "hello");
    }

    #[test]
    fn message_content_default() {
        let content = MessageContent::default();
        assert_eq!(content, "");
    }
}
