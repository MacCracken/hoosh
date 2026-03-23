//! Core inference types: requests, responses, model metadata.

use serde::{Deserialize, Serialize};

/// An inference request.
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
}

/// A single message in a conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

/// Message role.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
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
    /// Provider that handled the request.
    pub provider: String,
    /// Latency in milliseconds.
    pub latency_ms: u64,
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
}

/// Analyze response text for sentiment (requires `sentiment` feature).
#[cfg(feature = "sentiment")]
#[must_use]
pub fn analyze_response_sentiment(text: &str) -> SentimentAnalysis {
    let result = bhava::sentiment::analyze(text);
    SentimentAnalysis {
        valence: result.valence,
        confidence: result.confidence,
        is_positive: result.is_positive(),
        is_negative: result.is_negative(),
    }
}

/// Extension trait for analyzing sentiment on inference responses.
#[cfg(feature = "sentiment")]
pub trait ResponseSentiment {
    /// Analyze the sentiment of this response's text.
    fn sentiment(&self) -> SentimentAnalysis;
}

#[cfg(feature = "sentiment")]
impl ResponseSentiment for InferenceResponse {
    fn sentiment(&self) -> SentimentAnalysis {
        analyze_response_sentiment(&self.text)
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
        let msg = Message {
            role: Role::User,
            content: "What is Rust?".into(),
        };
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
                Message {
                    role: Role::System,
                    content: "You are helpful.".into(),
                },
                Message {
                    role: Role::User,
                    content: "Hi".into(),
                },
            ],
            max_tokens: Some(1000),
            temperature: Some(0.5),
            top_p: Some(0.9),
            stream: true,
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
        };
        let json = serde_json::to_string(&sa).unwrap();
        let back: SentimentAnalysis = serde_json::from_str(&json).unwrap();
        assert!((back.valence - 0.8).abs() < f32::EPSILON);
        assert!(back.is_positive);
    }
}
