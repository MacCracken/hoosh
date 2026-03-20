//! Text-to-speech provider.
//!
//! Supports local TTS via an HTTP backend (e.g. openedai-speech, piper-http)
//! or remote APIs (OpenAI /v1/audio/speech). The TTS endpoint is OpenAI-compatible.

use std::path::PathBuf;

use crate::inference::{SpeechRequest, SpeechResponse};

/// TTS provider that calls an HTTP TTS backend.
pub struct TtsProvider {
    client: reqwest::Client,
    base_url: String,
    api_key: Option<String>,
}

impl TtsProvider {
    /// Create a TTS provider pointing at an OpenAI-compatible TTS endpoint.
    pub fn new(base_url: impl Into<String>, api_key: Option<String>) -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(120))
                .connect_timeout(std::time::Duration::from_secs(10))
                .build()
                .unwrap_or_default(),
            base_url: base_url.into().trim_end_matches('/').to_string(),
            api_key,
        }
    }

    /// Synthesize speech from text via the backend.
    pub async fn synthesize(&self, request: &SpeechRequest) -> anyhow::Result<SpeechResponse> {
        let url = format!("{}/v1/audio/speech", self.base_url);
        let body = serde_json::json!({
            "input": request.input,
            "voice": request.voice,
            "speed": request.speed,
            "response_format": request.response_format,
            "model": "tts-1",
        });

        let mut rb = self.client.post(&url).json(&body);
        if let Some(key) = &self.api_key {
            rb = rb.bearer_auth(key);
        }

        let resp = rb.send().await?.error_for_status()?;

        let content_type = resp
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("audio/wav")
            .to_string();

        let audio = resp.bytes().await?.to_vec();

        // Estimate duration from WAV header or raw PCM
        let (sample_rate, duration_secs) = if audio.len() > 44 && &audio[0..4] == b"RIFF" {
            let sr = u32::from_le_bytes([audio[24], audio[25], audio[26], audio[27]]);
            let data_size = audio.len().saturating_sub(44);
            let duration = data_size as f64 / (sr as f64 * 2.0); // 16-bit mono
            (sr, duration)
        } else {
            (22050, audio.len() as f64 / (22050.0 * 2.0))
        };

        let format = if content_type.contains("wav") {
            "wav"
        } else if content_type.contains("pcm") {
            "pcm"
        } else if content_type.contains("mp3") {
            "mp3"
        } else {
            "wav"
        };

        Ok(SpeechResponse {
            audio,
            format: format.to_string(),
            sample_rate,
            duration_secs,
        })
    }
}

/// Generate a silent WAV file (useful for testing).
pub fn silent_wav(duration_secs: f64, sample_rate: u32) -> Vec<u8> {
    let num_samples = (duration_secs * sample_rate as f64) as usize;
    let data_size = (num_samples * 2) as u32;
    let file_size = 36 + data_size;

    let mut wav = Vec::with_capacity(44 + data_size as usize);
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&file_size.to_le_bytes());
    wav.extend_from_slice(b"WAVE");
    wav.extend_from_slice(b"fmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes()); // PCM
    wav.extend_from_slice(&1u16.to_le_bytes()); // mono
    wav.extend_from_slice(&sample_rate.to_le_bytes());
    wav.extend_from_slice(&(sample_rate * 2).to_le_bytes());
    wav.extend_from_slice(&2u16.to_le_bytes());
    wav.extend_from_slice(&16u16.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_size.to_le_bytes());
    wav.resize(44 + data_size as usize, 0); // silence
    wav
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silent_wav_valid_header() {
        let wav = silent_wav(1.0, 22050);
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        let sr = u32::from_le_bytes([wav[24], wav[25], wav[26], wav[27]]);
        assert_eq!(sr, 22050);
        // ~1 second at 22050 Hz mono 16-bit = 44100 bytes + 44 header
        assert_eq!(wav.len(), 44 + 44100);
    }

    #[test]
    fn provider_creation() {
        let p = TtsProvider::new("http://localhost:5500", None);
        assert_eq!(p.base_url, "http://localhost:5500");
    }

    #[test]
    fn provider_with_api_key() {
        let p = TtsProvider::new("https://api.openai.com", Some("sk-test".into()));
        assert!(p.api_key.is_some());
    }
}
