//! Whisper provider — speech-to-text via whisper.cpp (whisper-rs).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use crate::inference::{TranscriptionRequest, TranscriptionResponse, TranscriptionSegment};

/// Whisper speech-to-text provider.
pub struct WhisperProvider {
    ctx: Arc<WhisperContext>,
    model_path: PathBuf,
}

impl WhisperProvider {
    /// Load a whisper model from disk.
    pub fn new(model_path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let path = model_path.as_ref().to_path_buf();
        let params = WhisperContextParameters::new();
        let ctx = WhisperContext::new_with_params(
            path.to_str().ok_or_else(|| anyhow::anyhow!("invalid model path"))?,
            params,
        )
        .map_err(|e| anyhow::anyhow!("failed to load whisper model: {e:?}"))?;

        Ok(Self {
            ctx: Arc::new(ctx),
            model_path: path,
        })
    }

    /// The model file path.
    pub fn model_path(&self) -> &Path {
        &self.model_path
    }

    /// Transcribe audio. Runs synchronously (call via spawn_blocking for async).
    pub fn transcribe(&self, request: &TranscriptionRequest) -> anyhow::Result<TranscriptionResponse> {
        let samples = decode_audio(&request.audio)?;

        let mut state = self
            .ctx
            .create_state()
            .map_err(|e| anyhow::anyhow!("failed to create whisper state: {e:?}"))?;

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_print_progress(false);
        params.set_print_special(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);

        if let Some(lang) = &request.language {
            params.set_language(Some(lang));
        } else {
            params.set_detect_language(true);
        }

        if request.word_timestamps {
            params.set_token_timestamps(true);
        }

        params.set_n_threads(std::thread::available_parallelism().map(|n| n.get() as i32).unwrap_or(4));

        state
            .full(params, &samples)
            .map_err(|e| anyhow::anyhow!("whisper transcription failed: {e:?}"))?;

        let n_segments = state
            .full_n_segments()
            .map_err(|e| anyhow::anyhow!("failed to get segments: {e:?}"))?;

        let mut text_parts = Vec::new();
        let mut segments = Vec::new();

        for i in 0..n_segments {
            let seg_text = state
                .full_get_segment_text_lossy(i)
                .map_err(|e| anyhow::anyhow!("failed to get segment text: {e:?}"))?;
            let t0 = state
                .full_get_segment_t0(i)
                .map_err(|e| anyhow::anyhow!("failed to get segment t0: {e:?}"))?;
            let t1 = state
                .full_get_segment_t1(i)
                .map_err(|e| anyhow::anyhow!("failed to get segment t1: {e:?}"))?;

            text_parts.push(seg_text.clone());
            segments.push(TranscriptionSegment {
                text: seg_text,
                start_secs: t0 as f64 / 100.0,
                end_secs: t1 as f64 / 100.0,
            });
        }

        let lang_id = state.full_lang_id_from_state().unwrap_or(0);
        let language = whisper_rs::get_lang_str(lang_id)
            .unwrap_or("unknown")
            .to_string();

        let duration_secs = samples.len() as f64 / 16000.0;

        Ok(TranscriptionResponse {
            text: text_parts.join("").trim().to_string(),
            language,
            duration_secs,
            segments,
        })
    }

    /// Async transcription wrapper.
    pub async fn transcribe_async(
        &self,
        request: TranscriptionRequest,
    ) -> anyhow::Result<TranscriptionResponse> {
        let ctx = self.ctx.clone();
        let provider = WhisperProvider {
            ctx,
            model_path: self.model_path.clone(),
        };
        tokio::task::spawn_blocking(move || provider.transcribe(&request)).await?
    }
}

/// Decode raw audio bytes (WAV) into 16kHz mono f32 samples.
fn decode_audio(data: &[u8]) -> anyhow::Result<Vec<f32>> {
    // Simple WAV decoder — expects 16-bit PCM WAV
    if data.len() < 44 {
        return Err(anyhow::anyhow!("audio data too short for WAV header"));
    }
    if &data[0..4] != b"RIFF" || &data[8..12] != b"WAVE" {
        return Err(anyhow::anyhow!("not a WAV file"));
    }

    // Find data chunk
    let mut pos = 12;
    while pos + 8 < data.len() {
        let chunk_id = &data[pos..pos + 4];
        let chunk_size = u32::from_le_bytes([
            data[pos + 4],
            data[pos + 5],
            data[pos + 6],
            data[pos + 7],
        ]) as usize;

        if chunk_id == b"data" {
            let audio_data = &data[pos + 8..pos + 8 + chunk_size.min(data.len() - pos - 8)];
            // Convert i16 PCM to f32
            let mut samples = Vec::with_capacity(audio_data.len() / 2);
            for chunk in audio_data.chunks_exact(2) {
                let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
                samples.push(sample as f32 / 32768.0);
            }
            return Ok(samples);
        }

        pos += 8 + chunk_size;
        // Align to even boundary
        if chunk_size % 2 != 0 {
            pos += 1;
        }
    }

    Err(anyhow::anyhow!("no data chunk found in WAV file"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_audio_rejects_short_data() {
        let result = decode_audio(&[0; 10]);
        assert!(result.is_err());
    }

    #[test]
    fn decode_audio_rejects_non_wav() {
        let result = decode_audio(&[0; 100]);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not a WAV"));
    }

    #[test]
    fn decode_audio_valid_wav() {
        // Minimal valid WAV: 44-byte header + 4 bytes of silence
        let mut wav = Vec::new();
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&40u32.to_le_bytes()); // file size - 8
        wav.extend_from_slice(b"WAVE");
        wav.extend_from_slice(b"fmt ");
        wav.extend_from_slice(&16u32.to_le_bytes()); // chunk size
        wav.extend_from_slice(&1u16.to_le_bytes()); // PCM
        wav.extend_from_slice(&1u16.to_le_bytes()); // mono
        wav.extend_from_slice(&16000u32.to_le_bytes()); // sample rate
        wav.extend_from_slice(&32000u32.to_le_bytes()); // byte rate
        wav.extend_from_slice(&2u16.to_le_bytes()); // block align
        wav.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&4u32.to_le_bytes()); // data size
        wav.extend_from_slice(&0i16.to_le_bytes()); // sample 1
        wav.extend_from_slice(&16384i16.to_le_bytes()); // sample 2

        let samples = decode_audio(&wav).unwrap();
        assert_eq!(samples.len(), 2);
        assert!((samples[0] - 0.0).abs() < f32::EPSILON);
        assert!((samples[1] - 0.5).abs() < 0.001);
    }
}
