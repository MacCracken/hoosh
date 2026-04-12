//! Audio route handlers — speech-to-text (Whisper) and text-to-speech (Piper).

#[cfg(any(feature = "whisper", feature = "piper"))]
use std::sync::Arc;

#[cfg(any(feature = "whisper", feature = "piper"))]
use axum::{Json, extract::State, http::StatusCode, response::IntoResponse};

#[cfg(any(feature = "whisper", feature = "piper"))]
use super::AppState;
#[cfg(any(feature = "whisper", feature = "piper"))]
use super::types::error_response;

// ---------------------------------------------------------------------------
// Speech-to-text: /v1/audio/transcriptions (OpenAI-compatible)
// ---------------------------------------------------------------------------

#[cfg(feature = "whisper")]
pub(crate) async fn transcribe(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    body: axum::body::Bytes,
) -> impl IntoResponse {
    if let Some(ct) = headers.get("content-type") {
        let ct_str = ct.to_str().unwrap_or("");
        if !ct_str.starts_with("audio/")
            && !ct_str.starts_with("application/octet-stream")
            && !ct_str.starts_with("multipart/form-data")
        {
            return error_response(
                StatusCode::UNSUPPORTED_MEDIA_TYPE,
                format!("expected audio/* content type, got: {ct_str}"),
            )
            .into_response();
        }
    }

    let whisper = match &state.whisper {
        Some(w) => w.clone(),
        None => {
            return error_response(
                StatusCode::SERVICE_UNAVAILABLE,
                "Whisper model not loaded. Set whisper_model in config.",
            )
            .into_response();
        }
    };

    let request = crate::inference::TranscriptionRequest {
        audio: body.to_vec(),
        language: None,
        word_timestamps: false,
    };

    match whisper.transcribe_async(request).await {
        Ok(result) => {
            let resp = serde_json::json!({
                "text": result.text,
                "language": result.language,
                "duration": result.duration_secs,
                "segments": result.segments.iter().map(|s| {
                    serde_json::json!({
                        "text": s.text,
                        "start": s.start_secs,
                        "end": s.end_secs,
                    })
                }).collect::<Vec<_>>(),
            });
            (StatusCode::OK, Json(resp)).into_response()
        }
        Err(e) => error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Transcription error: {e}"),
        )
        .into_response(),
    }
}

// ---------------------------------------------------------------------------
// Text-to-speech: /v1/audio/speech (OpenAI-compatible)
// ---------------------------------------------------------------------------

#[cfg(feature = "piper")]
pub(crate) async fn text_to_speech(
    State(state): State<Arc<AppState>>,
    Json(req): Json<crate::inference::SpeechRequest>,
) -> impl IntoResponse {
    let tts = match &state.tts {
        Some(t) => t.clone(),
        None => {
            return error_response(
                StatusCode::SERVICE_UNAVAILABLE,
                "TTS model not loaded. Set tts_model in config.",
            )
            .into_response();
        }
    };

    if req.input.is_empty() {
        return error_response(StatusCode::BAD_REQUEST, "input text is required").into_response();
    }
    if req.input.len() > 4096 {
        return error_response(
            StatusCode::BAD_REQUEST,
            format!("input too long: {} chars (max 4096)", req.input.len()),
        )
        .into_response();
    }
    if !(0.25..=4.0).contains(&req.speed) {
        return error_response(
            StatusCode::BAD_REQUEST,
            format!("speed must be between 0.25 and 4.0, got {}", req.speed),
        )
        .into_response();
    }

    match tts.synthesize(&req).await {
        Ok(result) => {
            let content_type = match result.format.as_str() {
                "pcm" => "audio/pcm",
                _ => "audio/wav",
            };
            (
                StatusCode::OK,
                [(axum::http::header::CONTENT_TYPE, content_type)],
                result.audio,
            )
                .into_response()
        }
        Err(e) => error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("TTS synthesis error: {e}"),
        )
        .into_response(),
    }
}
