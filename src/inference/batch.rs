//! Batch inference manager — concurrent request execution with progress tracking.
//!
//! Accepts a batch of inference requests and executes them concurrently with
//! configurable parallelism, progress tracking, and cancellation support.

use std::sync::Arc;

use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use tokio::sync::Semaphore;
use tokio_util::sync::CancellationToken;

use crate::inference::{InferenceRequest, InferenceResponse};

/// Status of a batch operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum BatchStatus {
    /// Batch is currently executing.
    Running,
    /// All requests completed (some may have failed).
    Completed,
    /// Batch was cancelled by the user.
    Cancelled,
}

/// Progress of a single item in the batch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchItemResult {
    /// Index of this item in the original request array.
    pub index: usize,
    /// The response, if the request succeeded.
    pub response: Option<InferenceResponse>,
    /// Error message, if the request failed.
    pub error: Option<String>,
}

/// Progress snapshot for a batch operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchProgress {
    /// Batch ID.
    pub id: String,
    /// Total requests in the batch.
    pub total: usize,
    /// Number of completed requests (success + failure).
    pub completed: usize,
    /// Number of failed requests.
    pub failed: usize,
    /// Current status.
    pub status: BatchStatus,
    /// Individual results (populated as they complete).
    pub results: Vec<BatchItemResult>,
}

/// Internal state for a running batch.
struct BatchState {
    progress: tokio::sync::Mutex<BatchProgress>,
    cancel: CancellationToken,
}

/// Batch inference manager with concurrency control and progress tracking.
pub struct BatchManager {
    /// Maximum concurrent inference requests across all batches.
    semaphore: Arc<Semaphore>,
    /// Active batch operations keyed by batch ID.
    batches: DashMap<String, Arc<BatchState>>,
}

impl BatchManager {
    /// Create a new batch manager with the given concurrency limit.
    #[must_use]
    pub fn new(max_concurrent: usize) -> Self {
        Self {
            semaphore: Arc::new(Semaphore::new(max_concurrent)),
            batches: DashMap::new(),
        }
    }

    /// Submit a batch of inference requests for execution.
    ///
    /// Returns the batch ID for tracking progress. The batch executes
    /// asynchronously — use [`get_progress`] to check status.
    pub fn submit<F, Fut>(
        &self,
        batch_id: String,
        requests: Vec<InferenceRequest>,
        infer_fn: F,
    ) -> String
    where
        F: Fn(InferenceRequest) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = anyhow::Result<InferenceResponse>> + Send + 'static,
    {
        let total = requests.len();
        let progress = BatchProgress {
            id: batch_id.clone(),
            total,
            completed: 0,
            failed: 0,
            status: BatchStatus::Running,
            results: (0..total)
                .map(|i| BatchItemResult {
                    index: i,
                    response: None,
                    error: None,
                })
                .collect(),
        };

        let state = Arc::new(BatchState {
            progress: tokio::sync::Mutex::new(progress),
            cancel: CancellationToken::new(),
        });

        self.batches.insert(batch_id.clone(), state.clone());
        let semaphore = self.semaphore.clone();
        let infer_fn = Arc::new(infer_fn);

        tokio::spawn(async move {
            let mut handles = Vec::with_capacity(total);

            for (index, request) in requests.into_iter().enumerate() {
                let sem = semaphore.clone();
                let st = state.clone();
                let f = infer_fn.clone();

                let handle = tokio::spawn(async move {
                    // Check cancellation before acquiring semaphore
                    if st.cancel.is_cancelled() {
                        return;
                    }

                    let _permit = match sem.acquire().await {
                        Ok(p) => p,
                        Err(_) => return, // semaphore closed
                    };

                    if st.cancel.is_cancelled() {
                        return;
                    }

                    let result = f(request).await;
                    let mut prog = st.progress.lock().await;

                    match result {
                        Ok(response) => {
                            prog.results[index].response = Some(response);
                        }
                        Err(e) => {
                            prog.results[index].error = Some(e.to_string());
                            prog.failed += 1;
                        }
                    }
                    prog.completed += 1;
                });

                handles.push(handle);
            }

            // Wait for all tasks
            for handle in handles {
                let _ = handle.await;
            }

            // Mark batch as completed
            let mut prog = state.progress.lock().await;
            if state.cancel.is_cancelled() {
                prog.status = BatchStatus::Cancelled;
            } else {
                prog.status = BatchStatus::Completed;
            }
        });

        batch_id
    }

    /// Get the current progress of a batch.
    pub async fn get_progress(&self, batch_id: &str) -> Option<BatchProgress> {
        let state = self.batches.get(batch_id)?;
        let prog = state.progress.lock().await;
        Some(prog.clone())
    }

    /// Cancel a running batch.
    pub fn cancel(&self, batch_id: &str) -> bool {
        if let Some(state) = self.batches.get(batch_id) {
            state.cancel.cancel();
            true
        } else {
            false
        }
    }

    /// Remove a completed batch from tracking.
    pub fn remove(&self, batch_id: &str) -> bool {
        self.batches.remove(batch_id).is_some()
    }

    /// Number of active batches.
    #[must_use]
    pub fn active_count(&self) -> usize {
        self.batches.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::TokenUsage;

    fn make_request(model: &str) -> InferenceRequest {
        InferenceRequest {
            model: model.into(),
            prompt: "test".into(),
            ..Default::default()
        }
    }

    fn make_response(model: &str) -> InferenceResponse {
        InferenceResponse {
            text: "response".into(),
            model: model.into(),
            usage: TokenUsage::default(),
            tool_calls: Vec::new(),
            provider: "test".into(),
            latency_ms: 1,
        }
    }

    #[test]
    fn batch_manager_creation() {
        let mgr = BatchManager::new(10);
        assert_eq!(mgr.active_count(), 0);
    }

    #[tokio::test]
    async fn batch_submit_and_complete() {
        let mgr = BatchManager::new(4);
        let requests = vec![make_request("model1"), make_request("model2")];

        let batch_id = mgr.submit("batch-1".into(), requests, |req| async move {
            Ok(make_response(&req.model))
        });

        // Wait for completion
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let progress = mgr.get_progress(&batch_id).await.unwrap();
        assert_eq!(progress.total, 2);
        assert_eq!(progress.completed, 2);
        assert_eq!(progress.failed, 0);
        assert_eq!(progress.status, BatchStatus::Completed);
    }

    #[tokio::test]
    async fn batch_with_failures() {
        let mgr = BatchManager::new(4);
        let requests = vec![make_request("ok"), make_request("fail")];

        mgr.submit("batch-2".into(), requests, |req| async move {
            if req.model == "fail" {
                Err(anyhow::anyhow!("simulated failure"))
            } else {
                Ok(make_response(&req.model))
            }
        });

        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let progress = mgr.get_progress("batch-2").await.unwrap();
        assert_eq!(progress.completed, 2);
        assert_eq!(progress.failed, 1);
    }

    #[tokio::test]
    async fn batch_cancel() {
        let mgr = BatchManager::new(1); // only 1 concurrent
        let requests = vec![make_request("a"), make_request("b"), make_request("c")];

        mgr.submit("batch-3".into(), requests, |_req| async {
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            Ok(make_response("x"))
        });

        // Cancel quickly
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        assert!(mgr.cancel("batch-3"));

        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        let progress = mgr.get_progress("batch-3").await.unwrap();
        assert_eq!(progress.status, BatchStatus::Cancelled);
    }

    #[test]
    fn batch_remove() {
        let mgr = BatchManager::new(4);
        assert!(!mgr.remove("nonexistent"));
    }

    #[tokio::test]
    async fn batch_nonexistent_progress() {
        let mgr = BatchManager::new(4);
        assert!(mgr.get_progress("nope").await.is_none());
    }
}
