//! Priority request queue — queue inference requests when providers are busy.
//!
//! Uses majra's [`ConcurrentPriorityQueue`] for async-native, tokio-aware
//! locking instead of `std::sync::Mutex`.

use serde::{Deserialize, Serialize};

use majra::queue::{ConcurrentPriorityQueue, Priority, QueueItem};

use crate::inference::InferenceRequest;

/// Re-export TaskId for callers.
pub use majra::queue::TaskId;

/// A queued inference request with routing metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueuedRequest {
    /// The inference request.
    pub request: InferenceRequest,
    /// Model being requested.
    pub model: String,
    /// Token budget pool name.
    pub pool: String,
    /// Request ID for tracking.
    pub request_id: String,
}

/// Inference request queue with priority tiers.
///
/// Backed by majra's [`ConcurrentPriorityQueue`] — async-aware tokio mutex
/// with built-in notify for `dequeue_wait`.
pub struct InferenceQueue {
    inner: ConcurrentPriorityQueue<QueuedRequest>,
}

impl InferenceQueue {
    pub fn new() -> Self {
        Self {
            inner: ConcurrentPriorityQueue::new(),
        }
    }

    /// Enqueue a request with the given priority.
    pub async fn enqueue(&self, request: QueuedRequest, priority: Priority) -> TaskId {
        let item = QueueItem::new(priority, request);
        let id = item.id;
        self.inner.enqueue(item).await;
        id
    }

    /// Dequeue the highest-priority request.
    pub async fn dequeue(&self) -> Option<QueueItem<QueuedRequest>> {
        self.inner.dequeue().await
    }

    /// Number of queued requests.
    pub async fn len(&self) -> usize {
        self.inner.len().await
    }

    pub async fn is_empty(&self) -> bool {
        self.inner.is_empty().await
    }

    /// Wait for a new item to be available, then dequeue it (async).
    pub async fn dequeue_wait(&self) -> QueueItem<QueuedRequest> {
        self.inner.dequeue_wait().await
    }
}

impl Default for InferenceQueue {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn enqueue_dequeue_ordering() {
        let queue = InferenceQueue::new();

        let normal_req = QueuedRequest {
            request: InferenceRequest::default(),
            model: "llama3".into(),
            pool: "default".into(),
            request_id: "req-1".into(),
        };
        let critical_req = QueuedRequest {
            request: InferenceRequest::default(),
            model: "gpt-4o".into(),
            pool: "default".into(),
            request_id: "req-2".into(),
        };

        queue.enqueue(normal_req, Priority::Normal).await;
        queue.enqueue(critical_req, Priority::Critical).await;

        // Critical should come out first
        let first = queue.dequeue().await.unwrap();
        assert_eq!(first.payload.request_id, "req-2");
        assert_eq!(first.payload.model, "gpt-4o");

        let second = queue.dequeue().await.unwrap();
        assert_eq!(second.payload.request_id, "req-1");
        assert_eq!(second.payload.model, "llama3");
    }

    #[tokio::test]
    async fn empty_queue_returns_none() {
        let queue = InferenceQueue::new();
        assert!(queue.dequeue().await.is_none());
        assert!(queue.is_empty().await);
    }

    #[tokio::test]
    async fn len_tracking() {
        let queue = InferenceQueue::new();
        assert_eq!(queue.len().await, 0);
        assert!(queue.is_empty().await);

        let req = QueuedRequest {
            request: InferenceRequest::default(),
            model: "test".into(),
            pool: "default".into(),
            request_id: "req-1".into(),
        };
        queue.enqueue(req.clone(), Priority::Normal).await;
        assert_eq!(queue.len().await, 1);
        assert!(!queue.is_empty().await);

        queue.enqueue(req, Priority::High).await;
        assert_eq!(queue.len().await, 2);

        queue.dequeue().await;
        assert_eq!(queue.len().await, 1);

        queue.dequeue().await;
        assert_eq!(queue.len().await, 0);
        assert!(queue.is_empty().await);
    }

    #[tokio::test]
    async fn all_five_priority_tiers_ordering() {
        let queue = InferenceQueue::new();

        let make_req = |id: &str| QueuedRequest {
            request: InferenceRequest::default(),
            model: "test".into(),
            pool: "default".into(),
            request_id: id.into(),
        };

        // Enqueue in arbitrary order
        queue.enqueue(make_req("normal"), Priority::Normal).await;
        queue
            .enqueue(make_req("background"), Priority::Background)
            .await;
        queue
            .enqueue(make_req("critical"), Priority::Critical)
            .await;
        queue.enqueue(make_req("low"), Priority::Low).await;
        queue.enqueue(make_req("high"), Priority::High).await;

        assert_eq!(queue.len().await, 5);

        // Should dequeue in priority order: Critical > High > Normal > Low > Background
        assert_eq!(
            queue.dequeue().await.unwrap().payload.request_id,
            "critical"
        );
        assert_eq!(queue.dequeue().await.unwrap().payload.request_id, "high");
        assert_eq!(queue.dequeue().await.unwrap().payload.request_id, "normal");
        assert_eq!(queue.dequeue().await.unwrap().payload.request_id, "low");
        assert_eq!(
            queue.dequeue().await.unwrap().payload.request_id,
            "background"
        );

        assert!(queue.is_empty().await);
        assert!(queue.dequeue().await.is_none());
    }

    #[tokio::test]
    async fn dequeue_returns_none_on_empty() {
        let queue = InferenceQueue::new();
        assert!(queue.dequeue().await.is_none());
        assert!(queue.dequeue().await.is_none()); // multiple calls
    }

    #[tokio::test]
    async fn is_empty_reflects_state() {
        let queue = InferenceQueue::new();
        assert!(queue.is_empty().await);

        let req = QueuedRequest {
            request: InferenceRequest::default(),
            model: "m".into(),
            pool: "p".into(),
            request_id: "r".into(),
        };
        queue.enqueue(req, Priority::Normal).await;
        assert!(!queue.is_empty().await);

        queue.dequeue().await;
        assert!(queue.is_empty().await);
    }

    #[tokio::test]
    async fn same_priority_fifo_order() {
        let queue = InferenceQueue::new();

        let make_req = |id: &str| QueuedRequest {
            request: InferenceRequest::default(),
            model: "test".into(),
            pool: "default".into(),
            request_id: id.into(),
        };

        queue.enqueue(make_req("first"), Priority::Normal).await;
        queue.enqueue(make_req("second"), Priority::Normal).await;
        queue.enqueue(make_req("third"), Priority::Normal).await;

        assert_eq!(queue.dequeue().await.unwrap().payload.request_id, "first");
        assert_eq!(queue.dequeue().await.unwrap().payload.request_id, "second");
        assert_eq!(queue.dequeue().await.unwrap().payload.request_id, "third");
    }

    #[tokio::test]
    async fn enqueue_returns_unique_task_ids() {
        let queue = InferenceQueue::new();

        let req = QueuedRequest {
            request: InferenceRequest::default(),
            model: "test".into(),
            pool: "default".into(),
            request_id: "r".into(),
        };

        let id1 = queue.enqueue(req.clone(), Priority::Normal).await;
        let id2 = queue.enqueue(req.clone(), Priority::Normal).await;
        let id3 = queue.enqueue(req, Priority::High).await;

        assert_ne!(id1, id2);
        assert_ne!(id2, id3);
        assert_ne!(id1, id3);
    }

    #[tokio::test]
    async fn default_creates_empty_queue() {
        let queue = InferenceQueue::default();
        assert!(queue.is_empty().await);
        assert_eq!(queue.len().await, 0);
        assert!(queue.dequeue().await.is_none());
    }

    #[tokio::test]
    async fn mixed_priority_interleaved() {
        let queue = InferenceQueue::new();

        let make_req = |id: &str| QueuedRequest {
            request: InferenceRequest::default(),
            model: "test".into(),
            pool: "default".into(),
            request_id: id.into(),
        };

        // Enqueue, dequeue, enqueue more
        queue.enqueue(make_req("low1"), Priority::Low).await;
        queue.enqueue(make_req("high1"), Priority::High).await;

        // High comes first
        assert_eq!(queue.dequeue().await.unwrap().payload.request_id, "high1");

        // Enqueue a Critical while Low is still pending
        queue.enqueue(make_req("crit1"), Priority::Critical).await;

        // Critical should come before low
        assert_eq!(queue.dequeue().await.unwrap().payload.request_id, "crit1");
        assert_eq!(queue.dequeue().await.unwrap().payload.request_id, "low1");
        assert!(queue.is_empty().await);
    }

    #[tokio::test]
    async fn dequeue_wait_wakes_on_enqueue() {
        let queue = std::sync::Arc::new(InferenceQueue::new());
        let q2 = queue.clone();

        let handle = tokio::spawn(async move { q2.dequeue_wait().await.payload.request_id });

        // Small delay to ensure the waiter is parked.
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;

        let req = QueuedRequest {
            request: InferenceRequest::default(),
            model: "test".into(),
            pool: "default".into(),
            request_id: "woke".into(),
        };
        queue.enqueue(req, Priority::Normal).await;

        let result = handle.await.unwrap();
        assert_eq!(result, "woke");
    }
}
