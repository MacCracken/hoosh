//! Priority request queue — queue inference requests when providers are busy.

use serde::{Deserialize, Serialize};

use majra::queue::{Priority, PriorityQueue, QueueItem};

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
pub struct InferenceQueue {
    inner: std::sync::Mutex<PriorityQueue<QueuedRequest>>,
    notify: tokio::sync::Notify,
}

impl InferenceQueue {
    pub fn new() -> Self {
        Self {
            inner: std::sync::Mutex::new(PriorityQueue::new()),
            notify: tokio::sync::Notify::new(),
        }
    }

    /// Enqueue a request with the given priority.
    pub fn enqueue(&self, request: QueuedRequest, priority: Priority) -> TaskId {
        let item = QueueItem::new(priority, request);
        let id = item.id;
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .enqueue(item);
        self.notify.notify_one();
        id
    }

    /// Dequeue the highest-priority request.
    pub fn dequeue(&self) -> Option<QueueItem<QueuedRequest>> {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .dequeue()
    }

    /// Number of queued requests.
    pub fn len(&self) -> usize {
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_empty()
    }

    /// Wait for a new item to be available (async).
    pub async fn wait_for_item(&self) {
        self.notify.notified().await;
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

    #[test]
    fn enqueue_dequeue_ordering() {
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

        queue.enqueue(normal_req, Priority::Normal);
        queue.enqueue(critical_req, Priority::Critical);

        // Critical should come out first
        let first = queue.dequeue().unwrap();
        assert_eq!(first.payload.request_id, "req-2");
        assert_eq!(first.payload.model, "gpt-4o");

        let second = queue.dequeue().unwrap();
        assert_eq!(second.payload.request_id, "req-1");
        assert_eq!(second.payload.model, "llama3");
    }

    #[test]
    fn empty_queue_returns_none() {
        let queue = InferenceQueue::new();
        assert!(queue.dequeue().is_none());
        assert!(queue.is_empty());
    }

    #[test]
    fn len_tracking() {
        let queue = InferenceQueue::new();
        assert_eq!(queue.len(), 0);
        assert!(queue.is_empty());

        let req = QueuedRequest {
            request: InferenceRequest::default(),
            model: "test".into(),
            pool: "default".into(),
            request_id: "req-1".into(),
        };
        queue.enqueue(req.clone(), Priority::Normal);
        assert_eq!(queue.len(), 1);
        assert!(!queue.is_empty());

        queue.enqueue(req, Priority::High);
        assert_eq!(queue.len(), 2);

        queue.dequeue();
        assert_eq!(queue.len(), 1);

        queue.dequeue();
        assert_eq!(queue.len(), 0);
        assert!(queue.is_empty());
    }
}
