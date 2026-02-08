//! GC-aware mutex with race detection and shared thread state.

use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc, Mutex, MutexGuard,
};

use super::gc::ThreadGC;

// ---------------------------------------------------------------------------
// Conditional trace logging
// ---------------------------------------------------------------------------

macro_rules! thread_trace {
    ($($arg:tt)*) => {
        #[cfg(feature = "thread-debug")]
        tracing::trace!($($arg)*);
    };
}

// ---------------------------------------------------------------------------
// MutexGcHook — trait for type-erased GC cleanup
// ---------------------------------------------------------------------------

/// Trait allowing the GC to perform dead-owner cleanup on mutexes without
/// knowing their inner type `T`.
pub(crate) trait MutexGcHook: Send + Sync {
    /// Release the mutex if its owning thread is dead.
    fn force_unlock_if_dead(&self);

    /// The unique identifier of this mutex.
    #[allow(dead_code)]
    fn mutex_id(&self) -> u64;
}

// ---------------------------------------------------------------------------
// GcMutex<T>
// ---------------------------------------------------------------------------

/// A GC-tracked mutex with owner tracking and race detection.
///
/// Always constructed behind an `Arc` so the GC can hold `Weak` references.
#[derive(Debug)]
pub struct GcMutex<T: Send + 'static> {
    id: u64,
    data: Mutex<T>,
    owner_thread: Mutex<Option<u64>>,
    poisoned: AtomicBool,
    /// Atomic busy flag for lightweight race detection.
    busy: AtomicBool,
}

/// Global mutex ID counter.
static NEXT_MUTEX_ID: AtomicU64 = AtomicU64::new(1);

impl<T: Send + 'static> GcMutex<T> {
    /// Create a new GC-tracked mutex and register it with the thread GC.
    pub fn new(initial: T) -> Arc<Self> {
        let id = NEXT_MUTEX_ID.fetch_add(1, Ordering::Relaxed);

        let mutex = Arc::new(GcMutex {
            id,
            data: Mutex::new(initial),
            owner_thread: Mutex::new(None),
            poisoned: AtomicBool::new(false),
            busy: AtomicBool::new(false),
        });

        ThreadGC::register_mutex(id, mutex.clone() as Arc<dyn MutexGcHook>);
        thread_trace!(mutex_id = id, "registered mutex");

        mutex
    }

    /// Acquire the lock with race detection.
    ///
    /// The `busy` flag is used as a lightweight race detector: if another thread
    /// is already in the process of acquiring, we log a warning. The underlying
    /// `Mutex` still serialises access correctly regardless.
    pub fn lock(&self, thread_id: u64) -> std::sync::LockResult<MutexGuard<'_, T>> {
        // Acquire ordering synchronises with the Release in the success path below.
        if self
            .busy
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            tracing::warn!(mutex_id = self.id, "race detected on mutex (already busy)");
        }

        let guard = self.data.lock();

        match guard {
            Ok(g) => {
                *self.owner_thread.lock().unwrap() = Some(thread_id);
                self.busy.store(false, Ordering::Release);
                Ok(g)
            }
            Err(poisoned) => {
                self.poisoned.store(true, Ordering::SeqCst);
                self.busy.store(false, Ordering::Release);
                Err(poisoned)
            }
        }
    }

    /// Manually release the lock (clears owner tracking).
    pub fn unlock(&self) {
        *self.owner_thread.lock().unwrap() = None;
        thread_trace!(mutex_id = self.id, "mutex unlocked manually");
    }

    /// Returns the unique identifier of this mutex.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Check whether the owning thread is dead.
    ///
    /// Lock ordering: reads `owner_thread` first and releases it before
    /// querying `ThreadGC` to avoid deadlock.
    fn owner_dead(&self) -> bool {
        let owner_id = { *self.owner_thread.lock().unwrap() };

        if let Some(owner) = owner_id {
            let threads = ThreadGC::global().threads().lock().unwrap();
            if let Some(weak_handle) = threads.get(&owner) {
                if let Some(handle) = weak_handle.upgrade() {
                    return handle.is_finished();
                } else {
                    thread_trace!(thread_id = owner, "thread already collected (Weak expired)");
                    return true;
                }
            }
        }
        false
    }
}

impl<T: Send + 'static> MutexGcHook for GcMutex<T> {
    fn force_unlock_if_dead(&self) {
        if self.owner_dead() {
            tracing::debug!(mutex_id = self.id, "releasing mutex (owner thread dead)");
            *self.owner_thread.lock().unwrap() = None;
        }
    }

    fn mutex_id(&self) -> u64 {
        self.id
    }
}

// ---------------------------------------------------------------------------
// ThreadState<T>
// ---------------------------------------------------------------------------

/// Thread-safe shared state wrapper.
///
/// A simple `Arc<Mutex<T>>` wrapper providing `get`, `set`, and `update`
/// operations for cross-thread shared values.
#[derive(Clone)]
pub struct ThreadState<T: Send + Clone + 'static> {
    inner: Arc<Mutex<T>>,
}

impl<T: Send + Clone + 'static> ThreadState<T> {
    pub fn new(initial: T) -> Self {
        Self {
            inner: Arc::new(Mutex::new(initial)),
        }
    }

    pub fn get(&self) -> T {
        self.inner.lock().unwrap().clone()
    }

    pub fn set(&self, val: T) {
        *self.inner.lock().unwrap() = val;
    }

    pub fn update<F: FnOnce(&mut T)>(&self, f: F) {
        let mut data = self.inner.lock().unwrap();
        f(&mut data);
    }
}
