//! GC-managed thread handle with auto-join-on-drop semantics.

use std::{
    any::Any,
    cell::RefCell,
    mem,
    os::raw::c_void,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, Mutex,
    },
    thread,
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
// Global state
// ---------------------------------------------------------------------------

/// Counts total threads ever spawned — enables a fast-path in `join_all`.
pub(crate) static THREADS_EVER_SPAWNED: AtomicU64 = AtomicU64::new(0);

/// Global thread ID counter.
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

thread_local! {
    /// Per-thread ancestry stack for recursion prevention.
    static THREAD_ANCESTRY: RefCell<Vec<u64>> = const { RefCell::new(Vec::new()) };
}

// ---------------------------------------------------------------------------
// AncestryGuard — RAII cleanup for thread ancestry
// ---------------------------------------------------------------------------

/// Automatically removes a thread ID from the ancestry stack on drop,
/// ensuring cleanup even if the thread panics.
struct AncestryGuard(u64);

impl Drop for AncestryGuard {
    fn drop(&mut self) {
        THREAD_ANCESTRY.with(|ancestry| {
            let mut a = ancestry.borrow_mut();
            if let Some(pos) = a.iter().position(|x| *x == self.0) {
                a.remove(pos);
            }
        });
    }
}

// ---------------------------------------------------------------------------
// ThreadHandle
// ---------------------------------------------------------------------------

/// A GC-managed thread handle.
///
/// Threads are registered with [`ThreadGC`] on spawn and automatically joined
/// when the handle is dropped (if not already finished).
#[derive(Debug)]
pub struct ThreadHandle {
    pub(crate) id: u64,
    pub(crate) finished: Arc<AtomicBool>,
    #[allow(dead_code)]
    pub(crate) result: Arc<Mutex<Option<Box<dyn Any + Send + 'static>>>>,
    pub(crate) join_handle: Mutex<Option<thread::JoinHandle<()>>>,
    #[allow(dead_code)]
    pub(crate) ref_count: Arc<AtomicU64>,
}

impl ThreadHandle {
    /// Spawn a new GC-managed thread that calls `func_ptr`.
    ///
    /// The function pointer must have C calling convention `extern "C" fn()`.
    /// Returns an error sentinel (`id: 0`, `finished: true`) on null pointer
    /// or ID exhaustion.
    pub(crate) fn spawn(func_ptr: *const c_void) -> Arc<ThreadHandle> {
        if func_ptr.is_null() {
            tracing::error!("thread spawn called with null function pointer");
            ThreadGC::collect_now();
            return Self::error_sentinel(0);
        }

        let id = match NEXT_ID.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
            current.checked_add(1)
        }) {
            Ok(prev) => prev,
            Err(_) => {
                tracing::error!("thread ID overflow — maximum thread count reached");
                return Self::error_sentinel(0);
            }
        };

        // Recursion prevention: check if this thread ID is in the ancestry.
        let mut recursion_violation = false;
        THREAD_ANCESTRY.with(|ancestry| {
            let ancestors = ancestry.borrow();
            if ancestors.contains(&id) {
                recursion_violation = true;
            }
        });
        if recursion_violation {
            tracing::error!(thread_id = id, "recursion prevented: thread tried to spawn itself");
            return Self::error_sentinel(id);
        }

        THREAD_ANCESTRY.with(|ancestry| ancestry.borrow_mut().push(id));

        let finished = Arc::new(AtomicBool::new(false));
        let result = Arc::new(Mutex::new(None));
        let ref_count = Arc::new(AtomicU64::new(1));

        let func: extern "C" fn() = unsafe { mem::transmute(func_ptr) };
        let fin_clone = finished.clone();

        THREADS_EVER_SPAWNED.fetch_add(1, Ordering::Relaxed);

        thread_trace!(thread_id = id, "spawning GC-managed thread");

        let join_handle = Mutex::new(Some(thread::spawn(move || {
            let _ancestry_guard = AncestryGuard(id);

            let run_result = std::panic::catch_unwind(|| func());
            match run_result {
                Ok(_) => {
                    thread_trace!(thread_id = id, "thread finished normally");
                }
                Err(_) => {
                    tracing::error!(thread_id = id, "thread panicked");
                }
            }
            fin_clone.store(true, Ordering::SeqCst);
        })));

        let handle = Arc::new(ThreadHandle {
            id,
            finished,
            result,
            join_handle,
            ref_count,
        });

        ThreadGC::register(handle.clone());
        handle
    }

    /// Join the thread, blocking until it completes.
    pub fn join(&self) {
        let mut guard = self.join_handle.lock().unwrap();
        if let Some(handle) = guard.take() {
            thread_trace!(thread_id = self.id, "joining thread");

            match handle.join() {
                Ok(_) => {
                    thread_trace!(thread_id = self.id, "successfully joined thread");
                }
                Err(panic_payload) => {
                    tracing::error!(thread_id = self.id, ?panic_payload, "thread panicked on join");
                }
            }

            self.finished.store(true, Ordering::SeqCst);
        }
    }

    /// Check whether the thread has finished executing.
    pub fn is_finished(&self) -> bool {
        self.finished.load(Ordering::SeqCst)
    }

    /// Increment the external reference count.
    #[allow(dead_code)]
    pub(crate) fn add_ref(&self) {
        self.ref_count.fetch_add(1, Ordering::SeqCst);
    }

    /// Decrement the external reference count. Triggers GC collection when it
    /// reaches zero.
    #[allow(dead_code)]
    pub(crate) fn release(&self) {
        let mut current = self.ref_count.load(Ordering::SeqCst);
        loop {
            if current == 0 {
                tracing::warn!(thread_id = self.id, "attempted to release already-freed thread");
                return;
            }

            match self.ref_count.compare_exchange(
                current,
                current - 1,
                Ordering::SeqCst,
                Ordering::SeqCst,
            ) {
                Ok(old_value) => {
                    if old_value == 1 {
                        thread_trace!(thread_id = self.id, "thread handle released (0 refs)");
                        ThreadGC::collect_now();
                    }
                    return;
                }
                Err(actual) => {
                    current = actual;
                }
            }
        }
    }

    /// Create an error sentinel handle (finished, no join handle).
    fn error_sentinel(id: u64) -> Arc<ThreadHandle> {
        Arc::new(ThreadHandle {
            id,
            finished: Arc::new(AtomicBool::new(true)),
            result: Arc::new(Mutex::new(None)),
            join_handle: Mutex::new(None),
            ref_count: Arc::new(AtomicU64::new(0)),
        })
    }
}

impl Drop for ThreadHandle {
    fn drop(&mut self) {
        if !self.is_finished() {
            thread_trace!(thread_id = self.id, "auto-joining unfinished thread on drop");

            match self.join_handle.lock() {
                Ok(mut guard) => {
                    if let Some(handle) = guard.take() {
                        let _ = handle.join();
                    }
                    self.finished.store(true, Ordering::SeqCst);
                }
                Err(_) => {
                    tracing::error!(
                        thread_id = self.id,
                        "poisoned lock on thread handle during drop, cannot join"
                    );
                    self.finished.store(true, Ordering::SeqCst);
                }
            }
        }
    }
}
