//! Thread garbage collector — global singleton with mark-and-sweep collection.

use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, LazyLock, Mutex, Weak,
    },
    thread,
    time::Duration,
};

use super::handle::{ThreadHandle, THREADS_EVER_SPAWNED};
use super::mutex::MutexGcHook;
use super::GC_DAEMON_INTERVAL_SECS;

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
// Daemon shutdown flag
// ---------------------------------------------------------------------------

static DAEMON_SHUTDOWN: AtomicBool = AtomicBool::new(false);

// ---------------------------------------------------------------------------
// ThreadGC
// ---------------------------------------------------------------------------

/// Global garbage collector for threads and mutexes.
///
/// Tracks all spawned threads and GC-aware mutexes via `Weak` references.
/// Periodically (or on demand) sweeps finished threads and cleans up mutexes
/// whose owning threads have died.
pub struct ThreadGC {
    threads: Mutex<HashMap<u64, Weak<ThreadHandle>>>,
    mutexes: Mutex<HashMap<u64, Weak<dyn MutexGcHook>>>,
}

/// The global singleton instance.
static INSTANCE: LazyLock<ThreadGC> = LazyLock::new(|| ThreadGC {
    threads: Mutex::new(HashMap::new()),
    mutexes: Mutex::new(HashMap::new()),
});

impl ThreadGC {
    /// Returns the global `ThreadGC` singleton.
    pub(crate) fn global() -> &'static ThreadGC {
        &INSTANCE
    }

    /// Provides access to the threads map (used by `GcMutex::owner_dead`).
    pub(crate) fn threads(&self) -> &Mutex<HashMap<u64, Weak<ThreadHandle>>> {
        &self.threads
    }

    // -----------------------------------------------------------------------
    // Registration
    // -----------------------------------------------------------------------

    /// Register a thread handle with the GC.
    pub(crate) fn register(handle: Arc<ThreadHandle>) {
        let id = handle.id;
        Self::global()
            .threads
            .lock()
            .unwrap()
            .insert(id, Arc::downgrade(&handle));
        thread_trace!(thread_id = id, "registered thread handle");
    }

    /// Register a type-erased mutex with the GC.
    pub(crate) fn register_mutex(id: u64, hook: Arc<dyn MutexGcHook>) {
        Self::global()
            .mutexes
            .lock()
            .unwrap()
            .insert(id, Arc::downgrade(&hook));
        thread_trace!(mutex_id = id, "registered mutex");
    }

    // -----------------------------------------------------------------------
    // Collection
    // -----------------------------------------------------------------------

    /// Perform an immediate mark-and-sweep collection.
    ///
    /// - Removes finished threads with no external `Arc` references.
    /// - Removes expired mutex `Weak` references.
    /// - Calls `force_unlock_if_dead()` on live mutexes whose owners have died.
    #[allow(unused_variables)]
    pub(crate) fn collect_now() {
        let gc = Self::global();
        let mut threads = gc.threads.lock().unwrap();
        let mut mutexes = gc.mutexes.lock().unwrap();

        let mut thread_collected = 0u32;

        threads.retain(|id, weak_handle| {
            if let Some(handle) = weak_handle.upgrade() {
                if handle.is_finished() && Arc::strong_count(&handle) == 1 {
                    thread_trace!(thread_id = id, "collecting finished thread");
                    thread_collected += 1;
                    false
                } else {
                    true
                }
            } else {
                thread_trace!(thread_id = id, "thread fully dropped");
                false
            }
        });

        let mut mutex_cleaned = 0u32;

        mutexes.retain(|id, weak_hook| {
            if let Some(hook) = weak_hook.upgrade() {
                hook.force_unlock_if_dead();
                true
            } else {
                thread_trace!(mutex_id = id, "mutex weak reference expired");
                mutex_cleaned += 1;
                false
            }
        });

        if thread_collected > 0 || mutex_cleaned > 0 {
            tracing::debug!(
                threads = thread_collected,
                mutexes = mutex_cleaned,
                "GC collected"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Join all
    // -----------------------------------------------------------------------

    /// Join all remaining threads. Typically called at program exit.
    ///
    /// Uses a snapshot-then-join pattern: takes a snapshot of all thread weak
    /// references, releases the GC lock, then joins each thread outside the
    /// lock to avoid blocking while holding it.
    pub(crate) fn join_all() {
        if THREADS_EVER_SPAWNED.load(Ordering::Relaxed) == 0 {
            thread_trace!("no threads ever spawned, skipping join_all");
            return;
        }

        let gc = Self::global();
        let mut threads_lock = gc.threads.lock().unwrap();

        if threads_lock.is_empty() {
            thread_trace!("no threads to join");
            return;
        }

        // Snapshot and clear under lock.
        let snapshot: Vec<(u64, Weak<ThreadHandle>)> = threads_lock
            .iter()
            .map(|(id, w)| (*id, w.clone()))
            .collect();

        threads_lock.clear();
        drop(threads_lock);

        tracing::debug!(count = snapshot.len(), "joining all remaining GC threads");

        let mut joined = 0usize;
        for (id, weak_handle) in snapshot {
            if let Some(handle_arc) = weak_handle.upgrade() {
                let mut join_guard = handle_arc.join_handle.lock().unwrap();
                if let Some(join_handle) = join_guard.take() {
                    thread_trace!(thread_id = id, "joining thread");
                    match join_handle.join() {
                        Ok(_) => {
                            thread_trace!(thread_id = id, "joined thread");
                            joined += 1;
                        }
                        Err(_) => {
                            tracing::error!(thread_id = id, "thread panicked during join_all");
                        }
                    }
                } else {
                    thread_trace!(thread_id = id, "thread already finished or detached");
                }
            } else {
                thread_trace!(thread_id = id, "thread already collected (Weak expired)");
            }
        }

        tracing::debug!(joined, "all GC threads joined");
    }

    // -----------------------------------------------------------------------
    // Daemon
    // -----------------------------------------------------------------------

    /// Start a background GC daemon that periodically collects threads and
    /// checks mutexes for dead owners.
    pub(crate) fn start_daemon() {
        DAEMON_SHUTDOWN.store(false, Ordering::Release);

        thread::spawn(|| loop {
            if DAEMON_SHUTDOWN.load(Ordering::Acquire) {
                thread_trace!("GC daemon shutting down");
                break;
            }

            let gc = Self::global();

            // Collect finished threads.
            {
                let mut threads = gc.threads.lock().unwrap();
                let mut to_remove = Vec::new();

                for (id, weak_handle) in threads.iter() {
                    if let Some(mut handle_arc) = weak_handle.upgrade()
                        && handle_arc.is_finished()
                    {
                        if let Some(handle_mut) = Arc::get_mut(&mut handle_arc) {
                            let mut guard = handle_mut.join_handle.lock().unwrap();
                            if let Some(join_handle) = guard.take() {
                                thread_trace!(thread_id = id, "daemon auto-joining finished thread");
                                let _ = join_handle.join();
                            }
                        }
                        to_remove.push(*id);
                    }
                    // If upgrade fails, the Weak expired — will be cleaned below.
                }

                let collected = to_remove.len();
                for id in to_remove {
                    threads.remove(&id);
                }

                if collected > 0 {
                    tracing::debug!(collected, "GC daemon collected threads");
                }
            }

            // Check mutexes for dead owners.
            {
                let mutexes = gc.mutexes.lock().unwrap();
                for (_, weak_hook) in mutexes.iter() {
                    if let Some(hook) = weak_hook.upgrade() {
                        hook.force_unlock_if_dead();
                    }
                }
            }

            thread::sleep(Duration::from_secs(GC_DAEMON_INTERVAL_SECS));
        });
    }

    /// Request graceful shutdown of the GC daemon.
    pub(crate) fn stop_daemon() {
        tracing::debug!("requesting GC daemon shutdown");
        DAEMON_SHUTDOWN.store(true, Ordering::Release);
    }
}
