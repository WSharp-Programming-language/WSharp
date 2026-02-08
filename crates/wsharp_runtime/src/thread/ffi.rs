//! C FFI surface for the thread GC system.
//!
//! All functions use the `wsharp_` prefix to match existing runtime conventions
//! (see `executor.rs` for the pattern).

use std::os::raw::{c_int, c_void};
use std::sync::Arc;

use super::gc::ThreadGC;
use super::handle::ThreadHandle;
use super::mutex::{GcMutex, ThreadState};

// ===========================================================================
// Thread management
// ===========================================================================

/// Spawn a new GC-managed thread that calls `fn_ptr`.
///
/// Returns a raw pointer to the `ThreadHandle` (caller owns the `Arc` ref).
/// Returns a sentinel handle (id 0, already finished) on null input.
///
/// # Safety
/// `fn_ptr` must be a valid `extern "C" fn()` pointer, or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn wsharp_thread_spawn_gc(fn_ptr: *const c_void) -> *mut ThreadHandle {
    let handle_arc = ThreadHandle::spawn(fn_ptr);
    Arc::into_raw(handle_arc) as *mut ThreadHandle
}

/// Join (block on) a thread and consume the handle.
///
/// This decrements the `Arc` reference count after joining.
///
/// # Safety
/// `ptr` must be a valid pointer returned by `wsharp_thread_spawn_gc`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn wsharp_thread_join(ptr: *mut ThreadHandle) {
    if ptr.is_null() {
        tracing::warn!("wsharp_thread_join: null pointer");
        return;
    }

    let arc = unsafe { Arc::from_raw(ptr) };
    arc.join();
    // Arc drops here, decrementing reference count.
}

/// Poll whether a thread has finished (non-blocking).
///
/// Returns 1 if finished, 0 otherwise (or on null input).
///
/// # Safety
/// `ptr` must be a valid pointer returned by `wsharp_thread_spawn_gc`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn wsharp_thread_poll(ptr: *mut ThreadHandle) -> c_int {
    if ptr.is_null() {
        return 0;
    }
    let handle = unsafe { &*ptr };
    handle.is_finished() as c_int
}

/// Release a thread handle without joining.
///
/// Decrements the `Arc` reference count. If this was the last reference, the
/// GC will eventually collect the thread.
///
/// # Safety
/// `ptr` must be a valid pointer returned by `wsharp_thread_spawn_gc`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn wsharp_thread_release(ptr: *mut ThreadHandle) {
    if ptr.is_null() {
        tracing::warn!("wsharp_thread_release: null pointer");
        return;
    }

    let _arc = unsafe { Arc::from_raw(ptr) };
    // Arc drops here, decrementing reference count.
}

/// Join all remaining GC-managed threads. Typically called at program exit.
#[unsafe(no_mangle)]
pub extern "C" fn wsharp_thread_join_all() {
    ThreadGC::join_all();
}

// ===========================================================================
// Mutex operations
// ===========================================================================

/// Create a new GC-tracked mutex with an initial `c_int` value.
///
/// Returns an opaque pointer. The caller must use `wsharp_mutex_destroy` to
/// free it (or let the GC collect it).
#[unsafe(no_mangle)]
pub extern "C" fn wsharp_mutex_new(initial: c_int) -> *mut c_void {
    let mtx = GcMutex::new(initial);
    Arc::into_raw(mtx) as *mut c_void
}

/// Acquire the mutex lock with race detection.
///
/// # Safety
/// `ptr` must be a valid pointer returned by `wsharp_mutex_new`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn wsharp_mutex_lock(ptr: *mut c_void, thread_id: c_int) {
    if ptr.is_null() {
        tracing::warn!("wsharp_mutex_lock: null pointer");
        return;
    }
    let mtx = unsafe { &*(ptr as *const GcMutex<c_int>) };
    match mtx.lock(thread_id as u64) {
        Ok(_guard) => {
            // Guard dropped immediately — in a real usage the locked
            // section would be managed by the compiled code.
        }
        Err(_) => {
            tracing::error!(mutex_id = mtx.id(), "mutex lock failed (poisoned)");
        }
    }
}

/// Manually unlock a mutex (clears owner tracking).
///
/// # Safety
/// `ptr` must be a valid pointer returned by `wsharp_mutex_new`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn wsharp_mutex_unlock(ptr: *mut c_void) {
    if ptr.is_null() {
        tracing::warn!("wsharp_mutex_unlock: null pointer");
        return;
    }
    let mtx = unsafe { &*(ptr as *const GcMutex<c_int>) };
    mtx.unlock();
}

/// Destroy a mutex, consuming the `Arc` reference.
///
/// # Safety
/// `ptr` must be a valid pointer returned by `wsharp_mutex_new`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn wsharp_mutex_destroy(ptr: *mut c_void) {
    if ptr.is_null() {
        tracing::warn!("wsharp_mutex_destroy: null pointer");
        return;
    }
    let _arc = unsafe { Arc::from_raw(ptr as *const GcMutex<c_int>) };
    // Arc drops here.
}

// ===========================================================================
// Thread state operations
// ===========================================================================

/// Create a new thread-safe shared state with an initial `c_int` value.
#[unsafe(no_mangle)]
pub extern "C" fn wsharp_thread_state_new(initial: c_int) -> *mut c_void {
    let state = ThreadState::new(initial);
    Box::into_raw(Box::new(state)) as *mut c_void
}

/// Read the current value of a thread state.
///
/// # Safety
/// `ptr` must be a valid pointer returned by `wsharp_thread_state_new`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn wsharp_thread_state_get(ptr: *mut c_void) -> c_int {
    if ptr.is_null() {
        tracing::warn!("wsharp_thread_state_get: null pointer");
        return 0;
    }
    let state = unsafe { &*(ptr as *mut ThreadState<c_int>) };
    state.get()
}

/// Set the value of a thread state.
///
/// # Safety
/// `ptr` must be a valid pointer returned by `wsharp_thread_state_new`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn wsharp_thread_state_set(ptr: *mut c_void, val: c_int) {
    if ptr.is_null() {
        tracing::warn!("wsharp_thread_state_set: null pointer");
        return;
    }
    let state = unsafe { &*(ptr as *mut ThreadState<c_int>) };
    state.set(val);
}

// ===========================================================================
// GC daemon control
// ===========================================================================

/// Start the background GC daemon thread.
#[unsafe(no_mangle)]
pub extern "C" fn wsharp_gc_daemon_start() {
    ThreadGC::start_daemon();
}

/// Request graceful shutdown of the GC daemon.
#[unsafe(no_mangle)]
pub extern "C" fn wsharp_gc_daemon_stop() {
    ThreadGC::stop_daemon();
}
