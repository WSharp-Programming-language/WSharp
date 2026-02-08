//! GC-managed thread system for the W# programming language.
//!
//! Provides automatic thread lifecycle management, mutex tracking, and race detection.
//!
//! ## Modules
//!
//! - [`handle`]: Thread handle with auto-join-on-drop semantics
//! - [`gc`]: Global thread garbage collector (mark-and-sweep)
//! - [`mutex`]: GC-aware mutex with race detection and shared state
//! - [`ffi`]: C FFI surface for codegen integration

pub mod ffi;
pub mod gc;
pub mod handle;
pub mod mutex;

/// Interval in seconds between GC daemon collection sweeps.
pub(crate) const GC_DAEMON_INTERVAL_SECS: u64 = 3;

pub use gc::ThreadGC;
pub use handle::ThreadHandle;
pub use mutex::{GcMutex, ThreadState};
