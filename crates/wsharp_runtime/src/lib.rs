//! Runtime library for the W# programming language.
//!
//! This crate provides:
//! - Reference counting with cycle detection
//! - Async runtime with coroutine support
//! - GC-managed thread system with mutex tracking
//! - Prototype object system

pub mod cycle;
pub mod executor;
pub mod rc;
pub mod thread;

pub use cycle::{collect_cycles, possible_root, CycleCollector, Trace};
pub use executor::{
    Context, CoroutineHandle, Executor, Pollable, PollResult, Ready, TaskId, Waker, YieldOnce,
};
pub use rc::*;
pub use thread::{GcMutex, ThreadGC, ThreadHandle, ThreadState};
