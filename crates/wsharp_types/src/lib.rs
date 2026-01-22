//! Type system and dispatch resolution for the W# programming language.
//!
//! This crate provides:
//! - Type definitions for W#'s type system
//! - Type inference with Hindley-Milner unification
//! - Multiple dispatch resolution

pub mod types;
pub mod dispatch;
pub mod http_status;
pub mod inference;

pub use types::*;
pub use dispatch::*;
pub use http_status::*;
pub use inference::*;
