//! Compilation driver for the W# programming language.
//!
//! This crate orchestrates the compilation pipeline from source to executable:
//!
//! ```text
//! Source (.ws) → Lexer → Parser → AST → HIR → MIR → LLVM IR → Native/JIT
//! ```
//!
//! # Example
//!
//! ```ignore
//! use wsharp_driver::{Driver, CompileOptions};
//!
//! let driver = Driver::new();
//! let result = driver.compile_file("hello.ws", &CompileOptions::default())?;
//! ```

mod compile;
mod error;
mod session;

pub use compile::*;
pub use error::*;
pub use session::*;

// Re-export commonly used types from dependencies
pub use wsharp_ast::SourceFile;
pub use wsharp_hir::HirModule;
pub use wsharp_mir::MirModule;
