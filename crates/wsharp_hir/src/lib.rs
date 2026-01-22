//! High-level Intermediate Representation for the W# programming language.
//!
//! The HIR is a desugared representation of the AST that's easier to analyze
//! and transform. Key simplifications:
//!
//! - All loops are converted to `loop` with `break`
//! - Pattern matching is simplified
//! - Implicit returns are made explicit
//! - Method calls are converted to function calls
//! - Operator overloading is desugared to method calls

mod hir;
mod lower;
mod resolve;
mod typecheck;

pub use hir::*;
pub use lower::*;
pub use resolve::*;
pub use typecheck::*;
