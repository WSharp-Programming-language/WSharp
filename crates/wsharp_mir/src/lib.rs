//! Mid-level Intermediate Representation for the W# programming language.
//!
//! The MIR is a control-flow graph representation suitable for optimization
//! and code generation. It uses SSA-like form with explicit places and
//! basic block terminators.
//!
//! Key features:
//! - Control flow graph with basic blocks
//! - Explicit control flow via terminators
//! - Places for memory locations (locals, fields, indices)
//! - Operands for values (copy, move, constant)
//! - Rvalues for computations

pub mod build;
pub mod cfg;
pub mod coroutine;
pub mod mir;
pub mod pretty;

pub use build::*;
pub use cfg::*;
pub use coroutine::*;
pub use mir::*;
pub use pretty::*;
