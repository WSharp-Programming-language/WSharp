//! LLVM code generation for the W# programming language.
//!
//! This crate handles the translation from MIR to LLVM IR and supports
//! both AOT (ahead-of-time) and JIT (just-in-time) compilation.
//!
//! # Architecture
//!
//! The code generator works in the following stages:
//!
//! 1. **Type Translation**: W# types are converted to LLVM types
//! 2. **Function Declaration**: All functions are declared in the module
//! 3. **Body Generation**: MIR basic blocks are converted to LLVM IR
//! 4. **Optimization**: LLVM optimization passes are run
//! 5. **Code Emission**: Output as object file, assembly, or JIT execution
//!
//! # Example
//!
//! ```ignore
//! use wsharp_codegen::{CodeGenerator, AotCompiler};
//! use inkwell::context::Context;
//!
//! let context = Context::create();
//! let mut aot = AotCompiler::new(&context)?;
//! aot.compile_to_object(&mir_module, Path::new("output.o"))?;
//! ```

pub mod aot;
pub mod codegen;
pub mod error;
#[cfg(feature = "jit")]
pub mod jit;

pub use aot::*;
pub use codegen::*;
pub use error::*;
#[cfg(feature = "jit")]
pub use jit::*;
