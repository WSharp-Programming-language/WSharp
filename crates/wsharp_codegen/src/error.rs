//! Error types for code generation.

use thiserror::Error;

/// A code generation error.
#[derive(Debug, Error)]
pub enum CodegenError {
    #[error("unsupported type: {0}")]
    UnsupportedType(String),

    #[error("unsupported operation: {0}")]
    UnsupportedOperation(String),

    #[error("undefined function: {0}")]
    UndefinedFunction(String),

    #[error("LLVM error: {0}")]
    LlvmError(String),

    #[error("builder error: {0}")]
    BuilderError(#[from] inkwell::builder::BuilderError),
}

/// Result type for code generation.
pub type CodegenResult<T> = Result<T, CodegenError>;
