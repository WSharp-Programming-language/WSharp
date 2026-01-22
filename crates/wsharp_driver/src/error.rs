//! Error types for the compilation driver.

use thiserror::Error;

/// Errors that can occur during compilation.
#[derive(Debug, Error)]
pub enum CompileError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Lexer error: {0}")]
    Lexer(String),

    #[error("Parser error: {0}")]
    Parser(String),

    #[error("Type error: {0}")]
    Type(String),

    #[error("Lowering error: {0}")]
    Lowering(String),

    #[error("Code generation error: {0}")]
    Codegen(#[from] wsharp_codegen::CodegenError),

    #[error("Multiple errors occurred:\n{0}")]
    Multiple(String),
}

/// Result type for compilation operations.
pub type CompileResult<T> = Result<T, CompileError>;
