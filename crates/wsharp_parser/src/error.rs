//! Parser error types.

use wsharp_lexer::{Span, TokenKind};
use thiserror::Error;

/// Result type for parser operations.
pub type ParseResult<T> = Result<T, ParseError>;

/// A parse error.
#[derive(Clone, Debug, Error)]
pub enum ParseError {
    #[error("unexpected token: expected {expected}, found {found}")]
    UnexpectedToken {
        expected: String,
        found: String,
        span: Span,
    },

    #[error("unexpected end of file")]
    UnexpectedEof { span: Span },

    #[error("expected expression")]
    ExpectedExpression { span: Span },

    #[error("expected identifier")]
    ExpectedIdent { span: Span },

    #[error("expected type")]
    ExpectedType { span: Span },

    #[error("expected '{expected}'")]
    Expected { expected: &'static str, span: Span },

    #[error("invalid number literal")]
    InvalidNumber { span: Span },

    #[error("invalid HTTP status code: {code}")]
    InvalidHttpStatus { code: u16, span: Span },

    #[error("{message}")]
    Custom { message: String, span: Span },
}

impl ParseError {
    pub fn span(&self) -> Span {
        match self {
            ParseError::UnexpectedToken { span, .. } => *span,
            ParseError::UnexpectedEof { span } => *span,
            ParseError::ExpectedExpression { span } => *span,
            ParseError::ExpectedIdent { span } => *span,
            ParseError::ExpectedType { span } => *span,
            ParseError::Expected { span, .. } => *span,
            ParseError::InvalidNumber { span } => *span,
            ParseError::InvalidHttpStatus { span, .. } => *span,
            ParseError::Custom { span, .. } => *span,
        }
    }

    pub fn unexpected_token(expected: impl Into<String>, found: &TokenKind, span: Span) -> Self {
        ParseError::UnexpectedToken {
            expected: expected.into(),
            found: format!("{}", found),
            span,
        }
    }
}
