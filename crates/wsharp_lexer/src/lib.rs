//! Lexical analyzer for the W# programming language.
//!
//! This crate provides the lexer (tokenizer) that converts W# source code
//! into a stream of tokens for parsing.

mod token;
mod span;
mod lexer;

pub use token::{Token, TokenKind};
pub use span::Span;
pub use lexer::Lexer;
