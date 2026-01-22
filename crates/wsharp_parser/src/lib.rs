//! Parser for the W# programming language.
//!
//! This crate provides a recursive descent parser with Pratt parsing for
//! expressions.

mod parser;
mod expr;
mod stmt;
mod decl;
mod error;

pub use parser::Parser;
pub use error::{ParseError, ParseResult};
