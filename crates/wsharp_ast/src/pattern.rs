//! Pattern AST nodes for pattern matching.

use crate::{Ident, Literal, TypeExpr};
use wsharp_lexer::Span;

/// A pattern for matching.
#[derive(Clone, Debug)]
pub struct Pattern {
    pub kind: PatternKind,
    pub span: Span,
}

impl Pattern {
    pub fn new(kind: PatternKind, span: Span) -> Self {
        Self { kind, span }
    }
}

/// The kind of pattern.
#[derive(Clone, Debug)]
pub enum PatternKind {
    /// A literal pattern (42, "hello", true)
    Literal(Literal),

    /// An identifier pattern (binds to a name)
    Ident {
        name: Ident,
        mutable: bool,
    },

    /// Wildcard pattern (_)
    Wildcard,

    /// HTTP status pattern
    HttpStatus(HttpStatusPattern),

    /// Prototype/object pattern ({ name, age })
    Prototype {
        name: Option<Ident>,
        fields: Vec<(Ident, Pattern)>,
        rest: bool,
    },

    /// Tuple pattern (a, b, c)
    Tuple(Vec<Pattern>),

    /// Array pattern [a, b, c]
    Array {
        elements: Vec<Pattern>,
        rest: Option<Box<Pattern>>,
    },

    /// Or pattern (a | b)
    Or(Vec<Pattern>),

    /// Range pattern (a..b or a..=b)
    Range {
        start: Option<Box<Pattern>>,
        end: Option<Box<Pattern>>,
        inclusive: bool,
    },

    /// Binding pattern (name @ pattern)
    Binding {
        name: Ident,
        pattern: Box<Pattern>,
    },

    /// Type ascription pattern (pattern: Type)
    Typed {
        pattern: Box<Pattern>,
        ty: TypeExpr,
    },
}

/// HTTP status pattern.
#[derive(Clone, Debug)]
pub enum HttpStatusPattern {
    /// Exact status code
    Exact(u16),
    /// Range of codes
    Range { start: u16, end: u16 },
    /// Category (2xx, 4xx, etc.)
    Category(crate::HttpStatusCategory),
    /// Any status
    Any,
}
