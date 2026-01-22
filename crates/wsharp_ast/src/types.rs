//! Type expression AST nodes.

use crate::Ident;
use wsharp_lexer::Span;

/// A type expression in the source code.
#[derive(Clone, Debug)]
pub struct TypeExpr {
    pub kind: TypeExprKind,
    pub span: Span,
}

impl TypeExpr {
    pub fn new(kind: TypeExprKind, span: Span) -> Self {
        Self { kind, span }
    }
}

/// The kind of type expression.
#[derive(Clone, Debug)]
pub enum TypeExprKind {
    /// A named type (possibly with generics)
    Named {
        name: Ident,
        generics: Vec<TypeExpr>,
    },

    /// A function type
    Function {
        params: Vec<TypeExpr>,
        return_type: Box<TypeExpr>,
        is_async: bool,
    },

    /// An HTTP status type (e.g., HttpStatus<200>, HttpStatus<2xx>)
    HttpStatus(HttpStatusTypeExpr),

    /// A prototype type with inline definition
    Prototype {
        parent: Option<Box<TypeExpr>>,
        members: Vec<(Ident, TypeExpr)>,
    },

    /// An array type [T; N] or [T]
    Array {
        element: Box<TypeExpr>,
        size: Option<usize>,
    },

    /// A tuple type (A, B, C)
    Tuple(Vec<TypeExpr>),

    /// A reference type
    Ref {
        inner: Box<TypeExpr>,
        mutable: bool,
    },

    /// An optional/nullable type (T?)
    Optional(Box<TypeExpr>),

    /// A union type (A | B)
    Union(Vec<TypeExpr>),

    /// An intersection type (A & B)
    Intersection(Vec<TypeExpr>),

    /// Type inference placeholder (_)
    Infer,

    /// Never type (!)
    Never,

    /// Unit type ()
    Unit,
}

/// HTTP status type expression.
#[derive(Clone, Debug)]
pub enum HttpStatusTypeExpr {
    /// Exact status code (e.g., 200, 404)
    Exact(u16),
    /// Category (e.g., 1xx, 2xx, 3xx, 4xx, 5xx)
    Category(HttpStatusCategory),
    /// Range (e.g., 400..499)
    Range { start: u16, end: u16 },
    /// Union of status types
    Union(Vec<HttpStatusTypeExpr>),
    /// Any status
    Any,
}

/// HTTP status code category.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HttpStatusCategory {
    /// 1xx - Informational
    Informational,
    /// 2xx - Success
    Success,
    /// 3xx - Redirection
    Redirection,
    /// 4xx - Client Error
    ClientError,
    /// 5xx - Server Error
    ServerError,
}

impl HttpStatusCategory {
    pub fn as_str(&self) -> &'static str {
        match self {
            HttpStatusCategory::Informational => "1xx",
            HttpStatusCategory::Success => "2xx",
            HttpStatusCategory::Redirection => "3xx",
            HttpStatusCategory::ClientError => "4xx",
            HttpStatusCategory::ServerError => "5xx",
        }
    }

    pub fn range(&self) -> (u16, u16) {
        match self {
            HttpStatusCategory::Informational => (100, 199),
            HttpStatusCategory::Success => (200, 299),
            HttpStatusCategory::Redirection => (300, 399),
            HttpStatusCategory::ClientError => (400, 499),
            HttpStatusCategory::ServerError => (500, 599),
        }
    }
}
