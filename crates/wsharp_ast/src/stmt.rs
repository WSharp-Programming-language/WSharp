//! Statement AST nodes.

use crate::{Expr, Ident, NodeId, TypeExpr};
use wsharp_lexer::Span;

/// A statement in W#.
#[derive(Clone, Debug)]
pub struct Stmt {
    pub kind: StmtKind,
    pub span: Span,
    pub id: NodeId,
}

impl Stmt {
    pub fn new(kind: StmtKind, span: Span, id: NodeId) -> Self {
        Self { kind, span, id }
    }
}

/// The kind of statement.
#[derive(Clone, Debug)]
pub enum StmtKind {
    /// Expression statement (expr;)
    Expr(Expr),

    /// Let binding (let x = expr; or let x: Type = expr;)
    Let {
        name: Ident,
        ty: Option<TypeExpr>,
        value: Option<Expr>,
        mutable: bool,
    },

    /// Empty statement (;)
    Empty,
}
