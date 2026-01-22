//! Expression AST nodes.

use crate::{Ident, NodeId, Pattern, Stmt, TypeExpr};
use wsharp_lexer::Span;

/// An expression in W#.
#[derive(Clone, Debug)]
pub struct Expr {
    pub kind: ExprKind,
    pub span: Span,
    pub id: NodeId,
}

impl Expr {
    pub fn new(kind: ExprKind, span: Span, id: NodeId) -> Self {
        Self { kind, span, id }
    }
}

/// The kind of expression.
#[derive(Clone, Debug)]
pub enum ExprKind {
    /// A literal value (42, "hello", true, etc.)
    Literal(Literal),

    /// An identifier reference
    Ident(Ident),

    /// A binary operation (a + b, x == y, etc.)
    Binary {
        op: BinaryOp,
        left: Box<Expr>,
        right: Box<Expr>,
    },

    /// A unary operation (-x, !b, etc.)
    Unary {
        op: UnaryOp,
        operand: Box<Expr>,
    },

    /// A function call
    Call {
        callee: Box<Expr>,
        args: Vec<Expr>,
    },

    /// A method call (obj.method(args))
    MethodCall {
        receiver: Box<Expr>,
        method: Ident,
        args: Vec<Expr>,
    },

    /// Property access (obj.prop)
    PropertyAccess {
        object: Box<Expr>,
        property: Ident,
    },

    /// Index access (arr[i])
    IndexAccess {
        object: Box<Expr>,
        index: Box<Expr>,
    },

    /// Object literal { name: value, ... }
    ObjectLiteral {
        prototype: Option<Box<Expr>>,
        members: Vec<ObjectMember>,
    },

    /// Array literal [a, b, c]
    ArrayLiteral(Vec<Expr>),

    /// Tuple (a, b, c)
    Tuple(Vec<Expr>),

    /// If expression
    If {
        condition: Box<Expr>,
        then_branch: Box<Block>,
        else_branch: Option<Box<Block>>,
    },

    /// Match expression
    Match {
        scrutinee: Box<Expr>,
        arms: Vec<MatchArm>,
    },

    /// While loop
    While {
        condition: Box<Expr>,
        body: Box<Block>,
    },

    /// For loop
    For {
        binding: Ident,
        iterator: Box<Expr>,
        body: Box<Block>,
    },

    /// Loop (infinite)
    Loop {
        body: Box<Block>,
    },

    /// Break expression
    Break(Option<Box<Expr>>),

    /// Continue expression
    Continue,

    /// Return expression
    Return(Option<Box<Expr>>),

    /// Await expression
    Await(Box<Expr>),

    /// Lambda/closure expression
    Lambda {
        params: Vec<Parameter>,
        return_type: Option<TypeExpr>,
        body: Box<Expr>,
        is_async: bool,
    },

    /// Block expression
    Block(Box<Block>),

    /// Range expression (a..b or a..=b)
    Range {
        start: Option<Box<Expr>>,
        end: Option<Box<Expr>>,
        inclusive: bool,
    },

    /// Type cast (expr as Type)
    Cast {
        expr: Box<Expr>,
        ty: TypeExpr,
    },

    /// HTTP status literal (e.g., http 200, http 404)
    HttpStatus(u16),

    /// New expression for creating prototype instances
    New {
        prototype: Box<Expr>,
        args: Vec<Expr>,
    },

    /// Assignment expression
    Assign {
        target: Box<Expr>,
        value: Box<Expr>,
    },

    /// Compound assignment (+=, -=, etc.)
    CompoundAssign {
        op: BinaryOp,
        target: Box<Expr>,
        value: Box<Expr>,
    },
}

/// A literal value.
#[derive(Clone, Debug)]
pub enum Literal {
    Int(i128),
    Float(f64),
    String(String),
    Char(char),
    Bool(bool),
    Null,
}

/// Binary operators.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinaryOp {
    // Arithmetic
    Add,
    Sub,
    Mul,
    Div,
    Mod,

    // Comparison
    Eq,
    NotEq,
    Lt,
    LtEq,
    Gt,
    GtEq,

    // Logical
    And,
    Or,

    // Bitwise
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
}

impl BinaryOp {
    /// Returns the precedence of this operator (higher = binds tighter).
    pub fn precedence(&self) -> u8 {
        match self {
            BinaryOp::Or => 1,
            BinaryOp::And => 2,
            BinaryOp::Eq | BinaryOp::NotEq => 3,
            BinaryOp::Lt | BinaryOp::LtEq | BinaryOp::Gt | BinaryOp::GtEq => 4,
            BinaryOp::BitOr => 5,
            BinaryOp::BitXor => 6,
            BinaryOp::BitAnd => 7,
            BinaryOp::Shl | BinaryOp::Shr => 8,
            BinaryOp::Add | BinaryOp::Sub => 9,
            BinaryOp::Mul | BinaryOp::Div | BinaryOp::Mod => 10,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            BinaryOp::Add => "+",
            BinaryOp::Sub => "-",
            BinaryOp::Mul => "*",
            BinaryOp::Div => "/",
            BinaryOp::Mod => "%",
            BinaryOp::Eq => "==",
            BinaryOp::NotEq => "!=",
            BinaryOp::Lt => "<",
            BinaryOp::LtEq => "<=",
            BinaryOp::Gt => ">",
            BinaryOp::GtEq => ">=",
            BinaryOp::And => "&&",
            BinaryOp::Or => "||",
            BinaryOp::BitAnd => "&",
            BinaryOp::BitOr => "|",
            BinaryOp::BitXor => "^",
            BinaryOp::Shl => "<<",
            BinaryOp::Shr => ">>",
        }
    }
}

/// Unary operators.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnaryOp {
    Neg,
    Not,
    BitNot,
    Ref,
    RefMut,
    Deref,
}

impl UnaryOp {
    pub fn as_str(&self) -> &'static str {
        match self {
            UnaryOp::Neg => "-",
            UnaryOp::Not => "!",
            UnaryOp::BitNot => "~",
            UnaryOp::Ref => "&",
            UnaryOp::RefMut => "&mut",
            UnaryOp::Deref => "*",
        }
    }
}

/// A member in an object literal.
#[derive(Clone, Debug)]
pub struct ObjectMember {
    pub key: Ident,
    pub value: Expr,
    pub span: Span,
}

/// A match arm.
#[derive(Clone, Debug)]
pub struct MatchArm {
    pub pattern: Pattern,
    pub guard: Option<Expr>,
    pub body: Expr,
    pub span: Span,
}

/// A function parameter.
#[derive(Clone, Debug)]
pub struct Parameter {
    pub name: Ident,
    pub ty: Option<TypeExpr>,
    pub default: Option<Expr>,
    pub span: Span,
}

/// A block of statements.
#[derive(Clone, Debug)]
pub struct Block {
    pub stmts: Vec<Stmt>,
    pub expr: Option<Box<Expr>>,
    pub span: Span,
}
