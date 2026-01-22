//! HIR node definitions.

use wsharp_lexer::Span;
use wsharp_types::Type;

/// A unique identifier for HIR nodes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HirId(pub u32);

/// A unique identifier for definitions (functions, types, etc.).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DefId(pub u32);

/// A unique identifier for local variables.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LocalId(pub u32);

/// A complete HIR module.
#[derive(Clone, Debug)]
pub struct HirModule {
    pub name: String,
    pub items: Vec<HirItem>,
    pub span: Span,
}

/// A top-level item in HIR.
#[derive(Clone, Debug)]
pub enum HirItem {
    Function(HirFunction),
    Prototype(HirPrototype),
    TypeAlias(HirTypeAlias),
    Module(HirModule),
}

/// A function in HIR.
#[derive(Clone, Debug)]
pub struct HirFunction {
    pub id: DefId,
    pub name: String,
    pub params: Vec<HirParam>,
    pub return_type: Type,
    pub body: Option<HirBody>,
    pub is_async: bool,
    pub span: Span,
}

/// A function parameter.
#[derive(Clone, Debug)]
pub struct HirParam {
    pub id: LocalId,
    pub name: String,
    pub ty: Type,
    pub span: Span,
}

/// A function body.
#[derive(Clone, Debug)]
pub struct HirBody {
    pub locals: Vec<HirLocal>,
    pub expr: HirExpr,
}

/// A local variable declaration.
#[derive(Clone, Debug)]
pub struct HirLocal {
    pub id: LocalId,
    pub name: String,
    pub ty: Type,
    pub mutable: bool,
    pub span: Span,
}

/// An expression in HIR.
#[derive(Clone, Debug)]
pub struct HirExpr {
    pub id: HirId,
    pub kind: HirExprKind,
    pub ty: Type,
    pub span: Span,
}

/// The kind of HIR expression.
#[derive(Clone, Debug)]
pub enum HirExprKind {
    /// A literal value.
    Literal(HirLiteral),

    /// A local variable reference.
    Local(LocalId),

    /// A global reference (function, constant, etc.).
    Global(DefId),

    /// A binary operation.
    Binary {
        op: HirBinaryOp,
        left: Box<HirExpr>,
        right: Box<HirExpr>,
    },

    /// A unary operation.
    Unary {
        op: HirUnaryOp,
        operand: Box<HirExpr>,
    },

    /// A function call.
    Call {
        callee: Box<HirExpr>,
        args: Vec<HirExpr>,
    },

    /// A method call (desugared to function call + receiver).
    MethodCall {
        receiver: Box<HirExpr>,
        method: DefId,
        args: Vec<HirExpr>,
    },

    /// Field access on a prototype.
    Field {
        object: Box<HirExpr>,
        field: String,
    },

    /// Array/slice indexing.
    Index {
        object: Box<HirExpr>,
        index: Box<HirExpr>,
    },

    /// Block expression.
    Block(HirBlock),

    /// If expression (always has both branches in HIR).
    If {
        condition: Box<HirExpr>,
        then_branch: Box<HirExpr>,
        else_branch: Box<HirExpr>,
    },

    /// Simple loop (all loops are desugared to this).
    Loop {
        body: Box<HirExpr>,
    },

    /// Break from a loop with optional value.
    Break {
        value: Option<Box<HirExpr>>,
    },

    /// Continue to next loop iteration.
    Continue,

    /// Return from function.
    Return {
        value: Option<Box<HirExpr>>,
    },

    /// Let binding.
    Let {
        local: LocalId,
        init: Box<HirExpr>,
        body: Box<HirExpr>,
    },

    /// Assignment.
    Assign {
        target: Box<HirExpr>,
        value: Box<HirExpr>,
    },

    /// Tuple construction.
    Tuple(Vec<HirExpr>),

    /// Array construction.
    Array(Vec<HirExpr>),

    /// Object/Prototype literal.
    Object {
        prototype: Option<DefId>,
        fields: Vec<(String, HirExpr)>,
    },

    /// Lambda expression.
    Lambda {
        params: Vec<HirParam>,
        body: Box<HirExpr>,
        captures: Vec<LocalId>,
        is_async: bool,
    },

    /// Await expression.
    Await(Box<HirExpr>),

    /// Type cast.
    Cast {
        expr: Box<HirExpr>,
        target_ty: Type,
    },

    /// Match expression (simplified).
    Match {
        scrutinee: Box<HirExpr>,
        arms: Vec<HirMatchArm>,
    },

    /// HTTP status literal.
    HttpStatus(u16),
}

/// A block of statements with optional result.
#[derive(Clone, Debug)]
pub struct HirBlock {
    pub stmts: Vec<HirStmt>,
    pub expr: Option<Box<HirExpr>>,
}

/// A statement in HIR.
#[derive(Clone, Debug)]
pub enum HirStmt {
    /// Expression statement.
    Expr(HirExpr),

    /// Let binding without continuation (statement form).
    Let {
        local: LocalId,
        init: HirExpr,
    },
}

/// A match arm.
#[derive(Clone, Debug)]
pub struct HirMatchArm {
    pub pattern: HirPattern,
    pub guard: Option<HirExpr>,
    pub body: HirExpr,
}

/// A simplified pattern for matching.
#[derive(Clone, Debug)]
pub enum HirPattern {
    /// Wildcard pattern (_).
    Wildcard,

    /// Bind to a local variable.
    Binding(LocalId),

    /// Literal pattern.
    Literal(HirLiteral),

    /// Tuple pattern.
    Tuple(Vec<HirPattern>),

    /// HTTP status pattern.
    HttpStatus(HirHttpStatusPattern),

    /// Or pattern.
    Or(Vec<HirPattern>),
}

/// HTTP status pattern in HIR.
#[derive(Clone, Debug)]
pub enum HirHttpStatusPattern {
    Exact(u16),
    Range { start: u16, end: u16 },
    Any,
}

/// A literal value.
#[derive(Clone, Debug)]
pub enum HirLiteral {
    Int(i128),
    Float(f64),
    Bool(bool),
    Char(char),
    String(String),
    Unit,
}

/// Binary operators in HIR.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HirBinaryOp {
    // Arithmetic
    Add,
    Sub,
    Mul,
    Div,
    Rem,

    // Bitwise
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,

    // Comparison
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,

    // Logical
    And,
    Or,
}

/// Unary operators in HIR.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HirUnaryOp {
    Neg,
    Not,
    BitNot,
    Deref,
    Ref,
    RefMut,
}

/// A prototype definition in HIR.
#[derive(Clone, Debug)]
pub struct HirPrototype {
    pub id: DefId,
    pub name: String,
    pub parent: Option<DefId>,
    pub fields: Vec<HirField>,
    pub methods: Vec<HirFunction>,
    pub constructor: Option<HirFunction>,
    pub span: Span,
}

/// A field in a prototype.
#[derive(Clone, Debug)]
pub struct HirField {
    pub name: String,
    pub ty: Type,
    pub default: Option<HirExpr>,
    pub span: Span,
}

/// A type alias in HIR.
#[derive(Clone, Debug)]
pub struct HirTypeAlias {
    pub id: DefId,
    pub name: String,
    pub ty: Type,
    pub span: Span,
}
