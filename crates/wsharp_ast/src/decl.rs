//! Declaration AST nodes.

use crate::{Block, Expr, Ident, NodeId, Parameter, TypeExpr, Visibility};
use wsharp_lexer::Span;

/// A function declaration.
#[derive(Clone, Debug)]
pub struct FunctionDecl {
    pub name: Ident,
    pub generics: Vec<GenericParam>,
    pub params: Vec<Parameter>,
    pub return_type: Option<TypeExpr>,
    pub body: Option<Block>,
    pub is_async: bool,
    pub visibility: Visibility,
    pub dispatch_params: Vec<DispatchParam>,
    pub span: Span,
    pub id: NodeId,
}

/// A generic type parameter.
#[derive(Clone, Debug)]
pub struct GenericParam {
    pub name: Ident,
    pub bounds: Vec<TypeExpr>,
    pub default: Option<TypeExpr>,
    pub span: Span,
}

/// Information about a parameter's role in dispatch.
#[derive(Clone, Debug)]
pub struct DispatchParam {
    pub param_index: usize,
    pub dispatch_kind: DispatchKind,
}

/// The kind of dispatch for a parameter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DispatchKind {
    Static,
    Dynamic,
    Multiple,
    Function,
    HttpStatus,
}

/// A prototype (class-like) declaration.
#[derive(Clone, Debug)]
pub struct PrototypeDecl {
    pub name: Ident,
    pub parent: Option<TypeExpr>,
    pub members: Vec<PrototypeMember>,
    pub constructor: Option<FunctionDecl>,
    pub visibility: Visibility,
    pub span: Span,
    pub id: NodeId,
}

/// A member of a prototype.
#[derive(Clone, Debug)]
pub enum PrototypeMember {
    /// A property with optional default value
    Property {
        name: Ident,
        ty: Option<TypeExpr>,
        default: Option<Expr>,
        span: Span,
    },
    /// A method
    Method(FunctionDecl),
}

/// An extension declaration (adding methods to existing types).
#[derive(Clone, Debug)]
pub struct ExtensionDecl {
    pub target_type: TypeExpr,
    pub methods: Vec<FunctionDecl>,
    pub span: Span,
    pub id: NodeId,
}

/// A type alias declaration.
#[derive(Clone, Debug)]
pub struct TypeAliasDecl {
    pub name: Ident,
    pub generics: Vec<GenericParam>,
    pub ty: TypeExpr,
    pub visibility: Visibility,
    pub span: Span,
    pub id: NodeId,
}

/// A module declaration.
#[derive(Clone, Debug)]
pub struct ModuleDecl {
    pub name: Ident,
    pub items: Vec<crate::Item>,
    pub visibility: Visibility,
    pub span: Span,
    pub id: NodeId,
}

/// An import declaration.
#[derive(Clone, Debug)]
pub struct ImportDecl {
    pub path: Vec<Ident>,
    pub alias: Option<Ident>,
    pub items: ImportItems,
    pub span: Span,
    pub id: NodeId,
}

/// Items being imported.
#[derive(Clone, Debug)]
pub enum ImportItems {
    /// Import all items (use foo::*)
    All,
    /// Import specific items (use foo::{a, b})
    Specific(Vec<ImportItem>),
    /// Import the module itself
    Module,
}

/// A single import item.
#[derive(Clone, Debug)]
pub struct ImportItem {
    pub name: Ident,
    pub alias: Option<Ident>,
}
