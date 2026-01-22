//! Abstract Syntax Tree definitions for the W# programming language.
//!
//! This crate provides the core AST data structures used throughout
//! the W# compiler pipeline.

mod types;
mod pattern;
mod stmt;
mod expr;
mod decl;

pub use types::*;
pub use pattern::*;
pub use stmt::*;
pub use expr::*;
pub use decl::*;

use wsharp_lexer::Span;

/// A unique identifier for an AST node.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NodeId(pub u32);

impl NodeId {
    pub fn new(id: u32) -> Self {
        Self(id)
    }
}

/// An identifier with its source span.
#[derive(Clone, Debug, PartialEq)]
pub struct Ident {
    pub name: String,
    pub span: Span,
}

impl Ident {
    pub fn new(name: String, span: Span) -> Self {
        Self { name, span }
    }
}

/// Visibility of a declaration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Visibility {
    #[default]
    Private,
    Public,
}

/// A complete W# source file.
#[derive(Clone, Debug)]
pub struct SourceFile {
    pub items: Vec<Item>,
    pub span: Span,
}

/// A top-level item in a source file.
#[derive(Clone, Debug)]
pub enum Item {
    Function(FunctionDecl),
    Prototype(PrototypeDecl),
    Extension(ExtensionDecl),
    TypeAlias(TypeAliasDecl),
    Module(ModuleDecl),
    Import(ImportDecl),
}

impl Item {
    pub fn span(&self) -> Span {
        match self {
            Item::Function(f) => f.span,
            Item::Prototype(p) => p.span,
            Item::Extension(e) => e.span,
            Item::TypeAlias(t) => t.span,
            Item::Module(m) => m.span,
            Item::Import(i) => i.span,
        }
    }
}
