//! Name resolution and symbol table management.
//!
//! This module handles:
//! - Building symbol tables from AST declarations
//! - Resolving names to their definitions
//! - Handling scopes (lexical scoping)

use crate::{DefId, LocalId};
use std::collections::HashMap;
use thiserror::Error;
use wsharp_lexer::Span;
use wsharp_types::Type;

/// Errors during name resolution.
#[derive(Clone, Debug, Error)]
pub enum ResolveError {
    #[error("undefined variable: {name}")]
    UndefinedVariable { name: String, span: Span },

    #[error("undefined type: {name}")]
    UndefinedType { name: String, span: Span },

    #[error("undefined function: {name}")]
    UndefinedFunction { name: String, span: Span },

    #[error("duplicate definition: {name}")]
    DuplicateDefinition { name: String, span: Span },

    #[error("cannot access private member: {name}")]
    PrivateAccess { name: String, span: Span },

    #[error("invalid assignment target")]
    InvalidAssignTarget { span: Span },
}

/// The result of resolution operations.
pub type ResolveResult<T> = Result<T, ResolveError>;

/// A symbol in the symbol table.
#[derive(Clone, Debug)]
pub enum Symbol {
    /// A local variable.
    Local {
        id: LocalId,
        ty: Type,
        mutable: bool,
    },

    /// A function.
    Function {
        id: DefId,
        ty: Type,
    },

    /// A prototype (type).
    Prototype {
        id: DefId,
    },

    /// A type alias.
    TypeAlias {
        id: DefId,
        ty: Type,
    },

    /// A module.
    Module {
        id: DefId,
    },

    /// A generic type parameter.
    TypeParam {
        index: usize,
    },
}

/// A scope in the symbol table.
#[derive(Clone, Debug, Default)]
pub struct Scope {
    /// Variable bindings.
    variables: HashMap<String, Symbol>,
    /// Type bindings.
    types: HashMap<String, Symbol>,
    /// Function overloads (multiple functions can have the same name).
    function_overloads: HashMap<String, Vec<Symbol>>,
    /// Whether this is a function scope (for return checking).
    is_function: bool,
    /// Whether this is a loop scope (for break/continue checking).
    is_loop: bool,
}

impl Scope {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn function_scope() -> Self {
        Self {
            is_function: true,
            ..Self::default()
        }
    }

    pub fn loop_scope() -> Self {
        Self {
            is_loop: true,
            ..Self::default()
        }
    }
}

/// Information about a prototype (type) including its fields and methods.
#[derive(Clone, Debug)]
pub struct PrototypeInfo {
    /// The prototype's name.
    pub name: String,
    /// Parent prototype for inheritance (if any).
    pub parent: Option<DefId>,
    /// Fields: name -> (index, type).
    pub fields: HashMap<String, (usize, Type)>,
    /// Methods: name -> method DefId.
    pub methods: HashMap<String, DefId>,
}

impl PrototypeInfo {
    pub fn new(name: String) -> Self {
        Self {
            name,
            parent: None,
            fields: HashMap::new(),
            methods: HashMap::new(),
        }
    }

    pub fn with_parent(name: String, parent: DefId) -> Self {
        Self {
            name,
            parent: Some(parent),
            fields: HashMap::new(),
            methods: HashMap::new(),
        }
    }
}

/// The resolver maintains symbol tables and resolves names.
pub struct Resolver {
    /// Stack of scopes.
    scopes: Vec<Scope>,

    /// Counter for generating local variable IDs.
    next_local: u32,

    /// Counter for generating definition IDs.
    next_def: u32,

    /// Global definitions.
    definitions: HashMap<DefId, DefinitionInfo>,

    /// Prototype information for method/field resolution.
    prototypes: HashMap<DefId, PrototypeInfo>,

    /// Collected errors.
    errors: Vec<ResolveError>,
}

/// Information about a definition.
#[derive(Clone, Debug)]
pub struct DefinitionInfo {
    pub name: String,
    pub kind: DefinitionKind,
    pub span: Span,
}

/// The kind of definition.
#[derive(Clone, Debug)]
pub enum DefinitionKind {
    Function,
    Prototype,
    TypeAlias,
    Module,
    Field,
    Method,
}

impl Resolver {
    pub fn new() -> Self {
        let mut resolver = Self {
            scopes: vec![Scope::new()],
            next_local: 0,
            next_def: 0,
            definitions: HashMap::new(),
            prototypes: HashMap::new(),
            errors: Vec::new(),
        };

        // Add built-in types
        resolver.add_builtin_types();

        resolver
    }

    fn add_builtin_types(&mut self) {
        let builtins = [
            "i8", "i16", "i32", "i64", "i128", "u8", "u16", "u32", "u64", "u128", "f32", "f64",
            "bool", "char", "String",
        ];

        for name in builtins {
            let id = self.fresh_def_id();
            self.define_type(
                name.to_string(),
                Symbol::TypeAlias {
                    id,
                    ty: Type::Unknown, // Will be properly set during type checking
                },
            );
        }
    }

    /// Generate a fresh local variable ID.
    pub fn fresh_local_id(&mut self) -> LocalId {
        let id = LocalId(self.next_local);
        self.next_local += 1;
        id
    }

    /// Generate a fresh definition ID.
    pub fn fresh_def_id(&mut self) -> DefId {
        let id = DefId(self.next_def);
        self.next_def += 1;
        id
    }

    /// Enter a new scope.
    pub fn enter_scope(&mut self) {
        self.scopes.push(Scope::new());
    }

    /// Enter a function scope.
    pub fn enter_function_scope(&mut self) {
        self.scopes.push(Scope::function_scope());
    }

    /// Enter a loop scope.
    pub fn enter_loop_scope(&mut self) {
        self.scopes.push(Scope::loop_scope());
    }

    /// Exit the current scope.
    pub fn exit_scope(&mut self) {
        if self.scopes.len() > 1 {
            self.scopes.pop();
        }
    }

    /// Define a variable in the current scope.
    pub fn define_variable(&mut self, name: String, symbol: Symbol) -> ResolveResult<()> {
        if let Some(scope) = self.scopes.last_mut() {
            if scope.variables.contains_key(&name) {
                return Err(ResolveError::DuplicateDefinition {
                    name,
                    span: Span::dummy(),
                });
            }
            scope.variables.insert(name, symbol);
        }
        Ok(())
    }

    /// Define a type in the current scope.
    pub fn define_type(&mut self, name: String, symbol: Symbol) {
        if let Some(scope) = self.scopes.last_mut() {
            // For functions, track all overloads
            if matches!(&symbol, Symbol::Function { .. }) {
                scope
                    .function_overloads
                    .entry(name.clone())
                    .or_default()
                    .push(symbol.clone());
            }
            scope.types.insert(name, symbol);
        }
    }

    /// Get all function overloads for a name.
    pub fn lookup_function_overloads(&self, name: &str) -> Vec<&Symbol> {
        let mut overloads = Vec::new();
        for scope in self.scopes.iter().rev() {
            if let Some(symbols) = scope.function_overloads.get(name) {
                overloads.extend(symbols.iter());
            }
        }
        overloads
    }

    /// Look up a variable by name.
    pub fn lookup_variable(&self, name: &str) -> Option<&Symbol> {
        for scope in self.scopes.iter().rev() {
            if let Some(symbol) = scope.variables.get(name) {
                return Some(symbol);
            }
        }
        None
    }

    /// Look up a type by name.
    pub fn lookup_type(&self, name: &str) -> Option<&Symbol> {
        for scope in self.scopes.iter().rev() {
            if let Some(symbol) = scope.types.get(name) {
                return Some(symbol);
            }
        }
        None
    }

    /// Check if we're inside a function.
    pub fn in_function(&self) -> bool {
        self.scopes.iter().any(|s| s.is_function)
    }

    /// Check if we're inside a loop.
    pub fn in_loop(&self) -> bool {
        self.scopes.iter().any(|s| s.is_loop)
    }

    // === Prototype management ===

    /// Register a prototype with its info.
    pub fn register_prototype(&mut self, id: DefId, info: PrototypeInfo) {
        self.prototypes.insert(id, info);
    }

    /// Get prototype info by DefId.
    pub fn get_prototype(&self, id: DefId) -> Option<&PrototypeInfo> {
        self.prototypes.get(&id)
    }

    /// Get mutable prototype info by DefId.
    pub fn get_prototype_mut(&mut self, id: DefId) -> Option<&mut PrototypeInfo> {
        self.prototypes.get_mut(&id)
    }

    /// Look up a prototype by name.
    pub fn lookup_prototype_by_name(&self, name: &str) -> Option<DefId> {
        for scope in self.scopes.iter().rev() {
            if let Some(Symbol::Prototype { id }) = scope.types.get(name) {
                return Some(*id);
            }
        }
        None
    }

    /// Look up a method on a prototype, with inheritance chain traversal.
    pub fn lookup_method(&self, prototype_id: DefId, method_name: &str) -> Option<DefId> {
        let proto = self.prototypes.get(&prototype_id)?;

        // First, check methods defined on this prototype
        if let Some(&method_id) = proto.methods.get(method_name) {
            return Some(method_id);
        }

        // If not found, check parent prototype (inheritance)
        if let Some(parent_id) = proto.parent {
            return self.lookup_method(parent_id, method_name);
        }

        None
    }

    /// Look up a field on a prototype, with inheritance chain traversal.
    pub fn lookup_field(&self, prototype_id: DefId, field_name: &str) -> Option<(usize, Type)> {
        let proto = self.prototypes.get(&prototype_id)?;

        // First, check fields on this prototype
        if let Some(&(idx, ref ty)) = proto.fields.get(field_name) {
            return Some((idx, ty.clone()));
        }

        // If not found, check parent prototype
        if let Some(parent_id) = proto.parent {
            return self.lookup_field(parent_id, field_name);
        }

        None
    }

    /// Register a method on a prototype.
    pub fn register_method(&mut self, prototype_id: DefId, method_name: String, method_id: DefId) {
        if let Some(proto) = self.prototypes.get_mut(&prototype_id) {
            proto.methods.insert(method_name, method_id);
        }
    }

    /// Register a field on a prototype.
    pub fn register_field(&mut self, prototype_id: DefId, field_name: String, index: usize, ty: Type) {
        if let Some(proto) = self.prototypes.get_mut(&prototype_id) {
            proto.fields.insert(field_name, (index, ty));
        }
    }

    /// Register a definition.
    pub fn register_definition(&mut self, id: DefId, info: DefinitionInfo) {
        self.definitions.insert(id, info);
    }

    /// Get definition info.
    pub fn get_definition(&self, id: DefId) -> Option<&DefinitionInfo> {
        self.definitions.get(&id)
    }

    /// Report an error.
    pub fn error(&mut self, error: ResolveError) {
        self.errors.push(error);
    }

    /// Get all collected errors.
    pub fn errors(&self) -> &[ResolveError] {
        &self.errors
    }

    /// Check if there are any errors.
    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }

    /// Take the errors, leaving an empty vector.
    pub fn take_errors(&mut self) -> Vec<ResolveError> {
        std::mem::take(&mut self.errors)
    }
}

impl Default for Resolver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wsharp_types::PrimitiveType;

    #[test]
    fn test_scope_management() {
        let mut resolver = Resolver::new();

        // Define in outer scope
        let id1 = resolver.fresh_local_id();
        resolver
            .define_variable(
                "x".to_string(),
                Symbol::Local {
                    id: id1,
                    ty: Type::Primitive(PrimitiveType::I32),
                    mutable: false,
                },
            )
            .unwrap();

        // Enter inner scope
        resolver.enter_scope();

        // Define in inner scope
        let id2 = resolver.fresh_local_id();
        resolver
            .define_variable(
                "y".to_string(),
                Symbol::Local {
                    id: id2,
                    ty: Type::Primitive(PrimitiveType::I32),
                    mutable: false,
                },
            )
            .unwrap();

        // Both should be visible
        assert!(resolver.lookup_variable("x").is_some());
        assert!(resolver.lookup_variable("y").is_some());

        // Exit inner scope
        resolver.exit_scope();

        // x still visible, y not
        assert!(resolver.lookup_variable("x").is_some());
        assert!(resolver.lookup_variable("y").is_none());
    }

    #[test]
    fn test_shadowing() {
        let mut resolver = Resolver::new();

        let id1 = resolver.fresh_local_id();
        resolver
            .define_variable(
                "x".to_string(),
                Symbol::Local {
                    id: id1,
                    ty: Type::Primitive(PrimitiveType::I32),
                    mutable: false,
                },
            )
            .unwrap();

        resolver.enter_scope();

        let id2 = resolver.fresh_local_id();
        resolver
            .define_variable(
                "x".to_string(),
                Symbol::Local {
                    id: id2,
                    ty: Type::Primitive(PrimitiveType::I64),
                    mutable: false,
                },
            )
            .unwrap();

        // Should get the inner x
        if let Some(Symbol::Local { id, .. }) = resolver.lookup_variable("x") {
            assert_eq!(*id, id2);
        } else {
            panic!("Expected local variable");
        }

        resolver.exit_scope();

        // Should get the outer x
        if let Some(Symbol::Local { id, .. }) = resolver.lookup_variable("x") {
            assert_eq!(*id, id1);
        } else {
            panic!("Expected local variable");
        }
    }

    #[test]
    fn test_loop_and_function_scope() {
        let mut resolver = Resolver::new();

        assert!(!resolver.in_function());
        assert!(!resolver.in_loop());

        resolver.enter_function_scope();
        assert!(resolver.in_function());
        assert!(!resolver.in_loop());

        resolver.enter_loop_scope();
        assert!(resolver.in_function());
        assert!(resolver.in_loop());

        resolver.exit_scope();
        assert!(resolver.in_function());
        assert!(!resolver.in_loop());

        resolver.exit_scope();
        assert!(!resolver.in_function());
    }
}
