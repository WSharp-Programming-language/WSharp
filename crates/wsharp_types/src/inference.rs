//! Type inference engine using Hindley-Milner algorithm with extensions.
//!
//! This module provides type inference through:
//! - Type variable generation
//! - Constraint collection
//! - Unification algorithm
//! - Substitution application

use crate::{
    DispatchRole, FunctionType, HttpStatusType, HttpStatusTypeKind, ParamType, PrimitiveType,
    PrototypeType, StatusCategory, Type, TypeVarId,
};
use std::collections::HashMap;
use thiserror::Error;

/// Errors that can occur during type inference.
#[derive(Clone, Debug, Error)]
pub enum TypeError {
    #[error("cannot unify {0} with {1}")]
    UnificationFailure(Type, Type),

    #[error("infinite type: {0} occurs in {1}")]
    InfiniteType(TypeVarId, Type),

    #[error("undefined variable: {0}")]
    UndefinedVariable(String),

    #[error("undefined type: {0}")]
    UndefinedType(String),

    #[error("type mismatch: expected {expected}, found {found}")]
    TypeMismatch { expected: Type, found: Type },

    #[error("cannot call non-function type: {0}")]
    NotCallable(Type),

    #[error("wrong number of arguments: expected {expected}, found {found}")]
    WrongArity { expected: usize, found: usize },

    #[error("cannot access property {property} on type {ty}")]
    NoSuchProperty { ty: Type, property: String },

    #[error("invalid HTTP status code: {0}")]
    InvalidHttpStatus(u16),

    #[error("ambiguous type: could not infer type for {0}")]
    AmbiguousType(String),

    #[error("cannot dispatch on type: {0}")]
    CannotDispatch(Type),
}

/// The result of type operations.
pub type TypeResult<T> = Result<T, TypeError>;

/// A substitution mapping type variables to types.
#[derive(Clone, Debug, Default)]
pub struct Substitution {
    map: HashMap<TypeVarId, Type>,
}

impl Substitution {
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
        }
    }

    /// Extend the substitution with a new binding.
    pub fn extend(&mut self, var: TypeVarId, ty: Type) {
        self.map.insert(var, ty);
    }

    /// Look up a type variable.
    pub fn lookup(&self, var: TypeVarId) -> Option<&Type> {
        self.map.get(&var)
    }

    /// Apply this substitution to a type.
    pub fn apply(&self, ty: &Type) -> Type {
        match ty {
            Type::TypeVar(var) => {
                if let Some(replacement) = self.lookup(*var) {
                    // Recursively apply in case of chained substitutions
                    self.apply(replacement)
                } else {
                    ty.clone()
                }
            }
            Type::Array { element, size } => Type::Array {
                element: Box::new(self.apply(element)),
                size: *size,
            },
            Type::Slice { element } => Type::Slice {
                element: Box::new(self.apply(element)),
            },
            Type::Tuple(types) => Type::Tuple(types.iter().map(|t| self.apply(t)).collect()),
            Type::Function(func) => Type::Function(FunctionType {
                params: func
                    .params
                    .iter()
                    .map(|p| ParamType {
                        ty: self.apply(&p.ty),
                        dispatch_role: p.dispatch_role,
                    })
                    .collect(),
                return_type: Box::new(self.apply(&func.return_type)),
                is_async: func.is_async,
            }),
            Type::Applied { base, args } => Type::Applied {
                base: Box::new(self.apply(base)),
                args: args.iter().map(|t| self.apply(t)).collect(),
            },
            Type::Ref { inner, mutable } => Type::Ref {
                inner: Box::new(self.apply(inner)),
                mutable: *mutable,
            },
            Type::Rc(inner) => Type::Rc(Box::new(self.apply(inner))),
            Type::Future(inner) => Type::Future(Box::new(self.apply(inner))),
            Type::Prototype(proto) => Type::Prototype(PrototypeType {
                name: proto.name.clone(),
                parent: proto.parent.as_ref().map(|p| Box::new(self.apply(p))),
                members: proto
                    .members
                    .iter()
                    .map(|(n, t)| (n.clone(), self.apply(t)))
                    .collect(),
            }),
            // Types that don't contain type variables
            Type::Primitive(_)
            | Type::HttpStatus(_)
            | Type::Never
            | Type::Unknown
            | Type::Unit => ty.clone(),
        }
    }

    /// Compose two substitutions: apply self first, then other.
    pub fn compose(&self, other: &Substitution) -> Substitution {
        let mut result = Substitution::new();

        // First, apply other to all types in self
        for (var, ty) in &self.map {
            result.extend(*var, other.apply(ty));
        }

        // Then add all bindings from other that aren't in self
        for (var, ty) in &other.map {
            if !result.map.contains_key(var) {
                result.extend(*var, ty.clone());
            }
        }

        result
    }
}

/// Type inference context.
pub struct InferenceContext {
    /// Counter for generating fresh type variables.
    next_var: u32,

    /// The current substitution.
    substitution: Substitution,

    /// Type environment: variable name -> type scheme.
    env: TypeEnvironment,
}

impl InferenceContext {
    pub fn new() -> Self {
        Self {
            next_var: 0,
            substitution: Substitution::new(),
            env: TypeEnvironment::new(),
        }
    }

    /// Generate a fresh type variable.
    pub fn fresh_var(&mut self) -> Type {
        let var = TypeVarId(self.next_var);
        self.next_var += 1;
        Type::TypeVar(var)
    }

    /// Get the current substitution.
    pub fn substitution(&self) -> &Substitution {
        &self.substitution
    }

    /// Get mutable reference to the environment.
    pub fn env_mut(&mut self) -> &mut TypeEnvironment {
        &mut self.env
    }

    /// Get reference to the environment.
    pub fn env(&self) -> &TypeEnvironment {
        &self.env
    }

    /// Apply the current substitution to a type.
    pub fn apply(&self, ty: &Type) -> Type {
        self.substitution.apply(ty)
    }

    /// Unify two types and update the substitution.
    pub fn unify(&mut self, t1: &Type, t2: &Type) -> TypeResult<()> {
        let t1 = self.apply(t1);
        let t2 = self.apply(t2);

        match (&t1, &t2) {
            // Same types unify trivially
            _ if t1 == t2 => Ok(()),

            // Type variable unification
            (Type::TypeVar(var), ty) | (ty, Type::TypeVar(var)) => {
                if let Type::TypeVar(other_var) = ty {
                    if var == other_var {
                        return Ok(());
                    }
                }
                // Occurs check
                if self.occurs_in(*var, ty) {
                    return Err(TypeError::InfiniteType(*var, ty.clone()));
                }
                self.substitution.extend(*var, ty.clone());
                Ok(())
            }

            // Unknown unifies with anything
            (Type::Unknown, _) | (_, Type::Unknown) => Ok(()),

            // Primitive types must match exactly
            (Type::Primitive(p1), Type::Primitive(p2)) if p1 == p2 => Ok(()),

            // Array types
            (
                Type::Array {
                    element: e1,
                    size: s1,
                },
                Type::Array {
                    element: e2,
                    size: s2,
                },
            ) => {
                if s1 != s2 {
                    return Err(TypeError::UnificationFailure(t1.clone(), t2.clone()));
                }
                self.unify(e1, e2)
            }

            // Tuple types
            (Type::Tuple(ts1), Type::Tuple(ts2)) => {
                if ts1.len() != ts2.len() {
                    return Err(TypeError::UnificationFailure(t1.clone(), t2.clone()));
                }
                for (t1, t2) in ts1.iter().zip(ts2.iter()) {
                    self.unify(t1, t2)?;
                }
                Ok(())
            }

            // Function types
            (Type::Function(f1), Type::Function(f2)) => {
                if f1.params.len() != f2.params.len() || f1.is_async != f2.is_async {
                    return Err(TypeError::UnificationFailure(t1.clone(), t2.clone()));
                }
                for (p1, p2) in f1.params.iter().zip(f2.params.iter()) {
                    self.unify(&p1.ty, &p2.ty)?;
                }
                self.unify(&f1.return_type, &f2.return_type)
            }

            // Reference types
            (
                Type::Ref {
                    inner: i1,
                    mutable: m1,
                },
                Type::Ref {
                    inner: i2,
                    mutable: m2,
                },
            ) => {
                if m1 != m2 {
                    return Err(TypeError::UnificationFailure(t1.clone(), t2.clone()));
                }
                self.unify(i1, i2)
            }

            // Rc types
            (Type::Rc(i1), Type::Rc(i2)) => self.unify(i1, i2),

            // Future types
            (Type::Future(i1), Type::Future(i2)) => self.unify(i1, i2),

            // Applied generic types
            (
                Type::Applied {
                    base: b1,
                    args: a1,
                },
                Type::Applied {
                    base: b2,
                    args: a2,
                },
            ) => {
                if a1.len() != a2.len() {
                    return Err(TypeError::UnificationFailure(t1.clone(), t2.clone()));
                }
                self.unify(b1, b2)?;
                for (arg1, arg2) in a1.iter().zip(a2.iter()) {
                    self.unify(arg1, arg2)?;
                }
                Ok(())
            }

            // HTTP status types
            (Type::HttpStatus(h1), Type::HttpStatus(h2)) => {
                if self.http_status_compatible(&h1.kind, &h2.kind) {
                    Ok(())
                } else {
                    Err(TypeError::UnificationFailure(t1.clone(), t2.clone()))
                }
            }

            // Never type unifies with anything (it's a bottom type)
            (Type::Never, _) | (_, Type::Never) => Ok(()),

            // Unit types
            (Type::Unit, Type::Unit) => Ok(()),

            // Failed to unify
            _ => Err(TypeError::UnificationFailure(t1.clone(), t2.clone())),
        }
    }

    /// Check if a type variable occurs in a type.
    fn occurs_in(&self, var: TypeVarId, ty: &Type) -> bool {
        match ty {
            Type::TypeVar(v) => *v == var,
            Type::Array { element, .. } => self.occurs_in(var, element),
            Type::Slice { element } => self.occurs_in(var, element),
            Type::Tuple(types) => types.iter().any(|t| self.occurs_in(var, t)),
            Type::Function(func) => {
                func.params.iter().any(|p| self.occurs_in(var, &p.ty))
                    || self.occurs_in(var, &func.return_type)
            }
            Type::Applied { base, args } => {
                self.occurs_in(var, base) || args.iter().any(|t| self.occurs_in(var, t))
            }
            Type::Ref { inner, .. } => self.occurs_in(var, inner),
            Type::Rc(inner) => self.occurs_in(var, inner),
            Type::Future(inner) => self.occurs_in(var, inner),
            Type::Prototype(proto) => {
                proto
                    .parent
                    .as_ref()
                    .is_some_and(|p| self.occurs_in(var, p))
                    || proto.members.iter().any(|(_, t)| self.occurs_in(var, t))
            }
            Type::Primitive(_)
            | Type::HttpStatus(_)
            | Type::Never
            | Type::Unknown
            | Type::Unit => false,
        }
    }

    /// Check if two HTTP status types are compatible.
    fn http_status_compatible(&self, k1: &HttpStatusTypeKind, k2: &HttpStatusTypeKind) -> bool {
        match (k1, k2) {
            (HttpStatusTypeKind::Any, _) | (_, HttpStatusTypeKind::Any) => true,
            (HttpStatusTypeKind::Exact(c1), HttpStatusTypeKind::Exact(c2)) => c1 == c2,
            (HttpStatusTypeKind::Exact(code), HttpStatusTypeKind::Category(cat))
            | (HttpStatusTypeKind::Category(cat), HttpStatusTypeKind::Exact(code)) => {
                cat.contains(*code)
            }
            (HttpStatusTypeKind::Category(c1), HttpStatusTypeKind::Category(c2)) => c1 == c2,
            (HttpStatusTypeKind::Exact(code), HttpStatusTypeKind::Range { start, end })
            | (HttpStatusTypeKind::Range { start, end }, HttpStatusTypeKind::Exact(code)) => {
                *code >= *start && *code <= *end
            }
            (
                HttpStatusTypeKind::Range {
                    start: s1,
                    end: e1,
                },
                HttpStatusTypeKind::Range {
                    start: s2,
                    end: e2,
                },
            ) => s1 == s2 && e1 == e2,
            _ => false,
        }
    }

    /// Instantiate a type scheme with fresh type variables.
    pub fn instantiate(&mut self, scheme: &TypeScheme) -> Type {
        let mut subst = Substitution::new();
        for var in &scheme.vars {
            subst.extend(*var, self.fresh_var());
        }
        subst.apply(&scheme.ty)
    }

    /// Generalize a type into a type scheme by abstracting over free variables.
    pub fn generalize(&self, ty: &Type) -> TypeScheme {
        let ty = self.apply(ty);
        let free_in_env = self.env.free_type_vars();
        let free_in_ty = self.free_type_vars(&ty);

        let vars: Vec<_> = free_in_ty
            .into_iter()
            .filter(|v| !free_in_env.contains(v))
            .collect();

        TypeScheme { vars, ty }
    }

    /// Get free type variables in a type.
    fn free_type_vars(&self, ty: &Type) -> Vec<TypeVarId> {
        let mut vars = Vec::new();
        self.collect_free_vars(ty, &mut vars);
        vars
    }

    fn collect_free_vars(&self, ty: &Type, vars: &mut Vec<TypeVarId>) {
        match ty {
            Type::TypeVar(var) => {
                if let Some(replacement) = self.substitution.lookup(*var) {
                    self.collect_free_vars(replacement, vars);
                } else if !vars.contains(var) {
                    vars.push(*var);
                }
            }
            Type::Array { element, .. } | Type::Slice { element } => {
                self.collect_free_vars(element, vars);
            }
            Type::Tuple(types) => {
                for t in types {
                    self.collect_free_vars(t, vars);
                }
            }
            Type::Function(func) => {
                for p in &func.params {
                    self.collect_free_vars(&p.ty, vars);
                }
                self.collect_free_vars(&func.return_type, vars);
            }
            Type::Applied { base, args } => {
                self.collect_free_vars(base, vars);
                for arg in args {
                    self.collect_free_vars(arg, vars);
                }
            }
            Type::Ref { inner, .. } | Type::Rc(inner) | Type::Future(inner) => {
                self.collect_free_vars(inner, vars);
            }
            Type::Prototype(proto) => {
                if let Some(parent) = &proto.parent {
                    self.collect_free_vars(parent, vars);
                }
                for (_, t) in &proto.members {
                    self.collect_free_vars(t, vars);
                }
            }
            _ => {}
        }
    }
}

impl Default for InferenceContext {
    fn default() -> Self {
        Self::new()
    }
}

/// A type scheme (polymorphic type).
#[derive(Clone, Debug)]
pub struct TypeScheme {
    /// Bound type variables.
    pub vars: Vec<TypeVarId>,
    /// The underlying type.
    pub ty: Type,
}

impl TypeScheme {
    /// Create a monomorphic type scheme (no bound variables).
    pub fn mono(ty: Type) -> Self {
        Self { vars: vec![], ty }
    }

    /// Create a polymorphic type scheme.
    pub fn poly(vars: Vec<TypeVarId>, ty: Type) -> Self {
        Self { vars, ty }
    }
}

/// Type environment mapping names to type schemes.
#[derive(Clone, Debug, Default)]
pub struct TypeEnvironment {
    scopes: Vec<HashMap<String, TypeScheme>>,
}

impl TypeEnvironment {
    pub fn new() -> Self {
        Self {
            scopes: vec![HashMap::new()],
        }
    }

    /// Enter a new scope.
    pub fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    /// Exit the current scope.
    pub fn pop_scope(&mut self) {
        if self.scopes.len() > 1 {
            self.scopes.pop();
        }
    }

    /// Define a variable in the current scope.
    pub fn define(&mut self, name: String, scheme: TypeScheme) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name, scheme);
        }
    }

    /// Look up a variable.
    pub fn lookup(&self, name: &str) -> Option<&TypeScheme> {
        for scope in self.scopes.iter().rev() {
            if let Some(scheme) = scope.get(name) {
                return Some(scheme);
            }
        }
        None
    }

    /// Get all free type variables in the environment.
    pub fn free_type_vars(&self) -> Vec<TypeVarId> {
        let mut vars = Vec::new();
        for scope in &self.scopes {
            for scheme in scope.values() {
                for var in &scheme.vars {
                    if !vars.contains(var) {
                        vars.push(*var);
                    }
                }
            }
        }
        vars
    }
}

/// Built-in type definitions.
impl InferenceContext {
    /// Initialize the context with built-in types.
    pub fn with_builtins() -> Self {
        let mut ctx = Self::new();

        // Add primitive type constructors
        let primitives = [
            ("i8", PrimitiveType::I8),
            ("i16", PrimitiveType::I16),
            ("i32", PrimitiveType::I32),
            ("i64", PrimitiveType::I64),
            ("i128", PrimitiveType::I128),
            ("u8", PrimitiveType::U8),
            ("u16", PrimitiveType::U16),
            ("u32", PrimitiveType::U32),
            ("u64", PrimitiveType::U64),
            ("u128", PrimitiveType::U128),
            ("f32", PrimitiveType::F32),
            ("f64", PrimitiveType::F64),
            ("bool", PrimitiveType::Bool),
            ("char", PrimitiveType::Char),
            ("String", PrimitiveType::Str),
        ];

        for (name, prim) in primitives {
            ctx.env_mut()
                .define(name.to_string(), TypeScheme::mono(Type::Primitive(prim)));
        }

        // Add print function
        ctx.env_mut().define(
            "print".to_string(),
            TypeScheme::mono(Type::Function(FunctionType {
                params: vec![ParamType {
                    ty: Type::Primitive(PrimitiveType::Str),
                    dispatch_role: DispatchRole::Static,
                }],
                return_type: Box::new(Type::Unit),
                is_async: false,
            })),
        );

        ctx
    }

    /// Create an HTTP status type from a code.
    pub fn http_status_exact(code: u16) -> Type {
        Type::HttpStatus(HttpStatusType {
            kind: HttpStatusTypeKind::Exact(code),
        })
    }

    /// Create an HTTP status category type.
    pub fn http_status_category(category: StatusCategory) -> Type {
        Type::HttpStatus(HttpStatusType {
            kind: HttpStatusTypeKind::Category(category),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unify_same_types() {
        let mut ctx = InferenceContext::new();
        assert!(ctx
            .unify(
                &Type::Primitive(PrimitiveType::I32),
                &Type::Primitive(PrimitiveType::I32)
            )
            .is_ok());
    }

    #[test]
    fn test_unify_different_primitives_fails() {
        let mut ctx = InferenceContext::new();
        assert!(ctx
            .unify(
                &Type::Primitive(PrimitiveType::I32),
                &Type::Primitive(PrimitiveType::I64)
            )
            .is_err());
    }

    #[test]
    fn test_unify_type_var() {
        let mut ctx = InferenceContext::new();
        let var = ctx.fresh_var();
        let i32_ty = Type::Primitive(PrimitiveType::I32);

        assert!(ctx.unify(&var, &i32_ty).is_ok());

        // After unification, applying substitution should give i32
        let result = ctx.apply(&var);
        assert_eq!(result, i32_ty);
    }

    #[test]
    fn test_unify_functions() {
        let mut ctx = InferenceContext::new();

        let f1 = Type::Function(FunctionType {
            params: vec![ParamType {
                ty: Type::Primitive(PrimitiveType::I32),
                dispatch_role: DispatchRole::Static,
            }],
            return_type: Box::new(Type::Primitive(PrimitiveType::I32)),
            is_async: false,
        });

        let var = ctx.fresh_var();
        let f2 = Type::Function(FunctionType {
            params: vec![ParamType {
                ty: var.clone(),
                dispatch_role: DispatchRole::Static,
            }],
            return_type: Box::new(var.clone()),
            is_async: false,
        });

        assert!(ctx.unify(&f1, &f2).is_ok());
        assert_eq!(ctx.apply(&var), Type::Primitive(PrimitiveType::I32));
    }

    #[test]
    fn test_occurs_check() {
        let mut ctx = InferenceContext::new();
        let var = ctx.fresh_var();

        // Try to unify T with [T] - should fail with occurs check
        let array_of_var = Type::Array {
            element: Box::new(var.clone()),
            size: None,
        };

        assert!(matches!(
            ctx.unify(&var, &array_of_var),
            Err(TypeError::InfiniteType(_, _))
        ));
    }

    #[test]
    fn test_http_status_unification() {
        let mut ctx = InferenceContext::new();

        // 200 should unify with 2xx
        let exact_200 = InferenceContext::http_status_exact(200);
        let success_category = InferenceContext::http_status_category(StatusCategory::Success);

        assert!(ctx.unify(&exact_200, &success_category).is_ok());
    }

    #[test]
    fn test_type_scheme_instantiation() {
        let mut ctx = InferenceContext::new();

        // Create a polymorphic identity function: forall a. a -> a
        let a = TypeVarId(100);
        let scheme = TypeScheme::poly(
            vec![a],
            Type::Function(FunctionType {
                params: vec![ParamType {
                    ty: Type::TypeVar(a),
                    dispatch_role: DispatchRole::Static,
                }],
                return_type: Box::new(Type::TypeVar(a)),
                is_async: false,
            }),
        );

        // Instantiate twice - should get different fresh variables
        let inst1 = ctx.instantiate(&scheme);
        let inst2 = ctx.instantiate(&scheme);

        // The two instances should have different type variables
        if let (Type::Function(f1), Type::Function(f2)) = (&inst1, &inst2) {
            assert_ne!(f1.params[0].ty, f2.params[0].ty);
        } else {
            panic!("Expected function types");
        }
    }
}
