//! Type checking for HIR.
//!
//! This module type-checks HIR expressions and fills in type annotations
//! using the inference engine from wsharp_types.

use crate::{
    DefId, HirBinaryOp, HirBlock, HirBody, HirExpr, HirExprKind, HirFunction, HirItem, HirLiteral,
    HirMatchArm, HirModule, HirPattern, HirPrototype, HirStmt, HirTypeAlias, HirUnaryOp, LocalId,
};
use std::collections::HashMap;
use wsharp_lexer::Span;
use wsharp_types::{
    DispatchRole, FunctionType, HttpStatusType, HttpStatusTypeKind, InferenceContext, ParamType,
    PrimitiveType, StatusCategory, Type, TypeError, TypeResult, TypeScheme,
};

/// Type checker for HIR.
pub struct TypeChecker {
    /// The inference context.
    ctx: InferenceContext,

    /// Function signatures.
    functions: HashMap<DefId, Type>,

    /// Local variable types.
    locals: HashMap<LocalId, Type>,

    /// Collected errors.
    errors: Vec<TypeCheckError>,

    /// Current function return type (for checking returns).
    current_return_type: Option<Type>,
}

/// Errors during type checking.
#[derive(Clone, Debug)]
pub enum TypeCheckError {
    TypeError(TypeError),
    UnexpectedType {
        expected: Type,
        found: Type,
        span: Span,
    },
    MissingReturnType {
        span: Span,
    },
    BreakOutsideLoop {
        span: Span,
    },
    ContinueOutsideLoop {
        span: Span,
    },
    ReturnOutsideFunction {
        span: Span,
    },
    InvalidOperandType {
        op: String,
        ty: Type,
        span: Span,
    },
}

impl From<TypeError> for TypeCheckError {
    fn from(e: TypeError) -> Self {
        TypeCheckError::TypeError(e)
    }
}

impl TypeChecker {
    pub fn new() -> Self {
        Self {
            ctx: InferenceContext::with_builtins(),
            functions: HashMap::new(),
            locals: HashMap::new(),
            errors: Vec::new(),
            current_return_type: None,
        }
    }

    /// Type check a module.
    pub fn check_module(&mut self, module: &mut HirModule) {
        // First pass: collect function signatures
        for item in &module.items {
            if let HirItem::Function(func) = item {
                self.register_function(func);
            }
        }

        // Second pass: type check function bodies
        for item in &mut module.items {
            match item {
                HirItem::Function(func) => self.check_function(func),
                HirItem::Prototype(proto) => self.check_prototype(proto),
                HirItem::TypeAlias(_) => {}
                HirItem::Module(submodule) => self.check_module(submodule),
            }
        }
    }

    /// Register a function's signature.
    fn register_function(&mut self, func: &HirFunction) {
        let param_types: Vec<ParamType> = func
            .params
            .iter()
            .map(|p| ParamType {
                ty: if p.ty == Type::Unknown {
                    self.ctx.fresh_var()
                } else {
                    p.ty.clone()
                },
                dispatch_role: DispatchRole::Static,
            })
            .collect();

        let return_type = if func.return_type == Type::Unknown {
            self.ctx.fresh_var()
        } else {
            func.return_type.clone()
        };

        let func_type = Type::Function(FunctionType {
            params: param_types,
            return_type: Box::new(return_type),
            is_async: func.is_async,
        });

        self.functions.insert(func.id, func_type.clone());

        // Also add to the inference context environment
        self.ctx
            .env_mut()
            .define(func.name.clone(), TypeScheme::mono(func_type));
    }

    /// Type check a function.
    fn check_function(&mut self, func: &mut HirFunction) {
        // Get the function type
        let func_type = self.functions.get(&func.id).cloned();

        if let Some(Type::Function(ft)) = func_type {
            // Register parameter types
            for (param, param_type) in func.params.iter_mut().zip(ft.params.iter()) {
                param.ty = param_type.ty.clone();
                self.locals.insert(param.id, param_type.ty.clone());
            }

            // Set current return type
            self.current_return_type = Some(*ft.return_type.clone());

            // Type check body
            if let Some(body) = &mut func.body {
                let body_type = self.check_body(body);

                // Unify body type with return type
                if let Err(e) = self.ctx.unify(&body_type, &ft.return_type) {
                    self.errors.push(e.into());
                }
            }

            // Apply substitution to function type
            func.return_type = self.ctx.apply(&ft.return_type);
            for param in &mut func.params {
                param.ty = self.ctx.apply(&param.ty);
            }

            self.current_return_type = None;
        }
    }

    /// Type check a prototype.
    fn check_prototype(&mut self, proto: &mut HirPrototype) {
        // Type check constructor
        if let Some(constructor) = &mut proto.constructor {
            self.check_function(constructor);
        }

        // Type check methods
        for method in &mut proto.methods {
            self.check_function(method);
        }

        // Type check field default values
        for field in &mut proto.fields {
            if let Some(default) = &mut field.default {
                let default_type = self.check_expr(default);
                if field.ty != Type::Unknown {
                    if let Err(e) = self.ctx.unify(&default_type, &field.ty) {
                        self.errors.push(e.into());
                    }
                } else {
                    field.ty = default_type;
                }
            }
        }
    }

    /// Type check a function body.
    fn check_body(&mut self, body: &mut HirBody) -> Type {
        let result = self.check_expr(&mut body.expr);

        // Update HIR locals with their inferred types
        for local in &mut body.locals {
            if let Some(ty) = self.locals.get(&local.id) {
                local.ty = self.ctx.apply(ty);
            }
        }

        result
    }

    /// Type check an expression.
    fn check_expr(&mut self, expr: &mut HirExpr) -> Type {
        let ty = match &mut expr.kind {
            HirExprKind::Literal(lit) => self.check_literal(lit),

            HirExprKind::Local(id) => {
                self.locals.get(id).cloned().unwrap_or(Type::Unknown)
            }

            HirExprKind::Global(id) => {
                self.functions.get(id).cloned().unwrap_or(Type::Unknown)
            }

            HirExprKind::Binary { op, left, right } => {
                let left_ty = self.check_expr(left);
                let right_ty = self.check_expr(right);
                self.check_binary_op(*op, &left_ty, &right_ty, expr.span)
            }

            HirExprKind::Unary { op, operand } => {
                let operand_ty = self.check_expr(operand);
                self.check_unary_op(*op, &operand_ty, expr.span)
            }

            HirExprKind::Call { callee, args } => {
                let callee_ty = self.check_expr(callee);
                let arg_types: Vec<Type> = args.iter_mut().map(|a| self.check_expr(a)).collect();
                self.check_call(&callee_ty, &arg_types, expr.span)
            }

            HirExprKind::MethodCall { receiver, method: _, args } => {
                let receiver_ty = self.check_expr(receiver);
                let _arg_types: Vec<Type> = args.iter_mut().map(|a| self.check_expr(a)).collect();

                // Method calls are desugared to regular function calls during lowering.
                // If we reach here, it means the method wasn't resolved.
                // For now, infer from receiver type if it's a prototype.
                match receiver_ty {
                    Type::Prototype(_) => {
                        // TODO: Look up method in prototype and return its return type
                        Type::Unknown
                    }
                    _ => Type::Unknown,
                }
            }

            HirExprKind::Field { object, field } => {
                let object_ty = self.check_expr(object);
                self.check_field_access(&object_ty, field, expr.span)
            }

            HirExprKind::Index { object, index } => {
                let object_ty = self.check_expr(object);
                let index_ty = self.check_expr(index);
                self.check_index_access(&object_ty, &index_ty, expr.span)
            }

            HirExprKind::Block(block) => self.check_block(block),

            HirExprKind::If { condition, then_branch, else_branch } => {
                let cond_ty = self.check_expr(condition);
                if let Err(e) = self.ctx.unify(&cond_ty, &Type::Primitive(PrimitiveType::Bool)) {
                    self.errors.push(e.into());
                }

                let then_ty = self.check_expr(then_branch);
                let else_ty = self.check_expr(else_branch);

                if let Err(e) = self.ctx.unify(&then_ty, &else_ty) {
                    self.errors.push(e.into());
                }

                self.ctx.apply(&then_ty)
            }

            HirExprKind::Loop { body } => {
                self.check_expr(body);
                // Loop returns unit unless broken with a value
                Type::Unit
            }

            HirExprKind::Break { value } => {
                if let Some(val) = value {
                    self.check_expr(val);
                }
                Type::Never
            }

            HirExprKind::Continue => Type::Never,

            HirExprKind::Return { value } => {
                let return_ty = if let Some(val) = value {
                    self.check_expr(val)
                } else {
                    Type::Unit
                };

                if let Some(expected) = &self.current_return_type {
                    if let Err(e) = self.ctx.unify(&return_ty, expected) {
                        self.errors.push(e.into());
                    }
                }

                Type::Never
            }

            HirExprKind::Let { local, init, body } => {
                let init_ty = self.check_expr(init);
                self.locals.insert(*local, init_ty);
                self.check_expr(body)
            }

            HirExprKind::Assign { target, value } => {
                let target_ty = self.check_expr(target);
                let value_ty = self.check_expr(value);

                if let Err(e) = self.ctx.unify(&target_ty, &value_ty) {
                    self.errors.push(e.into());
                }

                Type::Unit
            }

            HirExprKind::Tuple(elements) => {
                let element_types: Vec<Type> = elements
                    .iter_mut()
                    .map(|e| self.check_expr(e))
                    .collect();
                Type::Tuple(element_types)
            }

            HirExprKind::Array(elements) => {
                if elements.is_empty() {
                    Type::Array {
                        element: Box::new(self.ctx.fresh_var()),
                        size: Some(0),
                    }
                } else {
                    let first_ty = self.check_expr(&mut elements[0]);
                    for elem in elements.iter_mut().skip(1) {
                        let elem_ty = self.check_expr(elem);
                        if let Err(e) = self.ctx.unify(&first_ty, &elem_ty) {
                            self.errors.push(e.into());
                        }
                    }
                    Type::Array {
                        element: Box::new(self.ctx.apply(&first_ty)),
                        size: Some(elements.len()),
                    }
                }
            }

            HirExprKind::Object { prototype: _, fields } => {
                let field_types: Vec<(String, Type)> = fields
                    .iter_mut()
                    .map(|(name, expr)| (name.clone(), self.check_expr(expr)))
                    .collect();

                Type::Prototype(wsharp_types::PrototypeType {
                    name: None,
                    parent: None,
                    members: field_types,
                })
            }

            HirExprKind::Lambda { params, body, captures: _ } => {
                // Register parameter types
                for param in params.iter_mut() {
                    let ty = if param.ty == Type::Unknown {
                        self.ctx.fresh_var()
                    } else {
                        param.ty.clone()
                    };
                    self.locals.insert(param.id, ty.clone());
                    param.ty = ty;
                }

                let body_ty = self.check_expr(body);

                let param_types: Vec<ParamType> = params
                    .iter()
                    .map(|p| ParamType {
                        ty: self.ctx.apply(&p.ty),
                        dispatch_role: DispatchRole::Static,
                    })
                    .collect();

                Type::Function(FunctionType {
                    params: param_types,
                    return_type: Box::new(self.ctx.apply(&body_ty)),
                    is_async: false,
                })
            }

            HirExprKind::Await(inner) => {
                let inner_ty = self.check_expr(inner);
                match inner_ty {
                    Type::Future(result_ty) => *result_ty,
                    _ => {
                        self.errors.push(TypeCheckError::UnexpectedType {
                            expected: Type::Future(Box::new(Type::Unknown)),
                            found: inner_ty,
                            span: expr.span,
                        });
                        Type::Unknown
                    }
                }
            }

            HirExprKind::Cast { expr: inner, target_ty } => {
                self.check_expr(inner);
                target_ty.clone()
            }

            HirExprKind::Match { scrutinee, arms } => {
                let scrutinee_ty = self.check_expr(scrutinee);
                let mut result_ty: Option<Type> = None;

                for arm in arms.iter_mut() {
                    self.check_pattern(&arm.pattern, &scrutinee_ty);
                    if let Some(guard) = &mut arm.guard {
                        let guard_ty = self.check_expr(guard);
                        if let Err(e) = self.ctx.unify(&guard_ty, &Type::Primitive(PrimitiveType::Bool)) {
                            self.errors.push(e.into());
                        }
                    }
                    let arm_ty = self.check_expr(&mut arm.body);

                    if let Some(ref expected) = result_ty {
                        if let Err(e) = self.ctx.unify(&arm_ty, expected) {
                            self.errors.push(e.into());
                        }
                    } else {
                        result_ty = Some(arm_ty);
                    }
                }

                result_ty.map(|t| self.ctx.apply(&t)).unwrap_or(Type::Never)
            }

            HirExprKind::HttpStatus(code) => {
                Type::HttpStatus(HttpStatusType {
                    kind: HttpStatusTypeKind::Exact(*code),
                })
            }
        };

        expr.ty = ty.clone();
        ty
    }

    /// Type check a block.
    fn check_block(&mut self, block: &mut HirBlock) -> Type {
        for stmt in &mut block.stmts {
            self.check_stmt(stmt);
        }

        if let Some(expr) = &mut block.expr {
            self.check_expr(expr)
        } else {
            Type::Unit
        }
    }

    /// Type check a statement.
    fn check_stmt(&mut self, stmt: &mut HirStmt) {
        match stmt {
            HirStmt::Expr(expr) => {
                self.check_expr(expr);
            }
            HirStmt::Let { local, init } => {
                let init_ty = self.check_expr(init);
                self.locals.insert(*local, init_ty);
            }
        }
    }

    /// Type check a pattern.
    fn check_pattern(&mut self, pattern: &HirPattern, expected: &Type) {
        match pattern {
            HirPattern::Wildcard => {}
            HirPattern::Binding(local) => {
                self.locals.insert(*local, expected.clone());
            }
            HirPattern::Literal(lit) => {
                let lit_ty = self.check_literal(lit);
                if let Err(e) = self.ctx.unify(&lit_ty, expected) {
                    self.errors.push(e.into());
                }
            }
            HirPattern::Tuple(patterns) => {
                if let Type::Tuple(types) = expected {
                    for (pat, ty) in patterns.iter().zip(types.iter()) {
                        self.check_pattern(pat, ty);
                    }
                }
            }
            HirPattern::HttpStatus(_) => {
                // HTTP status patterns match HTTP status types
                if let Err(e) = self.ctx.unify(expected, &Type::HttpStatus(HttpStatusType {
                    kind: HttpStatusTypeKind::Any,
                })) {
                    self.errors.push(e.into());
                }
            }
            HirPattern::Or(patterns) => {
                for pat in patterns {
                    self.check_pattern(pat, expected);
                }
            }
        }
    }

    /// Get the type of a literal.
    fn check_literal(&self, lit: &HirLiteral) -> Type {
        match lit {
            HirLiteral::Int(_) => Type::Primitive(PrimitiveType::I64),
            HirLiteral::Float(_) => Type::Primitive(PrimitiveType::F64),
            HirLiteral::Bool(_) => Type::Primitive(PrimitiveType::Bool),
            HirLiteral::Char(_) => Type::Primitive(PrimitiveType::Char),
            HirLiteral::String(_) => Type::Primitive(PrimitiveType::Str),
            HirLiteral::Unit => Type::Unit,
        }
    }

    /// Type check a binary operation.
    fn check_binary_op(&mut self, op: HirBinaryOp, left: &Type, right: &Type, span: Span) -> Type {
        // Unify operand types
        if let Err(e) = self.ctx.unify(left, right) {
            self.errors.push(e.into());
        }

        let operand_ty = self.ctx.apply(left);

        match op {
            // Arithmetic operations return the same type
            HirBinaryOp::Add | HirBinaryOp::Sub | HirBinaryOp::Mul | HirBinaryOp::Div | HirBinaryOp::Rem => {
                match &operand_ty {
                    Type::Primitive(p) if p.is_numeric() => operand_ty,
                    _ => {
                        self.errors.push(TypeCheckError::InvalidOperandType {
                            op: format!("{:?}", op),
                            ty: operand_ty.clone(),
                            span,
                        });
                        Type::Unknown
                    }
                }
            }

            // Comparison operations return bool
            HirBinaryOp::Eq | HirBinaryOp::Ne | HirBinaryOp::Lt | HirBinaryOp::Le | HirBinaryOp::Gt | HirBinaryOp::Ge => {
                Type::Primitive(PrimitiveType::Bool)
            }

            // Logical operations require bool
            HirBinaryOp::And | HirBinaryOp::Or => {
                if let Err(e) = self.ctx.unify(&operand_ty, &Type::Primitive(PrimitiveType::Bool)) {
                    self.errors.push(e.into());
                }
                Type::Primitive(PrimitiveType::Bool)
            }

            // Bitwise operations require integers
            HirBinaryOp::BitAnd | HirBinaryOp::BitOr | HirBinaryOp::BitXor | HirBinaryOp::Shl | HirBinaryOp::Shr => {
                match &operand_ty {
                    Type::Primitive(p) if p.is_integer() => operand_ty,
                    _ => {
                        self.errors.push(TypeCheckError::InvalidOperandType {
                            op: format!("{:?}", op),
                            ty: operand_ty.clone(),
                            span,
                        });
                        Type::Unknown
                    }
                }
            }
        }
    }

    /// Type check a unary operation.
    fn check_unary_op(&mut self, op: HirUnaryOp, operand: &Type, span: Span) -> Type {
        let operand_ty = self.ctx.apply(operand);

        match op {
            HirUnaryOp::Neg => {
                match &operand_ty {
                    Type::Primitive(p) if p.is_numeric() => operand_ty,
                    _ => {
                        self.errors.push(TypeCheckError::InvalidOperandType {
                            op: "negation".to_string(),
                            ty: operand_ty.clone(),
                            span,
                        });
                        Type::Unknown
                    }
                }
            }
            HirUnaryOp::Not => {
                if let Err(e) = self.ctx.unify(&operand_ty, &Type::Primitive(PrimitiveType::Bool)) {
                    self.errors.push(e.into());
                }
                Type::Primitive(PrimitiveType::Bool)
            }
            HirUnaryOp::BitNot => {
                match &operand_ty {
                    Type::Primitive(p) if p.is_integer() => operand_ty,
                    _ => {
                        self.errors.push(TypeCheckError::InvalidOperandType {
                            op: "bitwise not".to_string(),
                            ty: operand_ty.clone(),
                            span,
                        });
                        Type::Unknown
                    }
                }
            }
            HirUnaryOp::Ref => {
                Type::Ref {
                    inner: Box::new(operand_ty),
                    mutable: false,
                }
            }
            HirUnaryOp::RefMut => {
                Type::Ref {
                    inner: Box::new(operand_ty),
                    mutable: true,
                }
            }
            HirUnaryOp::Deref => {
                match operand_ty {
                    Type::Ref { inner, .. } => *inner,
                    _ => {
                        self.errors.push(TypeCheckError::InvalidOperandType {
                            op: "dereference".to_string(),
                            ty: operand_ty,
                            span,
                        });
                        Type::Unknown
                    }
                }
            }
        }
    }

    /// Type check a function call.
    fn check_call(&mut self, callee: &Type, args: &[Type], span: Span) -> Type {
        let callee_ty = self.ctx.apply(callee);

        match callee_ty {
            Type::Function(ft) => {
                if ft.params.len() != args.len() {
                    self.errors.push(TypeCheckError::TypeError(TypeError::WrongArity {
                        expected: ft.params.len(),
                        found: args.len(),
                    }));
                    return Type::Unknown;
                }

                for (param, arg) in ft.params.iter().zip(args.iter()) {
                    if let Err(e) = self.ctx.unify(&param.ty, arg) {
                        self.errors.push(e.into());
                    }
                }

                self.ctx.apply(&ft.return_type)
            }
            Type::TypeVar(_) => {
                // Create a fresh function type and unify
                let ret_ty = self.ctx.fresh_var();
                let param_types: Vec<ParamType> = args
                    .iter()
                    .map(|t| ParamType {
                        ty: t.clone(),
                        dispatch_role: DispatchRole::Static,
                    })
                    .collect();

                let expected_fn = Type::Function(FunctionType {
                    params: param_types,
                    return_type: Box::new(ret_ty.clone()),
                    is_async: false,
                });

                if let Err(e) = self.ctx.unify(&callee_ty, &expected_fn) {
                    self.errors.push(e.into());
                }

                self.ctx.apply(&ret_ty)
            }
            _ => {
                self.errors.push(TypeCheckError::TypeError(TypeError::NotCallable(callee_ty)));
                Type::Unknown
            }
        }
    }

    /// Type check field access.
    fn check_field_access(&mut self, object: &Type, field: &str, span: Span) -> Type {
        let object_ty = self.ctx.apply(object);

        match object_ty {
            Type::Prototype(proto) => {
                for (name, ty) in &proto.members {
                    if name == field {
                        return ty.clone();
                    }
                }
                self.errors.push(TypeCheckError::TypeError(TypeError::NoSuchProperty {
                    ty: Type::Prototype(proto),
                    property: field.to_string(),
                }));
                Type::Unknown
            }
            Type::Tuple(types) => {
                // Handle tuple field access like .0, .1, etc.
                if let Ok(index) = field.parse::<usize>() {
                    if index < types.len() {
                        return types[index].clone();
                    }
                }
                self.errors.push(TypeCheckError::TypeError(TypeError::NoSuchProperty {
                    ty: Type::Tuple(types),
                    property: field.to_string(),
                }));
                Type::Unknown
            }
            _ => {
                self.errors.push(TypeCheckError::TypeError(TypeError::NoSuchProperty {
                    ty: object_ty,
                    property: field.to_string(),
                }));
                Type::Unknown
            }
        }
    }

    /// Type check index access.
    fn check_index_access(&mut self, object: &Type, index: &Type, span: Span) -> Type {
        let object_ty = self.ctx.apply(object);
        let index_ty = self.ctx.apply(index);

        // Index must be an integer type
        match &index_ty {
            Type::Primitive(p) if p.is_integer() => {}
            _ => {
                self.errors.push(TypeCheckError::InvalidOperandType {
                    op: "index".to_string(),
                    ty: index_ty,
                    span,
                });
            }
        }

        match object_ty {
            Type::Array { element, .. } | Type::Slice { element } => *element,
            _ => {
                self.errors.push(TypeCheckError::TypeError(TypeError::NoSuchProperty {
                    ty: object_ty,
                    property: "[index]".to_string(),
                }));
                Type::Unknown
            }
        }
    }

    /// Get collected errors.
    pub fn errors(&self) -> &[TypeCheckError] {
        &self.errors
    }

    /// Check if there are errors.
    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }

    /// Get the inference context.
    pub fn inference_context(&self) -> &InferenceContext {
        &self.ctx
    }
}

impl Default for TypeChecker {
    fn default() -> Self {
        Self::new()
    }
}
