//! AST to HIR lowering.
//!
//! This module transforms the AST into HIR by:
//! - Desugaring loops (for, while -> loop)
//! - Making implicit returns explicit
//! - Resolving names to their definitions
//! - Simplifying patterns

use crate::{
    DefId, DefinitionInfo, DefinitionKind, HirBinaryOp, HirBlock, HirBody, HirExpr, HirExprKind,
    HirField, HirFunction, HirHttpStatusPattern, HirId, HirItem, HirLiteral, HirLocal,
    HirMatchArm, HirModule, HirParam, HirPattern, HirPrototype, HirStmt, HirTypeAlias, HirUnaryOp,
    LocalId, PrototypeInfo, ResolveError, Resolver, Symbol,
};
use wsharp_ast::{
    BinaryOp, Block, Expr, ExprKind, FunctionDecl, HttpStatusCategory as AstHttpStatusCategory,
    HttpStatusPattern as AstHttpStatusPattern, Item, Literal, Parameter,
    Pattern as AstPattern, PatternKind, PrototypeDecl, PrototypeMember, SourceFile, Stmt,
    StmtKind, TypeAliasDecl, TypeExpr, TypeExprKind, UnaryOp,
};
use wsharp_types::{PrimitiveType, Type};

/// The lowering context.
pub struct LoweringContext {
    /// Name resolver.
    resolver: Resolver,

    /// Counter for HIR node IDs.
    next_hir_id: u32,

    /// Errors collected during lowering.
    errors: Vec<LoweringError>,

    /// Locals collected for the current function.
    current_locals: Vec<HirLocal>,
}

/// Errors during lowering.
#[derive(Clone, Debug)]
pub enum LoweringError {
    Resolve(ResolveError),
    UnsupportedFeature(String),
}

impl From<ResolveError> for LoweringError {
    fn from(e: ResolveError) -> Self {
        LoweringError::Resolve(e)
    }
}

impl LoweringContext {
    pub fn new() -> Self {
        Self {
            resolver: Resolver::new(),
            next_hir_id: 0,
            errors: Vec::new(),
            current_locals: Vec::new(),
        }
    }

    /// Generate a fresh HIR ID.
    fn fresh_hir_id(&mut self) -> HirId {
        let id = HirId(self.next_hir_id);
        self.next_hir_id += 1;
        id
    }

    /// Lower a type expression from AST to a Type.
    fn lower_type_expr(&self, type_expr: &TypeExpr) -> Type {
        match &type_expr.kind {
            TypeExprKind::Named { name, generics } => {
                // Handle primitive types
                match name.name.as_str() {
                    "i8" => Type::Primitive(PrimitiveType::I8),
                    "i16" => Type::Primitive(PrimitiveType::I16),
                    "i32" => Type::Primitive(PrimitiveType::I32),
                    "i64" => Type::Primitive(PrimitiveType::I64),
                    "i128" => Type::Primitive(PrimitiveType::I128),
                    "u8" => Type::Primitive(PrimitiveType::U8),
                    "u16" => Type::Primitive(PrimitiveType::U16),
                    "u32" => Type::Primitive(PrimitiveType::U32),
                    "u64" => Type::Primitive(PrimitiveType::U64),
                    "u128" => Type::Primitive(PrimitiveType::U128),
                    "f32" => Type::Primitive(PrimitiveType::F32),
                    "f64" => Type::Primitive(PrimitiveType::F64),
                    "bool" => Type::Primitive(PrimitiveType::Bool),
                    "char" => Type::Primitive(PrimitiveType::Char),
                    "str" | "String" => Type::Primitive(PrimitiveType::Str),
                    _ => {
                        // Custom types - could be generics or user-defined
                        if !generics.is_empty() {
                            Type::Applied {
                                base: Box::new(Type::Unknown),
                                args: generics.iter().map(|g| self.lower_type_expr(g)).collect(),
                            }
                        } else {
                            Type::Unknown // Will be resolved later
                        }
                    }
                }
            }
            TypeExprKind::Unit => Type::Unit,
            TypeExprKind::Never => Type::Never,
            TypeExprKind::Infer => Type::Unknown,
            TypeExprKind::Array { element, size } => Type::Array {
                element: Box::new(self.lower_type_expr(element)),
                size: *size,
            },
            TypeExprKind::Tuple(types) => {
                Type::Tuple(types.iter().map(|t| self.lower_type_expr(t)).collect())
            }
            TypeExprKind::Ref { inner, mutable } => Type::Ref {
                inner: Box::new(self.lower_type_expr(inner)),
                mutable: *mutable,
            },
            TypeExprKind::Optional(inner) => {
                // Optional<T> is represented as a union with Unit
                Type::Unknown // Simplified for now
            }
            TypeExprKind::Function {
                params,
                return_type,
                is_async,
            } => {
                let param_types: Vec<Type> = params.iter().map(|p| self.lower_type_expr(p)).collect();
                let ret = self.lower_type_expr(return_type);
                Type::Function(wsharp_types::FunctionType {
                    params: param_types
                        .into_iter()
                        .map(|ty| wsharp_types::ParamType {
                            ty,
                            dispatch_role: wsharp_types::DispatchRole::Static,
                        })
                        .collect(),
                    return_type: Box::new(ret),
                    is_async: *is_async,
                })
            }
            TypeExprKind::HttpStatus(_) => {
                // HTTP status types - simplified for now
                Type::Unknown
            }
            TypeExprKind::Prototype { .. } => {
                // Prototype types - would need full resolution
                Type::Unknown
            }
            TypeExprKind::Union(_) | TypeExprKind::Intersection(_) => {
                // Union/intersection types - simplified for now
                Type::Unknown
            }
        }
    }

    /// Lower a source file to HIR.
    pub fn lower_source_file(&mut self, source: &SourceFile) -> HirModule {
        // First pass: collect all declarations
        self.collect_declarations(source);

        // Second pass: lower all items
        let items = source
            .items
            .iter()
            .filter_map(|item| self.lower_item(item))
            .collect();

        HirModule {
            name: "main".to_string(),
            items,
            span: source.span,
        }
    }

    /// First pass: collect all top-level declarations.
    fn collect_declarations(&mut self, source: &SourceFile) {
        for item in &source.items {
            match item {
                Item::Function(func) => {
                    let def_id = self.resolver.fresh_def_id();
                    self.resolver.register_definition(
                        def_id,
                        DefinitionInfo {
                            name: func.name.name.clone(),
                            kind: DefinitionKind::Function,
                            span: func.span,
                        },
                    );
                    self.resolver.define_type(
                        func.name.name.clone(),
                        Symbol::Function {
                            id: def_id,
                            ty: Type::Unknown, // Will be set during type checking
                        },
                    );
                }
                Item::Prototype(proto) => {
                    let def_id = self.resolver.fresh_def_id();
                    self.resolver.register_definition(
                        def_id,
                        DefinitionInfo {
                            name: proto.name.name.clone(),
                            kind: DefinitionKind::Prototype,
                            span: proto.span,
                        },
                    );
                    self.resolver
                        .define_type(proto.name.name.clone(), Symbol::Prototype { id: def_id });
                }
                Item::TypeAlias(alias) => {
                    let def_id = self.resolver.fresh_def_id();
                    self.resolver.register_definition(
                        def_id,
                        DefinitionInfo {
                            name: alias.name.name.clone(),
                            kind: DefinitionKind::TypeAlias,
                            span: alias.span,
                        },
                    );
                    self.resolver.define_type(
                        alias.name.name.clone(),
                        Symbol::TypeAlias {
                            id: def_id,
                            ty: Type::Unknown,
                        },
                    );
                }
                Item::Module(module) => {
                    let def_id = self.resolver.fresh_def_id();
                    self.resolver.register_definition(
                        def_id,
                        DefinitionInfo {
                            name: module.name.name.clone(),
                            kind: DefinitionKind::Module,
                            span: module.span,
                        },
                    );
                    self.resolver
                        .define_type(module.name.name.clone(), Symbol::Module { id: def_id });
                }
                _ => {}
            }
        }
    }

    /// Lower an item.
    fn lower_item(&mut self, item: &Item) -> Option<HirItem> {
        match item {
            Item::Function(func) => Some(HirItem::Function(self.lower_function(func))),
            Item::Prototype(proto) => Some(HirItem::Prototype(self.lower_prototype(proto))),
            Item::TypeAlias(alias) => Some(HirItem::TypeAlias(self.lower_type_alias(alias))),
            Item::Module(module) => {
                // Recursively lower module items
                self.resolver.enter_scope();
                let items = module
                    .items
                    .iter()
                    .filter_map(|item| self.lower_item(item))
                    .collect();
                self.resolver.exit_scope();

                Some(HirItem::Module(HirModule {
                    name: module.name.name.clone(),
                    items,
                    span: module.span,
                }))
            }
            _ => None, // Skip imports, extensions for now
        }
    }

    /// Lower a function declaration.
    fn lower_function(&mut self, func: &FunctionDecl) -> HirFunction {
        let def_id = self
            .resolver
            .lookup_type(&func.name.name)
            .and_then(|s| match s {
                Symbol::Function { id, .. } => Some(*id),
                _ => None,
            })
            .unwrap_or_else(|| self.resolver.fresh_def_id());

        self.resolver.enter_function_scope();

        // Clear locals for this function
        self.current_locals.clear();

        // Lower parameters
        let params: Vec<HirParam> = func
            .params
            .iter()
            .map(|p| self.lower_param(p))
            .collect();

        // Lower body and collect locals
        let body = func.body.as_ref().map(|block| {
            let expr = self.lower_block(block);
            HirBody {
                locals: std::mem::take(&mut self.current_locals),
                expr,
            }
        });

        self.resolver.exit_scope();

        // Lower return type if specified, otherwise Unknown for inference
        let return_type = func
            .return_type
            .as_ref()
            .map(|t| self.lower_type_expr(t))
            .unwrap_or(Type::Unknown);

        HirFunction {
            id: def_id,
            name: func.name.name.clone(),
            params,
            return_type,
            body,
            is_async: func.is_async,
            span: func.span,
        }
    }

    /// Lower a parameter.
    fn lower_param(&mut self, param: &Parameter) -> HirParam {
        let id = self.resolver.fresh_local_id();

        // Lower the type annotation if specified
        let ty = param
            .ty
            .as_ref()
            .map(|t| self.lower_type_expr(t))
            .unwrap_or(Type::Unknown);

        let _ = self.resolver.define_variable(
            param.name.name.clone(),
            Symbol::Local {
                id,
                ty: ty.clone(),
                mutable: false,
            },
        );

        HirParam {
            id,
            name: param.name.name.clone(),
            ty,
            span: param.span,
        }
    }

    /// Lower a prototype declaration.
    fn lower_prototype(&mut self, proto: &PrototypeDecl) -> HirPrototype {
        let def_id = self
            .resolver
            .lookup_type(&proto.name.name)
            .and_then(|s| match s {
                Symbol::Prototype { id } => Some(*id),
                _ => None,
            })
            .unwrap_or_else(|| self.resolver.fresh_def_id());

        // Create prototype info for method resolution
        let mut proto_info = PrototypeInfo::new(proto.name.name.clone());

        // Resolve parent prototype if specified
        if let Some(ref parent) = proto.parent {
            if let TypeExprKind::Named { name, .. } = &parent.kind {
                if let Some(parent_id) = self.resolver.lookup_prototype_by_name(&name.name) {
                    proto_info.parent = Some(parent_id);
                }
            }
        }

        let mut fields = Vec::new();
        let mut methods = Vec::new();

        for (idx, member) in proto.members.iter().enumerate() {
            match member {
                PrototypeMember::Property {
                    name,
                    ty,
                    default,
                    span,
                } => {
                    let field_ty = ty
                        .as_ref()
                        .map(|t| self.lower_type_expr(t))
                        .unwrap_or(Type::Unknown);

                    // Register field in prototype info
                    proto_info.fields.insert(name.name.clone(), (idx, field_ty.clone()));

                    fields.push(HirField {
                        name: name.name.clone(),
                        ty: field_ty,
                        default: default.as_ref().map(|e| self.lower_expr(e)),
                        span: *span,
                    });
                }
                PrototypeMember::Method(func) => {
                    let method = self.lower_function(func);
                    // Register method in prototype info
                    proto_info.methods.insert(func.name.name.clone(), method.id);
                    methods.push(method);
                }
            }
        }

        // Register prototype info with resolver
        self.resolver.register_prototype(def_id, proto_info);

        let constructor = proto.constructor.as_ref().map(|c| self.lower_function(c));

        HirPrototype {
            id: def_id,
            name: proto.name.name.clone(),
            parent: None, // TODO: resolve parent DefId
            fields,
            methods,
            constructor,
            span: proto.span,
        }
    }

    /// Lower a type alias.
    fn lower_type_alias(&mut self, alias: &TypeAliasDecl) -> HirTypeAlias {
        let def_id = self
            .resolver
            .lookup_type(&alias.name.name)
            .and_then(|s| match s {
                Symbol::TypeAlias { id, .. } => Some(*id),
                _ => None,
            })
            .unwrap_or_else(|| self.resolver.fresh_def_id());

        HirTypeAlias {
            id: def_id,
            name: alias.name.name.clone(),
            ty: Type::Unknown, // Will be resolved during type checking
            span: alias.span,
        }
    }

    /// Lower a block.
    fn lower_block(&mut self, block: &Block) -> HirExpr {
        self.resolver.enter_scope();

        let mut stmts = Vec::new();
        for stmt in &block.stmts {
            if let Some(hir_stmt) = self.lower_stmt(stmt) {
                stmts.push(hir_stmt);
            }
        }

        let expr = block.expr.as_ref().map(|e| Box::new(self.lower_expr(e)));

        self.resolver.exit_scope();

        HirExpr {
            id: self.fresh_hir_id(),
            kind: HirExprKind::Block(HirBlock { stmts, expr }),
            ty: Type::Unknown,
            span: block.span,
        }
    }

    /// Lower a statement.
    fn lower_stmt(&mut self, stmt: &Stmt) -> Option<HirStmt> {
        match &stmt.kind {
            StmtKind::Expr(expr) => Some(HirStmt::Expr(self.lower_expr(expr))),
            StmtKind::Let {
                name,
                ty,
                value,
                mutable,
            } => {
                let local_id = self.resolver.fresh_local_id();

                // Lower the type annotation if present
                let local_ty = ty
                    .as_ref()
                    .map(|t| self.lower_type_expr(t))
                    .unwrap_or(Type::Unknown);

                let _ = self.resolver.define_variable(
                    name.name.clone(),
                    Symbol::Local {
                        id: local_id,
                        ty: local_ty.clone(),
                        mutable: *mutable,
                    },
                );

                // Record this local for the current function
                self.current_locals.push(HirLocal {
                    id: local_id,
                    name: name.name.clone(),
                    ty: local_ty,
                    mutable: *mutable,
                    span: stmt.span,
                });

                let init = value
                    .as_ref()
                    .map(|e| self.lower_expr(e))
                    .unwrap_or_else(|| HirExpr {
                        id: self.fresh_hir_id(),
                        kind: HirExprKind::Literal(HirLiteral::Unit),
                        ty: Type::Unit,
                        span: stmt.span,
                    });

                Some(HirStmt::Let {
                    local: local_id,
                    init,
                })
            }
            StmtKind::Empty => None,
        }
    }

    /// Lower an expression.
    fn lower_expr(&mut self, expr: &Expr) -> HirExpr {
        let kind = match &expr.kind {
            ExprKind::Literal(lit) => HirExprKind::Literal(self.lower_literal(lit)),

            ExprKind::Ident(name) => {
                // First check local variables
                if let Some(symbol) = self.resolver.lookup_variable(&name.name) {
                    match symbol {
                        Symbol::Local { id, .. } => HirExprKind::Local(*id),
                        Symbol::Function { id, .. } => HirExprKind::Global(*id),
                        _ => HirExprKind::Literal(HirLiteral::Unit), // Error case
                    }
                // Then check type scope for functions
                } else if let Some(symbol) = self.resolver.lookup_type(&name.name) {
                    match symbol {
                        Symbol::Function { id, .. } => HirExprKind::Global(*id),
                        _ => {
                            self.errors.push(LoweringError::Resolve(
                                ResolveError::UndefinedVariable {
                                    name: name.name.clone(),
                                    span: expr.span,
                                },
                            ));
                            HirExprKind::Literal(HirLiteral::Unit)
                        }
                    }
                } else {
                    self.errors.push(LoweringError::Resolve(
                        ResolveError::UndefinedVariable {
                            name: name.name.clone(),
                            span: expr.span,
                        },
                    ));
                    HirExprKind::Literal(HirLiteral::Unit)
                }
            }

            ExprKind::Binary { op, left, right } => HirExprKind::Binary {
                op: self.lower_binary_op(*op),
                left: Box::new(self.lower_expr(left)),
                right: Box::new(self.lower_expr(right)),
            },

            ExprKind::Unary { op, operand } => HirExprKind::Unary {
                op: self.lower_unary_op(*op),
                operand: Box::new(self.lower_expr(operand)),
            },

            ExprKind::Call { callee, args } => HirExprKind::Call {
                callee: Box::new(self.lower_expr(callee)),
                args: args.iter().map(|a| self.lower_expr(a)).collect(),
            },

            ExprKind::MethodCall {
                receiver,
                method,
                args,
            } => {
                // For now, desugar method calls to regular calls
                // receiver.method(args) -> method(receiver, args)
                let all_args = std::iter::once(self.lower_expr(receiver))
                    .chain(args.iter().map(|a| self.lower_expr(a)))
                    .collect();

                HirExprKind::Call {
                    callee: Box::new(HirExpr {
                        id: self.fresh_hir_id(),
                        kind: HirExprKind::Global(DefId(0)), // TODO: resolve method
                        ty: Type::Unknown,
                        span: method.span,
                    }),
                    args: all_args,
                }
            }

            ExprKind::PropertyAccess { object, property } => HirExprKind::Field {
                object: Box::new(self.lower_expr(object)),
                field: property.name.clone(),
            },

            ExprKind::IndexAccess { object, index } => HirExprKind::Index {
                object: Box::new(self.lower_expr(object)),
                index: Box::new(self.lower_expr(index)),
            },

            ExprKind::If {
                condition,
                then_branch,
                else_branch,
            } => {
                let then_expr = self.lower_block(then_branch);
                let else_expr = else_branch
                    .as_ref()
                    .map(|b| self.lower_block(b))
                    .unwrap_or_else(|| HirExpr {
                        id: self.fresh_hir_id(),
                        kind: HirExprKind::Literal(HirLiteral::Unit),
                        ty: Type::Unit,
                        span: expr.span,
                    });

                HirExprKind::If {
                    condition: Box::new(self.lower_expr(condition)),
                    then_branch: Box::new(then_expr),
                    else_branch: Box::new(else_expr),
                }
            }

            ExprKind::While { condition, body } => {
                // Desugar while to loop + if + break
                // while cond { body } -> loop { if !cond { break } body }
                self.resolver.enter_loop_scope();

                let break_expr = HirExpr {
                    id: self.fresh_hir_id(),
                    kind: HirExprKind::Break { value: None },
                    ty: Type::Never,
                    span: expr.span,
                };

                let not_cond = HirExpr {
                    id: self.fresh_hir_id(),
                    kind: HirExprKind::Unary {
                        op: HirUnaryOp::Not,
                        operand: Box::new(self.lower_expr(condition)),
                    },
                    ty: Type::Unknown,
                    span: condition.span,
                };

                let check_and_break = HirExpr {
                    id: self.fresh_hir_id(),
                    kind: HirExprKind::If {
                        condition: Box::new(not_cond),
                        then_branch: Box::new(break_expr),
                        else_branch: Box::new(HirExpr {
                            id: self.fresh_hir_id(),
                            kind: HirExprKind::Literal(HirLiteral::Unit),
                            ty: Type::Unit,
                            span: expr.span,
                        }),
                    },
                    ty: Type::Unit,
                    span: expr.span,
                };

                let body_expr = self.lower_block(body);

                let loop_body = HirExpr {
                    id: self.fresh_hir_id(),
                    kind: HirExprKind::Block(HirBlock {
                        stmts: vec![HirStmt::Expr(check_and_break), HirStmt::Expr(body_expr)],
                        expr: None,
                    }),
                    ty: Type::Unit,
                    span: body.span,
                };

                self.resolver.exit_scope();

                HirExprKind::Loop {
                    body: Box::new(loop_body),
                }
            }

            ExprKind::Loop { body } => {
                self.resolver.enter_loop_scope();
                let body_expr = self.lower_block(body);
                self.resolver.exit_scope();

                HirExprKind::Loop {
                    body: Box::new(body_expr),
                }
            }

            ExprKind::For {
                binding,
                iterator,
                body,
            } => {
                // Desugar for to while over iterator
                // for x in iter { body } -> { let iter = iter.into_iter(); while let Some(x) = iter.next() { body } }
                // Simplified version: just create a loop for now
                self.resolver.enter_loop_scope();

                let local_id = self.resolver.fresh_local_id();
                let _ = self.resolver.define_variable(
                    binding.name.clone(),
                    Symbol::Local {
                        id: local_id,
                        ty: Type::Unknown,
                        mutable: false,
                    },
                );

                let body_expr = self.lower_block(body);
                self.resolver.exit_scope();

                // For now, just emit the body wrapped in a loop
                // Proper iterator support would require more complex lowering
                HirExprKind::Loop {
                    body: Box::new(body_expr),
                }
            }

            ExprKind::Break(value) => HirExprKind::Break {
                value: value.as_ref().map(|v| Box::new(self.lower_expr(v))),
            },

            ExprKind::Continue => HirExprKind::Continue,

            ExprKind::Return(value) => HirExprKind::Return {
                value: value.as_ref().map(|v| Box::new(self.lower_expr(v))),
            },

            ExprKind::Assign { target, value } => HirExprKind::Assign {
                target: Box::new(self.lower_expr(target)),
                value: Box::new(self.lower_expr(value)),
            },

            ExprKind::Tuple(elements) => {
                HirExprKind::Tuple(elements.iter().map(|e| self.lower_expr(e)).collect())
            }

            ExprKind::ArrayLiteral(elements) => {
                HirExprKind::Array(elements.iter().map(|e| self.lower_expr(e)).collect())
            }

            ExprKind::ObjectLiteral { prototype, members } => {
                let fields = members
                    .iter()
                    .map(|m| (m.key.name.clone(), self.lower_expr(&m.value)))
                    .collect();

                HirExprKind::Object {
                    prototype: None, // TODO: resolve prototype
                    fields,
                }
            }

            ExprKind::Lambda {
                params,
                return_type: _,
                body,
                is_async: _,
            } => {
                self.resolver.enter_function_scope();

                let hir_params: Vec<HirParam> =
                    params.iter().map(|p| self.lower_param(p)).collect();

                let body_expr = self.lower_expr(body);

                self.resolver.exit_scope();

                HirExprKind::Lambda {
                    params: hir_params,
                    body: Box::new(body_expr),
                    captures: Vec::new(), // TODO: capture analysis
                }
            }

            ExprKind::Await(inner) => HirExprKind::Await(Box::new(self.lower_expr(inner))),

            ExprKind::Match { scrutinee, arms } => {
                let hir_arms = arms
                    .iter()
                    .map(|arm| HirMatchArm {
                        pattern: self.lower_pattern(&arm.pattern),
                        guard: arm.guard.as_ref().map(|g| self.lower_expr(g)),
                        body: self.lower_expr(&arm.body),
                    })
                    .collect();

                HirExprKind::Match {
                    scrutinee: Box::new(self.lower_expr(scrutinee)),
                    arms: hir_arms,
                }
            }

            ExprKind::HttpStatus(code) => HirExprKind::HttpStatus(*code),

            ExprKind::Block(block) => {
                return self.lower_block(block);
            }

            // Handle remaining cases
            _ => {
                self.errors.push(LoweringError::UnsupportedFeature(format!(
                    "Unsupported expression: {:?}",
                    expr.kind
                )));
                HirExprKind::Literal(HirLiteral::Unit)
            }
        };

        HirExpr {
            id: self.fresh_hir_id(),
            kind,
            ty: Type::Unknown,
            span: expr.span,
        }
    }

    /// Lower a literal.
    fn lower_literal(&self, lit: &Literal) -> HirLiteral {
        match lit {
            Literal::Int(v) => HirLiteral::Int(*v),
            Literal::Float(v) => HirLiteral::Float(*v),
            Literal::Bool(v) => HirLiteral::Bool(*v),
            Literal::Char(v) => HirLiteral::Char(*v),
            Literal::String(v) => HirLiteral::String(v.clone()),
            Literal::Null => HirLiteral::Unit, // null maps to unit for now
        }
    }

    /// Lower a binary operator.
    fn lower_binary_op(&self, op: BinaryOp) -> HirBinaryOp {
        match op {
            BinaryOp::Add => HirBinaryOp::Add,
            BinaryOp::Sub => HirBinaryOp::Sub,
            BinaryOp::Mul => HirBinaryOp::Mul,
            BinaryOp::Div => HirBinaryOp::Div,
            BinaryOp::Mod => HirBinaryOp::Rem,
            BinaryOp::BitAnd => HirBinaryOp::BitAnd,
            BinaryOp::BitOr => HirBinaryOp::BitOr,
            BinaryOp::BitXor => HirBinaryOp::BitXor,
            BinaryOp::Shl => HirBinaryOp::Shl,
            BinaryOp::Shr => HirBinaryOp::Shr,
            BinaryOp::Eq => HirBinaryOp::Eq,
            BinaryOp::NotEq => HirBinaryOp::Ne,
            BinaryOp::Lt => HirBinaryOp::Lt,
            BinaryOp::LtEq => HirBinaryOp::Le,
            BinaryOp::Gt => HirBinaryOp::Gt,
            BinaryOp::GtEq => HirBinaryOp::Ge,
            BinaryOp::And => HirBinaryOp::And,
            BinaryOp::Or => HirBinaryOp::Or,
        }
    }

    /// Lower a unary operator.
    fn lower_unary_op(&self, op: UnaryOp) -> HirUnaryOp {
        match op {
            UnaryOp::Neg => HirUnaryOp::Neg,
            UnaryOp::Not => HirUnaryOp::Not,
            UnaryOp::BitNot => HirUnaryOp::BitNot,
            UnaryOp::Deref => HirUnaryOp::Deref,
            UnaryOp::Ref => HirUnaryOp::Ref,
            UnaryOp::RefMut => HirUnaryOp::RefMut,
        }
    }

    /// Lower a pattern.
    fn lower_pattern(&mut self, pattern: &AstPattern) -> HirPattern {
        match &pattern.kind {
            PatternKind::Wildcard => HirPattern::Wildcard,

            PatternKind::Ident { name, mutable } => {
                let id = self.resolver.fresh_local_id();
                let _ = self.resolver.define_variable(
                    name.name.clone(),
                    Symbol::Local {
                        id,
                        ty: Type::Unknown,
                        mutable: *mutable,
                    },
                );
                HirPattern::Binding(id)
            }

            PatternKind::Literal(lit) => HirPattern::Literal(self.lower_literal(lit)),

            PatternKind::Tuple(patterns) => {
                HirPattern::Tuple(patterns.iter().map(|p| self.lower_pattern(p)).collect())
            }

            PatternKind::HttpStatus(http_pattern) => {
                HirPattern::HttpStatus(self.lower_http_status_pattern(http_pattern))
            }

            PatternKind::Or(patterns) => {
                HirPattern::Or(patterns.iter().map(|p| self.lower_pattern(p)).collect())
            }

            _ => HirPattern::Wildcard, // Fallback for unsupported patterns
        }
    }

    /// Lower an HTTP status pattern.
    fn lower_http_status_pattern(&self, pattern: &AstHttpStatusPattern) -> HirHttpStatusPattern {
        match pattern {
            AstHttpStatusPattern::Exact(code) => HirHttpStatusPattern::Exact(*code),
            AstHttpStatusPattern::Range { start, end } => HirHttpStatusPattern::Range {
                start: *start,
                end: *end,
            },
            AstHttpStatusPattern::Category(cat) => {
                let (start, end) = match cat {
                    AstHttpStatusCategory::Informational => (100, 199),
                    AstHttpStatusCategory::Success => (200, 299),
                    AstHttpStatusCategory::Redirection => (300, 399),
                    AstHttpStatusCategory::ClientError => (400, 499),
                    AstHttpStatusCategory::ServerError => (500, 599),
                };
                HirHttpStatusPattern::Range { start, end }
            }
            AstHttpStatusPattern::Any => HirHttpStatusPattern::Any,
        }
    }

    /// Get collected errors.
    pub fn errors(&self) -> &[LoweringError] {
        &self.errors
    }

    /// Check if there are errors.
    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }
}

impl Default for LoweringContext {
    fn default() -> Self {
        Self::new()
    }
}
