//! HIR to MIR lowering.
//!
//! This module converts the tree-based HIR into the CFG-based MIR.

use crate::mir::{
    AggregateKind, BasicBlockId, BinOp, BodyId, BorrowKind, CastKind, Constant, Local, LocalDecl,
    MirBody, MirModule, Mutability, Operand, Place, PlaceElem, Rvalue, Statement, StatementKind,
    SwitchTargets, Terminator, TerminatorKind, UnOp,
};
use std::collections::HashMap;
use wsharp_hir::{
    DefId, HirBinaryOp, HirBlock, HirBody, HirExpr, HirExprKind, HirFunction, HirItem, HirLiteral,
    HirModule, HirStmt, HirUnaryOp, LocalId,
};
use wsharp_types::{HttpStatusType, HttpStatusTypeKind, PrimitiveType, Type};

/// Builder for converting HIR to MIR.
pub struct MirBuilder {
    /// The MIR module being built.
    module: MirModule,

    /// Mapping from HIR DefIds to MIR BodyIds.
    def_to_body: HashMap<DefId, BodyId>,

    /// Counter for generating unique body IDs.
    next_body_id: u32,
}

impl MirBuilder {
    pub fn new(name: String) -> Self {
        Self {
            module: MirModule::new(name),
            def_to_body: HashMap::new(),
            next_body_id: 0,
        }
    }

    /// Build MIR from an HIR module.
    pub fn build_module(mut self, hir_module: &HirModule) -> MirModule {
        // First pass: register all functions
        for item in &hir_module.items {
            if let HirItem::Function(func) = item {
                let body_id = BodyId(self.next_body_id);
                self.next_body_id += 1;
                self.def_to_body.insert(func.id, body_id);
            }
        }

        // Second pass: build function bodies
        for item in &hir_module.items {
            if let HirItem::Function(func) = item {
                if let Some(ref body) = func.body {
                    let mir_body = self.build_function(func, body);
                    let body_id = self.def_to_body[&func.id];
                    self.module.bodies.insert(body_id, mir_body);
                }
            }
        }

        // Find entry point (main function)
        for item in &hir_module.items {
            if let HirItem::Function(func) = item {
                if func.name == "main" {
                    self.module.entry_point = Some(self.def_to_body[&func.id]);
                }
            }
        }

        // Transform async functions to coroutine state machines
        crate::coroutine::transform_async_functions(&mut self.module);

        self.module
    }

    /// Build MIR for a single function.
    fn build_function(&self, func: &HirFunction, body: &HirBody) -> MirBody {
        let mut builder = FunctionBuilder::new(
            func.name.clone(),
            func.return_type.clone(),
            func.is_async,
            &self.def_to_body,
        );

        // Add parameters
        for param in &func.params {
            let local = builder.body.add_arg(param.ty.clone(), param.name.clone());
            builder.hir_to_local.insert(param.id, local);
        }

        // Note: entry block is already created by FunctionBuilder::new()

        // Add local variables
        for local in &body.locals {
            let mir_local = builder
                .body
                .add_local(local.ty.clone(), Some(local.name.clone()), local.mutable);
            builder.hir_to_local.insert(local.id, mir_local);
        }

        // Lower the body expression
        let result = builder.lower_expr(&body.expr);

        // Store result in return place and return
        if !builder.is_terminated() {
            let return_place = Place::return_place();
            builder.push_assign(return_place, Rvalue::Use(result));
            builder.terminate(TerminatorKind::Return);
        }

        builder.body
    }
}

/// Builder for a single function body.
struct FunctionBuilder<'a> {
    /// The MIR body being built.
    body: MirBody,

    /// Current basic block being built.
    current_block: BasicBlockId,

    /// Mapping from HIR LocalIds to MIR Locals.
    hir_to_local: HashMap<LocalId, Local>,

    /// Mapping from HIR DefIds to MIR BodyIds.
    def_to_body: &'a HashMap<DefId, BodyId>,

    /// Stack of loop contexts for break/continue.
    loop_stack: Vec<LoopContext>,
}

/// Context for a loop (for break/continue handling).
struct LoopContext {
    /// Block to jump to on break.
    break_block: BasicBlockId,

    /// Block to jump to on continue.
    continue_block: BasicBlockId,

    /// Place to store break value (if any).
    break_value: Option<Local>,
}

impl<'a> FunctionBuilder<'a> {
    fn new(
        name: String,
        return_ty: Type,
        is_async: bool,
        def_to_body: &'a HashMap<DefId, BodyId>,
    ) -> Self {
        let mut body = MirBody::new(name, return_ty, is_async);
        let entry = body.new_basic_block();

        Self {
            body,
            current_block: entry,
            hir_to_local: HashMap::new(),
            def_to_body,
            loop_stack: Vec::new(),
        }
    }

    /// Check if the current block is terminated.
    fn is_terminated(&self) -> bool {
        self.body.block(self.current_block).terminator.is_some()
    }

    /// Push a statement to the current block.
    fn push_stmt(&mut self, kind: StatementKind) {
        self.body
            .block_mut(self.current_block)
            .push_stmt(Statement { kind });
    }

    /// Push an assignment statement.
    fn push_assign(&mut self, place: Place, rvalue: Rvalue) {
        self.push_stmt(StatementKind::Assign(place, rvalue));
    }

    /// Set the terminator for the current block.
    fn terminate(&mut self, kind: TerminatorKind) {
        self.body
            .block_mut(self.current_block)
            .set_terminator(Terminator { kind });
    }

    /// Create a new temporary local.
    fn new_temp(&mut self, ty: Type) -> Local {
        self.body.add_local(ty, None, true)
    }

    /// Lower an HIR expression to MIR, returning an operand for the result.
    fn lower_expr(&mut self, expr: &HirExpr) -> Operand {
        match &expr.kind {
            HirExprKind::Literal(lit) => self.lower_literal(lit),

            HirExprKind::Local(id) => {
                let local = self.hir_to_local[id];
                Operand::Copy(Place::local(local))
            }

            HirExprKind::Global(def_id) => {
                if let Some(&body_id) = self.def_to_body.get(def_id) {
                    Operand::Constant(Constant::Function(body_id))
                } else {
                    // Unknown global - this shouldn't happen in well-typed code
                    Operand::const_unit()
                }
            }

            HirExprKind::Binary { op, left, right } => {
                self.lower_binary_op(*op, left, right, &expr.ty)
            }

            HirExprKind::Unary { op, operand } => self.lower_unary_op(*op, operand, &expr.ty),

            HirExprKind::Call { callee, args } => self.lower_call(callee, args, &expr.ty),

            HirExprKind::MethodCall {
                receiver,
                method,
                args,
            } => {
                // Method calls should be desugared to regular calls in HIR
                // but we handle them here anyway
                let receiver_op = self.lower_expr(receiver);
                let mut all_args = vec![receiver_op];
                for arg in args {
                    all_args.push(self.lower_expr(arg));
                }

                if let Some(&body_id) = self.def_to_body.get(method) {
                    let func = Operand::Constant(Constant::Function(body_id));
                    let dest = self.new_temp(expr.ty.clone());
                    let next_block = self.body.new_basic_block();

                    self.terminate(TerminatorKind::Call {
                        func,
                        args: all_args,
                        destination: Place::local(dest),
                        target: Some(next_block),
                    });

                    self.current_block = next_block;
                    Operand::Move(Place::local(dest))
                } else {
                    Operand::const_unit()
                }
            }

            HirExprKind::Field { object, field } => {
                let obj_op = self.lower_expr(object);

                // Get the field index from the type
                // For now, we need to look up the field in the object type
                if let Operand::Copy(place) | Operand::Move(place) = obj_op {
                    let field_idx = self.get_field_index(&object.ty, field);
                    let field_place = place.field(field_idx, expr.ty.clone());
                    Operand::Copy(field_place)
                } else {
                    // Constant object - need to materialize it first
                    let temp = self.new_temp(object.ty.clone());
                    self.push_assign(Place::local(temp), Rvalue::Use(obj_op));
                    let field_idx = self.get_field_index(&object.ty, field);
                    Operand::Copy(Place::local(temp).field(field_idx, expr.ty.clone()))
                }
            }

            HirExprKind::Index { object, index } => {
                let obj_op = self.lower_expr(object);
                let idx_op = self.lower_expr(index);

                // Materialize index into a local for Place::Index
                let idx_local = match idx_op {
                    Operand::Copy(Place { local, .. }) | Operand::Move(Place { local, .. })
                        if self.body.locals[local.0 as usize].ty == index.ty =>
                    {
                        local
                    }
                    _ => {
                        let temp = self.new_temp(index.ty.clone());
                        self.push_assign(Place::local(temp), Rvalue::Use(idx_op));
                        temp
                    }
                };

                if let Operand::Copy(place) | Operand::Move(place) = obj_op {
                    Operand::Copy(place.index(idx_local))
                } else {
                    let temp = self.new_temp(object.ty.clone());
                    self.push_assign(Place::local(temp), Rvalue::Use(obj_op));
                    Operand::Copy(Place::local(temp).index(idx_local))
                }
            }

            HirExprKind::Block(block) => self.lower_block(block),

            HirExprKind::If {
                condition,
                then_branch,
                else_branch,
            } => self.lower_if(condition, then_branch, else_branch, &expr.ty),

            HirExprKind::Loop { body } => self.lower_loop(body, &expr.ty),

            HirExprKind::Break { value } => {
                if let Some(ctx) = self.loop_stack.last() {
                    let break_block = ctx.break_block;
                    let break_value_local = ctx.break_value;

                    if let (Some(val_expr), Some(dest)) = (value, break_value_local) {
                        let val = self.lower_expr(val_expr);
                        self.push_assign(Place::local(dest), Rvalue::Use(val));
                    }

                    self.terminate(TerminatorKind::Goto {
                        target: break_block,
                    });
                }
                Operand::const_unit()
            }

            HirExprKind::Continue => {
                if let Some(ctx) = self.loop_stack.last() {
                    let continue_block = ctx.continue_block;
                    self.terminate(TerminatorKind::Goto {
                        target: continue_block,
                    });
                }
                Operand::const_unit()
            }

            HirExprKind::Return { value } => {
                let val = value
                    .as_ref()
                    .map(|v| self.lower_expr(v))
                    .unwrap_or_else(Operand::const_unit);

                self.push_assign(Place::return_place(), Rvalue::Use(val));
                self.terminate(TerminatorKind::Return);
                Operand::const_unit()
            }

            HirExprKind::Let { local, init, body } => {
                let mir_local = self.hir_to_local[local];
                let init_val = self.lower_expr(init);
                self.push_stmt(StatementKind::StorageLive(mir_local));
                self.push_assign(Place::local(mir_local), Rvalue::Use(init_val));
                let result = self.lower_expr(body);
                self.push_stmt(StatementKind::StorageDead(mir_local));
                result
            }

            HirExprKind::Assign { target, value } => {
                let val = self.lower_expr(value);
                let target_place = self.lower_place(target);
                self.push_assign(target_place, Rvalue::Use(val));
                Operand::const_unit()
            }

            HirExprKind::Tuple(elements) => {
                let operands: Vec<Operand> = elements.iter().map(|e| self.lower_expr(e)).collect();
                let temp = self.new_temp(expr.ty.clone());
                self.push_assign(
                    Place::local(temp),
                    Rvalue::Aggregate(AggregateKind::Tuple, operands),
                );
                Operand::Move(Place::local(temp))
            }

            HirExprKind::Array(elements) => {
                let operands: Vec<Operand> = elements.iter().map(|e| self.lower_expr(e)).collect();
                let elem_ty = if let Type::Array { element, .. } = &expr.ty {
                    (**element).clone()
                } else {
                    Type::Unknown
                };
                let temp = self.new_temp(expr.ty.clone());
                self.push_assign(
                    Place::local(temp),
                    Rvalue::Aggregate(AggregateKind::Array(elem_ty), operands),
                );
                Operand::Move(Place::local(temp))
            }

            HirExprKind::Object { prototype, fields } => {
                let operands: Vec<Operand> =
                    fields.iter().map(|(_, e)| self.lower_expr(e)).collect();
                let temp = self.new_temp(expr.ty.clone());
                self.push_assign(
                    Place::local(temp),
                    Rvalue::Aggregate(
                        AggregateKind::Adt {
                            name: prototype.map(|p| format!("proto_{}", p.0)).unwrap_or_default(),
                            variant: None,
                        },
                        operands,
                    ),
                );
                Operand::Move(Place::local(temp))
            }

            HirExprKind::Lambda {
                params,
                body,
                captures,
                is_async,
            } => {
                // For now, lambdas are lowered as closures
                // A full implementation would create a separate MIR body
                let capture_ops: Vec<Operand> = captures
                    .iter()
                    .map(|c| Operand::Copy(Place::local(self.hir_to_local[c])))
                    .collect();

                let capture_types: Vec<Type> = captures
                    .iter()
                    .map(|c| self.body.locals[self.hir_to_local[c].0 as usize].ty.clone())
                    .collect();

                let temp = self.new_temp(expr.ty.clone());
                // Placeholder body ID - proper implementation would create the lambda body
                // Note: is_async is tracked in the function type, not in the closure aggregate
                let _ = is_async; // Async info is captured in expr.ty (FunctionType with is_async)
                self.push_assign(
                    Place::local(temp),
                    Rvalue::Aggregate(
                        AggregateKind::Closure {
                            body_id: BodyId(u32::MAX),
                            captures: capture_types,
                        },
                        capture_ops,
                    ),
                );
                Operand::Move(Place::local(temp))
            }

            HirExprKind::Await(future) => {
                // Lower the future expression
                let future_op = self.lower_expr(future);

                // Create a temp to hold the awaited result
                let result_temp = self.new_temp(expr.ty.clone());

                // Create the resume block (where we continue after the await)
                let resume_block = self.body.new_basic_block();

                // Generate Yield terminator to suspend at this await point
                // The coroutine transformation will convert this to proper state machine code
                self.terminate(TerminatorKind::Yield {
                    value: future_op,
                    resume: resume_block,
                });

                // Switch to the resume block
                self.current_block = resume_block;

                // After resume, the awaited value is assumed to be in result_temp
                // The coroutine transformation will handle actually loading the result
                // For now, we return the temp (it will be filled in by the transformation)
                Operand::Move(Place::local(result_temp))
            }

            HirExprKind::Cast { expr: inner, target_ty } => {
                let val = self.lower_expr(inner);
                let cast_kind = self.get_cast_kind(&inner.ty, target_ty);
                let temp = self.new_temp(target_ty.clone());
                self.push_assign(
                    Place::local(temp),
                    Rvalue::Cast(cast_kind, val, target_ty.clone()),
                );
                Operand::Move(Place::local(temp))
            }

            HirExprKind::Match { scrutinee, arms } => {
                // Match is lowered to a series of conditional branches
                self.lower_match(scrutinee, arms, &expr.ty)
            }

            HirExprKind::HttpStatus(code) => {
                // HTTP status is represented as an integer constant with HttpStatus type
                let http_type = Type::HttpStatus(HttpStatusType {
                    kind: HttpStatusTypeKind::Exact(*code),
                });
                Operand::const_int(*code as i128, http_type)
            }
        }
    }

    /// Lower a literal to an operand.
    fn lower_literal(&self, lit: &HirLiteral) -> Operand {
        match lit {
            HirLiteral::Int(v) => {
                Operand::const_int(*v, Type::Primitive(PrimitiveType::I64))
            }
            HirLiteral::Float(v) => {
                Operand::const_float(*v, Type::Primitive(PrimitiveType::F64))
            }
            HirLiteral::Bool(v) => Operand::const_bool(*v),
            HirLiteral::Char(c) => Operand::Constant(Constant::Char(*c)),
            HirLiteral::String(s) => Operand::Constant(Constant::String(s.clone())),
            HirLiteral::Unit => Operand::const_unit(),
        }
    }

    /// Lower a binary operation.
    fn lower_binary_op(
        &mut self,
        op: HirBinaryOp,
        left: &HirExpr,
        right: &HirExpr,
        result_ty: &Type,
    ) -> Operand {
        // Short-circuit evaluation for && and ||
        match op {
            HirBinaryOp::And => return self.lower_short_circuit_and(left, right),
            HirBinaryOp::Or => return self.lower_short_circuit_or(left, right),
            _ => {}
        }

        let left_op = self.lower_expr(left);
        let right_op = self.lower_expr(right);

        let bin_op = match op {
            HirBinaryOp::Add => BinOp::Add,
            HirBinaryOp::Sub => BinOp::Sub,
            HirBinaryOp::Mul => BinOp::Mul,
            HirBinaryOp::Div => BinOp::Div,
            HirBinaryOp::Rem => BinOp::Rem,
            HirBinaryOp::BitAnd => BinOp::BitAnd,
            HirBinaryOp::BitOr => BinOp::BitOr,
            HirBinaryOp::BitXor => BinOp::BitXor,
            HirBinaryOp::Shl => BinOp::Shl,
            HirBinaryOp::Shr => BinOp::Shr,
            HirBinaryOp::Eq => BinOp::Eq,
            HirBinaryOp::Ne => BinOp::Ne,
            HirBinaryOp::Lt => BinOp::Lt,
            HirBinaryOp::Le => BinOp::Le,
            HirBinaryOp::Gt => BinOp::Gt,
            HirBinaryOp::Ge => BinOp::Ge,
            HirBinaryOp::And | HirBinaryOp::Or => unreachable!(),
        };

        let temp = self.new_temp(result_ty.clone());
        self.push_assign(
            Place::local(temp),
            Rvalue::BinaryOp(bin_op, left_op, right_op),
        );
        Operand::Move(Place::local(temp))
    }

    /// Lower short-circuit AND (&&).
    fn lower_short_circuit_and(&mut self, left: &HirExpr, right: &HirExpr) -> Operand {
        let result = self.new_temp(Type::Primitive(PrimitiveType::Bool));

        let left_op = self.lower_expr(left);

        // Store left value first
        self.push_assign(Place::local(result), Rvalue::Use(left_op.clone()));

        let eval_right = self.body.new_basic_block();
        let end = self.body.new_basic_block();

        // If left is true, evaluate right; otherwise skip
        self.terminate(TerminatorKind::SwitchInt {
            discr: left_op,
            targets: SwitchTargets::if_else(eval_right, end),
        });

        // Evaluate right and store result
        self.current_block = eval_right;
        let right_op = self.lower_expr(right);
        self.push_assign(Place::local(result), Rvalue::Use(right_op));
        self.terminate(TerminatorKind::Goto { target: end });

        self.current_block = end;
        Operand::Copy(Place::local(result))
    }

    /// Lower short-circuit OR (||).
    fn lower_short_circuit_or(&mut self, left: &HirExpr, right: &HirExpr) -> Operand {
        let result = self.new_temp(Type::Primitive(PrimitiveType::Bool));

        let left_op = self.lower_expr(left);

        // Store left value first
        self.push_assign(Place::local(result), Rvalue::Use(left_op.clone()));

        let eval_right = self.body.new_basic_block();
        let end = self.body.new_basic_block();

        // If left is false, evaluate right; otherwise skip
        self.terminate(TerminatorKind::SwitchInt {
            discr: left_op,
            targets: SwitchTargets::if_else(end, eval_right), // true -> end, false -> eval_right
        });

        // Evaluate right and store result
        self.current_block = eval_right;
        let right_op = self.lower_expr(right);
        self.push_assign(Place::local(result), Rvalue::Use(right_op));
        self.terminate(TerminatorKind::Goto { target: end });

        self.current_block = end;
        Operand::Copy(Place::local(result))
    }

    /// Lower a unary operation.
    fn lower_unary_op(&mut self, op: HirUnaryOp, operand: &HirExpr, result_ty: &Type) -> Operand {
        let operand_val = self.lower_expr(operand);

        match op {
            HirUnaryOp::Neg => {
                let temp = self.new_temp(result_ty.clone());
                self.push_assign(
                    Place::local(temp),
                    Rvalue::UnaryOp(UnOp::Neg, operand_val),
                );
                Operand::Move(Place::local(temp))
            }
            HirUnaryOp::Not | HirUnaryOp::BitNot => {
                let temp = self.new_temp(result_ty.clone());
                self.push_assign(
                    Place::local(temp),
                    Rvalue::UnaryOp(UnOp::Not, operand_val),
                );
                Operand::Move(Place::local(temp))
            }
            HirUnaryOp::Deref => {
                if let Operand::Copy(place) | Operand::Move(place) = operand_val {
                    Operand::Copy(place.deref())
                } else {
                    let temp = self.new_temp(operand.ty.clone());
                    self.push_assign(Place::local(temp), Rvalue::Use(operand_val));
                    Operand::Copy(Place::local(temp).deref())
                }
            }
            HirUnaryOp::Ref => {
                let place = self.lower_place(operand);
                let temp = self.new_temp(result_ty.clone());
                self.push_assign(Place::local(temp), Rvalue::Ref(place, BorrowKind::Shared));
                Operand::Move(Place::local(temp))
            }
            HirUnaryOp::RefMut => {
                let place = self.lower_place(operand);
                let temp = self.new_temp(result_ty.clone());
                self.push_assign(Place::local(temp), Rvalue::Ref(place, BorrowKind::Mut));
                Operand::Move(Place::local(temp))
            }
        }
    }

    /// Lower a function call.
    fn lower_call(&mut self, callee: &HirExpr, args: &[HirExpr], result_ty: &Type) -> Operand {
        let callee_op = self.lower_expr(callee);
        let arg_ops: Vec<Operand> = args.iter().map(|a| self.lower_expr(a)).collect();

        let dest = self.new_temp(result_ty.clone());
        let next_block = self.body.new_basic_block();

        self.terminate(TerminatorKind::Call {
            func: callee_op,
            args: arg_ops,
            destination: Place::local(dest),
            target: Some(next_block),
        });

        self.current_block = next_block;
        Operand::Move(Place::local(dest))
    }

    /// Lower a block expression.
    fn lower_block(&mut self, block: &HirBlock) -> Operand {
        for stmt in &block.stmts {
            self.lower_stmt(stmt);
            if self.is_terminated() {
                return Operand::const_unit();
            }
        }

        if let Some(ref expr) = block.expr {
            self.lower_expr(expr)
        } else {
            Operand::const_unit()
        }
    }

    /// Lower a statement.
    fn lower_stmt(&mut self, stmt: &HirStmt) {
        match stmt {
            HirStmt::Expr(expr) => {
                self.lower_expr(expr);
            }
            HirStmt::Let { local, init } => {
                let mir_local = self.hir_to_local[local];
                let init_val = self.lower_expr(init);
                self.push_stmt(StatementKind::StorageLive(mir_local));
                self.push_assign(Place::local(mir_local), Rvalue::Use(init_val));
            }
        }
    }

    /// Lower an if expression.
    fn lower_if(
        &mut self,
        condition: &HirExpr,
        then_branch: &HirExpr,
        else_branch: &HirExpr,
        result_ty: &Type,
    ) -> Operand {
        let cond = self.lower_expr(condition);

        let then_block = self.body.new_basic_block();
        let else_block = self.body.new_basic_block();
        let end_block = self.body.new_basic_block();

        // Result local for the if expression
        let result = self.new_temp(result_ty.clone());

        // Branch based on condition
        self.terminate(TerminatorKind::SwitchInt {
            discr: cond,
            targets: SwitchTargets::if_else(then_block, else_block),
        });

        // Then branch
        self.current_block = then_block;
        let then_val = self.lower_expr(then_branch);
        if !self.is_terminated() {
            self.push_assign(Place::local(result), Rvalue::Use(then_val));
            self.terminate(TerminatorKind::Goto { target: end_block });
        }

        // Else branch
        self.current_block = else_block;
        let else_val = self.lower_expr(else_branch);
        if !self.is_terminated() {
            self.push_assign(Place::local(result), Rvalue::Use(else_val));
            self.terminate(TerminatorKind::Goto { target: end_block });
        }

        self.current_block = end_block;
        Operand::Move(Place::local(result))
    }

    /// Lower a loop expression.
    fn lower_loop(&mut self, body: &HirExpr, result_ty: &Type) -> Operand {
        let loop_header = self.body.new_basic_block();
        let loop_exit = self.body.new_basic_block();

        // Result local for break values
        let result = if *result_ty != Type::Unit && *result_ty != Type::Never {
            Some(self.new_temp(result_ty.clone()))
        } else {
            None
        };

        // Jump to loop header
        self.terminate(TerminatorKind::Goto {
            target: loop_header,
        });

        // Push loop context for break/continue
        self.loop_stack.push(LoopContext {
            break_block: loop_exit,
            continue_block: loop_header,
            break_value: result,
        });

        // Loop body
        self.current_block = loop_header;
        self.lower_expr(body);

        // If not terminated, loop back
        if !self.is_terminated() {
            self.terminate(TerminatorKind::Goto {
                target: loop_header,
            });
        }

        self.loop_stack.pop();
        self.current_block = loop_exit;

        if let Some(r) = result {
            Operand::Move(Place::local(r))
        } else {
            Operand::const_unit()
        }
    }

    /// Lower a match expression.
    fn lower_match(
        &mut self,
        scrutinee: &HirExpr,
        arms: &[wsharp_hir::HirMatchArm],
        result_ty: &Type,
    ) -> Operand {
        let scrutinee_val = self.lower_expr(scrutinee);
        let scrutinee_local = match scrutinee_val {
            Operand::Copy(Place { local, .. }) | Operand::Move(Place { local, .. }) => local,
            _ => {
                let temp = self.new_temp(scrutinee.ty.clone());
                self.push_assign(Place::local(temp), Rvalue::Use(scrutinee_val));
                temp
            }
        };

        let result = self.new_temp(result_ty.clone());
        let end_block = self.body.new_basic_block();

        let mut arm_blocks: Vec<BasicBlockId> = Vec::new();
        let mut next_test_blocks: Vec<BasicBlockId> = Vec::new();

        // Create blocks for each arm
        for _ in arms {
            arm_blocks.push(self.body.new_basic_block());
            next_test_blocks.push(self.body.new_basic_block());
        }

        // Generate pattern tests
        for (i, arm) in arms.iter().enumerate() {
            if i > 0 {
                self.current_block = next_test_blocks[i - 1];
            }

            let body_block = arm_blocks[i];
            let next_block = if i + 1 < arms.len() {
                next_test_blocks[i]
            } else {
                end_block // Last arm - fallthrough to end (unreachable for exhaustive match)
            };

            // For now, simplified pattern matching - just jump to the arm
            // A full implementation would generate proper pattern tests
            self.terminate(TerminatorKind::Goto { target: body_block });
        }

        // Generate arm bodies
        for (i, arm) in arms.iter().enumerate() {
            self.current_block = arm_blocks[i];

            // TODO: Bind pattern variables
            let body_val = self.lower_expr(&arm.body);

            if !self.is_terminated() {
                self.push_assign(Place::local(result), Rvalue::Use(body_val));
                self.terminate(TerminatorKind::Goto { target: end_block });
            }
        }

        self.current_block = end_block;
        Operand::Move(Place::local(result))
    }

    /// Lower an expression to a place (for lvalues).
    fn lower_place(&mut self, expr: &HirExpr) -> Place {
        match &expr.kind {
            HirExprKind::Local(id) => Place::local(self.hir_to_local[id]),

            HirExprKind::Field { object, field } => {
                let obj_place = self.lower_place(object);
                let field_idx = self.get_field_index(&object.ty, field);
                obj_place.field(field_idx, expr.ty.clone())
            }

            HirExprKind::Index { object, index } => {
                let obj_place = self.lower_place(object);
                let idx_op = self.lower_expr(index);
                let idx_local = match idx_op {
                    Operand::Copy(Place { local, .. }) | Operand::Move(Place { local, .. }) => {
                        local
                    }
                    _ => {
                        let temp = self.new_temp(index.ty.clone());
                        self.push_assign(Place::local(temp), Rvalue::Use(idx_op));
                        temp
                    }
                };
                obj_place.index(idx_local)
            }

            HirExprKind::Unary {
                op: HirUnaryOp::Deref,
                operand,
            } => {
                let ptr_place = self.lower_place(operand);
                ptr_place.deref()
            }

            _ => {
                // For non-place expressions, materialize to a temporary
                let val = self.lower_expr(expr);
                let temp = self.new_temp(expr.ty.clone());
                self.push_assign(Place::local(temp), Rvalue::Use(val));
                Place::local(temp)
            }
        }
    }

    /// Get the field index for a field name in a type.
    fn get_field_index(&self, ty: &Type, field: &str) -> usize {
        match ty {
            Type::Prototype(proto) => proto
                .members
                .iter()
                .position(|(name, _)| name == field)
                .unwrap_or(0),
            Type::Tuple(_) => field.parse().unwrap_or(0),
            _ => 0,
        }
    }

    /// Determine the cast kind for a type conversion.
    fn get_cast_kind(&self, from: &Type, to: &Type) -> CastKind {
        match (from, to) {
            (Type::Primitive(p1), Type::Primitive(p2)) => {
                let from_float = matches!(p1, PrimitiveType::F32 | PrimitiveType::F64);
                let to_float = matches!(p2, PrimitiveType::F32 | PrimitiveType::F64);

                match (from_float, to_float) {
                    (false, false) => CastKind::IntToInt,
                    (false, true) => CastKind::IntToFloat,
                    (true, false) => CastKind::FloatToInt,
                    (true, true) => CastKind::FloatToFloat,
                }
            }
            (Type::Ref { .. }, Type::Ref { .. }) => CastKind::PtrToPtr,
            _ => CastKind::PtrToPtr, // Default
        }
    }
}
