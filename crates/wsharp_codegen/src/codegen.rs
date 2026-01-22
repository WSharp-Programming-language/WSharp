//! LLVM code generation from MIR.

use crate::error::{CodegenError, CodegenResult};
use inkwell::basic_block::BasicBlock;
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::types::{BasicMetadataTypeEnum, BasicType, BasicTypeEnum, FunctionType};
use inkwell::values::{AnyValue, BasicMetadataValueEnum, BasicValue, BasicValueEnum, FunctionValue, PointerValue};
use inkwell::{AddressSpace, FloatPredicate, IntPredicate};
use std::collections::HashMap;
use wsharp_mir::{
    AggregateKind, BasicBlockId, BinOp, BodyId, BorrowKind, CastKind, Constant, Local, MirBody,
    MirModule, Operand, Place, PlaceElem, Rvalue, StatementKind, SwitchTargets, TerminatorKind,
    UnOp,
};
use wsharp_types::{PrimitiveType, Type};

/// The LLVM code generator.
pub struct CodeGenerator<'ctx> {
    /// LLVM context.
    context: &'ctx Context,

    /// LLVM module being generated.
    module: Module<'ctx>,

    /// LLVM IR builder.
    builder: Builder<'ctx>,

    /// Mapping from MIR body IDs to LLVM functions.
    functions: HashMap<BodyId, FunctionValue<'ctx>>,

    /// Target triple.
    target_triple: String,
}

impl<'ctx> CodeGenerator<'ctx> {
    /// Create a new code generator.
    pub fn new(context: &'ctx Context, module_name: &str) -> Self {
        let module = context.create_module(module_name);
        let builder = context.create_builder();

        Self {
            context,
            module,
            builder,
            functions: HashMap::new(),
            target_triple: String::new(),
        }
    }

    /// Set the target triple.
    pub fn set_target_triple(&mut self, triple: &str) {
        self.target_triple = triple.to_string();
        self.module
            .set_triple(&inkwell::targets::TargetTriple::create(triple));
    }

    /// Get the generated LLVM module.
    pub fn into_module(self) -> Module<'ctx> {
        self.module
    }

    /// Get a reference to the module.
    pub fn module(&self) -> &Module<'ctx> {
        &self.module
    }

    /// Generate code for an entire MIR module.
    pub fn codegen_module(&mut self, mir_module: &MirModule) -> CodegenResult<()> {
        // First pass: declare all functions
        for (&body_id, body) in &mir_module.bodies {
            let func = self.declare_function(body)?;
            self.functions.insert(body_id, func);
        }

        // Second pass: generate function bodies
        for (&body_id, body) in &mir_module.bodies {
            let func = self.functions[&body_id];
            self.codegen_function(body, func)?;
        }

        // Verify the module
        if let Err(e) = self.module.verify() {
            return Err(CodegenError::LlvmError(e.to_string()));
        }

        Ok(())
    }

    /// Declare a function (without generating its body).
    fn declare_function(&self, body: &MirBody) -> CodegenResult<FunctionValue<'ctx>> {
        let return_ty = self.llvm_type(&body.return_ty)?;

        // Collect parameter types
        let param_types: Vec<BasicMetadataTypeEnum> = body.locals[1..=body.arg_count]
            .iter()
            .map(|decl| self.llvm_type(&decl.ty).map(|t| t.into()))
            .collect::<CodegenResult<Vec<_>>>()?;

        let fn_type = match return_ty {
            BasicTypeEnum::IntType(t) => t.fn_type(&param_types, false),
            BasicTypeEnum::FloatType(t) => t.fn_type(&param_types, false),
            BasicTypeEnum::PointerType(t) => t.fn_type(&param_types, false),
            BasicTypeEnum::StructType(t) => t.fn_type(&param_types, false),
            BasicTypeEnum::ArrayType(t) => t.fn_type(&param_types, false),
            BasicTypeEnum::VectorType(t) => t.fn_type(&param_types, false),
            BasicTypeEnum::ScalableVectorType(t) => t.fn_type(&param_types, false),
        };

        let func = self.module.add_function(&body.name, fn_type, None);

        // Set parameter names
        for (i, param) in func.get_param_iter().enumerate() {
            let decl = &body.locals[i + 1];
            if let Some(ref name) = decl.name {
                param.set_name(name);
            }
        }

        Ok(func)
    }

    /// Generate code for a function body.
    fn codegen_function(
        &mut self,
        body: &MirBody,
        func: FunctionValue<'ctx>,
    ) -> CodegenResult<()> {
        let mut func_ctx = FunctionContext::new(self, body, func)?;
        func_ctx.codegen_body()?;
        Ok(())
    }

    /// Convert a W# type to an LLVM type.
    fn llvm_type(&self, ty: &Type) -> CodegenResult<BasicTypeEnum<'ctx>> {
        match ty {
            Type::Primitive(prim) => self.llvm_primitive_type(*prim),
            Type::Array { element, size } => {
                let elem_ty = self.llvm_type(element)?;
                let size = size.unwrap_or(0) as u32;
                Ok(elem_ty.array_type(size).into())
            }
            Type::Slice { element } => {
                // Slices are represented as { ptr, len }
                let elem_ty = self.llvm_type(element)?;
                let ptr_ty = self.context.ptr_type(AddressSpace::default());
                let len_ty = self.context.i64_type();
                Ok(self
                    .context
                    .struct_type(&[ptr_ty.into(), len_ty.into()], false)
                    .into())
            }
            Type::Tuple(types) => {
                let field_types: Vec<BasicTypeEnum> = types
                    .iter()
                    .map(|t| self.llvm_type(t))
                    .collect::<CodegenResult<Vec<_>>>()?;
                Ok(self.context.struct_type(&field_types, false).into())
            }
            Type::Function(_) => {
                // Function pointers
                Ok(self.context.ptr_type(AddressSpace::default()).into())
            }
            Type::Prototype(proto) => {
                let field_types: Vec<BasicTypeEnum> = proto
                    .members
                    .iter()
                    .map(|(_, t)| self.llvm_type(t))
                    .collect::<CodegenResult<Vec<_>>>()?;
                Ok(self.context.struct_type(&field_types, false).into())
            }
            Type::HttpStatus(_) => Ok(self.context.i16_type().into()),
            Type::Ref { .. } | Type::Rc(_) => {
                Ok(self.context.ptr_type(AddressSpace::default()).into())
            }
            Type::Future(_) => {
                // Futures are represented as opaque pointers for now
                Ok(self.context.ptr_type(AddressSpace::default()).into())
            }
            Type::Unit => {
                // Unit type is an empty struct
                Ok(self.context.struct_type(&[], false).into())
            }
            Type::Never => {
                // Never type - use i8 as placeholder
                Ok(self.context.i8_type().into())
            }
            Type::Unknown => {
                Err(CodegenError::UnsupportedType(
                    "internal error: unresolved type 'Unknown' reached code generation (type checking may have failed)".into()
                ))
            }
            Type::TypeVar(id) => {
                Err(CodegenError::UnsupportedType(
                    format!("internal error: unresolved type variable 'T{}' reached code generation", id.0)
                ))
            }
            Type::Applied { .. } => {
                Err(CodegenError::UnsupportedType(
                    "generic type application not yet supported in code generation".into()
                ))
            }
        }
    }

    /// Convert a W# primitive type to an LLVM type.
    fn llvm_primitive_type(&self, prim: PrimitiveType) -> CodegenResult<BasicTypeEnum<'ctx>> {
        Ok(match prim {
            PrimitiveType::I8 => self.context.i8_type().into(),
            PrimitiveType::I16 => self.context.i16_type().into(),
            PrimitiveType::I32 => self.context.i32_type().into(),
            PrimitiveType::I64 => self.context.i64_type().into(),
            PrimitiveType::I128 => self.context.i128_type().into(),
            PrimitiveType::U8 => self.context.i8_type().into(),
            PrimitiveType::U16 => self.context.i16_type().into(),
            PrimitiveType::U32 => self.context.i32_type().into(),
            PrimitiveType::U64 => self.context.i64_type().into(),
            PrimitiveType::U128 => self.context.i128_type().into(),
            PrimitiveType::F32 => self.context.f32_type().into(),
            PrimitiveType::F64 => self.context.f64_type().into(),
            PrimitiveType::Bool => self.context.bool_type().into(),
            PrimitiveType::Char => self.context.i32_type().into(), // UTF-32
            PrimitiveType::Str => self.context.ptr_type(AddressSpace::default()).into(),
        })
    }

    /// Check if a primitive type is signed.
    fn is_signed_primitive(&self, prim: PrimitiveType) -> bool {
        matches!(
            prim,
            PrimitiveType::I8
                | PrimitiveType::I16
                | PrimitiveType::I32
                | PrimitiveType::I64
                | PrimitiveType::I128
        )
    }
}

/// Context for generating a single function.
struct FunctionContext<'a, 'ctx> {
    /// Reference to the code generator.
    codegen: &'a CodeGenerator<'ctx>,

    /// The MIR body being generated.
    body: &'a MirBody,

    /// The LLVM function being built.
    func: FunctionValue<'ctx>,

    /// Mapping from MIR locals to LLVM stack slots.
    locals: HashMap<Local, PointerValue<'ctx>>,

    /// Mapping from MIR basic blocks to LLVM basic blocks.
    blocks: HashMap<BasicBlockId, BasicBlock<'ctx>>,
}

impl<'a, 'ctx> FunctionContext<'a, 'ctx> {
    fn new(
        codegen: &'a CodeGenerator<'ctx>,
        body: &'a MirBody,
        func: FunctionValue<'ctx>,
    ) -> CodegenResult<Self> {
        Ok(Self {
            codegen,
            body,
            func,
            locals: HashMap::new(),
            blocks: HashMap::new(),
        })
    }

    fn context(&self) -> &'ctx Context {
        self.codegen.context
    }

    fn builder(&self) -> &Builder<'ctx> {
        &self.codegen.builder
    }

    /// Generate code for the function body.
    fn codegen_body(&mut self) -> CodegenResult<()> {
        // Create entry block
        let entry = self.context().append_basic_block(self.func, "entry");
        self.builder().position_at_end(entry);

        // Allocate all locals on the stack
        for (i, decl) in self.body.locals.iter().enumerate() {
            let local = Local(i as u32);
            let ty = self.codegen.llvm_type(&decl.ty)?;
            let default_name = format!("_{}", i);
            let name = decl.name.as_deref().unwrap_or(&default_name);
            let alloca = self.builder().build_alloca(ty, name)?;
            self.locals.insert(local, alloca);
        }

        // Store function arguments
        for (i, param) in self.func.get_param_iter().enumerate() {
            let local = Local((i + 1) as u32); // Locals 1..=arg_count are parameters
            let alloca = self.locals[&local];
            self.builder().build_store(alloca, param)?;
        }

        // Create LLVM basic blocks for each MIR block
        for &block_id in self.body.basic_blocks.keys() {
            let name = format!("bb{}", block_id.0);
            let bb = self.context().append_basic_block(self.func, &name);
            self.blocks.insert(block_id, bb);
        }

        // Jump from entry to the first MIR block
        let first_block = self.blocks[&BasicBlockId::ENTRY];
        self.builder().build_unconditional_branch(first_block)?;

        // Generate code for each basic block
        for (&block_id, block) in &self.body.basic_blocks {
            let llvm_block = self.blocks[&block_id];
            self.builder().position_at_end(llvm_block);

            // Generate statements
            for stmt in &block.statements {
                self.codegen_statement(stmt)?;
            }

            // Generate terminator
            if let Some(ref term) = block.terminator {
                self.codegen_terminator(&term.kind)?;
            }
        }

        Ok(())
    }

    /// Generate code for a statement.
    fn codegen_statement(&mut self, stmt: &wsharp_mir::Statement) -> CodegenResult<()> {
        match &stmt.kind {
            StatementKind::Assign(place, rvalue) => {
                let value = self.codegen_rvalue(rvalue)?;
                let ptr = self.place_to_ptr(place)?;
                self.builder().build_store(ptr, value)?;
            }
            StatementKind::StorageLive(_) | StatementKind::StorageDead(_) => {
                // No-op for now - could be used for stack coloring
            }
            StatementKind::Nop => {}
        }
        Ok(())
    }

    /// Generate code for a terminator.
    fn codegen_terminator(&mut self, kind: &TerminatorKind) -> CodegenResult<()> {
        match kind {
            TerminatorKind::Goto { target } => {
                let bb = self.blocks[target];
                self.builder().build_unconditional_branch(bb)?;
            }

            TerminatorKind::SwitchInt { discr, targets } => {
                let value = self.codegen_operand(discr)?;
                let int_value = value.into_int_value();

                if targets.targets.len() == 1 && targets.targets[0].0 == 0 {
                    // This is an if-else pattern
                    let then_bb = self.blocks[&targets.otherwise];
                    let else_bb = self.blocks[&targets.targets[0].1];
                    self.builder()
                        .build_conditional_branch(int_value, then_bb, else_bb)?;
                } else {
                    // General switch
                    let else_bb = self.blocks[&targets.otherwise];
                    let cases: Vec<_> = targets
                        .targets
                        .iter()
                        .map(|(val, bb)| {
                            let int_ty = int_value.get_type();
                            let const_val = int_ty.const_int(*val as u64, false);
                            (const_val, self.blocks[bb])
                        })
                        .collect();

                    self.builder().build_switch(int_value, else_bb, &cases)?;
                }
            }

            TerminatorKind::Return => {
                let return_place = self.locals[&Local::RETURN_PLACE];
                let return_ty = self.codegen.llvm_type(&self.body.return_ty)?;
                let return_val = self.builder().build_load(return_ty, return_place, "retval")?;
                self.builder().build_return(Some(&return_val))?;
            }

            TerminatorKind::Unreachable => {
                self.builder().build_unreachable()?;
            }

            TerminatorKind::Call {
                func,
                args,
                destination,
                target,
            } => {
                let callee = self.codegen_operand(func)?;
                let arg_values: Vec<BasicMetadataValueEnum> = args
                    .iter()
                    .map(|a| self.codegen_operand(a).map(|v| v.into()))
                    .collect::<CodegenResult<Vec<_>>>()?;

                // Get function type from callee
                let callee_ptr = callee.into_pointer_value();

                // Try to get the function from the callee
                if let Some(call_result) = self.build_call(callee_ptr, &arg_values)? {
                    let dest_ptr = self.place_to_ptr(destination)?;
                    self.builder().build_store(dest_ptr, call_result)?;
                }

                if let Some(next) = target {
                    let bb = self.blocks[next];
                    self.builder().build_unconditional_branch(bb)?;
                } else {
                    self.builder().build_unreachable()?;
                }
            }

            TerminatorKind::Drop { place, target } => {
                // For now, drops are no-ops (RC decrement would go here)
                let bb = self.blocks[target];
                self.builder().build_unconditional_branch(bb)?;
            }

            TerminatorKind::Assert {
                cond,
                expected,
                msg,
                target,
            } => {
                let cond_val = self.codegen_operand(cond)?.into_int_value();
                let expected_val = self.context().bool_type().const_int(*expected as u64, false);
                let check = self.builder().build_int_compare(
                    IntPredicate::EQ,
                    cond_val,
                    expected_val,
                    "assert_check",
                )?;

                let fail_bb = self.context().append_basic_block(self.func, "assert_fail");
                let ok_bb = self.blocks[target];

                self.builder()
                    .build_conditional_branch(check, ok_bb, fail_bb)?;

                // Assert failure block - just trap for now
                self.builder().position_at_end(fail_bb);
                self.builder().build_unreachable()?;
            }

            TerminatorKind::Yield { value, resume } => {
                // Yield is a no-op for now (async state machine would handle this)
                let bb = self.blocks[resume];
                self.builder().build_unconditional_branch(bb)?;
            }
        }
        Ok(())
    }

    /// Build a function call.
    fn build_call(
        &self,
        callee: PointerValue<'ctx>,
        args: &[BasicMetadataValueEnum<'ctx>],
    ) -> CodegenResult<Option<BasicValueEnum<'ctx>>> {
        // For now, assume all functions return i64 as a placeholder
        // In a full implementation, we'd track function types properly
        let fn_type = self.context().i64_type().fn_type(
            &args.iter().map(|_| self.context().i64_type().into()).collect::<Vec<_>>(),
            false,
        );

        let call = self
            .builder()
            .build_indirect_call(fn_type, callee, args, "call")?;

        // Get the return value - for non-void functions, this should work
        // try_as_basic_value returns Either which implements left() on nightly,
        // but for stable we need a different approach
        let call_site_val = call.as_any_value_enum();
        if call_site_val.is_int_value() {
            Ok(Some(call_site_val.into_int_value().into()))
        } else if call_site_val.is_float_value() {
            Ok(Some(call_site_val.into_float_value().into()))
        } else if call_site_val.is_pointer_value() {
            Ok(Some(call_site_val.into_pointer_value().into()))
        } else if call_site_val.is_struct_value() {
            Ok(Some(call_site_val.into_struct_value().into()))
        } else if call_site_val.is_array_value() {
            Ok(Some(call_site_val.into_array_value().into()))
        } else {
            // Void return or other non-basic value
            Ok(None)
        }
    }

    /// Generate code for an rvalue.
    fn codegen_rvalue(&mut self, rvalue: &Rvalue) -> CodegenResult<BasicValueEnum<'ctx>> {
        match rvalue {
            Rvalue::Use(op) => self.codegen_operand(op),

            Rvalue::Repeat(op, count) => {
                let elem = self.codegen_operand(op)?;
                let elem_ty = elem.get_type();
                let array_ty = elem_ty.array_type(*count as u32);

                let mut array = array_ty.const_zero();
                for i in 0..*count {
                    array = self.builder().build_insert_value(array, elem, i as u32, "repeat")?.into_array_value();
                }
                Ok(array.into())
            }

            Rvalue::Ref(place, _kind) => {
                let ptr = self.place_to_ptr(place)?;
                Ok(ptr.into())
            }

            Rvalue::AddressOf(place, _mutability) => {
                let ptr = self.place_to_ptr(place)?;
                Ok(ptr.into())
            }

            Rvalue::Len(place) => {
                // For arrays, return the compile-time length
                // For slices, load the length field
                let local = place.local;
                let decl = &self.body.locals[local.0 as usize];

                if let Type::Array { size: Some(len), .. } = &decl.ty {
                    Ok(self.context().i64_type().const_int(*len as u64, false).into())
                } else if let Type::Slice { .. } = &decl.ty {
                    // Slice is { ptr, len } - get the len field
                    let ptr = self.place_to_ptr(place)?;
                    let slice_ty = self.codegen.llvm_type(&decl.ty)?;
                    let slice = self.builder().build_load(slice_ty, ptr, "slice")?;
                    let len = self.builder().build_extract_value(slice.into_struct_value(), 1, "len")?;
                    Ok(len)
                } else {
                    Err(CodegenError::UnsupportedOperation("Len on non-array/slice".into()))
                }
            }

            Rvalue::BinaryOp(op, left, right) => {
                let lhs = self.codegen_operand(left)?;
                let rhs = self.codegen_operand(right)?;
                self.codegen_binop(*op, lhs, rhs)
            }

            Rvalue::CheckedBinaryOp(op, left, right) => {
                // For now, just do the operation without overflow checking
                let lhs = self.codegen_operand(left)?;
                let rhs = self.codegen_operand(right)?;
                let result = self.codegen_binop(*op, lhs, rhs)?;

                // Return (result, false) tuple
                let tuple_ty = self.context().struct_type(
                    &[result.get_type(), self.context().bool_type().into()],
                    false,
                );
                let mut tuple = tuple_ty.const_zero();
                tuple = self.builder().build_insert_value(tuple, result, 0, "res")?.into_struct_value();
                tuple = self.builder().build_insert_value(tuple, self.context().bool_type().const_zero(), 1, "overflow")?.into_struct_value();
                Ok(tuple.into())
            }

            Rvalue::UnaryOp(op, operand) => {
                let val = self.codegen_operand(operand)?;
                match op {
                    UnOp::Neg => {
                        if val.is_int_value() {
                            Ok(self.builder().build_int_neg(val.into_int_value(), "neg")?.into())
                        } else {
                            Ok(self.builder().build_float_neg(val.into_float_value(), "fneg")?.into())
                        }
                    }
                    UnOp::Not => {
                        if val.is_int_value() {
                            Ok(self.builder().build_not(val.into_int_value(), "not")?.into())
                        } else {
                            Err(CodegenError::UnsupportedOperation("Not on non-integer".into()))
                        }
                    }
                }
            }

            Rvalue::NullaryOp(op, ty) => {
                let llvm_ty = self.codegen.llvm_type(ty)?;
                match op {
                    wsharp_mir::NullOp::SizeOf => {
                        let size = llvm_ty.size_of().ok_or_else(|| {
                            CodegenError::UnsupportedOperation("SizeOf on unsized type".into())
                        })?;
                        Ok(size.into())
                    }
                    wsharp_mir::NullOp::AlignOf => {
                        // Placeholder - would need target data layout
                        Ok(self.context().i64_type().const_int(8, false).into())
                    }
                }
            }

            Rvalue::Cast(kind, operand, target_ty) => {
                let val = self.codegen_operand(operand)?;
                let target = self.codegen.llvm_type(target_ty)?;
                self.codegen_cast(*kind, val, target)
            }

            Rvalue::Discriminant(place) => {
                // For now, just return 0
                Ok(self.context().i64_type().const_zero().into())
            }

            Rvalue::Aggregate(kind, operands) => {
                let values: Vec<BasicValueEnum> = operands
                    .iter()
                    .map(|op| self.codegen_operand(op))
                    .collect::<CodegenResult<Vec<_>>>()?;

                match kind {
                    AggregateKind::Tuple => {
                        let types: Vec<BasicTypeEnum> = values.iter().map(|v| v.get_type()).collect();
                        let struct_ty = self.context().struct_type(&types, false);
                        let mut agg = struct_ty.const_zero();
                        for (i, val) in values.iter().enumerate() {
                            agg = self.builder().build_insert_value(agg, *val, i as u32, "tuple")?.into_struct_value();
                        }
                        Ok(agg.into())
                    }
                    AggregateKind::Array(elem_ty) => {
                        let llvm_elem_ty = self.codegen.llvm_type(elem_ty)?;
                        let array_ty = llvm_elem_ty.array_type(values.len() as u32);
                        let mut agg = array_ty.const_zero();
                        for (i, val) in values.iter().enumerate() {
                            agg = self.builder().build_insert_value(agg, *val, i as u32, "array")?.into_array_value();
                        }
                        Ok(agg.into())
                    }
                    AggregateKind::Adt { .. } => {
                        let types: Vec<BasicTypeEnum> = values.iter().map(|v| v.get_type()).collect();
                        let struct_ty = self.context().struct_type(&types, false);
                        let mut agg = struct_ty.const_zero();
                        for (i, val) in values.iter().enumerate() {
                            agg = self.builder().build_insert_value(agg, *val, i as u32, "adt")?.into_struct_value();
                        }
                        Ok(agg.into())
                    }
                    AggregateKind::Closure { .. } => {
                        // Closures are represented as structs containing captures
                        let types: Vec<BasicTypeEnum> = values.iter().map(|v| v.get_type()).collect();
                        let struct_ty = self.context().struct_type(&types, false);
                        let mut agg = struct_ty.const_zero();
                        for (i, val) in values.iter().enumerate() {
                            agg = self.builder().build_insert_value(agg, *val, i as u32, "closure")?.into_struct_value();
                        }
                        Ok(agg.into())
                    }
                }
            }

            Rvalue::ShallowInitBox(operand, ty) => {
                // Allocate memory for the box - placeholder implementation
                let val = self.codegen_operand(operand)?;
                Ok(val)
            }
        }
    }

    /// Generate code for a binary operation.
    fn codegen_binop(
        &self,
        op: BinOp,
        lhs: BasicValueEnum<'ctx>,
        rhs: BasicValueEnum<'ctx>,
    ) -> CodegenResult<BasicValueEnum<'ctx>> {
        let is_float = lhs.is_float_value();

        match op {
            BinOp::Add => {
                if is_float {
                    Ok(self
                        .builder()
                        .build_float_add(lhs.into_float_value(), rhs.into_float_value(), "fadd")?
                        .into())
                } else {
                    Ok(self
                        .builder()
                        .build_int_add(lhs.into_int_value(), rhs.into_int_value(), "add")?
                        .into())
                }
            }
            BinOp::Sub => {
                if is_float {
                    Ok(self
                        .builder()
                        .build_float_sub(lhs.into_float_value(), rhs.into_float_value(), "fsub")?
                        .into())
                } else {
                    Ok(self
                        .builder()
                        .build_int_sub(lhs.into_int_value(), rhs.into_int_value(), "sub")?
                        .into())
                }
            }
            BinOp::Mul => {
                if is_float {
                    Ok(self
                        .builder()
                        .build_float_mul(lhs.into_float_value(), rhs.into_float_value(), "fmul")?
                        .into())
                } else {
                    Ok(self
                        .builder()
                        .build_int_mul(lhs.into_int_value(), rhs.into_int_value(), "mul")?
                        .into())
                }
            }
            BinOp::Div => {
                if is_float {
                    Ok(self
                        .builder()
                        .build_float_div(lhs.into_float_value(), rhs.into_float_value(), "fdiv")?
                        .into())
                } else {
                    // TODO: Check signedness properly
                    Ok(self
                        .builder()
                        .build_int_signed_div(lhs.into_int_value(), rhs.into_int_value(), "sdiv")?
                        .into())
                }
            }
            BinOp::Rem => {
                if is_float {
                    Ok(self
                        .builder()
                        .build_float_rem(lhs.into_float_value(), rhs.into_float_value(), "frem")?
                        .into())
                } else {
                    Ok(self
                        .builder()
                        .build_int_signed_rem(lhs.into_int_value(), rhs.into_int_value(), "srem")?
                        .into())
                }
            }
            BinOp::BitAnd => Ok(self
                .builder()
                .build_and(lhs.into_int_value(), rhs.into_int_value(), "and")?
                .into()),
            BinOp::BitOr => Ok(self
                .builder()
                .build_or(lhs.into_int_value(), rhs.into_int_value(), "or")?
                .into()),
            BinOp::BitXor => Ok(self
                .builder()
                .build_xor(lhs.into_int_value(), rhs.into_int_value(), "xor")?
                .into()),
            BinOp::Shl => Ok(self
                .builder()
                .build_left_shift(lhs.into_int_value(), rhs.into_int_value(), "shl")?
                .into()),
            BinOp::Shr => Ok(self
                .builder()
                .build_right_shift(lhs.into_int_value(), rhs.into_int_value(), true, "shr")?
                .into()),
            BinOp::Eq => {
                if is_float {
                    Ok(self
                        .builder()
                        .build_float_compare(
                            FloatPredicate::OEQ,
                            lhs.into_float_value(),
                            rhs.into_float_value(),
                            "feq",
                        )?
                        .into())
                } else {
                    Ok(self
                        .builder()
                        .build_int_compare(
                            IntPredicate::EQ,
                            lhs.into_int_value(),
                            rhs.into_int_value(),
                            "eq",
                        )?
                        .into())
                }
            }
            BinOp::Ne => {
                if is_float {
                    Ok(self
                        .builder()
                        .build_float_compare(
                            FloatPredicate::ONE,
                            lhs.into_float_value(),
                            rhs.into_float_value(),
                            "fne",
                        )?
                        .into())
                } else {
                    Ok(self
                        .builder()
                        .build_int_compare(
                            IntPredicate::NE,
                            lhs.into_int_value(),
                            rhs.into_int_value(),
                            "ne",
                        )?
                        .into())
                }
            }
            BinOp::Lt => {
                if is_float {
                    Ok(self
                        .builder()
                        .build_float_compare(
                            FloatPredicate::OLT,
                            lhs.into_float_value(),
                            rhs.into_float_value(),
                            "flt",
                        )?
                        .into())
                } else {
                    Ok(self
                        .builder()
                        .build_int_compare(
                            IntPredicate::SLT,
                            lhs.into_int_value(),
                            rhs.into_int_value(),
                            "slt",
                        )?
                        .into())
                }
            }
            BinOp::Le => {
                if is_float {
                    Ok(self
                        .builder()
                        .build_float_compare(
                            FloatPredicate::OLE,
                            lhs.into_float_value(),
                            rhs.into_float_value(),
                            "fle",
                        )?
                        .into())
                } else {
                    Ok(self
                        .builder()
                        .build_int_compare(
                            IntPredicate::SLE,
                            lhs.into_int_value(),
                            rhs.into_int_value(),
                            "sle",
                        )?
                        .into())
                }
            }
            BinOp::Gt => {
                if is_float {
                    Ok(self
                        .builder()
                        .build_float_compare(
                            FloatPredicate::OGT,
                            lhs.into_float_value(),
                            rhs.into_float_value(),
                            "fgt",
                        )?
                        .into())
                } else {
                    Ok(self
                        .builder()
                        .build_int_compare(
                            IntPredicate::SGT,
                            lhs.into_int_value(),
                            rhs.into_int_value(),
                            "sgt",
                        )?
                        .into())
                }
            }
            BinOp::Ge => {
                if is_float {
                    Ok(self
                        .builder()
                        .build_float_compare(
                            FloatPredicate::OGE,
                            lhs.into_float_value(),
                            rhs.into_float_value(),
                            "fge",
                        )?
                        .into())
                } else {
                    Ok(self
                        .builder()
                        .build_int_compare(
                            IntPredicate::SGE,
                            lhs.into_int_value(),
                            rhs.into_int_value(),
                            "sge",
                        )?
                        .into())
                }
            }
            BinOp::Offset => {
                // Pointer offset
                let ptr = lhs.into_pointer_value();
                let offset = rhs.into_int_value();
                let result = unsafe {
                    self.builder().build_gep(
                        self.context().i8_type(),
                        ptr,
                        &[offset],
                        "offset",
                    )?
                };
                Ok(result.into())
            }
        }
    }

    /// Generate code for a type cast.
    fn codegen_cast(
        &self,
        kind: CastKind,
        val: BasicValueEnum<'ctx>,
        target: BasicTypeEnum<'ctx>,
    ) -> CodegenResult<BasicValueEnum<'ctx>> {
        match kind {
            CastKind::IntToInt => {
                let src = val.into_int_value();
                let dst_ty = target.into_int_type();
                let src_bits = src.get_type().get_bit_width();
                let dst_bits = dst_ty.get_bit_width();

                if src_bits < dst_bits {
                    // Sign extend (TODO: check signedness)
                    Ok(self.builder().build_int_s_extend(src, dst_ty, "sext")?.into())
                } else if src_bits > dst_bits {
                    Ok(self.builder().build_int_truncate(src, dst_ty, "trunc")?.into())
                } else {
                    Ok(val)
                }
            }
            CastKind::IntToFloat => {
                let src = val.into_int_value();
                let dst_ty = target.into_float_type();
                Ok(self
                    .builder()
                    .build_signed_int_to_float(src, dst_ty, "sitofp")?
                    .into())
            }
            CastKind::FloatToInt => {
                let src = val.into_float_value();
                let dst_ty = target.into_int_type();
                Ok(self
                    .builder()
                    .build_float_to_signed_int(src, dst_ty, "fptosi")?
                    .into())
            }
            CastKind::FloatToFloat => {
                let src = val.into_float_value();
                let dst_ty = target.into_float_type();
                let src_bits = src.get_type().size_of().get_zero_extended_constant().unwrap_or(0);
                let dst_bits = dst_ty.size_of().get_zero_extended_constant().unwrap_or(0);

                if src_bits < dst_bits {
                    Ok(self.builder().build_float_ext(src, dst_ty, "fpext")?.into())
                } else if src_bits > dst_bits {
                    Ok(self.builder().build_float_trunc(src, dst_ty, "fptrunc")?.into())
                } else {
                    Ok(val)
                }
            }
            CastKind::PtrToPtr | CastKind::FnPtrToPtr => {
                let src = val.into_pointer_value();
                let dst_ty = target.into_pointer_type();
                Ok(self.builder().build_pointer_cast(src, dst_ty, "ptrcast")?.into())
            }
            CastKind::PointerExposeAddress => {
                let src = val.into_pointer_value();
                let dst_ty = target.into_int_type();
                Ok(self.builder().build_ptr_to_int(src, dst_ty, "ptrtoint")?.into())
            }
            CastKind::PointerFromExposedAddress => {
                let src = val.into_int_value();
                let dst_ty = target.into_pointer_type();
                Ok(self.builder().build_int_to_ptr(src, dst_ty, "inttoptr")?.into())
            }
        }
    }

    /// Generate code for an operand.
    fn codegen_operand(&self, operand: &Operand) -> CodegenResult<BasicValueEnum<'ctx>> {
        match operand {
            Operand::Copy(place) | Operand::Move(place) => {
                let ptr = self.place_to_ptr(place)?;
                // Determine the type to load: if there's a field projection, use
                // the field's type; otherwise use the local's type
                let ty = if let Some(last_elem) = place.projection.last() {
                    match last_elem {
                        PlaceElem::Field(_, field_ty) => self.codegen.llvm_type(field_ty)?,
                        PlaceElem::Index(_) | PlaceElem::ConstantIndex { .. } => {
                            // For array indexing, get the element type
                            let local = place.local;
                            let decl = &self.body.locals[local.0 as usize];
                            if let Type::Array { element, .. } = &decl.ty {
                                self.codegen.llvm_type(element)?
                            } else {
                                self.codegen.llvm_type(&decl.ty)?
                            }
                        }
                        PlaceElem::Deref => {
                            // For deref, get the inner type
                            let local = place.local;
                            let decl = &self.body.locals[local.0 as usize];
                            if let Type::Ref { inner, .. } = &decl.ty {
                                self.codegen.llvm_type(inner)?
                            } else {
                                self.codegen.llvm_type(&decl.ty)?
                            }
                        }
                        _ => {
                            let local = place.local;
                            let decl = &self.body.locals[local.0 as usize];
                            self.codegen.llvm_type(&decl.ty)?
                        }
                    }
                } else {
                    let local = place.local;
                    let decl = &self.body.locals[local.0 as usize];
                    self.codegen.llvm_type(&decl.ty)?
                };
                Ok(self.builder().build_load(ty, ptr, "load")?)
            }
            Operand::Constant(c) => self.codegen_constant(c),
        }
    }

    /// Generate code for a constant.
    fn codegen_constant(&self, constant: &Constant) -> CodegenResult<BasicValueEnum<'ctx>> {
        match constant {
            Constant::Int(v, ty) => {
                let llvm_ty = self.codegen.llvm_type(ty)?.into_int_type();
                Ok(llvm_ty.const_int(*v as u64, *v < 0).into())
            }
            Constant::Float(v, ty) => {
                let llvm_ty = self.codegen.llvm_type(ty)?.into_float_type();
                Ok(llvm_ty.const_float(*v).into())
            }
            Constant::Bool(v) => Ok(self
                .context()
                .bool_type()
                .const_int(*v as u64, false)
                .into()),
            Constant::Char(c) => Ok(self
                .context()
                .i32_type()
                .const_int(*c as u64, false)
                .into()),
            Constant::String(s) => {
                let global = self.builder().build_global_string_ptr(s, "str")?;
                Ok(global.as_pointer_value().into())
            }
            Constant::Unit => {
                let unit_ty = self.context().struct_type(&[], false);
                Ok(unit_ty.const_zero().into())
            }
            Constant::Function(body_id) => {
                if let Some(&func) = self.codegen.functions.get(body_id) {
                    Ok(func.as_global_value().as_pointer_value().into())
                } else {
                    Err(CodegenError::UndefinedFunction(format!("body_{}", body_id.0)))
                }
            }
            Constant::Null => {
                let ptr_ty = self.context().ptr_type(AddressSpace::default());
                Ok(ptr_ty.const_null().into())
            }
        }
    }

    /// Convert a MIR place to an LLVM pointer.
    fn place_to_ptr(&self, place: &Place) -> CodegenResult<PointerValue<'ctx>> {
        let mut ptr = self.locals[&place.local];

        for elem in &place.projection {
            match elem {
                PlaceElem::Deref => {
                    let local = place.local;
                    let decl = &self.body.locals[local.0 as usize];
                    let inner_ty = if let Type::Ref { inner, .. } = &decl.ty {
                        self.codegen.llvm_type(inner)?
                    } else {
                        self.codegen.llvm_type(&decl.ty)?
                    };
                    let loaded = self.builder().build_load(
                        self.context().ptr_type(AddressSpace::default()),
                        ptr,
                        "deref",
                    )?;
                    ptr = loaded.into_pointer_value();
                }
                PlaceElem::Field(idx, ty) => {
                    let field_ty = self.codegen.llvm_type(ty)?;
                    let local = place.local;
                    let decl = &self.body.locals[local.0 as usize];
                    let struct_ty = self.codegen.llvm_type(&decl.ty)?;
                    ptr = self.builder().build_struct_gep(
                        struct_ty.into_struct_type(),
                        ptr,
                        *idx as u32,
                        "field",
                    )?;
                }
                PlaceElem::Index(idx_local) => {
                    let idx_ptr = self.locals[idx_local];
                    let idx = self.builder().build_load(
                        self.context().i64_type(),
                        idx_ptr,
                        "idx",
                    )?.into_int_value();

                    let local = place.local;
                    let decl = &self.body.locals[local.0 as usize];
                    let elem_ty = if let Type::Array { element, .. } = &decl.ty {
                        self.codegen.llvm_type(element)?
                    } else {
                        self.context().i64_type().into()
                    };

                    let zero = self.context().i64_type().const_zero();
                    ptr = unsafe {
                        self.builder().build_gep(
                            elem_ty,
                            ptr,
                            &[zero, idx],
                            "idx_ptr",
                        )?
                    };
                }
                PlaceElem::ConstantIndex { offset, from_end, .. } => {
                    let local = place.local;
                    let decl = &self.body.locals[local.0 as usize];
                    let elem_ty = if let Type::Array { element, .. } = &decl.ty {
                        self.codegen.llvm_type(element)?
                    } else {
                        self.context().i64_type().into()
                    };

                    let idx = self.context().i64_type().const_int(*offset as u64, false);
                    let zero = self.context().i64_type().const_zero();
                    ptr = unsafe {
                        self.builder().build_gep(
                            elem_ty,
                            ptr,
                            &[zero, idx],
                            "const_idx",
                        )?
                    };
                }
                PlaceElem::Subslice { .. } | PlaceElem::Downcast(_) => {
                    // Not yet implemented
                }
            }
        }

        Ok(ptr)
    }
}
