//! LLVM code generation from MIR.

use crate::error::{CodegenError, CodegenResult};
use inkwell::basic_block::BasicBlock;
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::types::{BasicMetadataTypeEnum, BasicType, BasicTypeEnum};
use inkwell::values::{AnyValue, BasicMetadataValueEnum, BasicValueEnum, FunctionValue, IntValue, PointerValue};
use inkwell::{AddressSpace, FloatPredicate, IntPredicate};
use std::collections::HashMap;
use wsharp_mir::{
    AggregateKind, BasicBlockId, BinOp, BodyId, CastKind, Constant,
    CoroutineInfo, CoroutineStateLayout, Local, MirBody, MirModule, Operand, Place, PlaceElem,
    Rvalue, StatementKind, TerminatorKind, UnOp,
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

    /// Mapping from MIR body IDs to their poll functions (for async).
    poll_functions: HashMap<BodyId, FunctionValue<'ctx>>,

    /// Mapping from MIR body IDs to their state struct types (for async).
    state_types: HashMap<BodyId, inkwell::types::StructType<'ctx>>,

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
            poll_functions: HashMap::new(),
            state_types: HashMap::new(),
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
        // First pass: declare all functions and poll functions for async
        for (&body_id, body) in &mir_module.bodies {
            if body.is_async && body.coroutine_info.is_some() {
                // For async functions, generate state type and poll function
                let coroutine_info = body.coroutine_info.as_ref().unwrap();
                let state_type = self.llvm_coroutine_state_type(&coroutine_info.state_layout)?;
                self.state_types.insert(body_id, state_type);

                let poll_func = self.declare_poll_function(body, state_type)?;
                self.poll_functions.insert(body_id, poll_func);

                // Also declare a wrapper function with the original signature
                let func = self.declare_function(body)?;
                self.functions.insert(body_id, func);
            } else {
                let func = self.declare_function(body)?;
                self.functions.insert(body_id, func);
            }
        }

        // Second pass: generate function bodies
        for (&body_id, body) in &mir_module.bodies {
            if body.is_async && body.coroutine_info.is_some() {
                // Generate poll function body
                let poll_func = self.poll_functions[&body_id];
                let state_type = self.state_types[&body_id];
                self.codegen_poll_function(body, poll_func, state_type)?;

                // Generate wrapper function that creates state and returns immediately
                // (for now, the wrapper just allocates state and calls poll in a loop)
                let func = self.functions[&body_id];
                self.codegen_async_wrapper_function(body, func, poll_func, state_type)?;
            } else {
                let func = self.functions[&body_id];
                self.codegen_function(body, func)?;
            }
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
        let mut func_ctx = FunctionContext::new(self, body, func, None)?;
        func_ctx.codegen_body()?;
        Ok(())
    }

    /// Declare a poll function for an async body.
    ///
    /// Poll function signature: `fn poll(state: *mut StateStruct) -> PollResult<T>`
    /// Where PollResult is `{ i8 discriminant, T value }`
    fn declare_poll_function(
        &self,
        body: &MirBody,
        state_type: inkwell::types::StructType<'ctx>,
    ) -> CodegenResult<FunctionValue<'ctx>> {
        // Get the result type (unwrap Future<T> to get T)
        let result_ty = match &body.return_ty {
            Type::Future(inner) => self.llvm_type(inner)?,
            other => self.llvm_type(other)?,
        };

        // PollResult<T> = { i8 discriminant, T value }
        let poll_result_ty = self.context.struct_type(
            &[self.context.i8_type().into(), result_ty],
            false,
        );

        // Parameter: pointer to state struct
        let state_ptr_ty = self.context.ptr_type(AddressSpace::default());
        let param_types: Vec<BasicMetadataTypeEnum> = vec![state_ptr_ty.into()];

        let fn_type = poll_result_ty.fn_type(&param_types, false);
        let poll_fn_name = format!("__poll_{}", body.name);
        let poll_func = self.module.add_function(&poll_fn_name, fn_type, None);

        // Name the parameter
        if let Some(param) = poll_func.get_first_param() {
            param.set_name("state");
        }

        Ok(poll_func)
    }

    /// Generate code for a poll function body.
    fn codegen_poll_function(
        &mut self,
        body: &MirBody,
        poll_func: FunctionValue<'ctx>,
        state_type: inkwell::types::StructType<'ctx>,
    ) -> CodegenResult<()> {
        let coroutine_info = body.coroutine_info.as_ref()
            .ok_or_else(|| CodegenError::LlvmError("missing coroutine_info".into()))?;

        let mut func_ctx = FunctionContext::new(self, body, poll_func, Some(CoroutineContext {
            state_type,
            coroutine_info: coroutine_info.clone(),
        }))?;
        func_ctx.codegen_poll_body()?;
        Ok(())
    }

    /// Generate a wrapper function for an async function.
    ///
    /// The wrapper allocates state on the stack, initializes it, and polls until complete.
    fn codegen_async_wrapper_function(
        &self,
        body: &MirBody,
        wrapper_func: FunctionValue<'ctx>,
        poll_func: FunctionValue<'ctx>,
        state_type: inkwell::types::StructType<'ctx>,
    ) -> CodegenResult<()> {
        // Create entry block
        let entry_bb = self.context.append_basic_block(wrapper_func, "entry");
        let poll_loop_bb = self.context.append_basic_block(wrapper_func, "poll_loop");
        let done_bb = self.context.append_basic_block(wrapper_func, "done");

        self.builder.position_at_end(entry_bb);

        // Allocate state struct on stack
        let state_ptr = self.builder.build_alloca(state_type, "state")
            .map_err(|e| CodegenError::LlvmError(e.to_string()))?;

        // Initialize __state field to 0
        let state_field_ptr = self.builder.build_struct_gep(state_type, state_ptr, 0, "__state")
            .map_err(|e| CodegenError::LlvmError(e.to_string()))?;
        let zero = self.context.i32_type().const_int(0, false);
        self.builder.build_store(state_field_ptr, zero)
            .map_err(|e| CodegenError::LlvmError(e.to_string()))?;

        // Store any function arguments into the state struct
        // Arguments are stored after __state and __result fields
        for (i, param) in wrapper_func.get_param_iter().enumerate() {
            // Field index: 2 + i (skip __state and __result)
            let field_idx = (2 + i) as u32;
            if field_idx < state_type.count_fields() {
                let arg_field_ptr = self.builder.build_struct_gep(state_type, state_ptr, field_idx, &format!("arg_{}", i))
                    .map_err(|e| CodegenError::LlvmError(e.to_string()))?;
                self.builder.build_store(arg_field_ptr, param)
                    .map_err(|e| CodegenError::LlvmError(e.to_string()))?;
            }
        }

        // Jump to poll loop
        self.builder.build_unconditional_branch(poll_loop_bb)
            .map_err(|e| CodegenError::LlvmError(e.to_string()))?;

        // Poll loop: call poll function until ready
        self.builder.position_at_end(poll_loop_bb);
        let poll_result = self.builder.build_call(poll_func, &[state_ptr.into()], "poll_result")
            .map_err(|e| CodegenError::LlvmError(e.to_string()))?;

        let poll_result_val = poll_result.as_any_value_enum();
        let poll_result_struct = if poll_result_val.is_struct_value() {
            poll_result_val.into_struct_value()
        } else {
            return Err(CodegenError::LlvmError("poll returned non-struct".into()));
        };

        // Extract discriminant (field 0)
        let discr = self.builder.build_extract_value(poll_result_struct, 0, "discr")
            .map_err(|e| CodegenError::LlvmError(e.to_string()))?;

        // Check if ready (discriminant == 1)
        let is_ready = self.builder.build_int_compare(
            IntPredicate::EQ,
            discr.into_int_value(),
            self.context.i8_type().const_int(1, false),
            "is_ready",
        ).map_err(|e| CodegenError::LlvmError(e.to_string()))?;

        self.builder.build_conditional_branch(is_ready, done_bb, poll_loop_bb)
            .map_err(|e| CodegenError::LlvmError(e.to_string()))?;

        // Done block: extract result and return
        self.builder.position_at_end(done_bb);
        let result_value = self.builder.build_extract_value(poll_result_struct, 1, "result")
            .map_err(|e| CodegenError::LlvmError(e.to_string()))?;

        self.builder.build_return(Some(&result_value))
            .map_err(|e| CodegenError::LlvmError(e.to_string()))?;

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

    // =========================================================================
    // Coroutine Codegen Support
    // =========================================================================

    /// Generate the LLVM struct type for a coroutine state.
    ///
    /// The state struct contains:
    /// - __state: u32 (current state index)
    /// - __result: T (the result type)
    /// - Saved locals that are live across yield points
    fn llvm_coroutine_state_type(
        &self,
        layout: &CoroutineStateLayout,
    ) -> CodegenResult<inkwell::types::StructType<'ctx>> {
        let field_types: Vec<BasicTypeEnum> = layout
            .fields
            .iter()
            .map(|(_, ty)| self.llvm_type(ty))
            .collect::<CodegenResult<Vec<_>>>()?;

        Ok(self.context.struct_type(&field_types, false))
    }

    /// Generate the PollResult<T> type.
    ///
    /// PollResult is represented as: { i8 discriminant, T value }
    /// - discriminant 0 = Pending
    /// - discriminant 1 = Ready
    fn llvm_poll_result_type(&self, result_ty: &Type) -> CodegenResult<inkwell::types::StructType<'ctx>> {
        let discr_ty = self.context.i8_type();
        let value_ty = self.llvm_type(result_ty)?;
        Ok(self.context.struct_type(&[discr_ty.into(), value_ty], false))
    }

    /// Create a Pending poll result value.
    fn create_pending_result(
        &self,
        poll_result_ty: inkwell::types::StructType<'ctx>,
    ) -> BasicValueEnum<'ctx> {
        let discr = self.context.i8_type().const_int(0, false);
        // Create undef for the value field since it's not used for Pending
        let value = poll_result_ty.get_field_type_at_index(1).unwrap();
        let undef_value = match value {
            BasicTypeEnum::IntType(t) => t.const_zero().into(),
            BasicTypeEnum::FloatType(t) => t.const_zero().into(),
            BasicTypeEnum::PointerType(t) => t.const_null().into(),
            BasicTypeEnum::StructType(t) => t.const_zero().into(),
            BasicTypeEnum::ArrayType(t) => t.const_zero().into(),
            BasicTypeEnum::VectorType(t) => t.const_zero().into(),
            BasicTypeEnum::ScalableVectorType(t) => t.const_zero().into(),
        };
        poll_result_ty.const_named_struct(&[discr.into(), undef_value]).into()
    }

    /// Create a Ready poll result value.
    fn create_ready_result(
        &self,
        poll_result_ty: inkwell::types::StructType<'ctx>,
        value: BasicValueEnum<'ctx>,
    ) -> BasicValueEnum<'ctx> {
        let discr = self.context.i8_type().const_int(1, false);
        poll_result_ty.const_named_struct(&[discr.into(), value]).into()
    }

    // =========================================================================
    // Executor Runtime Integration
    // =========================================================================

    /// Declare the executor runtime functions.
    ///
    /// These are extern "C" functions from wsharp_runtime that will be linked
    /// at runtime for async/await support.
    pub fn declare_executor_runtime(&self) {
        // wsharp_executor_new() -> *mut Executor
        let executor_ptr_ty = self.context.ptr_type(AddressSpace::default());
        let new_fn_ty = executor_ptr_ty.fn_type(&[], false);
        self.module.add_function("wsharp_executor_new", new_fn_ty, None);

        // wsharp_executor_destroy(*mut Executor)
        let destroy_fn_ty = self.context.void_type().fn_type(
            &[executor_ptr_ty.into()],
            false,
        );
        self.module.add_function("wsharp_executor_destroy", destroy_fn_ty, None);

        // wsharp_executor_run(*mut Executor)
        let run_fn_ty = self.context.void_type().fn_type(
            &[executor_ptr_ty.into()],
            false,
        );
        self.module.add_function("wsharp_executor_run", run_fn_ty, None);

        // wsharp_block_on_i64(*mut Executor, *mut state, poll_fn) -> i64
        let poll_fn_ptr_ty = self.context.ptr_type(AddressSpace::default());
        let state_ptr_ty = self.context.ptr_type(AddressSpace::default());
        let block_on_fn_ty = self.context.i64_type().fn_type(
            &[executor_ptr_ty.into(), state_ptr_ty.into(), poll_fn_ptr_ty.into()],
            false,
        );
        self.module.add_function("wsharp_block_on_i64", block_on_fn_ty, None);
    }

    /// Generate a synchronous wrapper for an async function.
    ///
    /// This creates a function that:
    /// 1. Allocates the coroutine state struct
    /// 2. Initializes the state
    /// 3. Creates an executor
    /// 4. Calls block_on to run the coroutine to completion
    /// 5. Returns the result
    pub fn generate_async_wrapper(
        &self,
        async_fn_name: &str,
        wrapper_name: &str,
        state_layout: &CoroutineStateLayout,
        return_ty: &Type,
    ) -> CodegenResult<FunctionValue<'ctx>> {
        // Create the wrapper function type
        let llvm_return_ty = self.llvm_type(return_ty)?;
        let wrapper_fn_ty = match llvm_return_ty {
            BasicTypeEnum::IntType(t) => t.fn_type(&[], false),
            BasicTypeEnum::FloatType(t) => t.fn_type(&[], false),
            BasicTypeEnum::PointerType(t) => t.fn_type(&[], false),
            BasicTypeEnum::StructType(t) => t.fn_type(&[], false),
            _ => return Err(CodegenError::UnsupportedType("unsupported async return type".into())),
        };

        let wrapper_fn = self.module.add_function(wrapper_name, wrapper_fn_ty, None);

        // Create entry block
        let entry_bb = self.context.append_basic_block(wrapper_fn, "entry");
        self.builder.position_at_end(entry_bb);

        // Allocate the coroutine state struct
        let state_ty = self.llvm_coroutine_state_type(state_layout)?;
        let state_ptr = self.builder.build_alloca(state_ty, "state")
            .map_err(|e| CodegenError::LlvmError(e.to_string()))?;

        // Initialize the state to 0 (initial state)
        let state_field_ptr = self.builder.build_struct_gep(state_ty, state_ptr, 0, "state_field")
            .map_err(|e| CodegenError::LlvmError(e.to_string()))?;
        let zero_state = self.context.i32_type().const_int(0, false);
        self.builder.build_store(state_field_ptr, zero_state)
            .map_err(|e| CodegenError::LlvmError(e.to_string()))?;

        // Get the poll function
        let poll_fn_name = format!("__poll_{}", async_fn_name);
        let poll_fn = self.module.get_function(&poll_fn_name);

        // Get executor runtime functions
        let executor_new = self.module.get_function("wsharp_executor_new");
        let executor_destroy = self.module.get_function("wsharp_executor_destroy");
        let block_on = self.module.get_function("wsharp_block_on_i64");

        // Check if we have the runtime functions
        if executor_new.is_none() || executor_destroy.is_none() || block_on.is_none() {
            // If executor runtime is not available, fall back to direct polling
            return self.generate_direct_poll_wrapper(
                wrapper_fn,
                state_ptr,
                state_ty,
                poll_fn,
                return_ty,
            );
        }

        let executor_new = executor_new.unwrap();
        let executor_destroy = executor_destroy.unwrap();
        let block_on = block_on.unwrap();

        // Create executor
        let executor_call = self.builder.build_call(executor_new, &[], "executor")
            .map_err(|e| CodegenError::LlvmError(e.to_string()))?;
        let executor_val = executor_call.as_any_value_enum();
        let executor = if executor_val.is_pointer_value() {
            executor_val.into_pointer_value().into()
        } else {
            return Err(CodegenError::LlvmError("executor_new returned non-pointer".into()));
        };

        // Cast state pointer to void*
        let void_state_ptr = self.builder.build_pointer_cast(
            state_ptr,
            self.context.ptr_type(AddressSpace::default()),
            "void_state",
        ).map_err(|e| CodegenError::LlvmError(e.to_string()))?;

        // Get poll function as pointer (or create a stub if not available)
        let poll_fn_ptr = match poll_fn {
            Some(f) => f.as_global_value().as_pointer_value(),
            None => {
                // Create a stub poll function that returns immediately
                self.context.ptr_type(AddressSpace::default()).const_null()
            }
        };

        // Call block_on
        let block_on_call = self.builder.build_call(
            block_on,
            &[
                executor,
                void_state_ptr.into(),
                poll_fn_ptr.into(),
            ],
            "result",
        )
        .map_err(|e| CodegenError::LlvmError(e.to_string()))?;
        let block_on_val = block_on_call.as_any_value_enum();
        let result = if block_on_val.is_int_value() {
            block_on_val.into_int_value().into()
        } else {
            return Err(CodegenError::LlvmError("block_on returned non-int".into()));
        };

        // Destroy executor
        self.builder.build_call(executor_destroy, &[executor.into()], "")
            .map_err(|e| CodegenError::LlvmError(e.to_string()))?;

        // Return result (may need type conversion)
        let final_result = self.convert_result_type(result, return_ty)?;
        self.builder.build_return(Some(&final_result))
            .map_err(|e| CodegenError::LlvmError(e.to_string()))?;

        Ok(wrapper_fn)
    }

    /// Generate a direct poll wrapper (fallback when executor runtime is not available).
    fn generate_direct_poll_wrapper(
        &self,
        wrapper_fn: FunctionValue<'ctx>,
        state_ptr: PointerValue<'ctx>,
        state_ty: inkwell::types::StructType<'ctx>,
        poll_fn: Option<FunctionValue<'ctx>>,
        return_ty: &Type,
    ) -> CodegenResult<FunctionValue<'ctx>> {
        // If no poll function exists, return a default value
        let poll_fn = match poll_fn {
            Some(f) => f,
            None => {
                let default_value = self.create_default_value(return_ty)?;
                self.builder.build_return(Some(&default_value))
                    .map_err(|e| CodegenError::LlvmError(e.to_string()))?;
                return Ok(wrapper_fn);
            }
        };

        // Cast state pointer to void*
        let void_state_ptr = self.builder.build_pointer_cast(
            state_ptr,
            self.context.ptr_type(AddressSpace::default()),
            "void_state",
        ).map_err(|e| CodegenError::LlvmError(e.to_string()))?;

        // Create a simple polling loop
        let loop_bb = self.context.append_basic_block(wrapper_fn, "poll_loop");
        let done_bb = self.context.append_basic_block(wrapper_fn, "done");

        self.builder.build_unconditional_branch(loop_bb)
            .map_err(|e| CodegenError::LlvmError(e.to_string()))?;

        // Poll loop
        self.builder.position_at_end(loop_bb);
        let null_ctx = self.context.ptr_type(AddressSpace::default()).const_null();
        let poll_call = self.builder.build_call(
            poll_fn,
            &[void_state_ptr.into(), null_ctx.into()],
            "poll_result",
        )
        .map_err(|e| CodegenError::LlvmError(e.to_string()))?;
        let poll_val = poll_call.as_any_value_enum();
        let poll_result_struct = if poll_val.is_struct_value() {
            poll_val.into_struct_value()
        } else {
            return Err(CodegenError::LlvmError("poll function returned non-struct".into()));
        };

        // Check discriminant (field 0 of the PollResult struct)
        let discr = self.builder.build_extract_value(poll_result_struct, 0, "discr")
            .map_err(|e| CodegenError::LlvmError(e.to_string()))?;

        let is_ready = self.builder.build_int_compare(
            IntPredicate::EQ,
            discr.into_int_value(),
            self.context.i8_type().const_int(1, false),
            "is_ready",
        ).map_err(|e| CodegenError::LlvmError(e.to_string()))?;

        self.builder.build_conditional_branch(is_ready, done_bb, loop_bb)
            .map_err(|e| CodegenError::LlvmError(e.to_string()))?;

        // Done block - extract and return the result
        self.builder.position_at_end(done_bb);
        let result_value = self.builder.build_extract_value(poll_result_struct, 1, "result")
            .map_err(|e| CodegenError::LlvmError(e.to_string()))?;

        self.builder.build_return(Some(&result_value))
            .map_err(|e| CodegenError::LlvmError(e.to_string()))?;

        Ok(wrapper_fn)
    }

    /// Convert result from block_on (i64) to the expected return type.
    fn convert_result_type(
        &self,
        value: BasicValueEnum<'ctx>,
        target_ty: &Type,
    ) -> CodegenResult<BasicValueEnum<'ctx>> {
        let llvm_target = self.llvm_type(target_ty)?;

        // If types match, return as-is
        if value.get_type() == llvm_target {
            return Ok(value);
        }

        // Handle integer type conversions
        if let BasicTypeEnum::IntType(target_int) = llvm_target {
            if let BasicValueEnum::IntValue(src_int) = value {
                let src_bits = src_int.get_type().get_bit_width();
                let target_bits = target_int.get_bit_width();

                if src_bits > target_bits {
                    return Ok(self.builder.build_int_truncate(src_int, target_int, "trunc")
                        .map_err(|e| CodegenError::LlvmError(e.to_string()))?.into());
                } else if src_bits < target_bits {
                    return Ok(self.builder.build_int_s_extend(src_int, target_int, "sext")
                        .map_err(|e| CodegenError::LlvmError(e.to_string()))?.into());
                }
            }
        }

        Ok(value)
    }

    /// Create a default value for a type (used when async function has no poll function).
    fn create_default_value(&self, ty: &Type) -> CodegenResult<BasicValueEnum<'ctx>> {
        let llvm_ty = self.llvm_type(ty)?;
        Ok(match llvm_ty {
            BasicTypeEnum::IntType(t) => t.const_zero().into(),
            BasicTypeEnum::FloatType(t) => t.const_zero().into(),
            BasicTypeEnum::PointerType(t) => t.const_null().into(),
            BasicTypeEnum::StructType(t) => t.const_zero().into(),
            BasicTypeEnum::ArrayType(t) => t.const_zero().into(),
            BasicTypeEnum::VectorType(t) => t.const_zero().into(),
            BasicTypeEnum::ScalableVectorType(t) => t.const_zero().into(),
        })
    }
}

/// Coroutine-specific context for poll function generation.
struct CoroutineContext<'ctx> {
    /// The state struct type.
    state_type: inkwell::types::StructType<'ctx>,
    /// Coroutine metadata from MIR.
    coroutine_info: CoroutineInfo,
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

    /// Coroutine context (for poll functions).
    coroutine_ctx: Option<CoroutineContext<'ctx>>,

    /// State pointer local (for poll functions).
    state_ptr: Option<PointerValue<'ctx>>,

    /// Pending block for coroutines.
    pending_block: Option<BasicBlock<'ctx>>,

    /// Ready block for coroutines.
    ready_block: Option<BasicBlock<'ctx>>,
}

impl<'a, 'ctx> FunctionContext<'a, 'ctx> {
    fn new(
        codegen: &'a CodeGenerator<'ctx>,
        body: &'a MirBody,
        func: FunctionValue<'ctx>,
        coroutine_ctx: Option<CoroutineContext<'ctx>>,
    ) -> CodegenResult<Self> {
        Ok(Self {
            codegen,
            body,
            func,
            locals: HashMap::new(),
            blocks: HashMap::new(),
            coroutine_ctx,
            state_ptr: None,
            pending_block: None,
            ready_block: None,
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

    /// Generate code for a poll function body (async/coroutine).
    fn codegen_poll_body(&mut self) -> CodegenResult<()> {
        // Extract state type and coroutine info (we need to copy because we'll borrow self mutably)
        let (state_type, coroutine_info) = match &self.coroutine_ctx {
            Some(ctx) => (ctx.state_type, ctx.coroutine_info.clone()),
            None => return Err(CodegenError::LlvmError("missing coroutine context".into())),
        };

        // Create special blocks for the poll function
        let entry = self.context().append_basic_block(self.func, "entry");
        let dispatch = self.context().append_basic_block(self.func, "dispatch");
        let pending = self.context().append_basic_block(self.func, "pending");
        let ready = self.context().append_basic_block(self.func, "ready");

        self.pending_block = Some(pending);
        self.ready_block = Some(ready);

        self.builder().position_at_end(entry);

        // Get the state pointer from the first parameter
        let state_ptr = self.func.get_first_param()
            .ok_or_else(|| CodegenError::LlvmError("poll function missing state param".into()))?
            .into_pointer_value();
        self.state_ptr = Some(state_ptr);

        // Allocate all locals on the stack
        for (i, decl) in self.body.locals.iter().enumerate() {
            let local = Local(i as u32);
            // For the state pointer local, use the function parameter instead of allocating
            if decl.name.as_deref() == Some("__state_ptr") {
                // Map this local to the state pointer parameter
                self.locals.insert(local, state_ptr);
                continue;
            }
            let ty = self.codegen.llvm_type(&decl.ty)?;
            let default_name = format!("_{}", i);
            let name = decl.name.as_deref().unwrap_or(&default_name);
            let alloca = self.builder().build_alloca(ty, name)?;
            self.locals.insert(local, alloca);
        }

        // Create LLVM basic blocks for each MIR block
        for &block_id in self.body.basic_blocks.keys() {
            let name = format!("bb{}", block_id.0);
            let bb = self.context().append_basic_block(self.func, &name);
            self.blocks.insert(block_id, bb);
        }

        // Jump from entry to dispatch
        self.builder().build_unconditional_branch(dispatch)?;

        // Setup dispatch block - switch on __state field
        self.builder().position_at_end(dispatch);

        // Load __state from state struct (field 0)
        let state_field_ptr = self.builder().build_struct_gep(
            state_type,
            state_ptr,
            0,
            "__state_ptr",
        ).map_err(|e| CodegenError::LlvmError(e.to_string()))?;
        let state_val = self.builder().build_load(
            self.context().i32_type(),
            state_field_ptr,
            "__state",
        ).map_err(|e| CodegenError::LlvmError(e.to_string()))?;

        // Build switch on state value
        // State 0 -> entry block of original function
        // State N -> resume block for yield point N
        let entry_block = self.blocks.get(&BasicBlockId::ENTRY)
            .copied()
            .unwrap_or(pending);

        let mut cases: Vec<(IntValue<'ctx>, BasicBlock<'ctx>)> = vec![
            (self.context().i32_type().const_int(0, false), entry_block),
        ];

        for yp in &coroutine_info.yield_points {
            if let Some(&resume_bb) = self.blocks.get(&yp.resume_block) {
                cases.push((
                    self.context().i32_type().const_int(yp.state_index as u64, false),
                    resume_bb,
                ));
            }
        }

        // Default case goes to ready (completed state)
        self.builder().build_switch(
            state_val.into_int_value(),
            ready,
            &cases,
        ).map_err(|e| CodegenError::LlvmError(e.to_string()))?;

        // Generate code for each basic block
        for (&block_id, block) in &self.body.basic_blocks {
            let llvm_block = self.blocks[&block_id];
            self.builder().position_at_end(llvm_block);

            // For resume blocks, first restore locals from state
            for yp in &coroutine_info.yield_points {
                if yp.resume_block == block_id {
                    self.restore_locals_from_state(state_type, &coroutine_info, &yp.live_locals)?;
                    break;
                }
            }

            // Generate statements
            for stmt in &block.statements {
                self.codegen_statement(stmt)?;
            }

            // Generate terminator
            if let Some(ref term) = block.terminator {
                self.codegen_poll_terminator(&term.kind, state_type, &coroutine_info)?;
            }
        }

        // Setup pending block - return Pending
        self.builder().position_at_end(pending);
        let result_ty = match &self.body.return_ty {
            Type::Future(inner) => self.codegen.llvm_type(inner)?,
            other => self.codegen.llvm_type(other)?,
        };
        let poll_result_ty = self.context().struct_type(
            &[self.context().i8_type().into(), result_ty],
            false,
        );
        let pending_result = poll_result_ty.const_named_struct(&[
            self.context().i8_type().const_int(0, false).into(), // Pending = 0
            self.codegen.create_default_value(&self.body.return_ty)?,
        ]);
        self.builder().build_return(Some(&pending_result))?;

        // Setup ready block - return Ready with result
        self.builder().position_at_end(ready);

        // Load result from state struct (field 1)
        let result_field_ptr = self.builder().build_struct_gep(
            state_type,
            state_ptr,
            1,
            "__result_ptr",
        ).map_err(|e| CodegenError::LlvmError(e.to_string()))?;
        let result_val = self.builder().build_load(
            result_ty,
            result_field_ptr,
            "__result",
        ).map_err(|e| CodegenError::LlvmError(e.to_string()))?;

        let ready_result = self.builder().build_insert_value(
            poll_result_ty.const_zero(),
            self.context().i8_type().const_int(1, false), // Ready = 1
            0,
            "ready_discr",
        ).map_err(|e| CodegenError::LlvmError(e.to_string()))?;

        let ready_result = self.builder().build_insert_value(
            ready_result.into_struct_value(),
            result_val,
            1,
            "ready_result",
        ).map_err(|e| CodegenError::LlvmError(e.to_string()))?;

        self.builder().build_return(Some(&ready_result.into_struct_value()))?;

        Ok(())
    }

    /// Restore locals from state struct after resuming.
    fn restore_locals_from_state(
        &self,
        state_type: inkwell::types::StructType<'ctx>,
        coroutine_info: &CoroutineInfo,
        live_locals: &[Local],
    ) -> CodegenResult<()> {
        let state_ptr = self.state_ptr
            .ok_or_else(|| CodegenError::LlvmError("missing state pointer".into()))?;

        for &local in live_locals {
            if let Some(&field_idx) = coroutine_info.local_to_state_field.get(&local) {
                if let Some(&local_ptr) = self.locals.get(&local) {
                    let field_ty = self.codegen.llvm_type(&self.body.locals[local.0 as usize].ty)?;
                    let field_ptr = self.builder().build_struct_gep(
                        state_type,
                        state_ptr,
                        field_idx as u32,
                        &format!("state_field_{}", field_idx),
                    ).map_err(|e| CodegenError::LlvmError(e.to_string()))?;

                    let val = self.builder().build_load(field_ty, field_ptr, "restored")
                        .map_err(|e| CodegenError::LlvmError(e.to_string()))?;
                    self.builder().build_store(local_ptr, val)
                        .map_err(|e| CodegenError::LlvmError(e.to_string()))?;
                }
            }
        }
        Ok(())
    }

    /// Save locals to state struct before yielding.
    fn save_locals_to_state(
        &self,
        state_type: inkwell::types::StructType<'ctx>,
        coroutine_info: &CoroutineInfo,
        live_locals: &[Local],
    ) -> CodegenResult<()> {
        let state_ptr = self.state_ptr
            .ok_or_else(|| CodegenError::LlvmError("missing state pointer".into()))?;

        for &local in live_locals {
            if let Some(&field_idx) = coroutine_info.local_to_state_field.get(&local) {
                if let Some(&local_ptr) = self.locals.get(&local) {
                    let field_ty = self.codegen.llvm_type(&self.body.locals[local.0 as usize].ty)?;

                    let val = self.builder().build_load(field_ty, local_ptr, "to_save")
                        .map_err(|e| CodegenError::LlvmError(e.to_string()))?;

                    let field_ptr = self.builder().build_struct_gep(
                        state_type,
                        state_ptr,
                        field_idx as u32,
                        &format!("state_field_{}", field_idx),
                    ).map_err(|e| CodegenError::LlvmError(e.to_string()))?;

                    self.builder().build_store(field_ptr, val)
                        .map_err(|e| CodegenError::LlvmError(e.to_string()))?;
                }
            }
        }
        Ok(())
    }

    /// Generate code for a terminator in a poll function.
    fn codegen_poll_terminator(
        &mut self,
        kind: &TerminatorKind,
        state_type: inkwell::types::StructType<'ctx>,
        coroutine_info: &CoroutineInfo,
    ) -> CodegenResult<()> {
        match kind {
            TerminatorKind::Yield { value: _, resume } => {
                // Find the yield point info for this yield
                let yp_info = coroutine_info.yield_points
                    .iter()
                    .find(|yp| yp.resume_block == *resume);

                if let Some(yp) = yp_info {
                    // Save live locals to state
                    self.save_locals_to_state(state_type, coroutine_info, &yp.live_locals)?;

                    // Update __state to the next state
                    let state_ptr = self.state_ptr
                        .ok_or_else(|| CodegenError::LlvmError("missing state pointer".into()))?;
                    let state_field_ptr = self.builder().build_struct_gep(
                        state_type,
                        state_ptr,
                        0,
                        "__state_ptr",
                    ).map_err(|e| CodegenError::LlvmError(e.to_string()))?;
                    let next_state = self.context().i32_type().const_int(yp.state_index as u64, false);
                    self.builder().build_store(state_field_ptr, next_state)
                        .map_err(|e| CodegenError::LlvmError(e.to_string()))?;
                }

                // Branch to pending block
                let pending = self.pending_block
                    .ok_or_else(|| CodegenError::LlvmError("missing pending block".into()))?;
                self.builder().build_unconditional_branch(pending)?;
            }

            TerminatorKind::Return => {
                // Store result in state struct and branch to ready
                let state_ptr = self.state_ptr
                    .ok_or_else(|| CodegenError::LlvmError("missing state pointer".into()))?;

                // Load the return value from _0
                let return_local = Local::RETURN_PLACE;
                if let Some(&return_ptr) = self.locals.get(&return_local) {
                    let result_ty = match &self.body.return_ty {
                        Type::Future(inner) => self.codegen.llvm_type(inner)?,
                        other => self.codegen.llvm_type(other)?,
                    };
                    let return_val = self.builder().build_load(result_ty, return_ptr, "return_val")
                        .map_err(|e| CodegenError::LlvmError(e.to_string()))?;

                    // Store in __result field (field 1)
                    let result_field_ptr = self.builder().build_struct_gep(
                        state_type,
                        state_ptr,
                        1,
                        "__result_ptr",
                    ).map_err(|e| CodegenError::LlvmError(e.to_string()))?;
                    self.builder().build_store(result_field_ptr, return_val)
                        .map_err(|e| CodegenError::LlvmError(e.to_string()))?;
                }

                // Set __state to completed (u32::MAX)
                let state_field_ptr = self.builder().build_struct_gep(
                    state_type,
                    state_ptr,
                    0,
                    "__state_ptr",
                ).map_err(|e| CodegenError::LlvmError(e.to_string()))?;
                let done_state = self.context().i32_type().const_int(u32::MAX as u64, false);
                self.builder().build_store(state_field_ptr, done_state)
                    .map_err(|e| CodegenError::LlvmError(e.to_string()))?;

                // Branch to ready block
                let ready = self.ready_block
                    .ok_or_else(|| CodegenError::LlvmError("missing ready block".into()))?;
                self.builder().build_unconditional_branch(ready)?;
            }

            // For other terminators, use the normal codegen
            _ => self.codegen_terminator(kind)?,
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

            TerminatorKind::Yield { value: _, resume } => {
                // After coroutine transformation, Yield terminators are converted to:
                // 1. Save live locals to state struct
                // 2. Update __state field
                // 3. Branch to pending block (return Pending)
                //
                // If we reach here with an untransformed Yield, it means the async
                // function had no yield points or transformation wasn't applied.
                // In that case, just branch to resume block.
                //
                // For transformed coroutines, the MIR transformation has already
                // replaced Yield with Goto to pending block, so this code path
                // should not be reached for properly transformed async functions.
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
        // Derive parameter types from the actual argument values
        let param_types: Vec<BasicMetadataTypeEnum> = args
            .iter()
            .map(|arg| match arg {
                BasicMetadataValueEnum::IntValue(v) => v.get_type().into(),
                BasicMetadataValueEnum::FloatValue(v) => v.get_type().into(),
                BasicMetadataValueEnum::PointerValue(v) => v.get_type().into(),
                BasicMetadataValueEnum::StructValue(v) => v.get_type().into(),
                BasicMetadataValueEnum::ArrayValue(v) => v.get_type().into(),
                BasicMetadataValueEnum::VectorValue(v) => v.get_type().into(),
                BasicMetadataValueEnum::ScalableVectorValue(v) => v.get_type().into(),
                BasicMetadataValueEnum::MetadataValue(_) => self.context().i64_type().into(),
            })
            .collect();

        // For now, assume all functions return i64
        // TODO: Track actual return types
        let fn_type = self.context().i64_type().fn_type(&param_types, false);

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
        let mut ptr = *self.locals.get(&place.local).ok_or_else(|| {
            CodegenError::LlvmError(format!(
                "missing local {:?} in function '{}', available locals: {:?}",
                place.local,
                self.body.name,
                self.locals.keys().collect::<Vec<_>>()
            ))
        })?;

        for elem in &place.projection {
            match elem {
                PlaceElem::Deref => {
                    let local = place.local;
                    let decl = &self.body.locals[local.0 as usize];

                    // Check if this is the __state_ptr local in a coroutine
                    // In that case, the local is already the state pointer (function param),
                    // so we don't need to load/deref - just continue with the current ptr
                    if decl.name.as_deref() == Some("__state_ptr") && self.coroutine_ctx.is_some() {
                        // The state pointer is already loaded as the function parameter
                        // No need to deref, ptr is already pointing to the state struct
                        continue;
                    }

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
                    let _field_ty = self.codegen.llvm_type(ty)?;
                    let local = place.local;
                    let decl = &self.body.locals[local.0 as usize];

                    // Check if this is accessing fields through the __state_ptr
                    // In that case, use the coroutine's state type instead of the local's type
                    let struct_ty = if decl.name.as_deref() == Some("__state_ptr") {
                        if let Some(ref ctx) = self.coroutine_ctx {
                            ctx.state_type.into()
                        } else {
                            self.codegen.llvm_type(&decl.ty)?
                        }
                    } else {
                        self.codegen.llvm_type(&decl.ty)?
                    };

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
