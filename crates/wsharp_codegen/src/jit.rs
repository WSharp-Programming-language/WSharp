//! JIT compilation support using LLVM's ORC JIT.

use crate::codegen::CodeGenerator;
use crate::error::{CodegenError, CodegenResult};
use inkwell::context::Context;
use inkwell::execution_engine::ExecutionEngine;
use inkwell::targets::{InitializationConfig, Target};
use inkwell::OptimizationLevel;
use wsharp_mir::MirModule;

/// A JIT compiler for W#.
pub struct JitCompiler<'ctx> {
    context: &'ctx Context,
    execution_engine: Option<ExecutionEngine<'ctx>>,
}

impl<'ctx> JitCompiler<'ctx> {
    /// Create a new JIT compiler.
    pub fn new(context: &'ctx Context) -> Self {
        Self {
            context,
            execution_engine: None,
        }
    }

    /// Compile a MIR module and prepare it for JIT execution.
    pub fn compile(&mut self, mir_module: &MirModule) -> CodegenResult<()> {
        // Initialize native target for JIT
        Target::initialize_native(&InitializationConfig::default())
            .map_err(|e| CodegenError::LlvmError(e))?;

        let mut codegen = CodeGenerator::new(self.context, &mir_module.name);

        // Set native target triple
        let triple = inkwell::targets::TargetMachine::get_default_triple();
        let triple_str = triple.as_str().to_str().unwrap_or("unknown-unknown-unknown");
        codegen.set_target_triple(triple_str);

        // Generate LLVM IR
        codegen.codegen_module(mir_module)?;

        // Get the module
        let module = codegen.into_module();

        // Create execution engine
        let ee = module
            .create_jit_execution_engine(OptimizationLevel::Default)
            .map_err(|e| CodegenError::LlvmError(e.to_string()))?;

        self.execution_engine = Some(ee);
        Ok(())
    }

    /// Run the main function if it exists.
    ///
    /// Returns the exit code from main.
    ///
    /// # Safety
    ///
    /// The main function must have the signature `fn() -> i64`.
    pub unsafe fn run_main(&self) -> CodegenResult<i64> {
        let ee = self
            .execution_engine
            .as_ref()
            .ok_or_else(|| CodegenError::LlvmError("No module compiled".into()))?;

        // Get main function
        let main_fn = ee
            .get_function_value("main")
            .map_err(|e| CodegenError::LlvmError(e.to_string()))?;

        // Run it with no arguments
        let result = unsafe { ee.run_function(main_fn, &[]) };

        Ok(result.as_int(true) as i64)
    }

    /// Check if the execution engine is ready.
    pub fn is_ready(&self) -> bool {
        self.execution_engine.is_some()
    }
}

/// REPL environment for interactive W# execution.
pub struct ReplEnvironment<'ctx> {
    context: &'ctx Context,
    counter: u32,
}

impl<'ctx> ReplEnvironment<'ctx> {
    /// Create a new REPL environment.
    pub fn new(context: &'ctx Context) -> Self {
        Self { context, counter: 0 }
    }

    /// Execute a snippet and return the result.
    pub fn eval(&mut self, mir_module: &MirModule) -> CodegenResult<String> {
        let mut jit = JitCompiler::new(self.context);
        jit.compile(mir_module)?;

        // Try to run main
        match unsafe { jit.run_main() } {
            Ok(v) => Ok(format!("{}", v)),
            Err(e) => Err(e),
        }
    }

    /// Generate a unique name for anonymous expressions.
    pub fn fresh_name(&mut self) -> String {
        let name = format!("__anon_{}", self.counter);
        self.counter += 1;
        name
    }
}
