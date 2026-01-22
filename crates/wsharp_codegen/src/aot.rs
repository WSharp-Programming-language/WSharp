//! Ahead-of-time compilation support.

use crate::codegen::CodeGenerator;
use crate::error::{CodegenError, CodegenResult};
use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::targets::{
    CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetMachine, TargetTriple,
};
use inkwell::OptimizationLevel;
use std::path::Path;
use wsharp_mir::MirModule;

/// AOT compiler for generating native executables.
pub struct AotCompiler<'ctx> {
    context: &'ctx Context,
    target_triple: TargetTriple,
    target_machine: Option<TargetMachine>,
    opt_level: OptimizationLevel,
}

impl<'ctx> AotCompiler<'ctx> {
    /// Create a new AOT compiler for the host platform.
    pub fn new(context: &'ctx Context) -> CodegenResult<Self> {
        // Initialize all targets
        Target::initialize_all(&InitializationConfig::default());

        let triple = TargetMachine::get_default_triple();

        Ok(Self {
            context,
            target_triple: triple,
            target_machine: None,
            opt_level: OptimizationLevel::Default,
        })
    }

    /// Set the target triple.
    pub fn set_target(&mut self, triple: &str) -> CodegenResult<()> {
        self.target_triple = TargetTriple::create(triple);
        self.target_machine = None; // Reset target machine
        Ok(())
    }

    /// Set the optimization level.
    pub fn set_opt_level(&mut self, level: OptimizationLevel) {
        self.opt_level = level;
        self.target_machine = None; // Reset target machine
    }

    /// Get or create the target machine.
    fn get_target_machine(&mut self) -> CodegenResult<&TargetMachine> {
        if self.target_machine.is_none() {
            let target = Target::from_triple(&self.target_triple)
                .map_err(|e| CodegenError::LlvmError(e.to_string()))?;

            let machine = target
                .create_target_machine(
                    &self.target_triple,
                    "generic",
                    "",
                    self.opt_level,
                    RelocMode::Default,
                    CodeModel::Default,
                )
                .ok_or_else(|| CodegenError::LlvmError("Failed to create target machine".into()))?;

            self.target_machine = Some(machine);
        }

        Ok(self.target_machine.as_ref().unwrap())
    }

    /// Compile a MIR module to LLVM IR.
    pub fn compile_to_ir(&self, mir_module: &MirModule) -> CodegenResult<Module<'ctx>> {
        let mut codegen = CodeGenerator::new(self.context, &mir_module.name);
        codegen.set_target_triple(&self.target_triple.to_string());
        codegen.codegen_module(mir_module)?;
        Ok(codegen.into_module())
    }

    /// Compile a MIR module to an LLVM IR string.
    pub fn compile_to_ir_string(&self, mir_module: &MirModule) -> CodegenResult<String> {
        let module = self.compile_to_ir(mir_module)?;
        Ok(module.print_to_string().to_string())
    }

    /// Compile a MIR module to an object file.
    pub fn compile_to_object(
        &mut self,
        mir_module: &MirModule,
        output_path: &Path,
    ) -> CodegenResult<()> {
        let module = self.compile_to_ir(mir_module)?;
        let target_machine = self.get_target_machine()?;

        target_machine
            .write_to_file(&module, FileType::Object, output_path)
            .map_err(|e| CodegenError::LlvmError(e.to_string()))?;

        Ok(())
    }

    /// Compile a MIR module to assembly.
    pub fn compile_to_assembly(
        &mut self,
        mir_module: &MirModule,
        output_path: &Path,
    ) -> CodegenResult<()> {
        let module = self.compile_to_ir(mir_module)?;
        let target_machine = self.get_target_machine()?;

        target_machine
            .write_to_file(&module, FileType::Assembly, output_path)
            .map_err(|e| CodegenError::LlvmError(e.to_string()))?;

        Ok(())
    }

    /// Write LLVM IR to a file.
    pub fn write_ir_to_file(
        &self,
        mir_module: &MirModule,
        output_path: &Path,
    ) -> CodegenResult<()> {
        let module = self.compile_to_ir(mir_module)?;
        module
            .print_to_file(output_path)
            .map_err(|e| CodegenError::LlvmError(e.to_string()))?;
        Ok(())
    }

    /// Write LLVM bitcode to a file.
    pub fn write_bitcode_to_file(
        &self,
        mir_module: &MirModule,
        output_path: &Path,
    ) -> CodegenResult<()> {
        let module = self.compile_to_ir(mir_module)?;
        module.write_bitcode_to_path(output_path);
        Ok(())
    }

    /// Get the default file extension for object files on this target.
    pub fn object_extension(&self) -> &'static str {
        let triple = self.target_triple.as_str().to_string_lossy();
        if triple.contains("windows") {
            "obj"
        } else {
            "o"
        }
    }

    /// Get the default file extension for executables on this target.
    pub fn executable_extension(&self) -> &'static str {
        let triple = self.target_triple.as_str().to_string_lossy();
        if triple.contains("windows") {
            "exe"
        } else {
            ""
        }
    }
}

/// Optimization passes for the compiled code.
pub struct OptimizationPipeline<'ctx> {
    module: Module<'ctx>,
}

impl<'ctx> OptimizationPipeline<'ctx> {
    /// Create a new optimization pipeline.
    pub fn new(module: Module<'ctx>) -> Self {
        Self { module }
    }

    /// Run the optimization pipeline.
    pub fn run(self, level: OptimizationLevel) -> Module<'ctx> {
        // Use the new pass manager approach
        // For now, just return the module as-is
        // A full implementation would use inkwell's pass manager
        self.module
    }

    /// Consume and return the module.
    pub fn into_module(self) -> Module<'ctx> {
        self.module
    }
}
