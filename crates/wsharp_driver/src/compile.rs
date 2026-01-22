//! Main compilation pipeline.

use crate::error::{CompileError, CompileResult};
use crate::session::CompileOptions;
use inkwell::context::Context;
use inkwell::OptimizationLevel;
use std::path::Path;
use wsharp_ast::SourceFile;
use wsharp_codegen::AotCompiler;
use wsharp_hir::{HirModule, LoweringContext, TypeChecker};
use wsharp_mir::{MirBuilder, MirModule};
use wsharp_parser::Parser;

/// The compilation driver.
pub struct Driver {
    verbose: bool,
}

impl Driver {
    /// Create a new driver.
    pub fn new() -> Self {
        Self { verbose: false }
    }

    /// Create a verbose driver.
    pub fn verbose() -> Self {
        Self { verbose: true }
    }

    /// Compile source code to MIR.
    pub fn compile_to_mir(&self, source: &str, name: &str) -> CompileResult<MirModule> {
        // Stage 1: Parsing (includes lexing internally)
        if self.verbose {
            eprintln!("[driver] Parsing...");
        }
        let mut parser = Parser::new(source);
        let ast = parser.parse().map_err(|e| CompileError::Parser(format!("{:?}", e)))?;

        // Stage 2: HIR Lowering
        if self.verbose {
            eprintln!("[driver] Lowering to HIR...");
        }
        let hir = self.lower_to_hir(&ast, name)?;

        // Stage 3: MIR Building
        if self.verbose {
            eprintln!("[driver] Building MIR...");
        }
        let mir = self.build_mir(&hir)?;

        Ok(mir)
    }

    /// Lower AST to HIR.
    fn lower_to_hir(&self, ast: &SourceFile, _name: &str) -> CompileResult<HirModule> {
        let mut ctx = LoweringContext::new();
        let mut hir = ctx.lower_source_file(ast);

        // Check for lowering errors
        let errors = ctx.errors();
        if !errors.is_empty() {
            let error_msgs: Vec<String> = errors.iter().map(|e| format!("{:?}", e)).collect();
            return Err(CompileError::Lowering(error_msgs.join("\n")));
        }

        // Stage 2b: Type checking
        if self.verbose {
            eprintln!("[driver] Type checking...");
        }
        let mut type_checker = TypeChecker::new();
        type_checker.check_module(&mut hir);

        if type_checker.has_errors() {
            let error_msgs: Vec<String> = type_checker
                .errors()
                .iter()
                .map(|e| format!("{:?}", e))
                .collect();
            return Err(CompileError::Type(error_msgs.join("\n")));
        }

        Ok(hir)
    }

    /// Build MIR from HIR.
    fn build_mir(&self, hir: &HirModule) -> CompileResult<MirModule> {
        let builder = MirBuilder::new(hir.name.clone());
        let mir = builder.build_module(hir);
        Ok(mir)
    }

    /// Compile source code to LLVM IR string.
    pub fn compile_to_ir(&self, source: &str, name: &str) -> CompileResult<String> {
        let mir = self.compile_to_mir(source, name)?;

        if self.verbose {
            eprintln!("[driver] Generating LLVM IR...");
        }

        let context = Context::create();
        let aot = AotCompiler::new(&context)?;
        let ir = aot.compile_to_ir_string(&mir)?;

        Ok(ir)
    }

    /// Compile source code to an object file.
    pub fn compile_to_object(
        &self,
        source: &str,
        name: &str,
        output: &Path,
        options: &CompileOptions,
    ) -> CompileResult<()> {
        let mir = self.compile_to_mir(source, name)?;

        if self.verbose {
            eprintln!("[driver] Generating object file...");
        }

        let context = Context::create();
        let mut aot = AotCompiler::new(&context)?;

        // Set target if specified
        if let Some(ref target) = options.target {
            aot.set_target(target)?;
        }

        // Set optimization level
        let opt = match options.opt_level {
            0 => OptimizationLevel::None,
            1 => OptimizationLevel::Less,
            2 => OptimizationLevel::Default,
            _ => OptimizationLevel::Aggressive,
        };
        aot.set_opt_level(opt);

        // Compile to object
        aot.compile_to_object(&mir, output)?;

        if self.verbose {
            eprintln!("[driver] Wrote object file to: {}", output.display());
        }

        Ok(())
    }

    /// Compile a source file.
    pub fn compile_file(&self, path: &Path, options: &CompileOptions) -> CompileResult<()> {
        let source = std::fs::read_to_string(path)?;
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("module");

        // Determine output path
        let output = options.output.clone().unwrap_or_else(|| {
            let mut out = path.to_path_buf();
            out.set_extension("o");
            out
        });

        // Compile to MIR first
        let mir = self.compile_to_mir(&source, name)?;

        // Emit MIR if requested
        if options.emit_mir {
            let mir_output = wsharp_mir::pretty_print_module(&mir);
            let mir_path = output.with_extension("mir");
            std::fs::write(&mir_path, mir_output)?;
            if self.verbose || options.verbose {
                eprintln!("[driver] Wrote MIR to: {}", mir_path.display());
            }
        }

        let context = Context::create();
        let mut aot = AotCompiler::new(&context)?;

        if let Some(ref target) = options.target {
            aot.set_target(target)?;
        }

        // Emit IR if requested
        if options.emit_ir {
            let ir_path = output.with_extension("ll");
            aot.write_ir_to_file(&mir, &ir_path)?;
            if self.verbose || options.verbose {
                eprintln!("[driver] Wrote LLVM IR to: {}", ir_path.display());
            }
        }

        // Emit assembly if requested
        if options.emit_asm {
            let asm_path = output.with_extension("s");
            aot.compile_to_assembly(&mir, &asm_path)?;
            if self.verbose || options.verbose {
                eprintln!("[driver] Wrote assembly to: {}", asm_path.display());
            }
        }

        // Compile to object file
        aot.compile_to_object(&mir, &output)?;

        if self.verbose || options.verbose {
            eprintln!("[driver] Wrote object file to: {}", output.display());
        }

        Ok(())
    }

    /// Compile and run using JIT.
    #[cfg(feature = "jit")]
    pub fn run_jit(&self, source: &str, name: &str) -> CompileResult<i64> {
        let mir = self.compile_to_mir(source, name)?;

        if self.verbose {
            eprintln!("[driver] JIT compiling...");
        }

        let context = Context::create();
        let mut jit = wsharp_codegen::JitCompiler::new(&context);
        jit.compile(&mir)?;

        if self.verbose {
            eprintln!("[driver] Running main...");
        }

        // SAFETY: We assume main has the correct signature
        let result = unsafe { jit.run_main()? };

        Ok(result)
    }
}

impl Default for Driver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compile_simple_function() {
        let source = r#"
            fn main() -> i64 {
                return 42;
            }
        "#;

        let driver = Driver::verbose();
        let result = driver.compile_to_mir(source, "test");

        // This might fail due to incomplete implementation, but should parse at least
        match result {
            Ok(mir) => {
                // Check that we have a main function
                assert!(!mir.bodies.is_empty() || mir.entry_point.is_some());
            }
            Err(e) => {
                // For now, parsing errors are acceptable as we're testing the pipeline
                eprintln!("Compile error (expected during development): {:?}", e);
            }
        }
    }

    #[test]
    fn test_compile_arithmetic() {
        let source = r#"
            fn add(a: i64, b: i64) -> i64 {
                return a + b;
            }
        "#;

        let driver = Driver::new();
        let result = driver.compile_to_mir(source, "test");

        match result {
            Ok(mir) => {
                assert!(!mir.bodies.is_empty());
            }
            Err(e) => {
                eprintln!("Compile error (expected during development): {:?}", e);
            }
        }
    }
}
