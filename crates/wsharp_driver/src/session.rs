//! Compilation session management.

use std::path::PathBuf;

/// Options for compilation.
#[derive(Clone, Debug)]
pub struct CompileOptions {
    /// The output file path.
    pub output: Option<PathBuf>,

    /// Whether to emit LLVM IR.
    pub emit_ir: bool,

    /// Whether to emit assembly.
    pub emit_asm: bool,

    /// Whether to emit MIR (for debugging).
    pub emit_mir: bool,

    /// Optimization level (0-3).
    pub opt_level: u32,

    /// Target triple (defaults to native).
    pub target: Option<String>,

    /// Whether to use JIT compilation instead of AOT.
    pub jit: bool,

    /// Verbose output.
    pub verbose: bool,
}

impl Default for CompileOptions {
    fn default() -> Self {
        Self {
            output: None,
            emit_ir: false,
            emit_asm: false,
            emit_mir: false,
            opt_level: 0,
            target: None,
            jit: false,
            verbose: false,
        }
    }
}

impl CompileOptions {
    /// Create options for JIT execution.
    pub fn jit() -> Self {
        Self {
            jit: true,
            ..Default::default()
        }
    }

    /// Create options for AOT compilation.
    pub fn aot(output: PathBuf) -> Self {
        Self {
            output: Some(output),
            ..Default::default()
        }
    }

    /// Set the optimization level.
    pub fn with_opt_level(mut self, level: u32) -> Self {
        self.opt_level = level.min(3);
        self
    }

    /// Enable verbose output.
    pub fn with_verbose(mut self) -> Self {
        self.verbose = true;
        self
    }
}

/// A compilation session holding state across multiple compilations.
pub struct Session {
    /// Files that have been compiled.
    pub compiled_files: Vec<PathBuf>,

    /// Errors encountered during compilation.
    pub errors: Vec<String>,

    /// Warnings encountered during compilation.
    pub warnings: Vec<String>,
}

impl Session {
    /// Create a new session.
    pub fn new() -> Self {
        Self {
            compiled_files: Vec::new(),
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }

    /// Check if the session has any errors.
    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }

    /// Add an error to the session.
    pub fn add_error(&mut self, error: impl Into<String>) {
        self.errors.push(error.into());
    }

    /// Add a warning to the session.
    pub fn add_warning(&mut self, warning: impl Into<String>) {
        self.warnings.push(warning.into());
    }

    /// Print all errors and warnings.
    pub fn report(&self) {
        for warning in &self.warnings {
            eprintln!("warning: {}", warning);
        }
        for error in &self.errors {
            eprintln!("error: {}", error);
        }
    }
}

impl Default for Session {
    fn default() -> Self {
        Self::new()
    }
}
