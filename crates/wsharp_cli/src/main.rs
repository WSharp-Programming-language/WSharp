//! Command-line interface for the W# compiler.

mod repl;

use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::process::ExitCode;
use wsharp_driver::{CompileOptions, Driver};

#[derive(Parser)]
#[command(name = "wsharp")]
#[command(author, version, about = "The W# programming language compiler", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Enable verbose output
    #[arg(short, long, global = true)]
    verbose: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Compile a W# source file
    Build {
        /// The input file to compile
        #[arg(required = true)]
        input: String,

        /// The output file
        #[arg(short, long)]
        output: Option<String>,

        /// Emit LLVM IR (.ll file)
        #[arg(long)]
        emit_llvm: bool,

        /// Emit assembly (.s file)
        #[arg(long)]
        emit_asm: bool,

        /// Emit MIR (.mir file) for debugging
        #[arg(long)]
        emit_mir: bool,

        /// Optimization level (0-3)
        #[arg(short = 'O', default_value = "0")]
        opt_level: u8,

        /// Target triple (defaults to native)
        #[arg(long)]
        target: Option<String>,
    },

    /// Run a W# source file using JIT compilation
    Run {
        /// The input file to run
        #[arg(required = true)]
        input: String,

        /// Arguments to pass to the program
        #[arg(trailing_var_arg = true)]
        args: Vec<String>,
    },

    /// Start the interactive REPL
    Repl,

    /// Check a W# source file for errors without compiling
    Check {
        /// The input file to check
        #[arg(required = true)]
        input: String,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let driver = if cli.verbose {
        Driver::verbose()
    } else {
        Driver::new()
    };

    match cli.command {
        Commands::Build {
            input,
            output,
            emit_llvm,
            emit_asm,
            emit_mir,
            opt_level,
            target,
        } => {
            let input_path = PathBuf::from(&input);

            if !input_path.exists() {
                eprintln!("error: file not found: {}", input);
                return ExitCode::FAILURE;
            }

            let output_path = output.map(PathBuf::from);

            let options = CompileOptions {
                output: output_path,
                emit_ir: emit_llvm,
                emit_asm,
                emit_mir,
                opt_level: opt_level as u32,
                target,
                jit: false,
                verbose: cli.verbose,
            };

            match driver.compile_file(&input_path, &options) {
                Ok(()) => {
                    if cli.verbose {
                        eprintln!("Build succeeded");
                    }
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("error: {}", e);
                    ExitCode::FAILURE
                }
            }
        }

        Commands::Run { input, args: _ } => {
            let input_path = PathBuf::from(&input);

            if !input_path.exists() {
                eprintln!("error: file not found: {}", input);
                return ExitCode::FAILURE;
            }

            // Read the source file
            let source = match std::fs::read_to_string(&input_path) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("error: failed to read file: {}", e);
                    return ExitCode::FAILURE;
                }
            };

            let name = input_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("module");

            #[cfg(feature = "jit")]
            {
                match driver.run_jit(&source, name) {
                    Ok(result) => {
                        println!("{}", result);
                        ExitCode::SUCCESS
                    }
                    Err(e) => {
                        eprintln!("error: {}", e);
                        ExitCode::FAILURE
                    }
                }
            }

            #[cfg(not(feature = "jit"))]
            {
                let _ = (source, name); // Silence unused warnings
                eprintln!("error: JIT compilation is not enabled. Rebuild with --features jit");
                ExitCode::FAILURE
            }
        }

        Commands::Repl => {
            match repl::Repl::new(cli.verbose) {
                Ok(mut repl) => {
                    if let Err(e) = repl.run() {
                        eprintln!("error: {}", e);
                        ExitCode::FAILURE
                    } else {
                        ExitCode::SUCCESS
                    }
                }
                Err(e) => {
                    eprintln!("error: failed to initialize REPL: {}", e);
                    ExitCode::FAILURE
                }
            }
        }

        Commands::Check { input } => {
            let input_path = PathBuf::from(&input);

            if !input_path.exists() {
                eprintln!("error: file not found: {}", input);
                return ExitCode::FAILURE;
            }

            // Read the source file
            let source = match std::fs::read_to_string(&input_path) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("error: failed to read file: {}", e);
                    return ExitCode::FAILURE;
                }
            };

            let name = input_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("module");

            // Compile to MIR (this does parsing + lowering + type checking)
            match driver.compile_to_mir(&source, name) {
                Ok(_) => {
                    println!("✓ No errors found in {}", input);
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("error: {}", e);
                    ExitCode::FAILURE
                }
            }
        }
    }
}
