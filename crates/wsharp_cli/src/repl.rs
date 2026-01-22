//! Interactive REPL for W#.

use anyhow::{anyhow, Result};
use rustyline::error::ReadlineError;
use rustyline::{DefaultEditor, Result as RlResult};
use std::collections::HashMap;
use wsharp_driver::Driver;

/// The W# REPL.
pub struct Repl {
    driver: Driver,
    editor: DefaultEditor,
    /// Accumulated function definitions
    definitions: Vec<String>,
    /// Map of defined variable names to their types (for future :type command)
    variables: HashMap<String, String>,
    /// Line counter for unique wrapper function names
    line_number: usize,
    /// Whether verbose mode is enabled
    verbose: bool,
}

impl Repl {
    /// Create a new REPL instance.
    pub fn new(verbose: bool) -> RlResult<Self> {
        let editor = DefaultEditor::new()?;
        let driver = if verbose {
            Driver::verbose()
        } else {
            Driver::new()
        };

        Ok(Self {
            driver,
            editor,
            definitions: Vec::new(),
            variables: HashMap::new(),
            line_number: 0,
            verbose,
        })
    }

    /// Run the REPL main loop.
    pub fn run(&mut self) -> Result<()> {
        println!("W# REPL v0.1.0");
        println!("Type :help for available commands, :quit to exit");
        println!();

        loop {
            let prompt = format!("wsharp:{:03}> ", self.line_number);

            match self.editor.readline(&prompt) {
                Ok(line) => {
                    let line = line.trim();

                    // Skip empty lines
                    if line.is_empty() {
                        continue;
                    }

                    // Add to history
                    let _ = self.editor.add_history_entry(line);

                    // Handle commands
                    if line.starts_with(':') {
                        match self.handle_command(line) {
                            Ok(should_continue) => {
                                if !should_continue {
                                    break;
                                }
                            }
                            Err(e) => {
                                eprintln!("error: {}", e);
                            }
                        }
                        continue;
                    }

                    // Evaluate code
                    match self.eval_line(line) {
                        Ok(Some(result)) => {
                            println!("{}", result);
                        }
                        Ok(None) => {
                            // No result to display (e.g., function definition)
                        }
                        Err(e) => {
                            eprintln!("error: {}", e);
                        }
                    }

                    self.line_number += 1;
                }
                Err(ReadlineError::Interrupted) => {
                    println!("^C");
                    // Continue on Ctrl+C
                }
                Err(ReadlineError::Eof) => {
                    println!("Goodbye!");
                    break;
                }
                Err(err) => {
                    eprintln!("error: {:?}", err);
                    break;
                }
            }
        }

        Ok(())
    }

    /// Handle a REPL command (lines starting with ':').
    fn handle_command(&mut self, cmd: &str) -> Result<bool> {
        let parts: Vec<&str> = cmd.splitn(2, ' ').collect();
        let command = parts[0];
        let _arg = parts.get(1).map(|s| s.trim());

        match command {
            ":quit" | ":q" | ":exit" => {
                println!("Goodbye!");
                Ok(false)
            }
            ":help" | ":h" | ":?" => {
                self.print_help();
                Ok(true)
            }
            ":clear" => {
                self.definitions.clear();
                self.variables.clear();
                self.line_number = 0;
                println!("Cleared all definitions.");
                Ok(true)
            }
            ":defs" => {
                if self.definitions.is_empty() {
                    println!("No definitions.");
                } else {
                    println!("Current definitions:");
                    for def in &self.definitions {
                        println!("  {}", def.lines().next().unwrap_or(def));
                    }
                }
                Ok(true)
            }
            ":type" | ":t" => {
                // TODO: Implement type query
                println!("Type query not yet implemented.");
                Ok(true)
            }
            _ => {
                Err(anyhow!("unknown command: {}. Type :help for available commands.", command))
            }
        }
    }

    /// Evaluate a line of W# code.
    fn eval_line(&mut self, input: &str) -> Result<Option<String>> {
        // Check if this is a function or let definition
        let trimmed = input.trim();

        if trimmed.starts_with("fn ") {
            // Function definition - add to definitions
            self.definitions.push(input.to_string());
            let name = self.extract_fn_name(trimmed);
            println!("defined function: {}", name.unwrap_or("<anonymous>"));
            return Ok(None);
        }

        if trimmed.starts_with("let ") && !trimmed.contains('=') {
            return Err(anyhow!("let binding requires initializer"));
        }

        if trimmed.starts_with("let ") {
            // Let binding at top level - wrap in main and execute
            // For now, we'll just evaluate the expression part
            self.definitions.push(input.to_string());
            let name = self.extract_let_name(trimmed);
            println!("defined: {}", name.unwrap_or("<unknown>"));
            return Ok(None);
        }

        // Expression - wrap in a main function and evaluate
        self.eval_expression(input)
    }

    /// Evaluate an expression by wrapping it in a main function.
    fn eval_expression(&self, expr: &str) -> Result<Option<String>> {
        // Build the full source with all definitions plus a main that returns the expression
        let mut source = String::new();

        // Add all accumulated definitions
        for def in &self.definitions {
            source.push_str(def);
            source.push('\n');
        }

        // Wrap expression in main function
        // For now, assume i64 return type (JIT limitation)
        let wrapper_name = format!("__repl_main_{}", self.line_number);
        source.push_str(&format!(
            "fn {}() -> i64 {{ return {}; }}\n",
            wrapper_name, expr
        ));

        // Also need a main that calls our wrapper
        source.push_str(&format!(
            "fn main() -> i64 {{ return {}(); }}\n",
            wrapper_name
        ));

        if self.verbose {
            eprintln!("[repl] Evaluating source:");
            for (i, line) in source.lines().enumerate() {
                eprintln!("  {:3}: {}", i + 1, line);
            }
        }

        // Try to JIT compile and run
        #[cfg(feature = "jit")]
        {
            match self.driver.run_jit(&source, "repl") {
                Ok(result) => Ok(Some(format!("{}", result))),
                Err(e) => Err(anyhow!("{}", e)),
            }
        }

        #[cfg(not(feature = "jit"))]
        {
            let _ = source;
            Err(anyhow!("JIT not enabled. Rebuild with --features jit"))
        }
    }

    /// Extract function name from a function definition.
    fn extract_fn_name<'a>(&self, input: &'a str) -> Option<&'a str> {
        // "fn name(...)" -> "name"
        let after_fn = input.strip_prefix("fn ")?.trim_start();
        let end = after_fn.find(|c: char| c == '(' || c == '<' || c.is_whitespace())?;
        Some(&after_fn[..end])
    }

    /// Extract variable name from a let binding.
    fn extract_let_name<'a>(&self, input: &'a str) -> Option<&'a str> {
        // "let name = ..." or "let name: Type = ..."
        let after_let = input.strip_prefix("let ")?.trim_start();
        let end = after_let.find(|c: char| c == ':' || c == '=' || c.is_whitespace())?;
        Some(&after_let[..end])
    }

    /// Print help message.
    fn print_help(&self) {
        println!("W# REPL Commands:");
        println!();
        println!("  :help, :h, :?    Show this help message");
        println!("  :quit, :q, :exit Exit the REPL");
        println!("  :clear           Clear all definitions");
        println!("  :defs            Show current definitions");
        println!("  :type <expr>     Show the type of an expression (not yet implemented)");
        println!();
        println!("Enter W# code to evaluate:");
        println!("  - Expressions like `1 + 2` are evaluated and printed");
        println!("  - Function definitions like `fn add(a: i64, b: i64) -> i64 {{ a + b }}`");
        println!("    are saved for use in subsequent expressions");
        println!();
    }
}
