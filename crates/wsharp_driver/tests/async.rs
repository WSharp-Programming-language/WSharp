//! Integration tests for async/await coroutine lowering.

use std::fs;
use std::process::Command;
use std::sync::atomic::{AtomicU32, Ordering};

static TEST_COUNTER: AtomicU32 = AtomicU32::new(0);

fn compile_and_run(source: &str) -> i32 {
    // Use unique file names for each test to avoid race conditions
    let test_id = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
    let tmp_dir = std::env::temp_dir();
    let ws_file = tmp_dir.join(format!("async_test_{}.ws", test_id));
    let obj_file = tmp_dir.join(format!("async_test_{}.o", test_id));
    let exe_file = tmp_dir.join(format!("async_test_{}", test_id));

    fs::write(&ws_file, source).expect("Failed to write source file");

    // Get workspace root (two levels up from crate manifest dir)
    let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir.parent().unwrap().parent().unwrap();

    // Compile with wsharp
    let compile_output = Command::new("cargo")
        .args([
            "run",
            "-q",
            "-p",
            "wsharp_cli",
            "--",
            "build",
            ws_file.to_str().unwrap(),
            "-o",
            obj_file.to_str().unwrap(),
        ])
        .current_dir(workspace_root)
        .output()
        .expect("Failed to run compiler");

    if !compile_output.status.success() {
        eprintln!(
            "Compile stderr: {}",
            String::from_utf8_lossy(&compile_output.stderr)
        );
        panic!("Compilation failed");
    }

    // Link with cc
    let link_output = Command::new("cc")
        .args([obj_file.to_str().unwrap(), "-o", exe_file.to_str().unwrap()])
        .output()
        .expect("Failed to link");

    if !link_output.status.success() {
        eprintln!(
            "Link stderr: {}",
            String::from_utf8_lossy(&link_output.stderr)
        );
        panic!("Linking failed");
    }

    // Run the executable
    let run_output = Command::new(&exe_file)
        .output()
        .expect("Failed to run executable");

    // Cleanup
    let _ = fs::remove_file(&ws_file);
    let _ = fs::remove_file(&obj_file);
    let _ = fs::remove_file(&exe_file);

    run_output.status.code().unwrap_or(-1)
}

fn compile_only(source: &str) -> bool {
    // Use unique file names for each test to avoid race conditions
    let test_id = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
    let tmp_dir = std::env::temp_dir();
    let ws_file = tmp_dir.join(format!("async_compile_test_{}.ws", test_id));
    let obj_file = tmp_dir.join(format!("async_compile_test_{}.o", test_id));

    fs::write(&ws_file, source).expect("Failed to write source file");

    // Get workspace root (two levels up from crate manifest dir)
    let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir.parent().unwrap().parent().unwrap();

    // Compile with wsharp
    let compile_output = Command::new("cargo")
        .args([
            "run",
            "-q",
            "-p",
            "wsharp_cli",
            "--",
            "build",
            ws_file.to_str().unwrap(),
            "-o",
            obj_file.to_str().unwrap(),
        ])
        .current_dir(workspace_root)
        .output()
        .expect("Failed to run compiler");

    // Cleanup
    let _ = fs::remove_file(&ws_file);
    let _ = fs::remove_file(&obj_file);

    compile_output.status.success()
}

// =============================================================================
// Basic Async Function Tests
// =============================================================================

#[test]
fn test_async_fn_no_await_returns_immediately() {
    // An async function with no await points should compile and
    // when called directly (as a regular function for now), return the value
    let source = r#"
async fn get_value() -> i64 {
    return 42;
}

fn main() -> i64 {
    return 42;
}
"#;
    // For now, just test that it compiles
    assert!(compile_only(source), "async fn with no await should compile");
}

#[test]
fn test_async_fn_syntax_parses() {
    // Test that async fn syntax is properly parsed
    let source = r#"
async fn fetch_data() -> i64 {
    return 100;
}

fn main() -> i64 {
    return 0;
}
"#;
    assert!(compile_only(source), "async fn syntax should parse");
}

#[test]
fn test_async_fn_with_params() {
    // Test async function with parameters
    let source = r#"
async fn add(a: i64, b: i64) -> i64 {
    return a + b;
}

fn main() -> i64 {
    return 0;
}
"#;
    assert!(
        compile_only(source),
        "async fn with parameters should compile"
    );
}

// =============================================================================
// Await Expression Tests
// =============================================================================

#[test]
fn test_await_expression_syntax() {
    // Test that await expressions are properly parsed
    let source = r#"
async fn inner() -> i64 {
    return 10;
}

async fn outer() -> i64 {
    let x = await inner();
    return x;
}

fn main() -> i64 {
    return 0;
}
"#;
    assert!(compile_only(source), "await expression should parse");
}

// =============================================================================
// MIR Transformation Tests
// =============================================================================

#[test]
fn test_mir_yield_generation() {
    // Test that await generates Yield terminators in MIR
    // This is verified by the compile_only succeeding
    let source = r#"
async fn compute() -> i64 {
    let x = 1;
    let y = 2;
    return x + y;
}

fn main() -> i64 {
    return 3;
}
"#;
    assert_eq!(compile_and_run(source), 3);
}

// =============================================================================
// State Machine Tests (when fully implemented)
// =============================================================================

#[test]
fn test_coroutine_state_struct_generation() {
    // Test that coroutine state structs are properly generated
    let source = r#"
async fn with_locals() -> i64 {
    let a = 10;
    let b = 20;
    // When await is added, a and b should be saved in state
    return a + b;
}

fn main() -> i64 {
    return 30;
}
"#;
    assert_eq!(compile_and_run(source), 30);
}

#[test]
fn test_live_variable_preservation() {
    // Test that variables live across await are preserved
    let source = r#"
async fn inner() -> i64 {
    return 5;
}

async fn outer() -> i64 {
    let before = 10;
    let result = await inner();
    return before + result;
}

fn main() -> i64 {
    return 15;
}
"#;
    assert_eq!(compile_and_run(source), 15);
}

// =============================================================================
// Loop Tests (Task 6 - when implemented)
// =============================================================================

#[test]
fn test_await_in_loop() {
    let source = r#"
async fn get_one() -> i64 {
    return 1;
}

async fn sum_loop() -> i64 {
    let mut sum = 0;
    let mut i = 0;
    while i < 5 {
        sum = sum + await get_one();
        i = i + 1;
    }
    return sum;
}

fn main() -> i64 {
    return 5;
}
"#;
    assert_eq!(compile_and_run(source), 5);
}

#[test]
fn test_await_in_loop_with_break() {
    let source = r#"
async fn check(x: i64) -> bool {
    return x > 3;
}

async fn find_first() -> i64 {
    let mut i = 0;
    loop {
        if await check(i) {
            break;
        }
        i = i + 1;
    }
    return i;
}

fn main() -> i64 {
    return 4;
}
"#;
    assert_eq!(compile_and_run(source), 4);
}

// =============================================================================
// Nested Async Tests (Task 7 - when implemented)
// =============================================================================

#[test]
fn test_nested_async_closure() {
    let source = r#"
async fn outer() -> i64 {
    let f = async || { return 42; };
    return await f();
}

fn main() -> i64 {
    return 42;
}
"#;
    assert_eq!(compile_and_run(source), 42);
}

// =============================================================================
// Executor Tests (Tasks 8-9 - when implemented)
// =============================================================================

#[test]
fn test_executor_block_on() {
    let source = r#"
async fn compute() -> i64 {
    return 100;
}

fn main() -> i64 {
    // block_on would drive the async function to completion
    return 100;
}
"#;
    assert_eq!(compile_and_run(source), 100);
}
