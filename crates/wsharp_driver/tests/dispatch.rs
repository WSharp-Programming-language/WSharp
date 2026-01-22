//! Integration tests for HTTP status dispatch.

use std::process::Command;
use std::fs;
use std::sync::atomic::{AtomicU32, Ordering};

static TEST_COUNTER: AtomicU32 = AtomicU32::new(0);

fn compile_and_run(source: &str) -> i32 {
    // Use unique file names for each test to avoid race conditions
    let test_id = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
    let tmp_dir = std::env::temp_dir();
    let ws_file = tmp_dir.join(format!("dispatch_test_{}.ws", test_id));
    let obj_file = tmp_dir.join(format!("dispatch_test_{}.o", test_id));
    let exe_file = tmp_dir.join(format!("dispatch_test_{}", test_id));

    fs::write(&ws_file, source).expect("Failed to write source file");

    // Get workspace root (two levels up from crate manifest dir)
    let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir.parent().unwrap().parent().unwrap();

    // Compile with wsharp
    let compile_output = Command::new("cargo")
        .args(["run", "-q", "-p", "wsharp_cli", "--", "build", ws_file.to_str().unwrap(), "-o", obj_file.to_str().unwrap()])
        .current_dir(workspace_root)
        .output()
        .expect("Failed to run compiler");

    if !compile_output.status.success() {
        eprintln!("Compile stderr: {}", String::from_utf8_lossy(&compile_output.stderr));
        panic!("Compilation failed");
    }

    // Link with cc
    let link_output = Command::new("cc")
        .args([obj_file.to_str().unwrap(), "-o", exe_file.to_str().unwrap()])
        .output()
        .expect("Failed to link");

    if !link_output.status.success() {
        eprintln!("Link stderr: {}", String::from_utf8_lossy(&link_output.stderr));
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

#[test]
fn test_exact_http_dispatch() {
    let code = r#"
fn handle(status: http 200) -> i64 { return 10; }
fn handle(status: http 404) -> i64 { return 40; }
fn main() -> i64 { return handle(http 404); }
"#;
    assert_eq!(compile_and_run(code), 40);
}

#[test]
fn test_category_fallback() {
    let code = r#"
fn handle(status: http 200) -> i64 { return 10; }
fn handle(status: http 2xx) -> i64 { return 20; }
fn main() -> i64 { return handle(http 201); }
"#;
    assert_eq!(compile_and_run(code), 20);
}

#[test]
fn test_specificity_ordering() {
    let code = r#"
fn handle(status: http 404) -> i64 { return 40; }
fn handle(status: http 4xx) -> i64 { return 41; }
fn main() -> i64 { return handle(http 404); }
"#;
    // Most specific (exact 404) should win over category 4xx
    assert_eq!(compile_and_run(code), 40);
}

#[test]
fn test_category_match() {
    let code = r#"
fn handle(status: http 404) -> i64 { return 40; }
fn handle(status: http 4xx) -> i64 { return 41; }
fn main() -> i64 { return handle(http 403); }
"#;
    // 403 doesn't match exact 404, falls back to 4xx category
    assert_eq!(compile_and_run(code), 41);
}

#[test]
fn test_single_function_no_dispatch() {
    let code = r#"
fn handle(status: http 200) -> i64 { return 10; }
fn main() -> i64 { return handle(http 200); }
"#;
    assert_eq!(compile_and_run(code), 10);
}

#[test]
fn test_server_error_category() {
    let code = r#"
fn handle(status: http 5xx) -> i64 { return 50; }
fn main() -> i64 { return handle(http 503); }
"#;
    assert_eq!(compile_and_run(code), 50);
}
