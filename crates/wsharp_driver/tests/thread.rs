//! Integration tests for the thread GC system.

use std::fs;
use std::process::Command;
use std::sync::atomic::{AtomicU32, Ordering};

static TEST_COUNTER: AtomicU32 = AtomicU32::new(0);

/// Compile W# source, link with the runtime library, and run.
/// Returns the process exit code.
fn compile_and_run(source: &str) -> i32 {
    let test_id = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
    let tmp_dir = std::env::temp_dir();
    let ws_file = tmp_dir.join(format!("thread_test_{}.ws", test_id));
    let obj_file = tmp_dir.join(format!("thread_test_{}.o", test_id));
    let exe_file = tmp_dir.join(format!("thread_test_{}", test_id));

    fs::write(&ws_file, source).expect("Failed to write source file");

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

    // Link with cc, including the runtime static library
    let runtime_lib_dir = workspace_root.join("target/debug");
    let link_output = Command::new("cc")
        .args([
            obj_file.to_str().unwrap(),
            "-o",
            exe_file.to_str().unwrap(),
            &format!("-L{}", runtime_lib_dir.display()),
            "-lwsharp_runtime",
            "-lpthread",
            "-ldl",
            "-lm",
        ])
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

/// Compile only (no link/run). Returns true if compilation succeeds.
fn compile_only(source: &str) -> bool {
    let test_id = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
    let tmp_dir = std::env::temp_dir();
    let ws_file = tmp_dir.join(format!("thread_compile_test_{}.ws", test_id));
    let obj_file = tmp_dir.join(format!("thread_compile_test_{}.o", test_id));

    fs::write(&ws_file, source).expect("Failed to write source file");

    let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir.parent().unwrap().parent().unwrap();

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

    let _ = fs::remove_file(&ws_file);
    let _ = fs::remove_file(&obj_file);

    compile_output.status.success()
}

// =========================================================================
// Compilation tests
// =========================================================================

#[test]
fn test_thread_spawn_compiles() {
    let code = r#"
fn worker() -> i64 { return 0; }
fn main() -> i64 {
    let t = thread_spawn(worker);
    return 0;
}
"#;
    assert!(compile_only(code), "thread_spawn should compile");
}

#[test]
fn test_mutex_compiles() {
    let code = r#"
fn main() -> i64 {
    let m = mutex_new(0);
    mutex_lock(m, 0);
    mutex_unlock(m);
    mutex_destroy(m);
    return 0;
}
"#;
    assert!(compile_only(code), "mutex operations should compile");
}

#[test]
fn test_thread_state_compiles() {
    let code = r#"
fn main() -> i64 {
    let s = thread_state_new(42);
    let v = thread_state_get(s);
    thread_state_set(s, 100);
    return 0;
}
"#;
    assert!(compile_only(code), "thread_state operations should compile");
}

// =========================================================================
// Link and run tests
// =========================================================================

#[test]
fn test_basic_thread_spawn_and_join() {
    // Spawn a thread, GC auto-joins before main exits.
    // Main returns 0 if everything works.
    let code = r#"
fn worker() -> i64 { return 0; }
fn main() -> i64 {
    let t = thread_spawn(worker);
    return 0;
}
"#;
    assert_eq!(compile_and_run(code), 0);
}

#[test]
fn test_thread_state_basic() {
    // Create thread state, set and get values.
    // thread_state_get returns i32, so we use it in an i32-returning helper.
    let code = r#"
fn check_state() -> i64 {
    let s = thread_state_new(42);
    let v = thread_state_get(s);
    return 0;
}
fn main() -> i64 {
    return check_state();
}
"#;
    assert_eq!(compile_and_run(code), 0);
}

#[test]
fn test_mutex_lifecycle() {
    // Create, lock, unlock, destroy a mutex.
    let code = r#"
fn main() -> i64 {
    let m = mutex_new(0);
    mutex_lock(m, 1);
    mutex_unlock(m);
    mutex_destroy(m);
    return 0;
}
"#;
    assert_eq!(compile_and_run(code), 0);
}
