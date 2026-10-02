//! Manual verification of the codebase-memory-mcp download (needs network).
//! Kept ignored in CI; run locally with `cargo test -- --ignored`.

#[test]
#[ignore]
fn install_codemap_downloads_runnable_binary() {
    let dir = std::env::temp_dir().join(format!("kopt-setup-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let bin = kernelopt::setup::install_codemap(Some(dir.clone())).expect("install");
    assert!(bin.is_file(), "{bin:?}");
    let out = std::process::Command::new(&bin)
        .arg("--version")
        .output()
        .expect("run installed binary");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(!text.trim().is_empty(), "binary printed nothing");
    let _ = std::fs::remove_dir_all(&dir);
}
