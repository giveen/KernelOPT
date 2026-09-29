//! A cross-process advisory lock that serializes GPU work.
//!
//! Benching must never run concurrently with other GPU work: two jobs on one
//! device wreck the timing and can exhaust VRAM. An advisory `flock` on a
//! lockfile also serializes *separate* `kernelopt` processes (a campaign plus a
//! single run, or two campaigns), so the device only ever holds one job.
//!
//! Declared directly against libc `flock(2)` — no new dependency.

use std::fs::{File, OpenOptions};
use std::io;
use std::os::unix::io::AsRawFd;
use std::path::Path;

extern "C" {
    fn flock(fd: i32, operation: i32) -> i32;
}

const LOCK_EX: i32 = 2;
const LOCK_UN: i32 = 8;

/// Holds the lock until dropped.
pub struct GpuGuard {
    file: File,
}

impl Drop for GpuGuard {
    fn drop(&mut self) {
        unsafe {
            flock(self.file.as_raw_fd(), LOCK_UN);
        }
    }
}

/// Acquire the exclusive GPU lock, blocking until it is available.
pub fn lock(path: &Path) -> io::Result<GpuGuard> {
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let file = OpenOptions::new().create(true).write(true).open(path)?;
    // SAFETY: `file` owns the fd for the lifetime of the guard.
    if unsafe { flock(file.as_raw_fd(), LOCK_EX) } != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(GpuGuard { file })
}

/// The default lockfile for a project rooted at `cwd`.
pub fn default_path(cwd: &Path) -> std::path::PathBuf {
    cwd.join(".kernelopt").join("gpu.lock")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lock_is_exclusive_across_handles() {
        let dir = std::env::temp_dir().join(format!("kopt-gpu-lock-{}", uuid::Uuid::new_v4()));
        let path = dir.join("gpu.lock");
        let held = lock(&path).unwrap();
        let p2 = path.clone();
        let waiter = std::thread::spawn(move || {
            let start = std::time::Instant::now();
            let _g2 = lock(&p2).unwrap();
            start.elapsed()
        });
        std::thread::sleep(std::time::Duration::from_millis(300));
        drop(held);
        let waited = waiter.join().unwrap();
        assert!(
            waited >= std::time::Duration::from_millis(200),
            "second lock did not block: waited {waited:?}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn default_path_is_under_kernelopt() {
        let p = default_path(Path::new("/tmp/proj"));
        assert!(p.ends_with(".kernelopt/gpu.lock"), "{}", p.display());
    }
}
