//! kernelopt library crate: dispatch-aware agentic GPU kernel optimization.

pub mod analyst;
pub mod attribution;
pub mod backend;
pub mod campaign;
pub mod config;
pub mod cuda_pipeline;
pub mod dotenv;
pub mod gpu_lock;
pub mod journal;
pub mod llamacpp;
pub mod llm;
pub mod memory;
pub mod models;
pub mod ninfer;
pub mod pipeline;
pub mod prompts;
pub mod runner_bridge;
pub mod search;
pub mod signals;
pub mod tools;
