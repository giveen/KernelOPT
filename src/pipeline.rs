//! Pipeline orchestration (compiled-model mode):
//! trace → profile → [plan → execute → verify(1–2) → bench → summarize → select]×T
//! → restitch → verify(3–4) → report. Every transition is journaled.
//!
//! Profiling is two-tier (attribution ≠ timing):
//!   Tier A  graphsignal sidecar on bench invocations (cheap, every candidate)
//!   Tier B  NCU deep context (planner-facing, on the baseline + beam leaders)

use crate::analyst;
use crate::config::Config;
use crate::journal::{Event, Journal};
use crate::llm::{Completion, LlmClient, ToolCall, ToolDef};
use crate::memory::{parse_summarizer_output, ExperienceItem, ExperienceMemory, MemoryUpdate, StrategyTracker};
use crate::runner_bridge::RunnerBridge;
use crate::search::{allocate_expansions, diverse_select_nodes, meltdown_detected, BeamNode, Candidate};
use crate::signals;
use anyhow::{Context, Result};
use serde_json::json;

pub struct Pipeline<'a> {
    pub cfg: &'a Config,
    pub llm: &'a dyn LlmClient,
    pub runner: &'a RunnerBridge,
    pub wrapped_runner: &'a RunnerBridge,
    pub journal: &'a mut Journal,
    pub session_id: String,
    pub model_file: String,
    pub baseline_weights: Option<String>,
    pub memory: ExperienceMemory,
    pub tracker: StrategyTracker,
    pub next_signals_port: u16,
    /// Raw /signals payload from the most recent wrapped bench (JSON string).
    pub last_bench_signals: Option<String>,
}

impl<'a> Pipeline<'a> {
    // ---------- runner convenience ----------

    fn runner_call(&self, command: &str, mut req: serde_json::Value) -> Result<serde_json::Value> {
        req["command"] = json!(command);
        self.runner.call(&req)
    }

    fn record_llm_usage(&mut self, agent: &str, c: &Completion) {
        if let Some(u) = &c.usage {
            let _ = self.journal.record(&Event::LlmCall {
                agent: agent.into(),
                prompt_tokens: u.prompt_tokens,
                completion_tokens: u.completion_tokens,
                cached_tokens: u.cached_tokens,
            });
        }
    }

    fn tmp_path(&self, name: &str) -> Result<String> {
        let root = std::env::current_dir().context("cwd")?;
        Ok(root
            .join(".kernelopt/tmp")
            .join(format!("{name}_{}", uuid::Uuid::new_v4().simple()))
            .to_string_lossy()
            .to_string())
    }

    // ---------- stage: trace ----------

    pub fn stage_trace(&mut self) -> Result<serde_json::Value> {
        let weights_path = self.tmp_path("baseline_weights")? + ".pt";
        let resp = self.runner_call(
            "trace",
            json!({
                "model_file": self.model_file(),
                "inputs_fn": self.cfg.inputs_fn,
                "mode": "max-autotune",
                "out_weights": weights_path,
                "seed": 0,
            }),
        )?;
        self.baseline_weights = resp["baseline_weights_path"]
            .as_str()
            .map(|s| s.to_string());
        self.journal.record(&Event::StageCompleted {
            stage: "trace".into(),
            data: resp.clone(),
        })?;
        Ok(resp)
    }

    // ---------- stage: baseline bench ----------

    pub fn stage_baseline_bench(&mut self) -> Result<f64> {
        let resp = self.bench_command(None)?;
        let ms = resp
            .get("mean_ms")
            .and_then(|v| v.as_f64())
            .context("bench returned no mean_ms")?;
        self.journal.record(&Event::StageCompleted {
            stage: "baseline_bench".into(),
            data: json!({"mean_ms": ms}),
        })?;
        Ok(ms)
    }

    /// Bench via the wrapped runner (graphsignal sidecar) when the profiler
    /// tier includes graphsignal; plain runner otherwise. Clean timing is in
    /// `mean_ms` either way — the payload is attribution only.
    fn bench_command(&mut self, variant_file: Option<&str>) -> Result<serde_json::Value> {
        let use_wrapper = matches!(
            self.cfg.profiler,
            crate::config::ProfilerMode::Graphsignal | crate::config::ProfilerMode::Both
        );
        let mut req = json!({
            "model_file": self.model_file(),
            "warmup_ms": 25.0,
            "rep_ms": 100.0,
        });
        if let Some(v) = variant_file {
            req["variant_file"] = json!(v);
        }
        if use_wrapper {
            let port = self.next_signals_port;
            req["graphsignal"] = json!({"enable": true, "listen_port": port, "settle_s": 3});
        }
        req["command"] = json!("bench");
        let bridge: &RunnerBridge = if use_wrapper { &self.wrapped_runner } else { &self.runner };
        let resp = bridge.call(&req)?;
        if use_wrapper {
            self.last_bench_signals = resp
                .get("graphsignal")
                .and_then(|g| g.get("payload"))
                .map(|p| serde_json::to_string(p).unwrap_or_default());
        }
        Ok(resp)
    }

    // ---------- stage: profile (two-tier context for the planner) ----------

    pub fn stage_profile(&mut self) -> Result<serde_json::Value> {
        let mut ctx = json!({"ncu": null, "graphsignal": null});

        // Tier B: NCU deep context (unless disabled or counters unavailable).
        if matches!(self.cfg.profiler, crate::config::ProfilerMode::Ncu | crate::config::ProfilerMode::Both) {
            let req = json!({
                "command": "ncu",
                "model_file": self.model_file(),
                "ncu_set": self.cfg.ncu_set,
                "inputs_fn": self.cfg.inputs_fn,
                "iters": 3,
                "config": {
                    // Profiler-agent hard rules for JIT kernels (enforced):
                    "replay_mode": "application",
                    "kernel_name": "",
                    "launch_skip": 6
                },
                "timeout_s": 900
            });
            match self.runner.call(&req) {
                Ok(resp) if resp.get("ok") == Some(&json!(true)) => {
                    let planning = analyst::planning_context(&resp["context"], 3);
                    ctx["ncu"] = planning;
                }
                Ok(resp) => {
                    let msg = resp
                        .pointer("/error/message")
                        .and_then(|m| m.as_str())
                        .unwrap_or("ncu failed")
                        .to_string();
                    ctx["ncu"] = json!({"available": false, "note": msg});
                }
                Err(e) => {
                    ctx["ncu"] = json!({"available": false, "note": e.to_string()});
                }
            }
        }

        // Tier A: graphsignal attribution from the baseline bench.
        let gs_summary = self.last_bench_signals.take();
        if let Some(payload) = gs_summary {
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&payload) {
                ctx["graphsignal"] = signals::summarize(&parsed, 5);
            }
        }

        self.journal.record(&Event::StageCompleted {
            stage: "profile".into(),
            data: json!({"ncu_available": ctx["ncu"]["available"].is_null()
                || ctx["ncu"]["available"] == json!(true)}),
        })?;
        Ok(ctx)
    }

    // ---------- stage: plan (Planner agent) ----------

    #[allow(clippy::too_many_arguments)]
    pub fn stage_plan(
        &mut self,
        kernel_source: &str,
        profiling_ctx: &serde_json::Value,
        _chain_idx: u32,
        diversity_hint: bool,
        recent_directions: &[String],
    ) -> Result<String> {
        let system = self.render_prompt(
            "planner.md",
            &json!({
                "diversity_hint": if diversity_hint {
                    "DIVERSITY ENFORCEMENT: recent plans collapsed to very few unique \
                     approaches. Choose a DIFFERENT strategy family this time."
                } else {
                    ""
                },
                "avoid_flags": self.tracker.avoid_flags("triton").join(", "),
                "memory_context": self.memory.context(),
            }),
        )?;
        let user = format!(
            "KERNEL SOURCE:\n```python\n{kernel_source}\n```\n\nPROFILING CONTEXT:\n{profiling_ctx}\n\nRECENT DIRECTIONS (avoid repeats): {}",
            if recent_directions.is_empty() {
                "(none)".to_string()
            } else {
                recent_directions.join("; ")
            }
        );

        let tools = vec![
            ToolDef {
                name: "submit_plan".into(),
                description: "Submit one optimization plan for the Executor".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "plan": {
                            "type": "object",
                            "properties": {
                                "change": {"type": "string"},
                                "implementation_hints": {"type": "string"},
                                "evidence": {"type": "string"}
                            },
                            "required": ["change"]
                        }
                    },
                    "required": ["plan"]
                }),
            },
            ToolDef {
                name: "search_memory".into(),
                description: "Query past optimization experience".into(),
                parameters: json!({"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}),
            },
        ];

        let completion = self
            .llm
            .complete(&system, &user, &tools, &self.session_id, Some("submit_plan"))
            .context("planner LLM call")?;
        self.record_llm_usage("planner", &completion);

        for tc in &completion.tool_calls {
            if tc.name == "submit_plan" {
                let change = tc
                    .arguments
                    .pointer("/plan/change")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unspecified change");
                return Ok(change.to_string());
            }
        }
        Ok(completion.content.unwrap_or_else(|| "no plan produced".into()))
    }

    // ---------- stage: execute (Executor agent) + gates 1–2 ----------

    #[allow(clippy::too_many_arguments)]
    pub fn stage_execute_and_verify(
        &mut self,
        kernel_source: &str,
        plan: &str,
        chain_idx: u32,
        iteration: u32,
        plan_no: u32,
    ) -> Result<(Candidate, Option<String>, Option<String>)> {
        // (candidate, graphsignal payload, last_error) — error is journaled.
        let system = self.render_prompt("executor.md", &json!({}))?;
        let user = format!(
            "OPTIMIZATION PLAN:\n{plan}\n\nCURRENT KERNEL SOURCE:\n```python\n{kernel_source}\n```"
        );
        let tools = vec![ToolDef {
            name: "submit_kernel".into(),
            description: "Submit the modified kernel source".into(),
            parameters: json!({
                "type":"object",
                "properties":{
                    "kernel_source":{"type":"string"},
                    "change_summary":{"type":"string"}
                },
                "required":["kernel_source"]
            }),
        }];

        let mut last_error: Option<String> = None;

        for attempt in 0..self.cfg.hyper.k_retries {
            let user_msg = match &last_error {
                Some(err) => format!(
                    "{user}\n\nPREVIOUS ATTEMPT FAILED VALIDATION (attempt {attempt}):\n{err}\n\nDiagnose and resubmit."
                ),
                None => user.clone(),
            };
            let completion = self
                .llm
                .complete(&system, &user_msg, &tools, &self.session_id, Some("submit_kernel"))
                .context("executor LLM call")?;
            self.record_llm_usage("executor", &completion);

            let submitted = completion
                .tool_calls
                .iter()
                .find(|tc| tc.name == "submit_kernel")
                .map(|tc: &ToolCall| {
                    tc.arguments
                        .get("kernel_source")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string()
                });

            let candidate_source = match submitted {
                Some(s) if !s.is_empty() => Self::strip_code_fences(&s),
                _ => {
                    last_error = Some("no submit_kernel tool call; you MUST call submit_kernel".into());
                    continue;
                }
            };

            // Postmortem: keep every submission, not just passers.
            let _ = std::fs::write(
                self.tmp_path("submission")? + ".py",
                &candidate_source,
            );

            // Gates 1–2 via runner `verify`.
            let verify_req = json!({
                "command": "verify",
                "model_file": self.model_file(),
                "candidate_code": candidate_source,
                "baseline_weights_path": self.baseline_weights,
                "baseline_seed": 0,
                "seeds": [0, 1, 2],
                "rtol": self.cfg.hyper.search_tol,
                "atol": self.cfg.hyper.search_tol,
            });
            let resp = self.runner.call(&verify_req)?;
            let passed = resp["all_passed"].as_bool().unwrap_or(false);
            if passed {
                let candidate_path = self.write_candidate_file(&candidate_source)?;
                let bench = self.bench_command(Some(&candidate_path))?;
                let latency = bench.get("mean_ms").and_then(|v| v.as_f64());
                // Keep the graphsignal payload for the summarizer + planner.
                let payload = bench.get("graphsignal").and_then(|g| g.get("payload")).cloned();
                return Ok((
                    Candidate {
                        id: format!("i{iteration}_c{chain_idx}_p{plan_no}_a{attempt}"),
                        chain: chain_idx,
                        iteration,
                        source: candidate_source,
                        plan: plan.to_string(),
                        latency_ms: latency,
                        passed: true,
                        commit: None,
                    change_summary: None,
                    hints: None,
                    evidence: None,
                    },
                    payload.map(|p| serde_json::to_string(&p).unwrap_or_default()),
                    None,
                ));
            }

            last_error = Some(
                resp.pointer("/gate1/error/message")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .or_else(|| {
                        resp.pointer("/gate2/error/message")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string())
                    })
                    .or_else(|| {
                        resp.pointer("/gate2/results")
                            .map(|r| format!("correctness failure: {r}"))
                    })
                    .unwrap_or_else(|| "verification failed".into()),
            );
        }

        Ok((
            Candidate {
                id: format!("i{iteration}_c{chain_idx}_p{plan_no}_failed"),
                chain: chain_idx,
                iteration,
                source: String::new(),
                plan: plan.to_string(),
                latency_ms: None,
                passed: false,
                commit: None,
                change_summary: None,
                hints: None,
                evidence: None,
            },
            None,
            last_error,
        ))
    }

    fn write_candidate_file(&self, source: &str) -> Result<String> {
        let path = self.tmp_path("candidate")? + ".py";
        std::fs::write(&path, source)?;
        Ok(path)
    }

    // ---------- stage: summarize (Summarizer agent → memory) ----------

    pub fn stage_summarize(
        &mut self,
        before_payload: Option<&str>,
        after_payload: Option<&str>,
        slow_source: &str,
        fast_source: &str,
        plan: &str,
        speedup: f64,
        iteration: u32,
    ) -> Result<()> {
        // Cheap-path: store even without the LLM when thresholds are met.
        let store = self.memory.should_store(speedup);
        // Marginal results aren't stored — skip the LLM call entirely.
        if !store {
            return Ok(());
        }
        let system = self.render_prompt("summarizer.md", &json!({}))?;
        let user = format!(
            "SLOW KERNEL:\n```python\n{slow_source}\n```\n\nFAST KERNEL:\n```python\n{fast_source}\n```\n\nPLAN APPLIED: {plan}\n\nPROFILE BEFORE:\n{before}\n\nPROFILE AFTER:\n{after}\n\nSPEEDUP: {speedup:.3}",
            before = before_payload.unwrap_or("(unavailable)"),
            after = after_payload.unwrap_or("(unavailable)"),
        );
        let completion = self
            .llm
            .complete(&system, &user, &[], &self.session_id, None)
            .context("summarizer LLM call")?;
        self.record_llm_usage("summarizer", &completion);
        let text = completion.content.unwrap_or_default();

        if let Ok((Some(item), update)) = parse_summarizer_output(&text) {
            let update = update.unwrap_or(MemoryUpdate {
                action: "append".into(),
                replace_item_id: None,
                reason: None,
            });
            let action = self.memory.apply(item, update);
            self.journal.record(&Event::MemoryUpdated {
                action: action.into(),
                direction: Some(plan.chars().take(80).collect()),
            })?;
        } else {
            // LLM output unparseable: record a minimal experience item so the
            // signal (win or regression) is not lost.
            let fallback = ExperienceItem {
                item_id: format!("fb_{}", uuid::Uuid::new_v4().simple()),
                iteration,
                speedup,
                rewrite_type: "unknown".into(),
                framework: "triton".into(),
                direction: plan.chars().take(80).collect(),
                profiling_signal: String::new(),
                strategy_title: format!("(unparsed summarizer) {plan:.40}"),
                strategy_description: String::new(),
                slow_pseudocode: String::new(),
                fast_pseudocode: String::new(),
                applicable_when: String::new(),
                do_not_apply_when: String::new(),
                framework_notes: String::new(),
            };
            self.memory.apply(
                fallback,
                MemoryUpdate { action: "append".into(), replace_item_id: None, reason: None },
            );
            self.journal.record(&Event::MemoryUpdated {
                action: "append_fallback".into(),
                direction: Some(plan.chars().take(80).collect()),
            })?;
        }
        Ok(())
    }

    // ---------- stage: restitch + e2e gates ----------

    pub fn stage_restitch_and_e2e(
        &mut self,
        best: &Candidate,
        baseline_ms: f64,
    ) -> Result<(bool, Option<f64>, serde_json::Value)> {
        let out_path = self.tmp_path("restitched")? + ".py";
        let resp = self.runner_call(
            "restitch",
            json!({
                "source_code": best.source,
                "out_path": out_path,
                "baseline_file": self.model_file(),
                "baseline_weights_path": self.baseline_weights,
                "baseline_seed": 0,
                "provenance": {"plan": best.plan, "id": best.id},
            }),
        )?;
        if resp.get("ok") != Some(&json!(true)) {
            return Ok((false, None, resp));
        }
        let restitched = resp["out_path"].as_str().unwrap_or("").to_string();

        // Gate 3: model-level correctness with float64 fallback.
        let e2e = self.runner_call(
            "e2e_verify",
            json!({
                "baseline_file": self.model_file(),
                "variant_file": restitched,
                "baseline_weights_path": self.baseline_weights,
                "baseline_seed": 0,
                "seeds": [0, 1, 2],
                "rtol": self.cfg.hyper.final_tol,
                "atol": self.cfg.hyper.final_tol,
                "rho_max": 10.0,
            }),
        )?;
        let gate3_passed = e2e["gate3"]["passed"].as_bool().unwrap_or(false);

        // Gate 4: performance (clean bench of re-stitched model).
        let mut gate4 = json!({});
        let mut final_ms: Option<f64> = None;
        if gate3_passed {
            let bench = self.bench_command(Some(&restitched))?;
            final_ms = bench.get("mean_ms").and_then(|v| v.as_f64());
            let perf_ok = final_ms
                .map(|ms| ms <= baseline_ms * self.cfg.hyper.gamma)
                .unwrap_or(false);
            gate4 = json!({"passed": perf_ok, "final_ms": final_ms, "baseline_ms": baseline_ms, "gamma": self.cfg.hyper.gamma});
        }

        let all_ok = gate3_passed && gate4["passed"].as_bool().unwrap_or(false);
        let detail = json!({"gate3": e2e, "gate4": gate4});
        self.journal.record(&Event::GatesVerdict {
            stage: "e2e".into(),
            passed: all_ok,
            detail: detail.clone(),
        })?;
        Ok((all_ok, final_ms, detail))
    }

    // ---------- full run ----------

    pub fn run(&mut self, model_file: &str) -> Result<serde_json::Value> {
        self.journal.record(&Event::RunStarted {
            run_id: self.session_id.clone(),
            model_file: model_file.into(),
            provider: self.cfg.provider.clone(),
            model: self.cfg.model.clone(),
        })?;

        // Stage 1: trace
        let trace = self.stage_trace()?;
        let classification = trace["classification"]
            .as_str()
            .unwrap_or("unknown")
            .to_string();
        if classification != "optimizable" {
            let root_cause = if classification == "extern_only" {
                "library dominance (cuBLAS/cuDNN) — no optimizable Triton kernels"
            } else {
                "no GPU kernels detected"
            };
            self.journal.record(&Event::RunFinished {
                outcome: "fallback".into(),
                speedup: None,
                root_cause: Some(root_cause.into()),
                stop_reason: Some("not_optimizable".into()),
            })?;
            return Ok(json!({
                "outcome": "fallback",
                "root_cause": root_cause,
                "classification": classification,
            }));
        }

        // Stage 2: baseline bench (+ graphsignal attribution payload).
        let baseline_ms = self.stage_baseline_bench()?;
        let baseline_payload = self.last_bench_signals.clone();

        // Stage 3: two-tier profile → planner context.
        let profiling_ctx = self.stage_profile()?;

        // Stage 4: beam search loop. The executor edits the REAL Inductor
        // Triton kernel (extracted at trace time) in a kernel workbench.
        let kernel_file_text = std::fs::read_to_string(model_file)?;
        let kernel_workbench = trace["triton_kernel_sources"]
            .as_array()
            .and_then(|a| a.first())
            .and_then(|k| k.get("source"))
            .and_then(|s| s.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| {
                eprintln!("[pipeline] no @triton.jit body extracted; falling back to model file as workbench");
                kernel_file_text.clone()
            });
        let mut best: Option<Candidate> = None;
        let mut recent_directions: Vec<String> = Vec::new();
        let mut no_improvement_streak = 0u32;
        let mut best_ms = baseline_ms;

        // Beam frontier: the extracted Inductor kernel is the root arm. Each arm
        // tracks how often its chain has been expanded, which drives UCB plan
        // allocation (paper §4.3).
        let mut frontier: Vec<BeamNode> = vec![BeamNode {
            expansions: 0,
            candidate: Candidate {
                id: "baseline".into(),
                chain: 0,
                iteration: 0,
                source: kernel_workbench.clone(),
                plan: "(baseline)".into(),
                latency_ms: if baseline_ms > 0.0 { Some(baseline_ms) } else { None },
                passed: true,
                commit: None,
                change_summary: None,
                hints: None,
                evidence: None,
            },
        }];

        'outer: for iteration in 0..self.cfg.hyper.t_iterations {
            let mut children: Vec<BeamNode> = Vec::new();
            let diversity_hint = meltdown_detected(&recent_directions, 6, 2);

            // Allocate this iteration's N plans across the beam by UCB(c).
            let n_plans =
                crate::search::plans_for_frontier(self.cfg.hyper.n_plans, frontier.len());
            let slots = allocate_expansions(&frontier, n_plans, self.cfg.hyper.ucb_c);
            let mut pending = vec![0u32; frontier.len()];
            for &a in &slots {
                pending[a] += 1;
            }

            let mut plan_no: u32 = 0;
            for (arm, &count) in pending.iter().enumerate() {
                if count == 0 {
                    continue;
                }
                let parent = frontier[arm].candidate.clone();
                let parent_expansions = frontier[arm].expansions;
                let chain = arm as u32;
                for _ in 0..count {
                    let this_plan = plan_no;
                    plan_no += 1;
                    let plan = self.stage_plan(
                        &parent.source,
                        &profiling_ctx,
                        chain,
                        diversity_hint,
                        &recent_directions,
                    )?;
                    recent_directions.push(plan.clone());

                    let (cand, payload, last_error) =
                        self.stage_execute_and_verify(&parent.source, &plan, chain, iteration, this_plan)?;
                    self.journal.record(&Event::CandidateEvaluated {
                        iteration,
                        chain,
                        plan: plan.clone(),
                        passed: cand.passed,
                        latency_ms: cand.latency_ms,
                        error: last_error.clone(),
                        commit: None,
                        change_summary: None,
                        hints: None,
                        evidence: None,
                    })?;

                    if cand.passed {
                        // Summarizer → experience memory (paper §4.5).
                        let speedup = cand
                            .latency_ms
                            .map(|m| baseline_ms / m)
                            .unwrap_or(1.0);
                        self.stage_summarize(
                            baseline_payload.as_deref(),
                            payload.as_deref(),
                            &parent.source,
                            &cand.source,
                            &plan,
                            speedup,
                            iteration,
                        )?;
                        if best.as_ref().map(|cur| cand.latency_ms < cur.latency_ms).unwrap_or(true) {
                            best = Some(cand.clone());
                        }
                        children.push(BeamNode {
                            expansions: parent_expansions + 1,
                            candidate: cand,
                        });
                    }
                }
            }

            let beam = diverse_select_nodes(&children, self.cfg.hyper.b_beam as usize);
            if let Some(b) = beam.first() {
                if let Some(ms) = b.candidate.latency_ms {
                    if ms < best_ms {
                        best_ms = ms;
                        no_improvement_streak = 0;
                    } else {
                        no_improvement_streak += 1;
                    }
                }
            } else {
                no_improvement_streak += 1;
            }
            if !beam.is_empty() {
                frontier = beam;
            }

            if no_improvement_streak >= 2 {
                break 'outer;
            }
        }

        match best {
            Some(cand) => {
                let (gates_ok, final_ms, _detail) =
                    self.stage_restitch_and_e2e(&cand, baseline_ms)?;
                if gates_ok {
                    let speedup = final_ms.map(|m| baseline_ms / m);
                    self.journal.record(&Event::RunFinished {
                        outcome: "optimized".into(),
                        speedup,
                        root_cause: None,
                        stop_reason: Some("completed".into()),
                    })?;
                    Ok(json!({
                        "outcome": "optimized",
                        "speedup": speedup,
                        "baseline_ms": baseline_ms,
                        "final_ms": final_ms,
                        "candidate_id": cand.id,
                    }))
                } else {
                    let cause = "e2e gates rejected candidate; compiler baseline preserved";
                    self.journal.record(&Event::RunFinished {
                        outcome: "fallback".into(),
                        speedup: None,
                        root_cause: Some(cause.into()),
                        stop_reason: Some("e2e_rejected".into()),
                    })?;
                    Ok(json!({"outcome": "fallback", "root_cause": cause}))
                }
            }
            None => {
                let cause = "no candidate passed gates 1-2";
                self.journal.record(&Event::RunFinished {
                    outcome: "fallback".into(),
                    speedup: None,
                    root_cause: Some(cause.into()),
                    stop_reason: Some("no_candidate".into()),
                })?;
                Ok(json!({"outcome": "fallback", "root_cause": cause}))
            }
        }
    }

    fn model_file(&self) -> String {
        self.model_file.clone()
    }

    /// Strip markdown code fences the model may wrap submissions in
    /// (```python ... ```). SyntaxError-at-line-1 is the telltale otherwise.
    fn strip_code_fences(s: &str) -> String {
        let trimmed = s.trim();
        if let Some(rest) = trimmed.strip_prefix("```") {
            // Skip the language tag on the opening line.
            let after_first_line = rest.splitn(2, '\n').nth(1).unwrap_or(rest);
            let body = after_first_line
                .strip_suffix("```")
                .unwrap_or(after_first_line);
            return body.trim().to_string();
        }
        trimmed.to_string()
    }

    fn render_prompt(&self, template: &str, vars: &serde_json::Value) -> Result<String> {
        let path = self.cfg.prompts_dir.join(template);
        let raw = std::fs::read_to_string(&path)
            .with_context(|| format!("reading prompt template {}", path.display()))?;
        Ok(crate::prompts::render(&raw, vars))
    }
}
