"""Parser tests for the ncu command (no GPU counters needed)."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kernelopt_runner.ncu import parse_ncu_csv, _extract_csv

FIXTURE = '''==PROF== Connected to process 1234
==PROF== Profiling "triton_poi_fused_addmm_gelu_0" - 0: 0%....50%....100% - 1 pass
"ID","Process ID","Process Name","Host Name","Kernel Name","Kernel Time","Context","Stream","Section Name","Metric Name","Metric Unit","Metric Value"
"0","1234","python3","localhost","triton_poi_fused_addmm_gelu_0","0","0","7","GPU Speed Of Light Throughput","Compute (SM) Throughput","%","18.25"
"0","1234","python3","localhost","triton_poi_fused_addmm_gelu_0","0","0","7","GPU Speed Of Light Throughput","Memory Throughput","%","46.10"
"0","1234","python3","localhost","triton_poi_fused_addmm_gelu_0","0","0","7","GPU Speed Of Light Throughput","Duration","us","12.40"
"0","1234","python3","localhost","triton_poi_fused_addmm_gelu_0","0","0","7","Launch Statistics","Registers Per Thread","register/thread","18"
"0","1234","python3","localhost","triton_poi_fused_addmm_gelu_0","0","0","7","Occupancy","Achieved Occupancy","%","61.30"
"0","1234","python3","localhost","triton_poi_fused_addmm_gelu_0","0","0","7","Optimization Rules","OPT Memory coalescing is ideal","","1,234.5"
'''


def test_extract_csv_skips_progress_lines():
    csv_text = _extract_csv(FIXTURE)
    assert csv_text.startswith('"ID"')


def test_parse_ncu_csv_kernel_metrics():
    ctx = parse_ncu_csv(_extract_csv(FIXTURE))
    assert len(ctx["kernels"]) == 1
    k = ctx["kernels"][0]
    assert k["name"] == "triton_poi_fused_addmm_gelu_0"
    assert k["sol_compute"] == 18.25
    assert k["sol_memory"] == 46.10
    assert k["duration_us"] == 12.40
    assert k["registers"] == 18.0
    assert k["achieved_occupancy"] == 61.30


def test_parse_ncu_csv_rules_ranked():
    ctx = parse_ncu_csv(_extract_csv(FIXTURE))
    assert ctx["rules"], "rule row missing"
    assert ctx["rules"][0]["estimated_speedup"] == 1234.5
