"""Tests for the NCU CSV parser (unit/section-aware SOL extraction)."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "runner"))

from kernelopt_runner import ncu  # noqa: E402

HEADER = (
    '"ID","Process ID","Process Name","Host Name","Kernel Name","Context","Stream",'
    '"Block Size","Grid Size","Device","CC","Section Name","Metric Name","Metric Unit",'
    '"Metric Value","Rule Name","Rule Type","Rule Description","Estimated Speedup Type",'
    '"Estimated Speedup"'
)


def _row(section, metric, unit, value):
    return (
        f'"0","1","app","host","void add_bias_bf16x8_kernel<256, 1>(...)","1","7",'
        f'"(256, 1, 1)","(1, 8, 1)","0","12.0","{section}","{metric}","{unit}","{value}",'
        f'"","","","",""'
    )


CSV = "\n".join(
    [
        HEADER,
        # SOL section: the real percentages
        _row("GPU Speed Of Light Throughput", "Memory Throughput", "%", "14.26"),
        _row("GPU Speed Of Light Throughput", "Duration", "ns", "2,848"),
        _row("GPU Speed Of Light Throughput", "Compute (SM) Throughput", "%", "0.09"),
        # Analysis section: same metric names, different units — must NOT clobber
        _row("Memory Workload Analysis", "Memory Throughput", "byte/s", "12,943,820,224.72"),
        _row("Launch Statistics", "Registers Per Thread", "register/thread", "19"),
        _row("Occupancy", "Achieved Occupancy", "%", "12.90"),
    ]
)


def test_sol_percent_not_clobbered_by_byte_rate():
    ctx = ncu.parse_ncu_csv(CSV)
    k = ctx["kernels"][0]
    assert k["sol_memory"] == 14.26  # not 12943820224.72
    assert k["sol_compute"] == 0.09
    assert k["achieved_occupancy"] == 12.90
    assert k["registers"] == 19


def test_duration_ns_converted_to_us():
    ctx = ncu.parse_ncu_csv(CSV)
    assert abs(ctx["kernels"][0]["duration_us"] - 2.848) < 1e-9
