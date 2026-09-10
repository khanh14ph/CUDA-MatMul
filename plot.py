"""
Roofline plot generator.

    python plot.py                     -> runs ./build/benchmark, writes roofline.png
    python plot.py bench.txt           -> parses a saved benchmark log
    ./build/benchmark | python plot.py -
    python plot.py --ncu               -> *measured* points via Nsight Compute (Volta+)
    python plot.py --nvprof            -> *measured* points via nvprof (pre-Volta, e.g. Pascal)

By default each kernel's point is *modeled*: arithmetic intensity from compulsory
GEMM traffic (2*M*N*K FLOPs / 4*(MN+NK+MK) bytes) and TFLOP/s from the benchmark's
wall-clock time.

--ncu / --nvprof replace those with hardware-measured values:
  achieved FLOP/s     = (2*FFMA + FADD + FMUL) / time            (fp64: DFMA/DADD/DMUL)
  achieved DRAM bytes = dram__bytes_read + dram__bytes_write     (nvprof: transactions*32)
so AI = FLOPs / DRAM bytes and TFLOP/s = FLOPs / time are what the hardware actually did.
Pick the tool by GPU: Nsight Compute supports only Volta (sm_70) and newer, so on a
Pascal card (GTX 10-series, sm_61) use --nvprof. --nvprof also measures cuBLAS; --ncu
leaves cuBLAS modeled (its vendor kernel name has no stable match).

GPU ceilings come from src/gpu_infor.cu (auto-compiled), kernels from '[Benchmark]' lines.
"""
import csv
import io
import json
import os
import re
import subprocess
import sys

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1. GPU ceilings — auto-detected by compiling & running gpu_infor.cu
# ---------------------------------------------------------------------------
SRC = "src/gpu_infor.cu"
BIN = "./peak_bw"


def query_gpu(device=0):
    """Compile gpu_infor.cu if needed, run it with --json, return a GPU dict."""
    if not os.path.exists(BIN) or os.path.getmtime(SRC) > os.path.getmtime(BIN):
        print(f"compiling {SRC} ...")
        subprocess.run(["nvcc", SRC, "-o", BIN], check=True)
    out = subprocess.run([BIN, "--json"], capture_output=True, text=True, check=True).stdout
    devices = [json.loads(line) for line in out.splitlines() if line.startswith("{")]
    if not devices:
        sys.exit("no CUDA devices reported")
    g = devices[device]
    return {
        "name": f"{g['name']} (sm_{g['cc'].replace('.', '')})",
        "mem_bw_gbps": g["mem_bw_gbps"],
        "ceilings": {
            "fp32 compute bound": g["fp32_tflops"],
        },
    }


GPU = query_gpu(device=0)

# ---------------------------------------------------------------------------
# 2. Kernels — parsed from ./build/benchmark output (any number of them)
#    Expected lines:   M=1024 N=1024 K=1024
#                      [Benchmark] gemm_v1 | Avg Time: 4.5482 ms | ...
# ---------------------------------------------------------------------------
BENCH = "./build/benchmark"
COLORS = ["tab:blue", "tab:green", "tab:orange", "tab:purple", "tab:brown",
          "tab:pink", "tab:olive", "tab:cyan"]


def parse_benchmark(text):
    m = re.search(r"M=(\d+)\s+N=(\d+)\s+K=(\d+)", text)
    if not m:
        sys.exit("could not find 'M=.. N=.. K=..' in benchmark output")
    M, N, K = map(int, m.groups())
    flops = 2.0 * M * N * K
    bytes_moved = 4 * (M * N + N * K + M * K)   # fp32, compulsory traffic

    kernels = []
    for i, (name, ms) in enumerate(
            re.findall(r"\[Benchmark\]\s*(\S+)\s*\|\s*Avg Time:\s*([\d.]+)\s*ms", text)):
        kernels.append({
            "name": name, "time_ms": float(ms),
            "flops": flops, "bytes": bytes_moved,
            "color": "tab:red" if "cublas" in name.lower() else COLORS[i % len(COLORS)],
        })
    if not kernels:
        sys.exit("no '[Benchmark]' lines found")
    return kernels, (M, N, K)


def load_kernels(source=None):
    """source: None -> run BENCH; a filename -> read saved output; '-' -> stdin."""
    if source is None:
        text = subprocess.run([BENCH], capture_output=True, text=True, check=True).stdout
    elif source == "-":
        text = sys.stdin.read()
    else:
        text = open(source).read()
    return parse_benchmark(text)


def resolve(k):
    """Return (arithmetic intensity FLOP/byte, achieved TFLOP/s)."""
    if "tflops" in k and "ai" in k:
        return k["ai"], k["tflops"]
    t = k["time_ms"] * 1e-3
    return k["flops"] / k["bytes"], k["flops"] / t / 1e12


# ---------------------------------------------------------------------------
# 2b. Measured points via Nsight Compute (--ncu)
#     Same definitions as the roofline annotations:
#       FLOPs = 2*FFMA + FADD + FMUL (+ fp64 variants), weighted 2/1/1
#       DRAM  = dram__bytes_read + dram__bytes_write
# ---------------------------------------------------------------------------
NCU = os.environ.get("NCU", "ncu")

# metric -> FLOP weight (FMA counts as 2, add/mul as 1); fp32 and fp64 ops
FLOP_METRICS = {
    "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum": 2,
    "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum": 1,
    "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum": 1,
    "smsp__sass_thread_inst_executed_op_dfma_pred_on.sum": 2,
    "smsp__sass_thread_inst_executed_op_dadd_pred_on.sum": 1,
    "smsp__sass_thread_inst_executed_op_dmul_pred_on.sum": 1,
}
BYTE_METRICS = ["dram__bytes_read.sum", "dram__bytes_write.sum"]
TIME_METRIC = "gpu__time_duration.sum"

# ncu prints scaled values with a unit column; normalise everything to base units
UNIT_SCALE = {
    "": 1, "inst": 1, "byte": 1,
    "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9, "Tbyte": 1e12,
    "second": 1, "msecond": 1e-3, "usecond": 1e-6, "nsecond": 1e-9,
}


def _parse_ncu_csv(text):
    """Parse `ncu --csv --page raw` output into {metric_name: value_in_base_units}."""
    rows = list(csv.reader(io.StringIO(text)))
    header = start = None
    for i, r in enumerate(rows):
        if "Metric Name" in r and "Metric Value" in r:
            header, start = r, i + 1
            break
    if header is None:
        raise RuntimeError("could not parse ncu CSV (no metric header found)")
    col = {name: header.index(name) for name in header}
    i_name, i_val = col["Metric Name"], col["Metric Value"]
    i_unit = col.get("Metric Unit")

    out = {}
    for r in rows[start:]:
        if len(r) <= i_val:
            continue
        raw = r[i_val].strip().replace(",", "")          # strip locale grouping
        if raw in ("", "n/a", "N/A", "inf"):
            continue
        try:
            v = float(raw)
        except ValueError:
            continue
        unit = r[i_unit].strip() if i_unit is not None and len(r) > i_unit else ""
        out[r[i_name].strip()] = v * UNIT_SCALE.get(unit, 1)
    return out


def measure_kernel(kernel_regex):
    """Profile the first launch matching `kernel_regex`; return (ai, tflops, info)."""
    metrics = list(FLOP_METRICS) + BYTE_METRICS + [TIME_METRIC]
    cmd = [NCU, "--csv", "--page", "raw", "-k", f"regex:{kernel_regex}",
           "-c", "1", "--metrics", ",".join(metrics), BENCH]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError((res.stderr.strip() or res.stdout.strip()
                            or "ncu exited non-zero").splitlines()[-1])
    m = _parse_ncu_csv(res.stdout)

    flops = sum(w * m.get(name, 0.0) for name, w in FLOP_METRICS.items())
    dram = sum(m.get(b, 0.0) for b in BYTE_METRICS)
    t = m.get(TIME_METRIC, 0.0)                          # seconds
    if flops <= 0 or t <= 0:
        raise RuntimeError("no FLOP/duration counters returned "
                           "(kernel name matched? counters permitted?)")
    ai = flops / dram if dram > 0 else float("nan")
    return ai, flops / t / 1e12, dict(flops=flops, dram=dram, t=t)


def ncu_kernel_regex(name):
    """Map a '[Benchmark]' name to an ncu -k regex, or None to skip (opaque kernel)."""
    low = name.lower()
    if "cublas" in low:
        return None            # real kernel name is internal/mangled; leave modeled
    tok = re.search(r"gemm_v\d+|v\d+", low)
    return tok.group(0) if tok else re.escape(name)


def annotate_measured(kernels):
    """Fill k['ai']/k['tflops'] from ncu so resolve() plots measured points."""
    print("profiling with Nsight Compute (ncu) ...")
    for k in kernels:
        rgx = ncu_kernel_regex(k["name"])
        if rgx is None:
            print(f"  {k['name']:>10}: opaque kernel, keeping modeled point")
            continue
        try:
            ai, tflops, info = measure_kernel(rgx)
        except Exception as e:                           # noqa: BLE001 - report & fall back
            print(f"  {k['name']:>10}: ncu failed ({e}); keeping modeled point")
            continue
        k["ai"], k["tflops"], k["measured"] = ai, tflops, True
        print(f"  {k['name']:>10}: measured {tflops:8.3f} TFLOP/s  AI={ai:7.2f}  "
              f"(FLOPs={info['flops']:.3e}, DRAM={info['dram']/1e6:.1f} MB, "
              f"t={info['t']*1e3:.4f} ms)")


# ---------------------------------------------------------------------------
# 2c. Measured points via nvprof (legacy profiler; the only option on pre-Volta
#     GPUs like Pascal/sm_61, where Nsight Compute is unsupported).
#     Counts come from nvprof; TIMING comes from the un-profiled benchmark run
#     (nvprof serialises kernels, so its own durations are not representative).
#     Unlike the ncu path this also measures cuBLAS, by attributing the vendor
#     gemm kernel (e.g. maxwell_sgemm_*) to the 'cuBLAS' benchmark entry.
# ---------------------------------------------------------------------------
NVPROF = os.environ.get("NVPROF", "nvprof")
NVPROF_FLOP = ["flop_count_sp", "flop_count_dp"]   # nvprof already weights FMA as 2
NVPROF_TXN = ["dram_read_transactions", "dram_write_transactions"]
DRAM_TXN_BYTES = 32                                # a DRAM transaction is 32 bytes


def _parse_nvprof_csv(text):
    """Parse `nvprof --metrics ... --csv` output -> {kernel_name: {metric: avg}}."""
    rows = list(csv.reader(io.StringIO(text)))
    header = start = None
    for i, r in enumerate(rows):
        if "Kernel" in r and "Metric Name" in r and "Avg" in r:
            header, start = r, i + 1
            break
    if header is None:
        raise RuntimeError("could not parse nvprof CSV (no metric header found)")
    c = {name: header.index(name) for name in header}
    i_kern, i_metric, i_avg = c["Kernel"], c["Metric Name"], c["Avg"]

    out = {}
    for r in rows[start:]:
        if len(r) <= i_avg:
            continue
        raw = r[i_avg].strip().replace(",", "")
        try:
            v = float(raw)
        except ValueError:
            continue
        out.setdefault(r[i_kern].strip(), {})[r[i_metric].strip()] = v
    return out


def _nvprof_matches(bench_name, profiled_names):
    """Which profiled kernel name(s) belong to this '[Benchmark]' entry."""
    low = bench_name.lower()
    if "cublas" in low:
        # cuBLAS dispatches vendor kernels (maxwell_sgemm_*, ...): gemm-ish but
        # not one of our own gemm_vN kernels.
        return [n for n in profiled_names
                if re.search(r"gemm|cutlass", n, re.I) and "gemm_v" not in n]
    tok = re.search(r"gemm_v\d+|v\d+", low)
    key = tok.group(0) if tok else re.escape(bench_name)
    return [n for n in profiled_names if re.search(key, n)]


def annotate_measured_nvprof(kernels):
    """Fill k['ai']/k['tflops'] from a single nvprof pass over all kernels."""
    print("profiling with nvprof (one pass, all kernels) ...")
    cmd = [NVPROF, "--csv", "--metrics", ",".join(NVPROF_FLOP + NVPROF_TXN), BENCH]
    res = subprocess.run(cmd, capture_output=True, text=True)
    blob = res.stderr + "\n" + res.stdout          # nvprof reports on stderr
    try:
        prof = _parse_nvprof_csv(blob)
    except Exception as e:                          # noqa: BLE001 - report & fall back
        tail = (res.stderr.strip().splitlines() or ["nvprof produced no output"])[-1]
        print(f"  nvprof failed ({e}: {tail}); keeping modeled points")
        return

    names = list(prof)
    for k in kernels:
        matched = _nvprof_matches(k["name"], names)
        flops = sum(prof[n].get(m, 0.0) for n in matched for m in NVPROF_FLOP)
        txns = sum(prof[n].get(m, 0.0) for n in matched for m in NVPROF_TXN)
        if not matched or flops <= 0:
            print(f"  {k['name']:>10}: no matching FLOP counters; keeping modeled point")
            continue
        dram = txns * DRAM_TXN_BYTES
        t = k["time_ms"] * 1e-3                      # real (un-profiled) wall time
        k["ai"] = flops / dram if dram > 0 else float("nan")
        k["tflops"] = flops / t / 1e12
        k["measured"] = True
        via = "+".join(n.split("(")[0].split("<")[0].strip().split()[-1] for n in matched)
        print(f"  {k['name']:>10}: measured {k['tflops']:8.3f} TFLOP/s  AI={k['ai']:7.2f}  "
              f"(FLOPs={flops:.3e}, DRAM={dram/1e6:.1f} MB, t={k['time_ms']:.4f} ms wall)"
              f"  [{via}]")


def plot_roofline(gpu, kernels, out="roofline.png", subtitle=""):
    bw = gpu["mem_bw_gbps"] / 1e3          # TB/s, so bw * AI is in TFLOP/s
    peak = max(gpu["ceilings"].values())

    ai = np.logspace(-2, 4, 400)
    fig, ax = plt.subplots(figsize=(11, 5.5))

    # Memory roof (diagonal), clipped at the top compute ceiling
    ax.plot(ai, np.minimum(bw * ai, peak), lw=2.5, color="#6baed6")
    ax.text(0.15, bw * 0.15 * 1.6, f"{gpu['mem_bw_gbps']:.0f} GB/s GMEM BW bound",
            rotation=np.degrees(np.arctan(1.0)) * 0.72, color="#3182bd", fontsize=10)

    # Compute roofs (horizontal), starting at their ridge points
    for name, tf in gpu["ceilings"].items():
        ridge = tf / bw
        ax.hlines(tf, ridge, ai[-1], lw=2.5, color="#6baed6")
        ax.hlines(tf, ai[0], ridge, lw=1, ls="--", color="#9ecae1")
        ax.plot(ridge, tf, "s", ms=6, color="#6baed6")
        ax.text(ai[0] * 1.3, tf * 1.15, f"{tf:g} TFLOP/s {name}", fontsize=10, color="#333")

    # Kernel points — numbers go in the legend, only a short name tags the marker
    for i, k in enumerate(kernels):
        x, y = resolve(k)
        bw_k = y * 1e3 / x if x and np.isfinite(x) else float("nan")  # GB/s = FLOP/s / (FLOP/B)
        pct = 100 * y / min(peak, bw * x)
        src = "meas" if k.get("measured") else "model"
        label = (f"{k['name']}  —  {y:.3g} TFLOP/s · {bw_k:.3g} GB/s · "
                 f"AI {x:.3g} · {pct:.0f}% roof  [{src}]")
        color = k.get("color", "tab:blue")
        ax.plot(x, y, "o", ms=10, color=color, mec="black", zorder=5, label=label)
        ax.vlines(x, 1e-3, y, ls="--", lw=0.8, color=color, alpha=0.4)
        ax.annotate(k["name"], (x, y), xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=8, color="#222")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(ai[0], ai[-1]); ax.set_ylim(1e-3, peak * 10)
    ax.set_xlabel("Arithmetic Intensity [FLOP/byte]")
    ax.set_ylabel("Performance [TFLOP/s]")
    ax.set_title(f"Roofline — {gpu['name']}  {subtitle}")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="lower right", fontsize=9, title="kernels (achieved)",
              title_fontsize=9, framealpha=0.92, borderpad=0.8, labelspacing=0.6)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved {out}")
    for k in kernels:
        x, y = resolve(k)
        bw_k = y * 1e3 / x if x and np.isfinite(x) else float("nan")
        bound = "memory" if x < peak / bw else "compute"
        print(f"{k['name']:>10}: {y:8.3f} TFLOP/s  {bw_k:7.1f} GB/s  AI={x:7.1f}  "
              f"({bound}-bound, {100*y/min(peak, bw*x):.1f}% of roof)")


if __name__ == "__main__":
    # python plot.py            -> runs ./build/benchmark (modeled points)
    # python plot.py bench.txt  -> parses a saved log
    # ./build/benchmark | python plot.py -
    # python plot.py --ncu      -> measure points with Nsight Compute
    argv = sys.argv[1:]
    use_ncu = "--ncu" in argv
    use_nvprof = "--nvprof" in argv
    argv = [a for a in argv if a not in ("--ncu", "--nvprof")]
    src = argv[0] if argv else None

    kernels, (M, N, K) = load_kernels(src)
    tag = ""
    if use_nvprof:                 # legacy profiler: works on Pascal, measures cuBLAS too
        annotate_measured_nvprof(kernels)
        tag = " (nvprof-measured)"
    elif use_ncu:                  # Nsight Compute: Volta+ only
        annotate_measured(kernels)
        tag = " (ncu-measured)"
    plot_roofline(GPU, kernels, subtitle=f"GEMM {M}x{N}x{K} fp32" + tag)