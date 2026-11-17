# gpu_selftest.py <out.json>
# Each section is independent: a failure is recorded and the next section still runs.
import json, subprocess, sys, time, traceback

R = {}


def sec(name):
    def deco(fn):
        t0 = time.time()
        try:
            R[name] = fn()
        except Exception as e:
            R[name] = {"ok": False, "error": f"{e.__class__.__name__}: {e}", "trace": traceback.format_exc()[-1500:]}
        if isinstance(R[name], dict):
            R[name]["secs"] = round(time.time() - t0, 1)
        return fn
    return deco


@sec("nvidia_smi")
def _():
    q = ("name,driver_version,memory.total,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,"
         "pcie.link.width.max,power.limit,power.max_limit,temperature.gpu,clocks.max.graphics")
    out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"], capture_output=True, timeout=30)
    vals = [v.strip() for v in out.stdout.decode("utf-8", "replace").splitlines()[0].split(",")]
    d = dict(zip(q.split(","), vals))
    full = subprocess.run(["nvidia-smi", "-q"], capture_output=True, timeout=30).stdout.decode("utf-8", "replace")
    lines = full.splitlines()
    # BAR1 total = 16384 MiB means Resizable BAR is on; 256 MiB means it is off.
    d["bar1_lines"] = [lines[i + 1].strip() for i, l in enumerate(lines[:-1]) if "BAR1 Memory Usage" in l]
    d["ok"] = True
    return d


@sec("torch")
def _():
    import torch
    p = torch.cuda.get_device_properties(0)
    return {"ok": torch.cuda.is_available(), "version": torch.__version__, "cuda": torch.version.cuda,
            "device": p.name, "capability": f"sm_{p.major}{p.minor}", "vram_gb": round(p.total_memory / 2**30, 1),
            "arch_list": torch.cuda.get_arch_list()}


@sec("matmul_bench")
def _():
    import torch
    a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16); b = torch.randn_like(a)
    for _ in range(3):
        a @ b
    torch.cuda.synchronize(); t0 = time.time(); n = 20
    for _ in range(n):
        a @ b
    torch.cuda.synchronize(); dt = (time.time() - t0) / n
    return {"ok": True, "bf16_tflops": round(2 * 8192**3 / dt / 1e12, 1)}


@sec("pcie_bandwidth")
def _():
    # Host->GPU copy speed (what weight offloading/streaming depends on) and the PCIe link while copying.
    import torch, threading
    n = 256 * 2**20
    dst = torch.empty(n, dtype=torch.uint8, device="cuda")
    res, links = {}, []

    def watch(stop):
        while not stop.is_set():
            o = subprocess.run(["nvidia-smi", "--query-gpu=pcie.link.gen.current,pcie.link.width.current",
                                "--format=csv,noheader"], capture_output=True, timeout=10)
            links.append(o.stdout.decode().strip()); time.sleep(0.3)
    for kind in ("pinned", "pageable"):
        src = torch.empty(n, dtype=torch.uint8, pin_memory=(kind == "pinned"))
        dst.copy_(src, non_blocking=True); torch.cuda.synchronize()
        stop = threading.Event(); th = threading.Thread(target=watch, args=(stop,)); th.start()
        t0 = time.time(); k = 0
        while time.time() - t0 < 3:
            dst.copy_(src, non_blocking=True); k += 1
        torch.cuda.synchronize(); dt = time.time() - t0
        stop.set(); th.join()
        res[f"{kind}_GBps"] = round(k * n / dt / 1e9, 2)
    res["link_during_copy"] = sorted(set(links))
    res["ok"] = True
    return res


@sec("triton")
def _():
    import triton, triton.language as tl, torch

    @triton.jit
    def add(x, y, o, n, B: tl.constexpr):
        i = tl.program_id(0) * B + tl.arange(0, B); m = i < n
        tl.store(o + i, tl.load(x + i, mask=m) + tl.load(y + i, mask=m), mask=m)
    x = torch.rand(1 << 20, device="cuda"); y = torch.rand_like(x); o = torch.empty_like(x)
    add[(triton.cdiv(x.numel(), 1024),)](x, y, o, x.numel(), B=1024)
    return {"ok": bool(torch.allclose(o, x + y)), "version": triton.__version__}


@sec("torch_compile")
def _():
    import torch
    f = torch.compile(lambda t: torch.nn.functional.gelu(t) * 2 + 1)
    t = torch.randn(4096, 4096, device="cuda")
    return {"ok": bool(torch.allclose(f(t), torch.nn.functional.gelu(t) * 2 + 1, atol=1e-4))}


@sec("sageattention")
def _():
    import torch, torch.nn.functional as F
    from sageattention import sageattn
    q, k, v = (torch.randn(2, 24, 4096, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    ref = F.scaled_dot_product_attention(q, k, v)
    out = sageattn(q, k, v, tensor_layout="HND", is_causal=False)
    err = ((out - ref).abs().mean() / ref.abs().mean()).item()

    def bench(fn):
        for _ in range(3):
            fn()
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(20):
            fn()
        torch.cuda.synchronize(); return (time.time() - t0) / 20 * 1000
    ms_sdpa = bench(lambda: F.scaled_dot_product_attention(q, k, v))
    ms_sage = bench(lambda: sageattn(q, k, v, tensor_layout="HND", is_causal=False))
    import importlib.metadata as md
    return {"ok": err < 0.1, "version": md.version("sageattention"), "mean_rel_err": round(err, 4),
            "sdpa_ms": round(ms_sdpa, 2), "sage_ms": round(ms_sage, 2), "speedup": round(ms_sdpa / ms_sage, 2)}


json.dump(R, open(sys.argv[1], "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(json.dumps({k: v.get("ok") if isinstance(v, dict) else v for k, v in R.items()}))
