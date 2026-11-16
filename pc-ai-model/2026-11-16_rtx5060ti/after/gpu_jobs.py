# gpu_jobs.py - GPU check job for the autopilot runner (added in runner 1.5 for the RTX 5060 Ti swap).
# Job type:
#   {"type": "gpu_check"}  -> runs gpu_selftest.py in a separate venv python process and reports
#                             driver / PCIe link / torch / triton / torch.compile / SageAttention status.
# Read-only: installs nothing, changes no settings.
import json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))


def h_gpu_check(ctx, j):
    out = ctx.out("gpu_check.json")
    script = os.path.join(HERE, "scripts", "gpu_selftest.py")
    ctx.run([sys.executable, "-I", script, out], cwd=os.path.dirname(script), timeout=j.get("timeout", 900))
    ctx.add(out)
    return json.load(open(out, encoding="utf-8"))


def register(runner):
    return {"gpu_check": h_gpu_check}
