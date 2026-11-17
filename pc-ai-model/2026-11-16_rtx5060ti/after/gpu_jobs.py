# gpu_jobs.py - extra jobs for the autopilot runner (added in runner 1.5 for the RTX 5060 Ti swap).
# Job types:
#   {"type": "gpu_check"}
#       read-only self test (driver / PCIe / torch / triton / torch.compile / SageAttention)
#   {"type": "hf_download", "repo": "owner/name", "filename": "x.safetensors", "subdir": "diffusion_models"}
#       downloads one file from Hugging Face into ComfyUI\models\<subdir>\, verifies size + SHA-256
#       against the Hugging Face listing, never overwrites or deletes an existing different file.
#   {"type": "comfy_info", "patterns": ["Sparse", "QwenImage21"]}
#       ComfyUI version (/system_stats) and the input spec of nodes whose class name matches a pattern.
import hashlib, json, os, shutil, subprocess, sys, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
_R = None  # runner module, set in register()


def h_gpu_check(ctx, j):
    out = ctx.out("gpu_check.json")
    script = os.path.join(HERE, "scripts", "gpu_selftest.py")
    ctx.run([sys.executable, "-I", script, out], cwd=os.path.dirname(script), timeout=j.get("timeout", 900))
    ctx.add(out)
    return json.load(open(out, encoding="utf-8"))


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(16 << 20), b""):
            h.update(b)
    return h.hexdigest()


def h_hf_download(ctx, j):
    repo, fname, sub = j["repo"], j["filename"], j["subdir"]
    if "/" in fname or "\\" in fname or ".." in sub:
        raise ValueError("filename must be a bare file name; subdir must stay under models")
    api = f"https://huggingface.co/api/models/{repo}/tree/{j.get('revision', 'main')}?recursive=true"
    with urllib.request.urlopen(api, timeout=120) as r:
        tree = json.load(r)
    hits = [e for e in tree if e.get("type") == "file" and e["path"].split("/")[-1] == fname]
    if len(hits) != 1:
        raise RuntimeError(f"{fname}: found {len(hits)} matches in {repo}: {[e['path'] for e in hits]}")
    e = hits[0]
    size = e.get("lfs", {}).get("size", e.get("size"))
    sha = e.get("lfs", {}).get("oid")
    url = f"https://huggingface.co/{repo}/resolve/{j.get('revision', 'main')}/{e['path']}"
    dest_dir = _R.safe_path(os.path.join(_R.CFG["comfy_dir"], "models", sub))
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, fname)
    info = {"repo": repo, "path": e["path"], "url": url, "size": size, "sha256_hf": sha, "dest": dest}
    if os.path.exists(dest):
        if os.path.getsize(dest) == size and (not sha or _sha256(dest) == sha):
            info["status"] = "already_present_ok"; ctx.log(info); return info
        raise RuntimeError(f"{dest} exists but differs from the Hugging Face file; not overwriting")
    part = dest + ".part"
    ctx.run(["curl.exe", "-L", "-C", "-", "--retry", "20", "--retry-delay", "15", "--retry-all-errors",
             "-sS", "-o", part, url], timeout=j.get("timeout", 4 * 3600))
    got = os.path.getsize(part)
    if got != size:
        raise RuntimeError(f"size mismatch: got {got}, expected {size} (partial file kept: {part})")
    if sha:
        h = _sha256(part); info["sha256_local"] = h
        if h != sha:
            raise RuntimeError(f"sha256 mismatch: {h} != {sha} (file kept: {part})")
    os.replace(part, dest)
    info["status"] = "downloaded_ok"; ctx.log(info)
    return info


def h_comfy_info(ctx, j):
    _R.comfy_start_if_needed(ctx)
    base = _R.CFG["comfy_url"]
    stats = json.loads(_R.http(base + "/system_stats")[0])
    objs = json.loads(_R.http(base + "/object_info", timeout=300)[0])
    pats = [p.lower() for p in j.get("patterns", [])]
    nodes = {k: v for k, v in objs.items() if any(p in k.lower() for p in pats)}
    p = ctx.out("comfy_info.json")
    json.dump({"system": stats.get("system", {}), "devices": stats.get("devices", []), "nodes": nodes},
              open(p, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    ctx.add(p)
    return {"comfyui_version": stats.get("system", {}).get("comfyui_version"), "matched_nodes": sorted(nodes)}


def register(runner):
    global _R
    _R = runner
    return {"gpu_check": h_gpu_check, "hf_download": h_hf_download, "comfy_info": h_comfy_info}
