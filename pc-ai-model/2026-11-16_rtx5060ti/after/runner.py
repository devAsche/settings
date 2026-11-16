# Autopilot runner: executes whitelisted job files that Claude drops into inbox\.
# Start with start_runner.bat (uses C:\AI model\venv python). Stop: create a file named STOP in this folder.
#
# Folder layout (ROOT = folder of this file, e.g. C:\AI model\autopilot):
#   inbox\<id>.json            job written by Claude
#   outbox\<id>.result.json    result written by this runner
#   outbox\files\<id>\...      output files + log.txt
#   done\<id>.json             processed job files
#   status.json                heartbeat (alive_at, current job, queue)
#   handoff\<task>\            manual ChatGPT image steps: prompt.txt + reference images; the user saves the result here
#   keys.env                   API keys (written by the user, never by Claude)
#   config.json                paths and defaults
#
# Only the job types in HANDLERS are executed. Every path a job names must resolve inside
# config["allowed_roots"]. No arbitrary shell commands are accepted.
import glob, json, os, shutil, subprocess, sys, time, traceback, urllib.error, urllib.request, uuid

ROOT = os.path.dirname(os.path.abspath(__file__))
INBOX, OUTBOX, DONE = (os.path.join(ROOT, d) for d in ("inbox", "outbox", "done"))
HANDOFF = os.path.join(ROOT, "handoff")
for d in (INBOX, OUTBOX, DONE, HANDOFF, os.path.join(OUTBOX, "files")):
    os.makedirs(d, exist_ok=True)
CFG = {}


def load_cfg():
    CFG.clear(); CFG.update(json.load(open(os.path.join(ROOT, "config.json"), encoding="utf-8")))


load_cfg()
RESTART_CODE = 3   # start_runner.bat restarts the runner when it exits with this code
VERSION = "1.5"


def load_keys():
    keys = {}
    p = os.path.join(ROOT, "keys.env")
    if os.path.exists(p):
        for line in open(p, encoding="utf-8-sig"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                keys[k.strip()] = v.strip().strip('"')
    for k in ("ELEVENLABS_API_KEY",):
        keys.setdefault(k, os.environ.get(k, ""))
    return keys


# ---------------------------------------------------------------- helpers
def safe_path(p, must_exist=False):
    """Resolve p (absolute, or relative to the parent of ROOT) and require it to sit inside an allowed root."""
    base = os.path.dirname(ROOT)
    full = os.path.realpath(p if os.path.isabs(p) else os.path.join(base, p))
    for r in CFG["allowed_roots"]:
        r = os.path.realpath(r)
        if os.path.commonpath([full.lower(), r.lower()]) == r.lower():
            if must_exist and not os.path.exists(full):
                raise FileNotFoundError(full)
            return full
    raise PermissionError(f"path outside allowed_roots: {full}")


class Ctx:
    def __init__(self, job):
        self.job = job
        self.dir = os.path.join(OUTBOX, "files", job["id"])
        os.makedirs(self.dir, exist_ok=True)
        self.logf = open(os.path.join(self.dir, "log.txt"), "a", encoding="utf-8")
        self.files = []

    def log(self, *a):
        s = " ".join(str(x) for x in a)
        print(f"[{self.job['id']}] {s}", flush=True)
        self.logf.write(s + "\n"); self.logf.flush()

    def out(self, name):
        p = os.path.join(self.dir, name)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        return p

    def add(self, p):
        self.files.append(p)

    def run(self, cmd, cwd=None, timeout=None):
        self.log("RUN", " ".join(f'"{c}"' if " " in str(c) else str(c) for c in cmd))
        pr = subprocess.Popen([str(c) for c in cmd], cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        t0 = time.time()
        for raw in pr.stdout:
            self.logf.write(raw.decode("utf-8", "replace")); self.logf.flush()
            if timeout and time.time() - t0 > timeout:
                pr.kill(); raise TimeoutError(f"timeout after {timeout}s")
        rc = pr.wait()
        if rc != 0:
            raise RuntimeError(f"exit code {rc}")


def http(url, data=None, headers=None, method=None, timeout=600):
    req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read(), r.headers.get("Content-Type", "")
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:2000]}")


def multipart(fields, files):
    b = uuid.uuid4().hex; body = bytearray()
    for k, v in fields.items():
        body += f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
    for k, path in files:
        ext = os.path.splitext(path)[1].lower().lstrip(".")
        mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "webp": "image/webp"}.get(ext, "image/png")
        body += f'--{b}\r\nContent-Disposition: form-data; name="{k}"; filename="{os.path.basename(path)}"\r\nContent-Type: {mime}\r\n\r\n'.encode()
        body += open(path, "rb").read() + b"\r\n"
    body += f"--{b}--\r\n".encode()
    return bytes(body), f"multipart/form-data; boundary={b}"


def copy_to(ctx, src, dest_rel):
    """Optionally copy a result to a second place (e.g. ComfyUI\\input) - must be inside allowed roots."""
    if dest_rel:
        d = safe_path(dest_rel)
        os.makedirs(os.path.dirname(d), exist_ok=True)
        shutil.copy2(src, d); ctx.add(d); ctx.log("copied to", d)


# ---------------------------------------------------------------- job handlers
def h_probe(ctx, j):
    info = {"runner": VERSION, "python": sys.version.split()[0]}
    info["blender_exists"] = os.path.exists(CFG["blender_exe"])
    for name, cmd in (("ffmpeg", ["ffmpeg", "-version"]), ("node", ["node", "-v"]), ("npx", [shutil.which("npx") or "npx", "-v"]),
                      ("gpu", ["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader"])):
        try:
            info[name] = subprocess.run(cmd, capture_output=True, timeout=30).stdout.decode("utf-8", "replace").splitlines()[0]
        except Exception as e:
            info[name] = f"missing ({e.__class__.__name__})"
    try:
        http(CFG["comfy_url"] + "/system_stats", timeout=5); info["comfyui"] = "running"
    except Exception:
        info["comfyui"] = "not running"
    k = load_keys()
    info["elevenlabs_key_set"] = bool(k.get("ELEVENLABS_API_KEY"))
    for mod in ("faster_whisper", "whisper"):
        try:
            __import__(mod); info["asr"] = mod; break
        except Exception:
            info["asr"] = "none"
    json.dump(info, open(ctx.out("probe.json"), "w"), indent=1); ctx.add(ctx.out("probe.json"))
    return info


def h_blender(ctx, j):
    script = safe_path(j["script"], must_exist=True)
    cmd = [CFG["blender_exe"], "-b"] + ([safe_path(j["blend"], True)] if j.get("blend") else []) + \
          ["-t", str(j.get("threads", 8)), "--python", script, "--"] + [str(a) for a in j.get("args", [])] + ["--outdir", ctx.dir]
    ctx.run(cmd, cwd=os.path.dirname(script), timeout=j.get("timeout", 4 * 3600))
    for p in j.get("collect", []):                    # glob patterns of outputs to report
        ctx.files += glob.glob(safe_path(p))


def comfy_start_if_needed(ctx):
    try:
        http(CFG["comfy_url"] + "/system_stats", timeout=5); return
    except Exception:
        pass
    if not CFG.get("auto_start_comfyui"):
        raise RuntimeError("ComfyUI is not running")
    ctx.log("starting ComfyUI:", CFG["comfy_start_bat"])
    subprocess.Popen(["cmd", "/c", "start", "", "/min", CFG["comfy_start_bat"]], cwd=os.path.dirname(CFG["comfy_start_bat"]))
    for _ in range(90):
        time.sleep(10)
        try:
            http(CFG["comfy_url"] + "/system_stats", timeout=5); ctx.log("ComfyUI up"); return
        except Exception:
            pass
    raise RuntimeError("ComfyUI did not start within 15 min")


def comfy_submit(ctx, graph, timeout):
    data, _ = http(CFG["comfy_url"] + "/prompt", json.dumps({"prompt": graph}).encode(), {"Content-Type": "application/json"})
    pid = json.loads(data)["prompt_id"]; ctx.log("queued", pid); t0 = time.time()
    while True:
        time.sleep(10)
        h = json.loads(http(CFG["comfy_url"] + "/history/" + pid)[0])
        if pid in h:
            st = h[pid].get("status", {})
            if st.get("completed") or st.get("status_str") in ("success", "error"):
                break
        if time.time() - t0 > timeout:
            raise TimeoutError("ComfyUI job timeout")
    if st.get("status_str") != "success":
        raise RuntimeError("ComfyUI error: " + json.dumps(st.get("messages", [])[-3:])[:3000])
    outs = []
    for node in h[pid].get("outputs", {}).values():
        for key in ("videos", "images", "gifs", "audio"):
            for it in node.get(key, []):
                src = os.path.join(CFG["comfy_dir"], it.get("type", "output"), it.get("subfolder", ""), it["filename"])
                if os.path.exists(src):
                    dst = ctx.out(os.path.basename(src)); shutil.copy2(src, dst); outs.append(dst)
    ctx.files += outs
    return outs


def frames_for(sec):
    n = max(5, round(sec * 24)); return n + (5 - n % 17) % 17


def encode_png_dir(ctx, src_dir, dst, fps=24):
    import av
    from PIL import Image
    files = sorted(glob.glob(os.path.join(src_dir, "*.png")))
    w, h = Image.open(files[0]).size
    c = av.open(dst, "w"); s = c.add_stream("libx264", rate=fps); s.width, s.height = w, h; s.pix_fmt = "yuv420p"; s.options = {"crf": "12"}
    for f in files:
        for p in s.encode(av.VideoFrame.from_image(Image.open(f).convert("RGB"))): c.mux(p)
    for p in s.encode(): c.mux(p)
    c.close(); ctx.log("control video", dst, len(files), "frames")


def h_comfy_h3(ctx, j):
    """MiniMax H3 image-to-video (+ optional Fun ControlNet from a PNG sequence). Same graph as ugc_sharpener\\h3_run.py."""
    comfy_start_if_needed(ctx)
    M = CFG["h3_models"]; W, H = j.get("width", 544), j.get("height", 960); L = frames_for(j.get("seconds", 5.0))
    ff = j["first_frame"]
    if not os.path.exists(os.path.join(CFG["comfy_dir"], "input", ff)) and j.get("first_frame_fallback"):
        ff = j["first_frame_fallback"]
    name = j.get("name", ctx.job["id"])
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": M["unet"], "weight_dtype": "default"}},
        "2": {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": M["lora"], "strength_model": 1.0, "model": ["1", 0]}},
        "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": M["clip"], "type": "minimax", "device": "default"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": M["video_vae"]}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": M["audio_vae"]}},
        "6": {"class_type": "LoadImage", "inputs": {"image": ff}},
        "8": {"class_type": "MiniMaxH3ImageToVideo", "inputs": {"prompt": j["prompt"], "width": W, "height": H, "length": L,
                                                               "clip": ["3", 0], "vae": ["4", 0], "first_frame": ["6", 0]}},
        "9": {"class_type": "BasicScheduler", "inputs": {"scheduler": "simple", "steps": j.get("steps", 8), "denoise": 1.0, "model": ["2", 0]}},
        "10": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "11": {"class_type": "RandomNoise", "inputs": {"noise_seed": j.get("seed", 12345)}},
        "12": {"class_type": "BasicGuider", "inputs": {"model": ["2", 0], "conditioning": ["8", 0]}},
        "13": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["11", 0], "guider": ["12", 0], "sampler": ["10", 0], "sigmas": ["9", 0], "latent_image": ["8", 1]}},
        "14": {"class_type": "VAEDecode", "inputs": {"samples": ["13", 0], "vae": ["4", 0]}},
        "15": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["13", 0], "vae": ["5", 0]}},
        "16": {"class_type": "CreateVideo", "inputs": {"fps": 24.0, "images": ["14", 0], "audio": ["15", 0]}},
        "17": {"class_type": "SaveVideo", "inputs": {"filename_prefix": "video/" + name, "format": "auto", "codec": "auto", "video": ["16", 0]}},
    }
    if j.get("last_frame"):
        g["7"] = {"class_type": "LoadImage", "inputs": {"image": j["last_frame"]}}
        g["8"]["inputs"]["last_frame"] = ["7", 0]
    ctl = j.get("control")
    if ctl:
        vname = ctl.get("video_name", name + "_control.mp4")
        encode_png_dir(ctx, os.path.join(CFG["comfy_dir"], "input", ctl["frames_dir"]), os.path.join(CFG["comfy_dir"], "input", vname))
        g["20"] = {"class_type": "ModelPatchLoader", "inputs": {"name": M["controlnet"]}}
        g["21"] = {"class_type": "LoadVideo", "inputs": {"file": vname}}
        g["22"] = {"class_type": "GetVideoComponents", "inputs": {"video": ["21", 0]}}
        g["23"] = {"class_type": "MiniMaxH3FunControlNetApply", "inputs": {"model": ["2", 0], "model_patch": ["20", 0], "vae": ["4", 0],
                   "strength": ctl.get("strength", 0.8), "start_percent": 0.0, "end_percent": ctl.get("end_percent", 0.7), "control_video": ["22", 0]}}
        g["9"]["inputs"]["model"] = ["23", 0]; g["12"]["inputs"]["model"] = ["23", 0]
    outs = comfy_submit(ctx, g, j.get("timeout", 3600))
    for o in outs:
        if o.lower().endswith(".mp4"):
            os.replace(o, ctx.out(name + ".mp4")); ctx.files[ctx.files.index(o)] = ctx.out(name + ".mp4")


def h_comfy_graph(ctx, j):
    """Submit a ComfyUI API-format graph stored in a JSON file (for other models / workflows)."""
    comfy_start_if_needed(ctx)
    g = json.load(open(safe_path(j["graph"], True), encoding="utf-8"))
    for nid, inputs in j.get("set_inputs", {}).items():
        g[nid]["inputs"].update(inputs)
    comfy_submit(ctx, g, j.get("timeout", 3600))


def _eleven(ctx, j, url, body):
    key = load_keys()["ELEVENLABS_API_KEY"]
    if not key:
        raise RuntimeError("ELEVENLABS_API_KEY is not set in keys.env")
    data, ct = http(url, json.dumps(body).encode(), {"xi-api-key": key, "Content-Type": "application/json", "Accept": "audio/mpeg"}, timeout=900)
    p = ctx.out(j.get("name", "audio") + ".mp3"); open(p, "wb").write(data); ctx.add(p)
    copy_to(ctx, p, j.get("copy_to"))


def h_elevenlabs_sfx(ctx, j):
    body = {"text": j["text"], "prompt_influence": j.get("prompt_influence", 0.5)}
    if j.get("duration_seconds"): body["duration_seconds"] = j["duration_seconds"]
    if j.get("model_id"): body["model_id"] = j["model_id"]
    _eleven(ctx, j, "https://api.elevenlabs.io/v1/sound-generation?output_format=mp3_44100_192", body)


def h_elevenlabs_music(ctx, j):
    body = {"prompt": j["prompt"], "music_length_ms": int(j.get("seconds", 15) * 1000)}
    if j.get("force_instrumental", True): body["force_instrumental"] = True
    if j.get("model_id"): body["model_id"] = j["model_id"]
    _eleven(ctx, j, "https://api.elevenlabs.io/v1/music?output_format=mp3_44100_192", body)


def h_elevenlabs_tts(ctx, j):
    body = {"text": j["text"], "model_id": j.get("model_id", "eleven_multilingual_v2")}
    if j.get("voice_settings"): body["voice_settings"] = j["voice_settings"]
    _eleven(ctx, j, f"https://api.elevenlabs.io/v1/text-to-speech/{j['voice_id']}?output_format=mp3_44100_192", body)


def h_remotion_install(ctx, j):
    proj = safe_path(j["project"], True)
    npm = shutil.which("npm") or "npm"
    ctx.run([npm, "ci" if os.path.exists(os.path.join(proj, "package-lock.json")) else "install"], cwd=proj, timeout=1800)


def h_remotion_render(ctx, j):
    proj = safe_path(j["project"], True)
    npx = shutil.which("npx") or "npx"
    out = ctx.out(j.get("name", "render") + ".mp4")
    cmd = [npx, "remotion", "render", j.get("entry", "src/index.ts"), j["composition"], out]
    if j.get("props"):
        pp = ctx.out("props.json"); json.dump(j["props"], open(pp, "w", encoding="utf-8")); cmd += ["--props", pp]
    for flag in ("codec", "crf", "concurrency", "frames", "scale"):
        if j.get(flag) is not None: cmd += [f"--{flag}", str(j[flag])]
    ctx.run(cmd, cwd=proj, timeout=j.get("timeout", 3 * 3600)); ctx.add(out)
    copy_to(ctx, out, j.get("copy_to"))


def h_transcribe(ctx, j):
    """faster-whisper (same Whisper weights as openai-whisper). GPU float16, falls back to CPU float32 (same accuracy, slower)."""
    src = safe_path(j["file"], True); res = {"segments": []}
    # CUDA 12 cuBLAS / cuDNN from the nvidia-*-cu12 pip packages (ComfyUI's torch ships CUDA 13 DLLs only)
    sp = os.path.join(os.path.dirname(os.path.dirname(sys.executable)), "Lib", "site-packages", "nvidia")
    for sub in ("cublas", "cudnn", "cuda_runtime", "cuda_nvrtc"):
        d = os.path.join(sp, sub, "bin")
        if os.path.isdir(d):
            os.add_dll_directory(d); os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", ""); ctx.log("dll dir", d)
    from faster_whisper import WhisperModel
    name = j.get("model", "large-v3")
    try:
        m = WhisperModel(name, device="cuda", compute_type="float16"); res["device"] = "cuda/float16"
        segs, _ = m.transcribe(src, word_timestamps=True, language=j.get("language", "en"), beam_size=5)
        segs = list(segs)
    except Exception as e:
        ctx.log("GPU transcribe failed, using CPU float32:", e)
        m = WhisperModel(name, device="cpu", compute_type="float32"); res["device"] = "cpu/float32"
        segs, _ = m.transcribe(src, word_timestamps=True, language=j.get("language", "en"), beam_size=5)
        segs = list(segs)
    for s in segs:
        res["segments"].append({"start": s.start, "end": s.end, "text": s.text,
                                "words": [{"start": w.start, "end": w.end, "word": w.word} for w in (s.words or [])]})
    res["model"] = name
    p = ctx.out("transcript.json"); json.dump(res, open(p, "w", encoding="utf-8"), indent=1, ensure_ascii=False); ctx.add(p)


def h_pip_install(ctx, j):
    """Install packages into the runner's own Python. Only names listed in config["pip_allowlist"]. dry_run=true reports what would change."""
    pk = j["packages"]
    bad = [p for p in pk if p.split("==")[0].split(">=")[0].lower() not in [a.lower() for a in CFG.get("pip_allowlist", [])]]
    if bad:
        raise PermissionError(f"not in pip_allowlist: {bad}")
    cmd = [sys.executable, "-m", "pip", "install"] + (["--dry-run", "--report", ctx.out("pip_report.json")] if j.get("dry_run", True) else []) + pk
    ctx.run(cmd, timeout=1800)
    if j.get("dry_run", True):
        ctx.add(ctx.out("pip_report.json"))
        rep = json.load(open(ctx.out("pip_report.json"), encoding="utf-8"))
        from importlib import metadata
        changes = []
        for it in rep.get("install", []):
            n, v = it["metadata"]["name"], it["metadata"]["version"]
            try:
                cur = metadata.version(n)
            except Exception:
                cur = None
            changes.append({"name": n, "new": v, "current": cur})
        return {"changes": changes}


def h_restart(ctx, j):
    """Ask the runner to restart itself after this job (picks up a new runner.py / config.json)."""
    ctx.restart = True
    return {"restarting": True}


def h_copy(ctx, j):
    """Copy files between allowed places (e.g. a staged-in asset into ComfyUI\\input). Never deletes or moves."""
    for src, dst in j["pairs"]:
        s, d = safe_path(src, True), safe_path(dst)
        if os.path.exists(d) and not j.get("overwrite", False):
            raise FileExistsError(d)
        os.makedirs(os.path.dirname(d), exist_ok=True); shutil.copy2(s, d); ctx.add(d)


HANDLERS = {"probe": h_probe, "blender": h_blender, "comfy_h3": h_comfy_h3, "comfy_graph": h_comfy_graph,
            "elevenlabs_sfx": h_elevenlabs_sfx, "elevenlabs_music": h_elevenlabs_music,
            "elevenlabs_tts": h_elevenlabs_tts, "remotion_install": h_remotion_install, "remotion_render": h_remotion_render,
            "transcribe": h_transcribe, "copy": h_copy, "pip_install": h_pip_install, "restart": h_restart}

# ---- v1.4: MV workflow job types live in mv_jobs.py next to this file. The jobs above are unchanged;
# ---- if mv_jobs.py is missing or broken, only the MV jobs are unavailable.
try:
    import mv_jobs
    for _k, _fn in mv_jobs.register(sys.modules[__name__]).items():
        HANDLERS.setdefault(_k, _fn)
except Exception as _e:
    print(f"mv_jobs not loaded ({_e.__class__.__name__}: {_e}); ad-video jobs work as before", flush=True)

# ---- v1.5: GPU check job (RTX 5060 Ti swap) lives in gpu_jobs.py. Read-only; other jobs are unaffected if it is missing.
try:
    import gpu_jobs
    for _k, _fn in gpu_jobs.register(sys.modules[__name__]).items():
        HANDLERS.setdefault(_k, _fn)
except Exception as _e:
    print(f"gpu_jobs not loaded ({_e.__class__.__name__}: {_e})", flush=True)


# ---------------------------------------------------------------- main loop
def heartbeat(current=None):
    st = {"alive_at": time.time(), "alive_at_text": time.strftime("%Y-%m-%d %H:%M:%S"), "runner": VERSION, "current": current,
          "queue": sorted(os.path.basename(p) for p in glob.glob(os.path.join(INBOX, "*.json")))}
    tmp = os.path.join(ROOT, "status.json.tmp"); json.dump(st, open(tmp, "w"), indent=1); os.replace(tmp, os.path.join(ROOT, "status.json"))


def process(path):
    try:
        job = json.load(open(path, encoding="utf-8-sig"))
    except Exception:
        time.sleep(2); job = json.load(open(path, encoding="utf-8-sig"))   # file may still be being written
    job.setdefault("id", os.path.splitext(os.path.basename(path))[0])
    res = {"id": job["id"], "type": job.get("type"), "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    load_cfg()
    t0 = time.time(); ctx = Ctx(job); ctx.restart = False; heartbeat(job["id"])
    try:
        fn = HANDLERS.get(job.get("type"))
        if not fn:
            raise ValueError(f"unknown job type: {job.get('type')} (allowed: {', '.join(HANDLERS)})")
        info = fn(ctx, job)
        res.update(status="success", info=info)
    except Exception as e:
        res.update(status="error", error=f"{e.__class__.__name__}: {e}", trace=traceback.format_exc()[-3000:])
        ctx.log("ERROR", res["error"])
    res.update(secs=int(time.time() - t0), files=sorted(set(ctx.files)), outdir=ctx.dir)
    ctx.logf.close()
    try:
        res["log_tail"] = open(os.path.join(ctx.dir, "log.txt"), encoding="utf-8", errors="replace").read()[-2500:]
    except Exception:
        pass
    tmp = os.path.join(OUTBOX, job["id"] + ".result.json.tmp")
    json.dump(res, open(tmp, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    os.replace(tmp, os.path.join(OUTBOX, job["id"] + ".result.json"))
    shutil.move(path, os.path.join(DONE, os.path.basename(path)))
    print(f"[{job['id']}] {res['status']} in {res['secs']}s", flush=True)
    return ctx.restart


def main():
    print(f"Autopilot runner {VERSION} watching {INBOX}  (create a file named STOP here to quit)", flush=True)
    while not os.path.exists(os.path.join(ROOT, "STOP")):
        heartbeat()
        jobs = sorted(glob.glob(os.path.join(INBOX, "*.json")), key=os.path.getmtime)
        if jobs:
            time.sleep(1)       # let the writer finish
            if process(jobs[0]):
                print("restarting runner ...", flush=True); sys.exit(RESTART_CODE)
        else:
            time.sleep(3)
    os.remove(os.path.join(ROOT, "STOP")); print("stopped")


if __name__ == "__main__":
    main()
