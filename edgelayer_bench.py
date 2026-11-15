"""
EdgeLayer post 8 benchmark helpers.

One eval protocol for every notebook, so every number in the post is comparable:
  * WikiText-2 test perplexity, ctx 2048, non-overlapping windows
  * KL divergence + top-1 agreement vs the FP16 model (same windows)
  * lm-eval zero-shot: arc_easy, hellaswag, piqa, winogrande
  * Speed: vLLM in its own venv (GPU) or llama-bench (GGUF), batch 1,
    512 prompt tokens, 128 generated tokens, median of 5 runs

Every config writes one row to results.jsonl. Rows with the same config_id are
merged by load_results(), so accuracy and speed can be logged separately.

One model (Qwen3-0.6B base) for every method, free GPUs only (Colab / Kaggle T4, CPU).
Part of the EdgeLayer quantization series. License: MIT.
"""
import gc
import glob
import importlib.metadata as md
import json
import math
import os
import platform
import random
import re
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

# --------------------------------------------------------------------------------------
# Config. Override in a notebook with eb.CFG.update(...)
# --------------------------------------------------------------------------------------
CFG = dict(
    model_id="Qwen/Qwen3-0.6B-Base",
    qat_model_id="Qwen/Qwen3-0.6B-Base",      # same model for QAT, so every row is comparable
    results_dir="/content/drive/MyDrive/edgelayer-bench",
    results_file="results.jsonl",
    work_dir="/content",                       # local scratch (set by setup(platform=...))
    models_dir="/content/models",
    seed=0,
    # perplexity / KLD
    ppl_ctx=2048,
    ppl_max_windows=None,                      # None = full WikiText-2 test (~140 windows)
    kld_windows=40,
    # task accuracy
    lm_eval_tasks=["arc_easy", "hellaswag"],
    lm_eval_limit=1000,                        # 1000 examples per task (say so in the post); None = full
    lm_eval_batch=16,
    # calibration
    calib_source="c4",
    calib_n=128,
    calib_len=2048,
    # domain text for the calibration-shift test (code). Calibration and eval docs never overlap.
    domain_dataset=("codeparrot/codeparrot-clean-valid", "content"),
    domain_eval_offset=5000,
    domain_eval_docs=300,
    domain_max_windows=40,
    # speed
    speed_prompt=512,
    speed_gen=128,
    speed_reps=5,
    vllm_env="/content/vllm-env",
)


# --------------------------------------------------------------------------------------
# Setup and environment
# --------------------------------------------------------------------------------------
def set_seed(seed=None):
    seed = CFG["seed"] if seed is None else seed
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


PLATFORM_WORK = {"colab": "/content", "kaggle": "/tmp/work",
                 "lightning": "/teamspace/studios/this_studio/work"}


def setup(results_dir=None, platform="colab", mount_drive=False, **overrides):
    """Set folders for the platform, create them, set the seed. Returns the results dir."""
    if results_dir:
        CFG["results_dir"] = results_dir
    w = PLATFORM_WORK[platform]
    CFG.update(platform=platform, work_dir=w, models_dir=f"{w}/models", vllm_env=f"{w}/vllm-env")
    CFG.update(overrides)
    Path(CFG["work_dir"]).mkdir(parents=True, exist_ok=True)
    if mount_drive:
        try:
            from google.colab import drive  # noqa
            if not os.path.ismount("/content/drive"):
                drive.mount("/content/drive")
        except ImportError:
            pass
    rd = Path(CFG["results_dir"])
    for sub in ["gguf", "logs", "blog_assets"]:
        (rd / sub).mkdir(parents=True, exist_ok=True)
    Path(CFG["models_dir"]).mkdir(parents=True, exist_ok=True)
    set_seed()
    print("results ->", rd / CFG["results_file"])
    return rd


def _ver(pkg):
    for name in (pkg, pkg.replace("_", "-")):
        try:
            return md.version(name)
        except md.PackageNotFoundError:
            continue
    return None


def env_info():
    info = dict(python=platform.python_version(), nproc=os.cpu_count())
    try:
        out = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
        m = re.search(r"Model name:\s*(.+)", out)
        info["cpu"] = m.group(1).strip() if m else None
    except Exception:
        info["cpu"] = None
    try:
        with open("/proc/meminfo") as f:
            kb = int(re.search(r"MemTotal:\s+(\d+)", f.read()).group(1))
        info["ram_gb"] = round(kb / 1e6, 1)
    except Exception:
        pass
    try:
        import torch
        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info.update(gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda,
                        gpu_mem_gb=round(p.total_memory / 1e9, 1), sm=f"{p.major}{p.minor}")
    except ImportError:
        pass
    for pkg in ["transformers", "llmcompressor", "compressed_tensors", "bitsandbytes",
                "peft", "lm_eval", "datasets", "accelerate", "torchao"]:
        v = _ver(pkg)
        if v:
            info[pkg] = v
    return info


def save_meta(**kw):
    p = Path(CFG["results_dir"]) / "meta.json"
    meta = json.loads(p.read_text()) if p.exists() else {}
    meta.update(kw)
    p.write_text(json.dumps(meta, indent=2))
    return meta


def ensure_meta():
    """Param counts for bits/weight. Computed on CPU once per results folder (0.6B loads in seconds)."""
    meta = get_meta()
    if meta.get("model_id") == CFG["model_id"] and meta.get("n_params"):
        return meta
    import torch
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(CFG["model_id"], dtype=torch.float16)
    n, nl = count_params(m)
    del m; gc.collect()
    return save_meta(model_id=CFG["model_id"], n_params=n, n_linear=nl)


def get_meta():
    p = Path(CFG["results_dir"]) / "meta.json"
    return json.loads(p.read_text()) if p.exists() else {}


def reset_llmc():
    """llm-compressor keeps a global session that can hold the last model; clear it between configs."""
    try:
        from llmcompressor.core import active_session
        active_session().reset()
    except Exception as e:
        print("session reset skipped:", e)


def ram_gb():
    try:
        with open("/proc/meminfo") as f:
            t = f.read()
        tot = int(re.search(r"MemTotal:\s+(\d+)", t).group(1)); av = int(re.search(r"MemAvailable:\s+(\d+)", t).group(1))
        return f"RAM used {(tot - av) / 1e6:.1f} / {tot / 1e6:.1f} GB"
    except Exception:
        return "RAM n/a"


def free_gpu():
    """Call after `del model` in the notebook."""
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def gpu_tag():
    """Short prefix for speed fields: t4, l4, a10g, a100, ... (cpu if no GPU)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return "cpu"
        name = torch.cuda.get_device_name(0).lower()
    except ImportError:
        return "cpu"
    for tag in ["l40s", "t4", "l4", "a10g", "a100", "h100", "p100", "v100"]:
        if tag in name.replace("-", "").replace(" ", ""):
            return tag
    return "gpu"


def models_dir():
    return Path(CFG["models_dir"])


def peak_vram_gb():
    import torch
    return round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None


class Timer:
    def __enter__(self):
        self.t = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.s = round(time.perf_counter() - self.t, 1)


# --------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------
def load_fp16(model_id=None, device="cuda"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mid = model_id or CFG["model_id"]
    tok = AutoTokenizer.from_pretrained(mid)
    model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16,
                                                 device_map=device if device else None)
    model.eval()
    return model, tok


def model_device(model):
    return next(model.parameters()).device


def count_params(model):
    """Total and Linear (excl. lm_head) parameter counts. Tied weights counted once."""
    import torch.nn as nn
    total = sum(p.numel() for p in model.parameters())
    linear = sum(m.weight.numel() for n, m in model.named_modules()
                 if isinstance(m, nn.Linear) and "lm_head" not in n)
    return total, linear


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
def _wikitext(split):
    from datasets import load_dataset
    return load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)["text"]


def wikitext_text(split="test"):
    return "\n\n".join(_wikitext(split))


_TOK_CACHE = {}


def wikitext_ids(tok, split="test"):
    key = ("wiki", split, tok.name_or_path)
    if key not in _TOK_CACHE:
        _TOK_CACHE[key] = tok(wikitext_text(split), return_tensors="pt").input_ids
    return _TOK_CACHE[key]


def domain_texts(n, offset=0):
    from datasets import load_dataset
    name, field = CFG["domain_dataset"]
    ds = load_dataset(name, split="train", streaming=True)
    out = []
    for i, ex in enumerate(ds):
        if i < offset:
            continue
        if ex[field] and ex[field].strip():
            out.append(ex[field])
        if len(out) >= n:
            break
    return out


def domain_text():
    return "\n\n".join(domain_texts(CFG["domain_eval_docs"], CFG["domain_eval_offset"]))


def domain_ids(tok):
    key = ("domain", tok.name_or_path)
    if key not in _TOK_CACHE:
        _TOK_CACHE[key] = tok(domain_text(), return_tensors="pt").input_ids
    return _TOK_CACHE[key]


def _source_texts(source, n_docs):
    from datasets import load_dataset
    if source == "c4":
        ds = load_dataset("allenai/c4", data_files="en/c4-train.00001-of-01024.json.gz", split="train")
        return ds["text"][: n_docs]
    if source == "wikitext":
        return [t for t in _wikitext("train") if t.strip()]
    if source == "domain":
        return domain_texts(n_docs, offset=0)       # eval docs start at domain_eval_offset
    raise ValueError(source)


def packed_chunks(tok, source="c4", n=None, seqlen=None, seed=0):
    """Concatenate docs (shuffled) and cut n chunks of seqlen tokens. Returns list[list[int]]."""
    n = n or CFG["calib_n"]
    seqlen = seqlen or CFG["calib_len"]
    texts = list(_source_texts(source, n_docs=max(2000, n * 20)))
    random.Random(seed).shuffle(texts)
    ids = []
    for t in texts:
        if not t.strip():
            continue
        ids.extend(tok(t).input_ids + [tok.eos_token_id])
        if len(ids) >= n * seqlen:
            break
    if len(ids) < n * seqlen:
        raise RuntimeError(f"not enough {source} text for {n} x {seqlen}")
    return [ids[i * seqlen:(i + 1) * seqlen] for i in range(n)]


def calib_dataset(tok, source=None, n=None, seqlen=None, seed=0):
    """Tokenized calibration set for llm-compressor oneshot()."""
    from datasets import Dataset
    chunks = packed_chunks(tok, source or CFG["calib_source"], n, seqlen, seed)
    return Dataset.from_dict({"input_ids": chunks, "attention_mask": [[1] * len(c) for c in chunks]})


def write_text_file(text, path):
    Path(path).write_text(text)
    return str(path)


# --------------------------------------------------------------------------------------
# Quality metrics
# --------------------------------------------------------------------------------------
def perplexity(model, ids, ctx=None, max_windows=None):
    import torch
    import torch.nn.functional as F
    ctx = ctx or CFG["ppl_ctx"]
    n = ids.size(1) // ctx
    if max_windows:
        n = min(n, max_windows)
    dev = model_device(model)
    nll, count = 0.0, 0
    with torch.no_grad():
        for i in range(n):
            x = ids[:, i * ctx:(i + 1) * ctx].to(dev)
            logits = model(x).logits[0, :-1].float()
            nll += F.cross_entropy(logits, x[0, 1:], reduction="sum").item()
            count += ctx - 1
    ppl = math.exp(nll / count)
    if not math.isfinite(ppl):
        print("WARNING: non-finite perplexity (fp16 overflow?)")
    return round(ppl, 4)


def kld_vs_ref(model, ref, ids, ctx=None, windows=None, chunk=512):
    """Mean KL(P_fp16 || P_quant) per token and top-1 agreement (%), on the first windows."""
    import torch
    import torch.nn.functional as F
    ctx = ctx or CFG["ppl_ctx"]
    windows = windows or CFG["kld_windows"]
    n = min(ids.size(1) // ctx, windows)
    dm, dr = model_device(model), model_device(ref)
    kl_sum, same, tokens = 0.0, 0, 0
    with torch.no_grad():
        for i in range(n):
            x = ids[:, i * ctx:(i + 1) * ctx]
            lq = model(x.to(dm)).logits[0]
            lr = ref(x.to(dr)).logits[0].to(lq.device)
            for s in range(0, ctx, chunk):
                q = F.log_softmax(lq[s:s + chunk].float(), -1)
                p = F.log_softmax(lr[s:s + chunk].float(), -1)
                kl_sum += (p.exp() * (p - q)).sum().item()
                same += (p.argmax(-1) == q.argmax(-1)).sum().item()
                tokens += q.size(0)
            del lq, lr
    return round(kl_sum / tokens, 5), round(100 * same / tokens, 2)


_PRIMARY = ["acc_norm,none", "acc,none", "exact_match,strict-match", "exact_match,flexible-extract"]


def lm_eval_scores(model, tok, tasks=None, limit="cfg", num_fewshot=0, batch_size=None):
    """Returns {task: score%, ..., 'avg': mean}. acc_norm where available, else acc / exact_match."""
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    tasks = tasks or CFG["lm_eval_tasks"]
    limit = CFG["lm_eval_limit"] if limit == "cfg" else limit
    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=batch_size or CFG["lm_eval_batch"])
    out = lm_eval.simple_evaluate(model=lm, tasks=tasks, limit=limit, num_fewshot=num_fewshot,
                                  random_seed=CFG["seed"], numpy_random_seed=CFG["seed"],
                                  torch_random_seed=CFG["seed"], log_samples=False)
    res = {}
    for t, v in out["results"].items():
        key = next((k for k in _PRIMARY if k in v), None)
        if key is None:
            continue
        res[t] = round(100 * v[key], 2)
        if "exact_match,flexible-extract" in v:
            res[t + "_flex"] = round(100 * v["exact_match,flexible-extract"], 2)
    main = [res[t] for t in tasks if t in res]
    res["avg"] = round(sum(main) / len(main), 2) if main else None
    return res


def evaluate(model, tok, ref=None, tasks="cfg", domain=True, ppl_windows="cfg"):
    """The standard protocol. Returns a dict ready for log_result(**...)."""
    ppl_windows = CFG["ppl_max_windows"] if ppl_windows == "cfg" else ppl_windows
    out = {}
    with Timer() as t:
        out["ppl_wiki"] = perplexity(model, wikitext_ids(tok), max_windows=ppl_windows)
        if domain:
            out["ppl_code"] = perplexity(model, domain_ids(tok), max_windows=CFG["domain_max_windows"])
        if ref is not None:
            out["kld"], out["top1_agree"] = kld_vs_ref(model, ref, wikitext_ids(tok))
        if tasks:
            out["tasks"] = lm_eval_scores(model, tok, tasks=None if tasks == "cfg" else tasks)
            out["avg_acc"] = out["tasks"].get("avg")
    out["eval_time_s"] = t.s
    print({k: v for k, v in out.items() if k != "tasks"}, out.get("tasks"))
    return out


# --------------------------------------------------------------------------------------
# Size and bits per weight
# --------------------------------------------------------------------------------------
def dir_size_gb(path):
    files = [f for ext in ("*.safetensors", "*.bin", "*.gguf") for f in glob.glob(f"{path}/**/{ext}", recursive=True)]
    return round(sum(os.path.getsize(f) for f in files) / 1e9, 3)


def file_size_gb(path):
    return round(os.path.getsize(path) / 1e9, 3)


def bpw(size_gb, n_params=None):
    n_params = n_params or get_meta().get("n_params")
    return round(size_gb * 1e9 * 8 / n_params, 3) if n_params else None


def analytic_size_gb(model, bits_linear, skip=("lm_head",)):
    """Size if every Linear (except skip) is stored at bits_linear and the rest in FP16."""
    import torch.nn as nn
    total, seen = 0.0, set()
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and not any(s in name for s in skip):
            total += m.weight.numel() * bits_linear / 8
            seen.add(id(m.weight))
    for p in model.parameters():
        if id(p) not in seen:
            total += p.numel() * 2
    return round(total / 1e9, 3)


# --------------------------------------------------------------------------------------
# Quantizers (for simulation and QAT). All work on a 2-D weight [out, in], grouped along in.
# --------------------------------------------------------------------------------------
_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
_E2M1_MID = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]


def round_e2m1(x):
    """Round to the nearest FP4 E2M1 value (sign kept), saturating at +-6."""
    import torch
    grid = torch.tensor(_E2M1, device=x.device, dtype=x.dtype)
    mids = torch.tensor(_E2M1_MID, device=x.device, dtype=x.dtype)
    a = x.abs().clamp(max=6.0)
    return torch.sign(x) * grid[torch.bucketize(a, mids)]


def mxfp4_qdq(w, block=32):
    """OCP MXFP4: E2M1 elements, one power-of-two (E8M0) scale per 32 values. 4.25 bits/weight."""
    import torch
    shp, dt = w.shape, w.dtype
    wf = w.float().reshape(-1, block)
    amax = wf.abs().amax(-1, keepdim=True).clamp(min=2.0 ** -126)
    scale = torch.exp2(torch.floor(torch.log2(amax)) - 2)      # 2 = emax of E2M1
    return (round_e2m1(wf / scale) * scale).reshape(shp).to(dt)


def _to_e4m3(x):
    import torch
    try:
        return x.to(torch.float8_e4m3fn).to(x.dtype)
    except Exception:                                           # cast on CPU if the GPU kernel is missing
        return x.cpu().to(torch.float8_e4m3fn).to(x.dtype).to(x.device)


def nvfp4_qdq(w, block=16):
    """NVFP4: E2M1 elements, FP8 E4M3 scale per 16 values, plus one FP32 scale per tensor. ~4.5 bits/weight."""
    shp, dt = w.shape, w.dtype
    wf = w.float().reshape(-1, block)
    s_t = wf.abs().amax().clamp(min=1e-12) / (6.0 * 448.0)
    s_b = _to_e4m3((wf.abs().amax(-1, keepdim=True) / 6.0) / s_t)
    eff = (s_b * s_t).clamp(min=1e-12)
    return (round_e2m1(wf / eff) * eff).reshape(shp).to(dt)


def intn_group_qdq(w, bits=4, group=128, symmetric=True):
    """Round-to-nearest INT-n with one scale per group. Returns dequantized weights."""
    import torch
    shp, dt = w.shape, w.dtype
    wg = w.float().reshape(-1, group)
    if symmetric:
        qmax = 2 ** (bits - 1) - 1
        scale = wg.abs().amax(-1, keepdim=True).clamp(min=1e-8) / qmax
        q = torch.clamp(torch.round(wg / scale), -qmax - 1, qmax)
        out = q * scale
    else:
        qmax = 2 ** bits - 1
        lo, hi = wg.amin(-1, keepdim=True), wg.amax(-1, keepdim=True)
        scale = ((hi - lo) / qmax).clamp(min=1e-8)
        zp = torch.round(-lo / scale)
        q = torch.clamp(torch.round(wg / scale) + zp, 0, qmax)
        out = (q - zp) * scale
    return out.reshape(shp).to(dt)


def intn_codes(w, bits=4, group=128):
    """Integer codes of symmetric INT-n (used to count weights that change tick after QAT)."""
    import torch
    wg = w.float().reshape(-1, group)
    qmax = 2 ** (bits - 1) - 1
    scale = wg.abs().amax(-1, keepdim=True).clamp(min=1e-8) / qmax
    return torch.clamp(torch.round(wg / scale), -qmax - 1, qmax).to(torch.int8)


def linear_layers(model, skip=("lm_head",)):
    import torch.nn as nn
    return [(n, m) for n, m in model.named_modules()
            if isinstance(m, nn.Linear) and not any(s in n for s in skip)]


def apply_weight_qdq(model, fn, skip=("lm_head",), **kw):
    """Replace every Linear weight (except skip) with fn(weight). Simulated quantization."""
    import torch
    with torch.no_grad():
        for _, m in linear_layers(model, skip):
            m.weight.copy_(fn(m.weight.data, **kw))
    return model


# ---- QAT: fake quant with a straight-through estimator --------------------------------
def _ste_module(bits, group):
    import torch.nn as nn

    class STEFakeQuant(nn.Module):
        def forward(self, w):
            return w + (intn_group_qdq(w, bits, group) - w).detach()
    return STEFakeQuant()


def add_fake_quant(model, bits=4, group=128, skip=("lm_head",)):
    import torch.nn.utils.parametrize as P
    for _, m in linear_layers(model, skip):
        P.register_parametrization(m, "weight", _ste_module(bits, group))
    return model


def remove_fake_quant(model, skip=("lm_head",)):
    """Drop the fake-quant wrapper and keep the latent (full-precision) weights."""
    import torch.nn.utils.parametrize as P
    for _, m in linear_layers(model, skip):
        if P.is_parametrized(m, "weight"):
            P.remove_parametrizations(m, "weight", leave_parametrized=False)
    return model


def train_lm(model, chunks, steps, lr=1e-5, micro_bs=4, accum=2, eval_fn=None, eval_every=None,
             warmup=20, log_every=25):
    """Tiny causal-LM training loop: fp16 autocast + GradScaler + 8-bit AdamW + grad checkpointing.
    chunks: list of token-id lists (all the same length). Returns time, peak VRAM, loss and eval curves."""
    import torch
    import bitsandbytes as bnb
    dev = model_device(model)
    model.train()
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    opt = bnb.optim.AdamW8bit([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warmup))
    scaler = torch.amp.GradScaler("cuda")
    data = torch.tensor(chunks)
    order = torch.randperm(len(data), generator=torch.Generator().manual_seed(CFG["seed"]))
    curve, losses, ptr = [], [], 0
    torch.cuda.reset_peak_memory_stats()
    with Timer() as t:
        if eval_fn:
            model.eval(); curve.append((0, eval_fn(model))); model.train()
        for step in range(1, steps + 1):
            for _ in range(accum):
                idx = order[ptr % len(order): ptr % len(order) + micro_bs]
                ptr += micro_bs
                x = data[idx].to(dev)
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = model(input_ids=x, labels=x).loss / accum
                scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
            losses.append(loss.item() * accum)
            if step % log_every == 0:
                print(f"step {step}/{steps}  loss {statistics.mean(losses[-log_every:]):.4f}")
            if eval_fn and eval_every and step % eval_every == 0:
                model.eval(); curve.append((step, eval_fn(model))); model.train()
    model.eval()
    model.config.use_cache = True
    return dict(train_time_s=t.s, train_peak_vram_gb=peak_vram_gb(), loss_curve=losses, eval_curve=curve)


# --------------------------------------------------------------------------------------
# llm-compressor recipe helpers
# --------------------------------------------------------------------------------------
def w_scheme(bits=4, group=128, symmetric=True):
    """config_groups for weight-only INT-n, group-wise (W{bits}A16)."""
    return {"group_0": {"targets": ["Linear"],
                        "weights": {"num_bits": bits, "type": "int", "symmetric": symmetric,
                                    "strategy": "group", "group_size": group}}}


# --------------------------------------------------------------------------------------
# Speed: vLLM in its own venv (keeps vLLM's torch away from llm-compressor's)
# --------------------------------------------------------------------------------------
_VLLM_SPEED = r'''
import json, sys, time, statistics
from vllm import LLM, SamplingParams
cfg = json.loads(sys.argv[1])
kw = dict(model=cfg["model"], dtype="half", max_model_len=cfg["prompt"] + cfg["gen"] + 64,
          gpu_memory_utilization=0.85, enable_prefix_caching=False, seed=0)
if cfg.get("quantization"):
    kw["quantization"] = cfg["quantization"]
llm = LLM(**kw)
prompt = {"prompt_token_ids": [100 + (i * 7) % 1000 for i in range(cfg["prompt"])]}
one = SamplingParams(max_tokens=1, temperature=0, ignore_eos=True)
full = SamplingParams(max_tokens=cfg["gen"], temperature=0, ignore_eos=True)
llm.generate([prompt], full, use_tqdm=False)                     # warm-up
pre, tot = [], []
for _ in range(cfg["reps"]):
    t = time.perf_counter(); llm.generate([prompt], one, use_tqdm=False); pre.append(time.perf_counter() - t)
    t = time.perf_counter(); llm.generate([prompt], full, use_tqdm=False); tot.append(time.perf_counter() - t)
tp, tt = statistics.median(pre), statistics.median(tot)
print("RESULT_JSON=" + json.dumps(dict(prefill_tok_s=round(cfg["prompt"] / tp, 1),
      decode_tok_s=round((cfg["gen"] - 1) / (tt - tp), 1), ttft_s=round(tp, 4), e2e_s=round(tt, 3))))
'''


def ensure_vllm_env(path=None):
    path = path or CFG["vllm_env"]
    py = f"{path}/bin/python"
    if not os.path.exists(py):
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
        subprocess.run(["uv", "venv", path], check=True)
        subprocess.run(["uv", "pip", "install", "--python", py, "vllm"], check=True)
    return py


def vllm_speed(model_path, quantization=None, timeout=2400):
    """Batch-1 prefill/decode tok/s. Free the GPU in the notebook before calling this."""
    py = ensure_vllm_env()
    script = Path("/tmp/el_vllm_speed.py")
    script.write_text(_VLLM_SPEED)
    cfg = dict(model=str(model_path), quantization=quantization, prompt=CFG["speed_prompt"],
               gen=CFG["speed_gen"], reps=CFG["speed_reps"])
    r = subprocess.run([py, str(script), json.dumps(cfg)], capture_output=True, text=True, timeout=timeout)
    m = re.search(r"RESULT_JSON=(\{.*\})", r.stdout)
    if not m:
        print("vLLM run failed:\n", r.stdout[-2000:], "\n", r.stderr[-4000:])
        return None
    res = json.loads(m.group(1))
    mm = re.search(r"Model loading took ([\d.]+) ?GiB", r.stdout + r.stderr)
    res["weights_gib"] = float(mm.group(1)) if mm else None
    return res


def prefixed(d, prefix):
    return {f"{prefix}_{k}": v for k, v in (d or {}).items()}


# --------------------------------------------------------------------------------------
# llama.cpp
# --------------------------------------------------------------------------------------
LLAMA_TARGETS = ["llama-quantize", "llama-imatrix", "llama-perplexity", "llama-bench"]


def sh(cmd, check=True, log=None):
    """Run a shell command, stream nothing, return combined output."""
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    out = r.stdout + r.stderr
    if log:
        Path(log).write_text(out)
    if check and r.returncode != 0:
        print(out[-4000:])
        raise RuntimeError(f"failed: {cmd}")
    return out


def build_llamacpp(path=None, cuda=True, cuda_arch="75", build_dir=None):
    """CUDA build -> <path>/build-cuda/bin, CPU-only build -> <path>/build-cpu/bin."""
    path = path or f"{CFG['work_dir']}/llama.cpp"
    build_dir = build_dir or ("build-cuda" if cuda else "build-cpu")
    if not os.path.exists(path):
        sh(f"git clone --depth 1 https://github.com/ggml-org/llama.cpp {path}")
    flags = f"-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES={cuda_arch}" if cuda else "-DGGML_CUDA=OFF"
    sh(f"cmake -S {path} -B {path}/{build_dir} {flags} -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release")
    sh(f"cmake --build {path}/{build_dir} --config Release -j {os.cpu_count()} --target {' '.join(LLAMA_TARGETS)}")
    print(sh(f"git -C {path} log -1 --format='llama.cpp %h %cd'"))
    return f"{path}/{build_dir}/bin"


def parse_ppl(text):
    m = re.search(r"Final estimate:\s*PPL\s*=\s*([\d.]+)\s*\+/-\s*([\d.]+)", text)
    if m:
        return float(m.group(1)), float(m.group(2))
    m = re.search(r"Mean PPL\(Q\)\s*:\s*([\d.]+)\s*±\s*([\d.]+)", text)
    return (float(m.group(1)), float(m.group(2))) if m else (None, None)


def parse_kld(text):
    k = re.search(r"Mean\s+KLD:\s*([\d.]+)\s*±\s*([\d.]+)", text)
    s = re.search(r"Same top p:\s*([\d.]+)\s*±\s*([\d.]+)\s*%", text)
    return (float(k.group(1)) if k else None, float(s.group(1)) if s else None)


def parse_llama_bench(json_text):
    """Returns dict(prefill_tok_s, decode_tok_s, threads, ...) from `llama-bench -o json`."""
    start = json_text.find("[")
    rows = json.loads(json_text[start:json_text.rfind("]") + 1])
    out = {}
    for r in rows:
        if r.get("n_prompt", 0) > 0 and r.get("n_gen", 0) == 0:
            out["prefill_tok_s"] = round(r["avg_ts"], 2); out["prefill_sd"] = round(r.get("stddev_ts", 0), 2)
        elif r.get("n_gen", 0) > 0 and r.get("n_prompt", 0) == 0:
            out["decode_tok_s"] = round(r["avg_ts"], 2); out["decode_sd"] = round(r.get("stddev_ts", 0), 2)
        out["threads"] = r.get("n_threads")
        out["backend"] = r.get("backends") or r.get("gpu_info")
    return out


def cpu_bandwidth_gbs(size_mb=512, reps=5, threads=None):
    """Rough multi-thread memory copy bandwidth (GB/s, read+write counted). Upper-bound helper for tok/s."""
    import numpy as np
    threads = threads or os.cpu_count()
    n = size_mb * 2 ** 20 // 8
    src = [np.ones(n) for _ in range(threads)]
    dst = [np.empty(n) for _ in range(threads)]
    best = 0.0
    for _ in range(reps):
        ts = [threading.Thread(target=np.copyto, args=(dst[i], src[i])) for i in range(threads)]
        t = time.perf_counter()
        [x.start() for x in ts]; [x.join() for x in ts]
        dt = time.perf_counter() - t
        best = max(best, 2 * threads * src[0].nbytes / dt / 1e9)
    return round(best, 1)


# --------------------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------------------
def log_result(config_id, **fields):
    """Append one row. Same config_id rows are merged later (last non-null value wins)."""
    row = dict(config_id=config_id, time=time.strftime("%Y-%m-%d %H:%M:%S"),
               model_id=fields.pop("model_id", CFG["model_id"]), **fields, env=env_info())
    path = Path(CFG["results_dir"]) / CFG["results_file"]
    with open(path, "a") as f:
        f.write(json.dumps(row, default=str) + "\n")
    short = {k: v for k, v in row.items() if k not in ("env", "tasks", "loss_curve", "eval_curve")}
    print("logged:", short)
    return row


def load_results(results_dir=None):
    """All results*.jsonl in the folder -> one DataFrame, one row per (model_id, config_id)."""
    import pandas as pd
    rd = Path(results_dir or CFG["results_dir"])
    rows = []
    for p in sorted(rd.glob("results*.jsonl")):
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                for k, v in (r.pop("tasks", None) or {}).items():
                    r[f"acc_{k}"] = v
                env = r.pop("env", {}) or {}
                r["hw"] = env.get("gpu") or env.get("cpu")
                rows.append(r)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.groupby(["model_id", "config_id"], sort=False).agg("last").reset_index()
    return df
