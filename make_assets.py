"""Builds every figure (figures/) and table (tables/ as .md, .csv and .png) from results/*.jsonl.

Run from the repo root:  python make_assets.py
License: MIT (code). The generated figures and tables are CC BY 4.0.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import edgelayer_bench as eb

OUT = HERE / "figures"
TAB = HERE / "tables"
OUT.mkdir(exist_ok=True); TAB.mkdir(exist_ok=True)
df = eb.load_results(HERE / "results")
m = df[df.model_id == "Qwen/Qwen3-0.6B-Base"].set_index("config_id")


def dppl(c, base):
    return 100 * (m[c] / m.loc[base, c] - 1)


m["dppl"] = dppl("ppl_wiki", "fp16").fillna(dppl("ppl_wiki_lcpp", "gguf_f16"))

C = ["#2a78d6", "#eb6834", "#1baf7a", "#e87ba4"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({"axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "axes.axisbelow": True,
                     "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def finish(fig, ax, title, name, ncol=4, legend="top"):
    n = len(ax.get_legend_handles_labels()[0])
    rows = -(-n // ncol) if (n and legend == "top") else 1
    ax.set_title(title, color=INK, loc="left", pad=12 + 15 * rows)
    if n and legend == "inside":
        ax.legend(frameon=False, loc="lower right", fontsize=8.5)
    elif n:
        ax.legend(frameon=False, ncol=ncol, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=9, borderaxespad=0.2)
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)


def label_bars(ax, bars, fmt="{:.0f}%"):
    for b in bars:
        h = b.get_height()
        if np.isfinite(h):
            ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h), ha="center", va="bottom",
                        fontsize=8, color=INK, xytext=(0, 2), textcoords="offset points")


def v(cid, col="dppl"):
    return m.loc[cid, col] if cid in m.index and col in m and pd.notna(m.loc[cid, col]) else np.nan


# F1 · size vs quality ------------------------------------------------------------------
LABEL = {"rtn_w4g128": ("RTN 4b", (4, 4)), "gptq_w4g128": ("GPTQ 4b", (6, 6)), "nf4_dq": ("NF4", (6, -4)),
         "awq_w4g128": ("AWQ 4b", (6, -10)), "rtn_w3g128": ("RTN 3b", (5, 2)), "gptq_w3g128": ("GPTQ 3b", (6, 4)),
         "awq_w3g128": ("AWQ 3b", (6, -10)), "gguf_q8_0": ("Q8_0", (5, 2)), "gguf_q5_k_m": ("Q5_K_M", (5, 2)),
         "gguf_q4_k_m": ("Q4_K_M", (5, -10)), "gguf_q4_0": ("Q4_0", (5, 4)), "gguf_iq4_xs": ("IQ4_XS", (-8, 8)),
         "gguf_q3_k_m": ("Q3_K_M", (5, 2))}
pts = m[m.index.isin(LABEL) & m.dppl.notna()]
fig, ax = plt.subplots(figsize=(7.2, 4.4))
for k, (lab, mask) in enumerate([("vLLM / HF formats (embeddings kept FP16)", ~pts.index.str.startswith("gguf")),
                                 ("GGUF (embeddings quantized too)", pts.index.str.startswith("gguf"))]):
    s = pts[mask]
    ax.scatter(s.size_gb, s.dppl, s=46, color=C[k], label=lab, edgecolor="white", linewidth=1.5, zorder=3)
for i, r in pts.iterrows():
    ax.annotate(LABEL[i][0], (r.size_gb, r.dppl), fontsize=7.5, color=MUTED, xytext=LABEL[i][1], textcoords="offset points")
ax.set_yscale("log"); ax.grid(True, axis="both")
ax.set_xlabel("File size (GB)  ·  FP16 = 1.19 GB"); ax.set_ylabel("Δ perplexity vs own FP16 (%, log)")
finish(fig, ax, "F1 · Size vs quality, Qwen3-0.6B (lower-left is better)", "F1_size_vs_quality", legend="inside")

# F2 · RTN vs GPTQ vs AWQ ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 3.8)); w = 0.38
for k, b in enumerate([4, 3]):
    vals = [v(f"{a}_w{b}g128") for a in ("rtn", "gptq", "awq")]
    label_bars(ax, ax.bar(np.arange(3) + (k - 0.5) * w, vals, w * 0.92, color=C[k], label=f"{b}-bit"))
ax.set_yscale("log"); ax.set_xticks(range(3), ["RTN (plain rounding)", "GPTQ", "AWQ"])
ax.set_ylabel("Δ perplexity vs FP16 (%, log)")
finish(fig, ax, "F2 · Plain rounding vs GPTQ vs AWQ", "F2_ptq_methods")

# F3 · calibration shift ----------------------------------------------------------------
ids = ["gptq_w4g128", "gptq_w4g128_code", "awq_w4g128"]
names = ["GPTQ · C4 calib", "GPTQ · code calib", "AWQ · C4 calib", ]
fig, ax = plt.subplots(figsize=(7.2, 3.8)); w = 0.38
for k, (col, lab) in enumerate([("ppl_wiki", "Eval on WikiText"), ("ppl_code", "Eval on code")]):
    vals = [100 * (v(i, col) / v("fp16", col) - 1) for i in ids]
    label_bars(ax, ax.bar(np.arange(len(ids)) + (k - 0.5) * w, vals, w * 0.92, color=C[k], label=lab))
ax.set_xticks(range(len(ids)), names); ax.set_ylabel("Δ perplexity vs FP16 (%)")
finish(fig, ax, "F3 · Calibration shift at 4-bit", "F3_calibration_shift")

# F4 · QAT vs PTQ -----------------------------------------------------------------------
rows = [("ptq_rtn_w{b}", "RTN"), ("ft_then_rtn_w{b}", "Fine-tune → RTN (control)"),
        ("gptq_w{b}g128", "GPTQ"), ("awq_w{b}g128", "AWQ"), ("qat_w{b}g128", "QAT → RTN")]
COLS = ["#a3a29d", "#5f5e5a", C[0], C[2], C[1]]
fig, ax = plt.subplots(figsize=(7.4, 4.0)); w = 0.16
for k, (pat, lab) in enumerate(rows):
    vals = [v(pat.format(b=b)) for b in (4, 3)]
    label_bars(ax, ax.bar(np.arange(2) + (k - 2) * w, vals, w * 0.92, color=COLS[k], label=lab))
ax.set_yscale("log"); ax.set_xticks(range(2), ["4-bit", "3-bit"]); ax.set_ylabel("Δ perplexity vs FP16 (%, log)")
finish(fig, ax, "F4 · QAT vs PTQ (same model, same quantizer for RTN / control / QAT)", "F4_qat_vs_ptq", ncol=3)

# F5 · QAT recovery curve ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 3.8))
for k, b in enumerate((4, 3)):
    xs, ys = zip(*m.loc[f"qat_w{b}g128", "eval_curve"])
    ax.plot(xs, ys, color=C[k], lw=2, marker="o", ms=5, label=f"{b}-bit", zorder=3)
    ax.annotate(f"{ys[0]:.0f}", (xs[0], ys[0]), fontsize=8, color=INK, xytext=(6, 0), textcoords="offset points")
    ax.annotate(f"{ys[-1]:.1f}", (xs[-1], ys[-1]), fontsize=8, color=INK, xytext=(-6, 7), textcoords="offset points")
ax.axhline(12.66, color=MUTED, lw=1, ls="--", label="FP16 = 12.7")
ax.set_yscale("log"); ax.grid(True, axis="both")
ax.set_xlabel("Training step (1 step = 8 × 1,024 tokens)"); ax.set_ylabel("WikiText ppl, fake quant on (log)")
finish(fig, ax, "F5 · QAT recovery: most of it happens in the first 100 steps", "F5_qat_curve")

# F6 · GGUF imatrix ---------------------------------------------------------------------
qs = ["q5_k_m", "q4_k_m", "q4_0", "iq4_xs", "q3_k_m"]
fig, ax = plt.subplots(figsize=(7.2, 3.8)); w = 0.27
for k, (suf, lab) in enumerate([("", "no imatrix"), ("_imat_wiki", "imatrix: WikiText"), ("_imat_code", "imatrix: code")]):
    vals = [v(f"gguf_{q}{suf}", "kld_lcpp") for q in qs]
    ax.bar(np.arange(len(qs)) + (k - 1) * w, vals, w * 0.92, color=C[k], label=lab)
ax.set_xticks(range(len(qs)), [q.upper() for q in qs]); ax.set_ylabel("KL divergence vs F16 (lower is better)")
finish(fig, ax, "F6 · GGUF quants: what the imatrix buys", "F6_gguf_imatrix")

# F7 · CPU speed vs ceiling -------------------------------------------------------------
gg = ["gguf_f16", "gguf_q8_0", "gguf_q5_k_m", "gguf_q4_k_m", "gguf_q4_0", "gguf_iq4_xs", "gguf_q3_k_m"]
names = ["F16", "Q8_0", "Q5_K_M", "Q4_K_M", "Q4_0", "IQ4_XS", "Q3_K_M"]
fig, ax = plt.subplots(figsize=(7.4, 3.9)); w = 0.38
meas = [v(i, "cpu_decode_tok_s") for i in gg]; ceil = [v(i, "cpu_ceiling_tok_s") for i in gg]
b1 = ax.bar(np.arange(len(gg)) - w / 2, meas, w * 0.92, color=C[0], label="Measured decode")
ax.bar(np.arange(len(gg)) + w / 2, ceil, w * 0.92, color="#c9c8c3", label="Ceiling = bandwidth ÷ size")
label_bars(ax, b1, "{:.0f}")
for i, (a, c) in enumerate(zip(meas, ceil)):
    ax.annotate(f"{100 * a / c:.0f}%", (i, 0), ha="center", va="bottom", fontsize=7.5, color="white",
                xytext=(-w * 30, 4), textcoords="offset points")
ax.set_xticks(range(len(gg)), names); ax.set_ylabel("Tokens / second (4 vCPU)")
finish(fig, ax, "F7 · CPU decode vs the bandwidth ceiling (% = share of ceiling reached)", "F7_cpu_speed_vs_ceiling")

# F8 · T4 GPU speed ---------------------------------------------------------------------
ids = ["fp16", "gptq_w4g128", "awq_w4g128"]; names = ["FP16", "GPTQ 4-bit", "AWQ 4-bit"]
fig, ax = plt.subplots(figsize=(6.2, 3.6))
vals = [v(i, "t4_decode_tok_s") for i in ids]
bars = ax.bar(range(3), vals, 0.55, color=[ "#a3a29d", C[0], C[2]])
for i, b in enumerate(bars):
    ax.annotate(f"{vals[i]:.0f} tok/s  ({vals[i] / vals[0]:.2f}×)", (b.get_x() + b.get_width() / 2, vals[i]),
                ha="center", va="bottom", fontsize=8.5, color=INK, xytext=(0, 2), textcoords="offset points")
ax.set_xticks(range(3), names); ax.set_ylabel("Decode tokens / second")
finish(fig, ax, "F8 · T4 decode speed (vLLM, batch 1; prefill ~6.7k tok/s for all)", "F8_t4_decode_speed")


# ======================================================================================
# Tables: .md (Substack / GitHub), .csv (data), .png (image for the post)
# ======================================================================================
def pct(x, d=0):
    return "–" if not np.isfinite(x) else (f"+{x:,.{d}f}%" if x >= 0 else f"{x:,.{d}f}%")


def num(x, fmt):
    return "–" if x is None or not np.isfinite(x) else fmt.format(x)


def save_table(rows, cols, name, title, bold_rows=()):
    t = pd.DataFrame(rows, columns=cols)
    t.to_csv(TAB / f"{name}.csv", index=False)
    (TAB / f"{name}.md").write_text(t.to_markdown(index=False))
    n, k = t.shape
    chars = [max(len(str(c)), *(len(str(x)) for x in t.iloc[:, j])) for j, c in enumerate(cols)]
    widths = [0.085 * c + 0.25 for c in chars]                      # inches per column
    W, H = sum(widths) + 0.2, 0.33 * (n + 1) + 0.55
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0.1 / W, 0.05 / H, 1 - 0.2 / W, 1 - 0.55 / H]); ax.axis("off")
    tb = ax.table(cellText=t.values, colLabels=cols, loc="upper left", cellLoc="left", colLoc="left",
                  colWidths=[w / sum(widths) for w in widths], bbox=[0, 0, 1, 1])
    tb.auto_set_font_size(False); tb.set_fontsize(9)
    for (r, c), cell in tb.get_celld().items():
        cell.set_edgecolor(GRID); cell.set_linewidth(0.6)
        cell.visible_edges = "B"
        if r == 0:
            cell.set_text_props(color=INK, weight="bold"); cell.set_facecolor("#f4f3ef"); cell.visible_edges = "TB"
        else:
            cell.set_text_props(color=INK, weight="bold" if (r - 1) in bold_rows else "normal")
    fig.text(0.1 / W, 1 - 0.12 / H, title, ha="left", va="top", color=INK, fontsize=11)
    fig.savefig(TAB / f"{name}.png", dpi=200)
    plt.close(fig)


fp = m.loc["fp16"]
# T1 · PTQ methods
rows = []
for cid, label in [("fp16", "FP16"), ("rtn_w4g128", "RTN 4-bit"), ("gptq_w4g128", "GPTQ 4-bit"),
                   ("awq_w4g128", "AWQ 4-bit"), ("nf4_dq", "NF4 + DQ"), ("rtn_w3g128", "RTN 3-bit"),
                   ("gptq_w3g128", "GPTQ 3-bit"), ("awq_w3g128", "AWQ 3-bit")]:
    r = m.loc[cid]
    t4 = v(cid, "t4_decode_tok_s")
    rows.append([label, num(r.size_gb, "{:.2f} GB"), "–" if cid == "fp16" else pct(r.dppl),
                 num(r.kld, "{:.3f}"), num(r.top1_agree, "{:.1f}%"), num(r.avg_acc, "{:.1f}"),
                 "–" if not np.isfinite(t4) else f"{t4:.0f} ({t4 / fp.t4_decode_tok_s:.2f}×)"])
save_table(rows, ["Method", "File size", "Δ ppl", "KLD", "Top-1 agree", "Task avg", "T4 decode tok/s"],
           "T1_ptq_methods", "T1 · PTQ methods on Qwen3-0.6B (Δ vs FP16 WikiText ppl 12.66)", bold_rows=(3,))

# T2 · calibration shift
rows = []
for cid, label in [("rtn_w4g128", "RTN (no calibration)"), ("gptq_w4g128", "GPTQ · calibrated on C4"),
                   ("gptq_w4g128_code", "GPTQ · calibrated on code"), ("awq_w4g128", "AWQ · calibrated on C4")]:
    rows.append([label, pct(100 * (v(cid, "ppl_wiki") / fp.ppl_wiki - 1)),
                 pct(100 * (v(cid, "ppl_code") / fp.ppl_code - 1))])
g = m.loc["gguf_f16"]
for cid, label in [("gguf_q4_k_m_imat_wiki", "GGUF Q4_K_M · imatrix WikiText"),
                   ("gguf_q4_k_m_imat_code", "GGUF Q4_K_M · imatrix code")]:
    rows.append([label, pct(100 * (v(cid, "ppl_wiki_lcpp") / g.ppl_wiki_lcpp - 1), 1),
                 pct(100 * (v(cid, "ppl_code_lcpp") / g.ppl_code_lcpp - 1), 1)])
save_table(rows, ["4-bit config", "Δ ppl on WikiText", "Δ ppl on code"], "T2_calibration_shift",
           "T2 · Does the calibration text matter?", bold_rows=(3,))

# T3 · QAT vs PTQ
rows = []
for pat, label in [("ptq_rtn_w{b}", "RTN (same quantizer)"), ("ft_then_rtn_w{b}", "Fine-tune → RTN (control)"),
                   ("gptq_w{b}g128", "GPTQ"), ("awq_w{b}g128", "AWQ"), ("qat_w{b}g128", "QAT → RTN")]:
    rows.append([label] + [x for b in (4, 3) for x in (pct(v(pat.format(b=b))), num(v(pat.format(b=b), "kld"), "{:.3f}"))])
save_table(rows, ["Method", "4-bit Δ ppl", "4-bit KLD", "3-bit Δ ppl", "3-bit KLD"], "T3_qat_vs_ptq",
           "T3 · QAT vs PTQ on the same model", bold_rows=(4,))

# T4 · GGUF
rows = []
for q, label in [("f16", "F16"), ("q8_0", "Q8_0"), ("q5_k_m", "Q5_K_M"), ("q4_k_m", "Q4_K_M"), ("q4_0", "Q4_0"),
                 ("iq4_xs", "IQ4_XS"), ("q3_k_m", "Q3_K_M")]:
    a, b = f"gguf_{q}", f"gguf_{q}_imat_wiki"
    imat = b in m.index
    dp = "–" if q == "f16" else pct(v(a), 1) + (f" → {pct(v(b), 1)}" if imat else "")
    kl = "–" if q == "f16" else num(v(a, "kld_lcpp"), "{:.3f}") + (f" → {v(b, 'kld_lcpp'):.3f}" if imat else "")
    rows.append([label, num(v(a, "size_gb"), "{:.2f} GB"), dp, kl, num(v(a, "cpu_decode_tok_s"), "{:.1f}"),
                 num(100 * v(a, "cpu_efficiency"), "{:.0f}%")])
save_table(rows, ["Quant", "Size", "Δ ppl (no imatrix → imatrix)", "KLD (no imatrix → imatrix)",
                  "CPU decode tok/s", "% of ceiling"], "T4_gguf_cpu",
           "T4 · GGUF on a 4-vCPU CPU (Δ vs llama.cpp F16)", bold_rows=(3,))

# T5 · cheat sheet
pv = lambda cid, col="dppl", d=0: pct(v(cid, col), d)
rows = [
    ["GPU, serving with vLLM", "AWQ 4-bit", f"{pv('awq_w4g128')} ppl, {v('awq_w4g128', 't4_decode_tok_s') / fp.t4_decode_tok_s:.2f}× decode"],
    ["GPU, AWQ too slow to calibrate", "GPTQ 4-bit", f"{pv('gptq_w4g128')} ppl, {v('gptq_w4g128', 'quant_time_s') / 60:.1f} min vs {v('awq_w4g128', 'quant_time_s') / 60:.0f} min"],
    ["Fine-tuning on a small GPU", "NF4 (QLoRA)", f"{pv('nf4_dq')} ppl, no calibration"],
    ["CPU / laptop", "GGUF Q4_K_M + imatrix", f"{pv('gguf_q4_k_m_imat_wiki', d=1)} ppl, {v('gguf_q4_k_m', 'cpu_decode_tok_s') / v('gguf_f16', 'cpu_decode_tok_s'):.1f}× F16 speed"],
    ["Quality first, still small", "Q5_K_M or Q8_0", f"{pv('gguf_q5_k_m_imat_wiki', d=1)} / {pv('gguf_q8_0', d=1)} ppl"],
    ["Must go to 3-bit", "QAT if you can train, else AWQ", f"{pv('qat_w3g128')} vs {pv('awq_w3g128')} ppl"],
    ["Never", "Plain 3-bit rounding", f"{pv('rtn_w3g128')} ppl"],
]
save_table(rows, ["You have / want", "Use", "Our numbers (Qwen3-0.6B)"], "T5_cheat_sheet",
           "T5 · Which one should I actually use?", bold_rows=(0, 3))

m.reset_index().to_csv(TAB / "all_results.csv", index=False)
print("figures ->", OUT, "| tables ->", TAB)
