"""Generate cost-comparison charts for AssessVox from measured usage + Oct-2026 pricing.

Run:  uv run --with matplotlib python cost_report/make_charts.py
Outputs PNGs into cost_report/.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

OUT = Path(__file__).parent
plt.rcParams.update({
    "figure.dpi": 130,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
})

INK = "#1b2a4a"
ACCENT = "#2f6df6"
GREEN = "#1aa06d"
AMBER = "#e8a23d"
RED = "#d9534f"
GREY = "#9aa4b2"

def usd(x, _=None):
    return f"${x:,.2f}"

# ── 1. Current stack cost breakdown ────────────────────────────────────────────
layers = ["TTS\nCartesia sonic-3.6", "LLM talk\nGemini 3.5 Flash",
          "Grading\nGemini 3.5 Flash", "STT\nCartesia ink-2", "Flow gen\n2.5 Flash-Lite"]
costs = [0.100, 0.090, 0.032, 0.039, 0.0005]
fig, ax = plt.subplots(figsize=(8, 4.2))
bars = ax.barh(layers[::-1], costs[::-1],
               color=[GREY, GREEN, GREY, AMBER, RED][::-1])
ax.set_title("Current stack — cost per 6-min exam  (total ≈ $0.26)")
ax.xaxis.set_major_formatter(FuncFormatter(usd))
for b, c in zip(bars, costs[::-1]):
    ax.text(c + 0.002, b.get_y() + b.get_height()/2, usd(c), va="center", fontsize=10)
ax.set_xlim(0, 0.12)
fig.tight_layout(); fig.savefig(OUT / "1_current_breakdown.png"); plt.close(fig)

# ── 2. LLM provider comparison (talk cost per exam) ────────────────────────────
llm = {"Gemini 3.5 Flash\n(current)": 0.090, "Cerebras\nLlama 3.3 70B": 0.041,
       "Gemini 2.5 Flash": 0.020, "Groq\nLlama 4 Scout": 0.006,
       "Gemini 2.5\nFlash-Lite": 0.0055}
fig, ax = plt.subplots(figsize=(8, 4.2))
colors = [RED, AMBER, ACCENT, GREEN, GREEN]
bars = ax.bar(list(llm), list(llm.values()), color=colors)
ax.set_title("LLM conversation cost per exam — current vs alternatives")
ax.yaxis.set_major_formatter(FuncFormatter(usd))
for b, v in zip(bars, llm.values()):
    ax.text(b.get_x()+b.get_width()/2, v+0.002, usd(v), ha="center", fontsize=10)
ax.set_ylim(0, 0.105)
ax.text(0.5, 0.098, "15× cheaper", color=GREEN, fontsize=10, ha="center")
fig.tight_layout(); fig.savefig(OUT / "2_llm_comparison.png"); plt.close(fig)

# ── 3. Current vs Optimized stacked ────────────────────────────────────────────
cats = ["Current\nstack", "Optimized\nstack"]
stt   = [0.039, 0.012]
tts   = [0.100, 0.100]
llm_t = [0.090, 0.006]
grade = [0.032, 0.007]
flow  = [0.0005, 0.0005]
fig, ax = plt.subplots(figsize=(7, 4.6))
b1 = ax.bar(cats, tts, label="TTS", color=GREY)
b2 = ax.bar(cats, llm_t, bottom=tts, label="LLM talk", color=GREEN)
bottom2 = [t+l for t, l in zip(tts, llm_t)]
b3 = ax.bar(cats, stt, bottom=bottom2, label="STT", color=AMBER)
bottom3 = [a+s for a, s in zip(bottom2, stt)]
b4 = ax.bar(cats, grade, bottom=bottom3, label="Grading", color=ACCENT)
bottom4 = [a+g for a, g in zip(bottom3, grade)]
b5 = ax.bar(cats, flow, bottom=bottom4, label="Flow gen", color=RED)
totals = [sum(x) for x in zip(stt, tts, llm_t, grade, flow)]
for i, t in enumerate(totals):
    ax.text(i, t+0.006, usd(t), ha="center", fontweight="bold", fontsize=12)
ax.set_title("Current vs optimized — cost per exam  (−52%)")
ax.yaxis.set_major_formatter(FuncFormatter(usd))
ax.set_ylim(0, 0.30)
ax.legend(ncol=5, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.08), frameon=False)
fig.tight_layout(); fig.savefig(OUT / "3_current_vs_optimized.png"); plt.close(fig)

# ── 4. Scale projection ────────────────────────────────────────────────────────
vols = [1_000, 10_000, 100_000]
cur = [v*0.26 for v in vols]
opt = [v*0.126 for v in vols]
x = range(len(vols))
fig, ax = plt.subplots(figsize=(8, 4.4))
w = 0.38
b1 = ax.bar([i-w/2 for i in x], cur, w, label="Current ($0.26/exam)", color=RED)
b2 = ax.bar([i+w/2 for i in x], opt, w, label="Optimized ($0.126/exam)", color=GREEN)
ax.set_yscale("log")
ax.set_xticks(list(x)); ax.set_xticklabels([f"{v:,}\nexams" for v in vols])
ax.set_title("Monthly cost at scale (log scale)")
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.0f}"))
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()*1.05,
                f"${b.get_height():,.0f}", ha="center", fontsize=9)
ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(OUT / "4_scale_projection.png"); plt.close(fig)

print("Wrote:", *[p.name for p in sorted(OUT.glob('*.png'))])
