"""
Figures and Effect Sizes for Process-Attestation Paper
======================================================
Generates publication-quality figures (PDF + PNG) and computes effect sizes
for key composition-vs-transcription claims.

Outputs:
  fig_clc_distribution.{pdf,png}
  fig_correlation_matrix.{pdf,png}
  fig_clc_composition_vs_transcription.{pdf,png}
  effect_sizes.json
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

np.random.seed(42)

# ── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
CSV_DIR = SCRIPT_DIR / "klicke" / "Files" / "WritingTask" / "WritingTask" / "keystrokelogs" / "csv"
QUANTIZATION_MS = 5
MIN_EVENTS = 50
COGNITIVE_LOAD = {"Nonproduction": 3, "Remove/Cut": 2, "Input": 1}

# ── Publication style ───────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
})

# ── Load and compute per-writer features ────────────────────────────────────
print("=" * 70)
print("Figures & Effect Sizes")
print("=" * 70)
print("\n[1/5] Loading KLiCKe and computing per-writer features...")

records = []
# Per-window records for composition vs transcription analysis
window_records = []

csv_files = sorted(CSV_DIR.glob("*.csv"))
n_files = len(csv_files)

for i, f in enumerate(csv_files):
    if (i + 1) % 1000 == 0:
        print(f"  {i+1}/{n_files} writers processed...")
    try:
        df = pd.read_csv(f)
    except Exception:
        continue

    df = df.sort_values("DownTime").reset_index(drop=True)
    iki = df["DownTime"].diff().dropna()
    iki = iki[(iki > 10) & (iki < 60000)]
    if len(iki) < MIN_EVENTS:
        continue

    # Entropy (quantised IKI)
    quantized = (iki // QUANTIZATION_MS) * QUANTIZATION_MS
    probs = quantized.value_counts(normalize=True).values
    entropy = -np.sum(probs * np.log2(probs + 1e-15))

    # CLC (Spearman: cognitive-load <-> IKI)
    df["iki_ms"] = df["DownTime"].diff()
    df["cog"] = df["Activity"].map(COGNITIVE_LOAD).fillna(1)
    valid = df.dropna(subset=["iki_ms"])
    valid = valid[(valid["iki_ms"] > 10) & (valid["iki_ms"] < 60000)]
    if len(valid) >= 30 and valid["cog"].std() > 0:
        clc_rho, _ = stats.spearmanr(valid["cog"], valid["iki_ms"])
    else:
        clc_rho = np.nan

    # IKI variance (log-transformed)
    iki_var = np.log(iki.var(ddof=0) + 1)

    # Pause frequency (fraction of IKIs > 2 s)
    pause_freq = (iki > 2000).mean()

    records.append({
        "writer_id": f.stem,
        "entropy": entropy,
        "clc_rho": clc_rho,
        "iki_log_var": iki_var,
        "pause_freq": pause_freq,
    })

    # ── Sliding-window analysis for composition vs transcription ─────────
    # Classify windows: "composition" = has Nonproduction AND Remove/Cut
    # (planning + revising), "transcription" = mostly Input
    if len(valid) < 60:
        continue
    WINDOW = 50
    STEP = 25
    for start in range(0, len(valid) - WINDOW, STEP):
        w = valid.iloc[start:start + WINDOW]
        act_counts = w["Activity"].value_counts()
        n_input = act_counts.get("Input", 0)
        n_nonprod = act_counts.get("Nonproduction", 0)
        n_remove = act_counts.get("Remove/Cut", 0)

        input_frac = n_input / WINDOW
        comp_frac = (n_nonprod + n_remove) / WINDOW

        if comp_frac >= 0.3:
            label = "composition"
        elif input_frac >= 0.8:
            label = "transcription"
        else:
            continue

        w_iki = w["iki_ms"].values
        w_cog = w["cog"].values
        if w_cog.std() > 0:
            w_rho, _ = stats.spearmanr(w_cog, w_iki)
        else:
            w_rho = np.nan

        w_quant = (pd.Series(w_iki) // QUANTIZATION_MS * QUANTIZATION_MS)
        w_probs = w_quant.value_counts(normalize=True).values
        w_entropy = -np.sum(w_probs * np.log2(w_probs + 1e-15))

        window_records.append({
            "writer_id": f.stem,
            "label": label,
            "clc_rho": w_rho,
            "entropy": w_entropy,
        })

df_feat = pd.DataFrame(records).dropna()
df_windows = pd.DataFrame(window_records).dropna()
print(f"  Writers with complete features: {len(df_feat)}")
print(f"  Windows: {len(df_windows)} "
      f"(composition={len(df_windows[df_windows.label=='composition'])}, "
      f"transcription={len(df_windows[df_windows.label=='transcription'])})")

# ── Figure 1: Per-Writer CLC Distribution ──────────────────────────────────
print("\n[2/5] Figure 1: Per-Writer CLC Distribution...")

fig, ax = plt.subplots(figsize=(5.5, 3.5))
clc_vals = df_feat["clc_rho"].values
pct_pos = (clc_vals > 0).mean() * 100
pct_neg = (clc_vals <= 0).mean() * 100

bins = np.linspace(clc_vals.min() - 0.02, clc_vals.max() + 0.02, 60)
n_hist, bin_edges, patches = ax.hist(clc_vals, bins=bins, color="#4878CF",
                                      edgecolor="white", linewidth=0.3, alpha=0.85)

# Shade negative region
for patch, left_edge in zip(patches, bin_edges[:-1]):
    if left_edge + (bin_edges[1] - bin_edges[0]) / 2 < 0:
        patch.set_facecolor("#E24A33")
        patch.set_alpha(0.7)

# Threshold line
ax.axvline(0.15, color="black", linestyle="--", linewidth=1.0, label=r"$\tau = 0.15$")
ax.axvline(0, color="gray", linestyle=":", linewidth=0.8)

# Annotations
y_top = n_hist.max()
ax.annotate(f"{pct_pos:.0f}% positive CLC", xy=(0.25, y_top * 0.85),
            fontsize=9, color="#4878CF", fontweight="bold")
ax.annotate(f"{pct_neg:.0f}% negative CLC", xy=(clc_vals.min() + 0.01, y_top * 0.85),
            fontsize=9, color="#E24A33", fontweight="bold")

ax.set_xlabel(r"Per-Writer CLC (Spearman $\rho$)")
ax.set_ylabel("Number of Writers")
ax.set_title("Per-Writer Cognitive-Load Correlation Distribution")
ax.legend(frameon=False)

fig.savefig(SCRIPT_DIR / "fig_clc_distribution.pdf")
fig.savefig(SCRIPT_DIR / "fig_clc_distribution.png")
plt.close(fig)
print("  Saved fig_clc_distribution.{pdf,png}")

# ── Figure 2: Cross-Domain Correlation Matrix Heatmap ──────────────────────
print("\n[3/5] Figure 2: Correlation Matrix Heatmap...")

features = ["entropy", "clc_rho", "iki_log_var", "pause_freq"]
labels = ["Entropy", "CLC", r"IKI $\log$ Var", "Pause Freq"]
n_f = len(features)
corr_mat = np.ones((n_f, n_f))
for i in range(n_f):
    for j in range(n_f):
        if i != j:
            rho, _ = stats.spearmanr(df_feat[features[i]], df_feat[features[j]])
            corr_mat[i, j] = rho

fig, ax = plt.subplots(figsize=(4.5, 4.0))
cax = ax.imshow(corr_mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")

for i in range(n_f):
    for j in range(n_f):
        color = "white" if abs(corr_mat[i, j]) > 0.5 else "black"
        ax.text(j, i, f"{corr_mat[i,j]:.2f}", ha="center", va="center",
                fontsize=10, color=color, fontweight="bold")

ax.set_xticks(range(n_f))
ax.set_yticks(range(n_f))
ax.set_xticklabels(labels, rotation=35, ha="right")
ax.set_yticklabels(labels)
ax.set_title(f"Cross-Domain Correlation Matrix (N={len(df_feat):,} KLiCKE Writers)",
             pad=12)
fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04, label=r"Spearman $\rho$")

fig.savefig(SCRIPT_DIR / "fig_correlation_matrix.pdf")
fig.savefig(SCRIPT_DIR / "fig_correlation_matrix.png")
plt.close(fig)
print("  Saved fig_correlation_matrix.{pdf,png}")

# ── Figure 3: Composition vs Transcription CLC ────────────────────────────
print("\n[4/5] Figure 3: Composition vs Transcription CLC...")

comp = df_windows[df_windows.label == "composition"]["clc_rho"].values
trans = df_windows[df_windows.label == "transcription"]["clc_rho"].values

fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

# Panel A: CLC distributions
ax = axes[0]
bins_c = np.linspace(-0.8, 0.8, 50)
ax.hist(comp, bins=bins_c, alpha=0.65, color="#4878CF", label="Composition", density=True,
        edgecolor="white", linewidth=0.3)
ax.hist(trans, bins=bins_c, alpha=0.65, color="#E24A33", label="Transcription", density=True,
        edgecolor="white", linewidth=0.3)
ax.set_xlabel(r"Window CLC (Spearman $\rho$)")
ax.set_ylabel("Density")
ax.set_title("A. CLC Distributions by Mode")
ax.legend(frameon=False)

# Panel B: Entropy distributions
comp_ent = df_windows[df_windows.label == "composition"]["entropy"].values
trans_ent = df_windows[df_windows.label == "transcription"]["entropy"].values

ax = axes[1]
bins_e = np.linspace(
    min(comp_ent.min(), trans_ent.min()) - 0.2,
    max(comp_ent.max(), trans_ent.max()) + 0.2,
    50,
)
ax.hist(comp_ent, bins=bins_e, alpha=0.65, color="#4878CF", label="Composition", density=True,
        edgecolor="white", linewidth=0.3)
ax.hist(trans_ent, bins=bins_e, alpha=0.65, color="#E24A33", label="Transcription", density=True,
        edgecolor="white", linewidth=0.3)
ax.set_xlabel("Window Entropy (bits)")
ax.set_ylabel("Density")
ax.set_title("B. Entropy Distributions by Mode")
ax.legend(frameon=False)

fig.suptitle("Composition vs Transcription-like Windows", y=1.02, fontsize=12)
fig.tight_layout()

fig.savefig(SCRIPT_DIR / "fig_clc_composition_vs_transcription.pdf")
fig.savefig(SCRIPT_DIR / "fig_clc_composition_vs_transcription.png")
plt.close(fig)
print("  Saved fig_clc_composition_vs_transcription.{pdf,png}")

# ── Effect sizes ───────────────────────────────────────────────────────────
print("\n[5/5] Computing effect sizes...")


def cohens_d(a, b):
    na, nb = len(a), len(b)
    pooled_std = np.sqrt(((na - 1) * a.std(ddof=1)**2 + (nb - 1) * b.std(ddof=1)**2)
                         / (na + nb - 2))
    return (a.mean() - b.mean()) / pooled_std


def glass_delta(a, b):
    """Glass's Delta using b (control/transcription) SD."""
    return (a.mean() - b.mean()) / b.std(ddof=1)


def cliffs_delta(a, b):
    """Cliff's delta via rank-biserial correlation: r = 1 - (2*U)/(n1*n2)."""
    n1, n2 = len(a), len(b)
    u_stat, _ = stats.mannwhitneyu(a, b, alternative="two-sided")
    return 1 - (2 * u_stat) / (n1 * n2)


effects = {
    "n_composition_windows": int(len(comp)),
    "n_transcription_windows": int(len(trans)),
    "entropy": {
        "composition_mean": round(float(comp_ent.mean()), 4),
        "composition_sd": round(float(comp_ent.std(ddof=1)), 4),
        "transcription_mean": round(float(trans_ent.mean()), 4),
        "transcription_sd": round(float(trans_ent.std(ddof=1)), 4),
        "cohens_d": round(float(cohens_d(comp_ent, trans_ent)), 4),
        "glass_delta": round(float(glass_delta(comp_ent, trans_ent)), 4),
        "cliffs_delta": round(float(cliffs_delta(comp_ent, trans_ent)), 4),
        "mann_whitney_p": float(f"{stats.mannwhitneyu(comp_ent, trans_ent).pvalue:.4e}"),
    },
    "clc": {
        "composition_mean": round(float(comp.mean()), 4),
        "composition_sd": round(float(comp.std(ddof=1)), 4),
        "transcription_mean": round(float(trans.mean()), 4),
        "transcription_sd": round(float(trans.std(ddof=1)), 4),
        "cohens_d": round(float(cohens_d(comp, trans)), 4),
        "glass_delta": round(float(glass_delta(comp, trans)), 4),
        "cliffs_delta": round(float(cliffs_delta(comp, trans)), 4),
        "mann_whitney_p": float(f"{stats.mannwhitneyu(comp, trans).pvalue:.4e}"),
    },
}

# Interpret effect sizes
for key in ["entropy", "clc"]:
    d = abs(effects[key]["cohens_d"])
    if d >= 0.8:
        effects[key]["magnitude"] = "large"
    elif d >= 0.5:
        effects[key]["magnitude"] = "medium"
    elif d >= 0.2:
        effects[key]["magnitude"] = "small"
    else:
        effects[key]["magnitude"] = "negligible"

out_path = SCRIPT_DIR / "effect_sizes.json"
with open(out_path, "w") as fout:
    json.dump(effects, fout, indent=2)

print(f"\n  Entropy  Cohen's d = {effects['entropy']['cohens_d']:.4f} ({effects['entropy']['magnitude']})")
print(f"  Entropy  Glass's Δ = {effects['entropy']['glass_delta']:.4f}")
print(f"  Entropy  Cliff's δ = {effects['entropy']['cliffs_delta']:.4f}")
print(f"  CLC      Cohen's d = {effects['clc']['cohens_d']:.4f} ({effects['clc']['magnitude']})")
print(f"  CLC      Glass's Δ = {effects['clc']['glass_delta']:.4f}")
print(f"  CLC      Cliff's δ = {effects['clc']['cliffs_delta']:.4f}")
print(f"\nEffect sizes saved to {out_path}")
print("\nDone.")
