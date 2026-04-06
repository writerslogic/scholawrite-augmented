"""
Likelihood Ratios & KL Divergence (Script 4)
==============================================
Computes empirical likelihood ratios P(features|genuine) / P(features|forged)
using KDE, and KL divergence between composition vs. transcription IKI
distributions (from KLiCKe activity labels).

Replaces:
  - Paper 04 assumed 3:1/8:1/12:1 likelihood ratios with measured values
  - Paper 07 uncited "Delta_T ≈ 0.8 bits" with measured KL divergence

Output: likelihood_ratios_results.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon

np.random.seed(42)

print("=" * 70)
print("Script 4: Likelihood Ratios & KL Divergence")
print("=" * 70)

# ---------- load session data from Script 2 -------------------------------
session_csv = Path(__file__).parent / "retype_sessions.csv"
if not session_csv.exists():
    raise FileNotFoundError(f"Run retype_simulation.py first: {session_csv}")

df = pd.read_csv(session_csv)
print(f"\n[1/3] Loaded {len(df)} sessions")

feature_cols = ["entropy", "autocorr_lag1", "iki_std", "iki_log_var", "pause_freq"]

genuine = df[df["label"] == 0]
attack_types = ["attack_constant", "attack_iid", "attack_cross"]

# ---------- KDE-based likelihood ratios -----------------------------------
print("\n[2/3] Computing KDE-based likelihood ratios per feature...")

lr_results = {}

for attack in attack_types:
    forged = df[df["session_type"] == attack]
    lr_per_feat = {}

    for feat in feature_cols:
        g_vals = genuine[feat].dropna().values
        f_vals = forged[feat].dropna().values

        if len(g_vals) < 10 or len(f_vals) < 10:
            continue

        # Fit KDE
        try:
            kde_g = stats.gaussian_kde(g_vals)
            kde_f = stats.gaussian_kde(f_vals)
        except Exception:
            continue

        # Evaluate LR at genuine data points
        eval_points = np.linspace(
            min(g_vals.min(), f_vals.min()),
            max(g_vals.max(), f_vals.max()),
            500,
        )
        p_g = kde_g(eval_points)
        p_f = kde_f(eval_points)

        # Likelihood ratio at genuine mean
        g_mean_val = np.atleast_1d(genuine[feat].mean())
        lr_at_mean = float((kde_g(g_mean_val) / (kde_f(g_mean_val) + 1e-15))[0])

        # Median LR for genuine samples
        lr_genuine = kde_g(g_vals) / (kde_f(g_vals) + 1e-15)
        median_lr = float(np.median(lr_genuine))

        # KL divergence D(genuine || forged) via numerical integration
        dx = eval_points[1] - eval_points[0]
        mask = (p_g > 1e-15) & (p_f > 1e-15)
        kl_div = float(np.sum(p_g[mask] * np.log2(p_g[mask] / p_f[mask]) * dx))

        lr_per_feat[feat] = {
            "lr_at_genuine_mean": round(lr_at_mean, 2),
            "median_lr_genuine": round(median_lr, 2),
            "kl_divergence_bits": round(kl_div, 4),
            "genuine_mean": round(float(g_vals.mean()), 4),
            "forged_mean": round(float(f_vals.mean()), 4),
        }

        print(f"  {attack} / {feat}: median LR = {median_lr:.1f}:1, KL = {kl_div:.3f} bits")

    lr_results[attack] = lr_per_feat

# ---------- Combined likelihood ratios (multi-feature) --------------------
print("\n  Combined multi-feature LRs (product of per-feature medians):")
combined_lrs = {}
for attack in attack_types:
    if attack in lr_results:
        medians = [lr_results[attack][f]["median_lr_genuine"]
                   for f in feature_cols if f in lr_results[attack]]
        combined = float(np.prod(medians)) if medians else 1.0
        combined_lrs[attack] = round(combined, 1)
        print(f"    {attack}: {combined:.1f}:1")

# ---------- KL divergence: composition vs transcription (from KLiCKe) -----
print("\n[3/3] KL divergence: demanding vs simple activities...")

CSV_DIR = Path(__file__).parent / "klicke" / "Files" / "WritingTask" / "WritingTask" / "keystrokelogs" / "csv"

demanding_ikis = []
simple_ikis = []

csv_files = sorted(CSV_DIR.glob("*.csv"))
for i, f in enumerate(csv_files[:500]):  # subsample for speed
    try:
        wdf = pd.read_csv(f)
    except Exception:
        continue
    wdf = wdf.sort_values("DownTime").reset_index(drop=True)
    wdf["iki"] = wdf["DownTime"].diff()
    valid = wdf.dropna(subset=["iki"])
    valid = valid[(valid["iki"] > 10) & (valid["iki"] < 60000)]

    dem = valid[valid["Activity"].isin(["Nonproduction", "Remove/Cut"])]["iki"].values
    sim = valid[valid["Activity"] == "Input"]["iki"].values

    if len(dem) > 0:
        demanding_ikis.extend(dem.tolist())
    if len(sim) > 0:
        simple_ikis.extend(sim.tolist())

demanding_arr = np.array(demanding_ikis)
simple_arr = np.array(simple_ikis)

print(f"  Demanding IKIs: {len(demanding_arr):,}")
print(f"  Simple IKIs:    {len(simple_arr):,}")

# KL divergence via histogram (more robust for large samples)
bins = np.linspace(10, 5000, 200)
hist_d, _ = np.histogram(demanding_arr, bins=bins, density=True)
hist_s, _ = np.histogram(simple_arr, bins=bins, density=True)
dx = bins[1] - bins[0]

# PY-H003: Use scipy's Jensen-Shannon distance which handles zero bins correctly,
# instead of manual epsilon addition that biases the divergence.
# Original manual approach (kept for reference):
#   hist_d = hist_d + 1e-10
#   hist_s = hist_s + 1e-10
#   kl_demanding_simple = float(np.sum(hist_d * np.log2(hist_d / hist_s) * dx))
#   kl_simple_demanding = float(np.sum(hist_s * np.log2(hist_s / hist_d) * dx))
#   js_divergence = 0.5 * kl_demanding_simple + 0.5 * kl_simple_demanding

# Normalize histograms to proper probability distributions (sum to 1)
p_d = hist_d * dx
p_s = hist_s * dx
p_d = p_d / p_d.sum() if p_d.sum() > 0 else p_d
p_s = p_s / p_s.sum() if p_s.sum() > 0 else p_s

# Jensen-Shannon divergence (squared distance, base 2) handles zeros correctly
js_divergence = float(jensenshannon(p_d, p_s, base=2) ** 2)

# KL divergences computed from non-zero bins only (no epsilon needed)
mask_both = (p_d > 0) & (p_s > 0)
kl_demanding_simple = float(np.sum(p_d[mask_both] * np.log2(p_d[mask_both] / p_s[mask_both])))
kl_simple_demanding = float(np.sum(p_s[mask_both] * np.log2(p_s[mask_both] / p_d[mask_both])))

print(f"  KL(demanding || simple) = {kl_demanding_simple:.4f} bits")
print(f"  KL(simple || demanding) = {kl_simple_demanding:.4f} bits")
print(f"  JS divergence           = {js_divergence:.4f} bits")

# ---------- save ----------------------------------------------------------
results = {
    "likelihood_ratios_by_attack": lr_results,
    "combined_likelihood_ratios": combined_lrs,
    "transcriptive_gap": {
        "kl_demanding_to_simple_bits": round(kl_demanding_simple, 4),
        "kl_simple_to_demanding_bits": round(kl_simple_demanding, 4),
        "js_divergence_bits": round(js_divergence, 4),
        "n_demanding_ikis": len(demanding_arr),
        "n_simple_ikis": len(simple_arr),
        "demanding_mean_ms": round(float(demanding_arr.mean()), 1),
        "simple_mean_ms": round(float(simple_arr.mean()), 1),
    },
    "interpretation": (
        "Empirical likelihood ratios replace assumed 3:1/8:1/12:1 values. "
        f"Transcriptive gap KL = {kl_demanding_simple:.2f} bits (Paper 07 claimed 0.8 bits)."
    ),
}

out = Path(__file__).parent / "likelihood_ratios_results.json"
with open(out, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out}")

print("\n" + "=" * 70)
print("KEY NUMBERS FOR PAPER UPDATES")
print("=" * 70)
print(f"  Paper 04: Replace assumed LRs with {combined_lrs}")
print(f"  Paper 07: Replace 'Delta_T ≈ 0.8 bits' with KL = {kl_demanding_simple:.2f} bits")
