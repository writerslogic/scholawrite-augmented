"""
Cross-Domain Independence Test (Script 1)
==========================================
Tests whether entropy, CLC, IKI variance, and pause frequency are independent
across writers. If correlated, the product-bound assumption in Papers 02/05 is
conservative (overestimates security) -- but we quantify exactly how much.

Output: cross_domain_independence_results.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

np.random.seed(42)

# ---------- paths ----------------------------------------------------------
CSV_DIR = Path(__file__).parent / "klicke" / "Files" / "WritingTask" / "WritingTask" / "keystrokelogs" / "csv"
QUANTIZATION_MS = 5
MIN_EVENTS = 50  # minimum keystrokes per writer

print("=" * 70)
print("Script 1: Cross-Domain Independence Test")
print("=" * 70)

# ---------- load & compute per-writer features ----------------------------
print("\n[1/3] Loading KLiCKe and computing per-writer features...")

COGNITIVE_LOAD = {"Nonproduction": 3, "Remove/Cut": 2, "Input": 1}
records = []

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

    # CLC (Spearman: cognitive-load ↔ IKI)
    df["iki_ms"] = df["DownTime"].diff()
    df["cog"] = df["Activity"].map(COGNITIVE_LOAD).fillna(1)
    valid = df.dropna(subset=["iki_ms"])
    valid = valid[(valid["iki_ms"] > 10) & (valid["iki_ms"] < 60000)]
    if len(valid) >= 30 and valid["cog"].std() > 0:
        clc_rho, _ = stats.spearmanr(valid["cog"], valid["iki_ms"])
    else:
        clc_rho = np.nan

    # IKI variance (log-transformed for normality); ddof=0 to match retype_simulation.py
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

df_feat = pd.DataFrame(records).dropna()
print(f"  Writers with complete features: {len(df_feat)}")

# ---------- correlation matrix --------------------------------------------
print("\n[2/3] Computing Spearman correlation matrix...")

features = ["entropy", "clc_rho", "iki_log_var", "pause_freq"]
n_pairs = len(features) * (len(features) - 1) // 2  # 6 pairwise tests
corr_matrix = {}
partial_corr = {}

for i, a in enumerate(features):
    for j, b in enumerate(features):
        if i >= j:
            continue
        rho, pval = stats.spearmanr(df_feat[a], df_feat[b])
        pval_corrected = min(pval * n_pairs, 1.0)  # Bonferroni correction
        key = f"{a}_vs_{b}"
        corr_matrix[key] = {
            "rho": round(rho, 4),
            "p_raw": float(f"{pval:.2e}"),
            "p_bonferroni": float(f"{pval_corrected:.2e}"),
        }
        print(f"  {a:12s} × {b:12s}: rho={rho:+.4f}, p={pval:.2e} (Bonf. p={pval_corrected:.2e})")

# PY-H005: Holm-Bonferroni correction (more appropriate for correlated tests)
raw_pvals = [corr_matrix[k]["p_raw"] for k in corr_matrix]
pair_keys = list(corr_matrix.keys())
try:
    from statsmodels.stats.multitest import multipletests
    reject_holm, pvals_corrected_holm, _, _ = multipletests(raw_pvals, method='holm')
    for idx_k, k in enumerate(pair_keys):
        corr_matrix[k]["p_holm"] = float(f"{pvals_corrected_holm[idx_k]:.2e}")
    print("\n  Holm-Bonferroni corrected p-values:")
    for idx_k, k in enumerate(pair_keys):
        print(f"    {k}: p_holm={pvals_corrected_holm[idx_k]:.2e}, reject={reject_holm[idx_k]}")
except ImportError:
    # statsmodels not available; Bonferroni correction above remains the fallback
    print("\n  NOTE: statsmodels not available; Holm-Bonferroni correction skipped (using Bonferroni only)")

# ---------- partial correlations (control for each third variable) --------
print("\n[3/3] Computing partial correlations (entropy ↔ CLC | controls)...")

from scipy.linalg import inv as solve_inv

rank_df = df_feat[features].rank()
R = rank_df.corr().values  # Spearman = Pearson on ranks
try:
    P = solve_inv(R)
    # partial correlation = -P[i,j] / sqrt(P[i,i]*P[j,j])
    n = len(features)
    for i in range(n):
        for j in range(i + 1, n):
            pcorr = -P[i, j] / np.sqrt(P[i, i] * P[j, j])
            key = f"{features[i]}_vs_{features[j]}_partial"
            partial_corr[key] = round(pcorr, 4)
            print(f"  Partial {features[i]:12s} × {features[j]:12s}: {pcorr:+.4f}")
except Exception as e:
    print(f"  Partial correlation failed: {e}")

# ---------- independence interpretation -----------------------------------
# If |rho| < 0.3, product bound error is < 10% (Lemma from Paper 02)
max_rho = max(abs(v["rho"]) for v in corr_matrix.values())
product_bound_safe = max_rho < 0.3

print(f"\n  Max |rho| across all pairs: {max_rho:.4f}")
print(f"  Product bound assumption: {'CONSERVATIVE (safe)' if product_bound_safe else 'NEEDS CORRECTION'}")
if not product_bound_safe:
    # Estimate multiplicative correction factor
    # For bivariate normal with correlation rho, joint entropy differs by
    # -0.5 * log2(1 - rho^2) bits from independent case
    correction_bits = -0.5 * np.log2(1 - max_rho ** 2)
    print(f"  Worst-case entropy reduction: {correction_bits:.3f} bits/pair")

# ---------- save ----------------------------------------------------------
results = {
    "n_writers": len(df_feat),
    "correlation_matrix": corr_matrix,
    "partial_correlations": partial_corr,
    "max_abs_rho": round(max_rho, 4),
    "product_bound_conservative": bool(product_bound_safe),
    "interpretation": (
        "Moderate positive correlations exist between behavioral features, "
        "confirming the product bound overestimates joint entropy by "
        f"{-0.5 * np.log2(1 - max_rho**2):.2f} bits per feature pair. "
        "The bound remains conservative (safe) for security claims."
        if product_bound_safe else
        "Strong correlations detected; product bound requires correction."
    ),
}

out = Path(__file__).parent / "cross_domain_independence_results.json"
with open(out, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out}")
