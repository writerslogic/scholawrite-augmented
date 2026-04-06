"""
Bootstrap Confidence Intervals & Bayes Factors
===============================================
Computes 10,000-resample bootstrap CIs and Jeffreys (1961) Bayes factors
for key claims across the research papers, using KLiCKE corpus features.

Output: bootstrap_results.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import hyp2f1

np.random.seed(42)

# ---------- paths & constants ------------------------------------------------
CSV_DIR = Path(__file__).parent / "klicke" / "Files" / "WritingTask" / "WritingTask" / "keystrokelogs" / "csv"
QUANTIZATION_MS = 5
MIN_EVENTS = 50
N_BOOTSTRAP = 10_000
COGNITIVE_LOAD = {"Nonproduction": 3, "Remove/Cut": 2, "Input": 1}

print("=" * 70)
print("Bootstrap CI & Bayes Factor Analysis")
print("=" * 70)

# ---------- load & compute per-writer features (same as cross_domain_independence.py)
print("\n[1/5] Loading KLiCKE and computing per-writer features...")

records = []
all_window_clcs = []  # per-window CLC values for window-level stats

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

    # Per-window CLC: split valid events into windows of 100
    window_size = 100
    for w_start in range(0, len(valid) - window_size + 1, window_size):
        w = valid.iloc[w_start:w_start + window_size]
        if w["cog"].std() > 0:
            w_rho, _ = stats.spearmanr(w["cog"], w["iki_ms"])
            all_window_clcs.append(w_rho)

    # IKI variance (log-transformed)
    iki_var = np.log(iki.var(ddof=0) + 1)

    # Pause frequency (fraction of IKIs > 2s)
    pause_freq = (iki > 2000).mean()

    records.append({
        "writer_id": f.stem,
        "entropy": entropy,
        "clc_rho": clc_rho,
        "iki_log_var": iki_var,
        "pause_freq": pause_freq,
    })

df_feat = pd.DataFrame(records).dropna()
n_writers = len(df_feat)
all_window_clcs = np.array(all_window_clcs)
print(f"  Writers with complete features: {n_writers}")
print(f"  Total per-window CLC values: {len(all_window_clcs)}")


# ---------- helper: Bayes factor for independence (Jeffreys 1961) -----------
def bayes_factor_independence(r: float, n: int) -> float:
    """BF_01 = ((1-r^2)^((n-1)/2)) * hyp2f1(1/2, 1/2, n/2, r^2).
    BF_01 > 1 favours H0 (independence); < 1 favours H1 (dependence).
    """
    r2 = r ** 2
    term1 = (1 - r2) ** ((n - 1) / 2)
    term2 = float(hyp2f1(0.5, 0.5, n / 2, r2))
    return term1 * term2


# ---------- helper: bootstrap CI -------------------------------------------
# PY-C002: Use scipy.stats.bootstrap with BCa bias correction instead of
# manual np.percentile bootstrap. BCa corrects for bias and skewness in the
# bootstrap distribution (Efron 1987).

def bootstrap_ci(data: np.ndarray, stat_fn, n_boot: int = N_BOOTSTRAP, alpha: float = 0.05):
    """Returns (point_estimate, ci_lower, ci_upper) using BCa bootstrap."""
    point = stat_fn(data)
    try:
        res = stats.bootstrap(
            (data,),
            statistic=lambda d, axis=None: stat_fn(d) if axis is None else np.apply_along_axis(stat_fn, axis, d),
            n_resamples=n_boot,
            confidence_level=1 - alpha,
            method='BCa',
            random_state=np.random.default_rng(42),
        )
        lo, hi = res.confidence_interval.low, res.confidence_interval.high
    except Exception:
        # Fallback to basic method if BCa fails (e.g. degenerate data)
        res = stats.bootstrap(
            (data,),
            statistic=lambda d, axis=None: stat_fn(d) if axis is None else np.apply_along_axis(stat_fn, axis, d),
            n_resamples=n_boot,
            confidence_level=1 - alpha,
            method='basic',
            random_state=np.random.default_rng(42),
        )
        lo, hi = res.confidence_interval.low, res.confidence_interval.high
    return float(point), float(lo), float(hi)


def bootstrap_corr_ci(x: np.ndarray, y: np.ndarray, n_boot: int = N_BOOTSTRAP, alpha: float = 0.05):
    """Bootstrap CI for Spearman correlation using BCa method.

    scipy.stats.bootstrap requires a single data argument, so we stack x and y
    into a 2-column array and index columns inside the statistic function.
    """
    rho, _ = stats.spearmanr(x, y)

    def _spearman_stat(pairs, axis=None):
        if axis is None:
            r, _ = stats.spearmanr(pairs[:, 0], pairs[:, 1])
            return r
        # vectorised path: pairs has shape (n, 2, n_resamples) when axis=0
        results = np.empty(pairs.shape[2])
        for k in range(pairs.shape[2]):
            r, _ = stats.spearmanr(pairs[:, 0, k], pairs[:, 1, k])
            results[k] = r
        return results

    paired = np.column_stack([x, y])
    try:
        res = stats.bootstrap(
            (paired,),
            statistic=_spearman_stat,
            n_resamples=n_boot,
            confidence_level=1 - alpha,
            method='BCa',
            random_state=np.random.default_rng(42),
        )
        lo, hi = res.confidence_interval.low, res.confidence_interval.high
    except Exception:
        # Fallback to basic if BCa fails
        res = stats.bootstrap(
            (paired,),
            statistic=_spearman_stat,
            n_resamples=n_boot,
            confidence_level=1 - alpha,
            method='basic',
            random_state=np.random.default_rng(42),
        )
        lo, hi = res.confidence_interval.low, res.confidence_interval.high
    return float(rho), float(lo), float(hi)


# ---------- [2/5] Cross-domain correlation CIs & Bayes factors ---------------
print("\n[2/5] Bootstrap CIs for cross-domain correlations...")

features = ["entropy", "clc_rho", "iki_log_var", "pause_freq"]
feature_pairs = []
for i, a in enumerate(features):
    for j, b in enumerate(features):
        if i < j:
            feature_pairs.append((a, b))

cross_domain = {}
for a, b in feature_pairs:
    x = df_feat[a].values
    y = df_feat[b].values
    rho, ci_lo, ci_hi = bootstrap_corr_ci(x, y)
    bf01 = bayes_factor_independence(rho, n_writers)
    key = f"{a}_vs_{b}"
    cross_domain[key] = {
        "rho": round(rho, 4),
        "ci_95_lower": round(ci_lo, 4),
        "ci_95_upper": round(ci_hi, 4),
        "bayes_factor_for_independence": round(bf01, 4),
    }
    bf_interp = "strong FOR independence" if bf01 > 10 else ("strong AGAINST independence" if bf01 < 0.1 else "inconclusive")
    print(f"  {a:12s} x {b:12s}: rho={rho:+.4f} [{ci_lo:+.4f}, {ci_hi:+.4f}]  BF01={bf01:.4f} ({bf_interp})")

# ---------- [3/5] CLC statistics ---------------------------------------------
print("\n[3/5] Bootstrap CIs for CLC statistics...")

clc_vals = df_feat["clc_rho"].values

# Per-writer median CLC
pw_med, pw_lo, pw_hi = bootstrap_ci(clc_vals, np.median)
print(f"  Per-writer median CLC: {pw_med:.4f} [{pw_lo:.4f}, {pw_hi:.4f}]")

# Per-window median CLC
ww_med, ww_lo, ww_hi = bootstrap_ci(all_window_clcs, np.median)
print(f"  Per-window median CLC: {ww_med:.4f} [{ww_lo:.4f}, {ww_hi:.4f}]")

# Percentage of writers with positive CLC
pct_pos_fn = lambda d: 100.0 * np.mean(d > 0)
pct_val, pct_lo, pct_hi = bootstrap_ci(clc_vals, pct_pos_fn)
print(f"  %% writers with positive CLC: {pct_val:.1f}% [{pct_lo:.1f}%, {pct_hi:.1f}%]")

# ---------- [4/5] Entropy statistics -----------------------------------------
print("\n[4/5] Bootstrap CIs for entropy statistics...")

ent_vals = df_feat["entropy"].values
ent_mean, ent_lo, ent_hi = bootstrap_ci(ent_vals, np.mean)
print(f"  Per-writer mean entropy: {ent_mean:.4f} [{ent_lo:.4f}, {ent_hi:.4f}]")

# ---------- [5/5] Save results -----------------------------------------------
print("\n[5/5] Saving results...")

results = {
    "n_writers": n_writers,
    "n_windows": len(all_window_clcs),
    "n_bootstrap": N_BOOTSTRAP,
    "cross_domain_correlations": cross_domain,
    "clc_statistics": {
        "per_writer_median": {
            "value": round(pw_med, 4),
            "ci_95_lower": round(pw_lo, 4),
            "ci_95_upper": round(pw_hi, 4),
        },
        "per_window_median": {
            "value": round(ww_med, 4),
            "ci_95_lower": round(ww_lo, 4),
            "ci_95_upper": round(ww_hi, 4),
        },
        "pct_positive": {
            "value": round(pct_val, 2),
            "ci_95_lower": round(pct_lo, 2),
            "ci_95_upper": round(pct_hi, 2),
        },
    },
    "entropy_statistics": {
        "per_writer_mean": {
            "value": round(ent_mean, 4),
            "ci_95_lower": round(ent_lo, 4),
            "ci_95_upper": round(ent_hi, 4),
        },
    },
}

out = Path(__file__).parent / "bootstrap_results.json"
with open(out, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out}")

# ---------- summary ----------------------------------------------------------
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  Writers analyzed: {n_writers}")
print(f"  Windows analyzed: {len(all_window_clcs)}")
for key, val in cross_domain.items():
    bf = val["bayes_factor_for_independence"]
    label = "INDEPENDENT" if bf > 10 else ("DEPENDENT" if bf < 0.1 else "INCONCLUSIVE")
    print(f"  {key}: BF01={bf:.4f} -> {label}")
print(f"  CLC median (per-writer): {pw_med:.4f} CI=[{pw_lo:.4f}, {pw_hi:.4f}]")
print(f"  CLC positive writers: {pct_val:.1f}% CI=[{pct_lo:.1f}%, {pct_hi:.1f}%]")
print(f"  Entropy mean: {ent_mean:.4f} CI=[{ent_lo:.4f}, {ent_hi:.4f}]")
