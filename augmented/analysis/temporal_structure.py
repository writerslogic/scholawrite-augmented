"""
Temporal Structure Analysis (Script 5)
=======================================
Quantifies temporal dependencies in genuine keystroke sequences that are
destroyed by iid retype attacks.

Computes:
  - Autocorrelation function (ACF) at lags 1-10
  - Conditional entropy H(IKI_n | IKI_{n-1}) vs marginal H(IKI_n)
  - Permutation tests for temporal structure significance

Replaces:
  - Paper 09 C4: delta-entropy assumption with measured conditional entropy
  - Temporal anchor claims across papers

Output: temporal_structure_results.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

print("=" * 70)
print("Script 5: Temporal Structure Analysis")
print("=" * 70)

CSV_DIR = Path(__file__).parent / "klicke" / "Files" / "WritingTask" / "WritingTask" / "keystrokelogs" / "csv"
QUANTIZATION_MS = 5
MAX_WRITERS = 300
N_PERMUTATIONS = 1000

np.random.seed(42)

# ---------- load genuine IKI sequences ------------------------------------
print("\n[1/4] Loading genuine IKI sequences...")

writer_sequences = {}
csv_files = sorted(CSV_DIR.glob("*.csv"))

sampled_files = csv_files[:MAX_WRITERS] if len(csv_files) <= MAX_WRITERS else \
    [csv_files[i] for i in np.random.choice(len(csv_files), MAX_WRITERS, replace=False)]
for f in sampled_files:
    try:
        df = pd.read_csv(f)
    except Exception:
        continue
    df = df.sort_values("DownTime").reset_index(drop=True)
    iki = df["DownTime"].diff().dropna().values
    valid = (iki > 10) & (iki < 60000)
    iki = iki[valid].astype(float)
    if len(iki) >= 50:
        writer_sequences[f.stem] = iki

print(f"  Loaded {len(writer_sequences)} writers with >=50 valid IKIs")

# ---------- ACF at lags 1-10 ---------------------------------------------
print("\n[2/4] Computing autocorrelation functions...")

MAX_LAG = 10

def acf(x, max_lag=MAX_LAG):
    """Compute sample autocorrelation at lags 1..max_lag."""
    n = len(x)
    xm = x - x.mean()
    c0 = np.dot(xm, xm) / n
    if c0 == 0:
        return np.zeros(max_lag)
    return np.array([np.dot(xm[:n-k], xm[k:]) / (n * c0) for k in range(1, max_lag + 1)])


# Genuine ACF
genuine_acfs = []
for wid, ikis in writer_sequences.items():
    genuine_acfs.append(acf(ikis))

genuine_acfs = np.array(genuine_acfs)
mean_genuine_acf = genuine_acfs.mean(axis=0)

print("  Genuine mean ACF:")
for lag in range(MAX_LAG):
    print(f"    Lag {lag+1}: {mean_genuine_acf[lag]:+.4f}")

# Shuffled ACF (destroy temporal structure)
shuffled_acfs = []
for wid, ikis in writer_sequences.items():
    shuf = ikis.copy()
    np.random.shuffle(shuf)
    shuffled_acfs.append(acf(shuf))

shuffled_acfs = np.array(shuffled_acfs)
mean_shuffled_acf = shuffled_acfs.mean(axis=0)

print("\n  Shuffled (iid) mean ACF:")
for lag in range(MAX_LAG):
    print(f"    Lag {lag+1}: {mean_shuffled_acf[lag]:+.4f}")

# ---------- conditional entropy -------------------------------------------
print("\n[3/4] Computing conditional entropy...")

def marginal_entropy(ikis, q=QUANTIZATION_MS):
    quantized = (ikis // q).astype(int)
    _, counts = np.unique(quantized, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs + 1e-15))


def conditional_entropy(ikis, q=QUANTIZATION_MS):
    """H(IKI_n | IKI_{n-1}) via bigram counts."""
    quantized = (ikis // q).astype(int)
    bigrams = {}
    prev_counts = {}
    for i in range(1, len(quantized)):
        prev = quantized[i - 1]
        curr = quantized[i]
        bigrams[(prev, curr)] = bigrams.get((prev, curr), 0) + 1
        prev_counts[prev] = prev_counts.get(prev, 0) + 1

    total = sum(prev_counts.values())
    h_cond = 0.0
    for (prev, curr), count in bigrams.items():
        p_joint = count / total
        p_cond = count / prev_counts[prev]
        h_cond -= p_joint * np.log2(p_cond + 1e-15)

    return h_cond


genuine_marginal = []
genuine_conditional = []
shuffled_conditional = []

for wid, ikis in writer_sequences.items():
    h_m = marginal_entropy(ikis)
    h_c = conditional_entropy(ikis)
    genuine_marginal.append(h_m)
    genuine_conditional.append(h_c)

    # Shuffled version
    shuf = ikis.copy()
    np.random.shuffle(shuf)
    shuffled_conditional.append(conditional_entropy(shuf))

genuine_marginal = np.array(genuine_marginal)
genuine_conditional = np.array(genuine_conditional)
shuffled_conditional = np.array(shuffled_conditional)

# Temporal redundancy = H(X) - H(X|X_{-1})
temporal_redundancy = genuine_marginal - genuine_conditional
shuffled_redundancy = genuine_marginal - shuffled_conditional

print(f"  Marginal entropy H(IKI):          mean = {genuine_marginal.mean():.3f} bits")
print(f"  Conditional entropy H(IKI|prev):  mean = {genuine_conditional.mean():.3f} bits")
print(f"  Temporal redundancy (genuine):    mean = {temporal_redundancy.mean():.3f} bits")
print(f"  Temporal redundancy (shuffled):   mean = {shuffled_redundancy.mean():.3f} bits")
print(f"  Delta (genuine - shuffled):       {temporal_redundancy.mean() - shuffled_redundancy.mean():.3f} bits")

# ---------- permutation test (all 10 lags, Bonferroni-corrected) ----------
print("\n[4/4] Permutation test for temporal structure significance...")

# Subsample writers for permutation test (use same population for observed + null)
PERM_SUBSAMPLE = 100
all_wids = list(writer_sequences.items())
if len(all_wids) > PERM_SUBSAMPLE:
    subsample_idx = np.random.choice(len(all_wids), PERM_SUBSAMPLE, replace=False)
    subsample_wids = [all_wids[i] for i in subsample_idx]
else:
    subsample_wids = all_wids

print(f"  Running {N_PERMUTATIONS} permutations across {MAX_LAG} lags on {len(subsample_wids)} writers...")

# Observed mean ACF at each lag — computed from the SAME subsample as the null
subsample_acfs = np.array([acf(ikis) for _, ikis in subsample_wids])
observed_acfs = subsample_acfs.mean(axis=0)  # shape: (MAX_LAG,)

# Build null distribution for each lag
null_distributions = np.zeros((N_PERMUTATIONS, MAX_LAG))

for perm_idx in range(N_PERMUTATIONS):
    perm_acfs = []
    for wid, ikis in subsample_wids:
        shuf = ikis.copy()
        np.random.shuffle(shuf)
        perm_acfs.append(acf(shuf))  # all lags
    null_distributions[perm_idx, :] = np.mean(perm_acfs, axis=0)

# Raw p-values per lag (PY-H006: two-sided test using absolute values)
raw_p_values = np.zeros(MAX_LAG)
for lag_idx in range(MAX_LAG):
    p = float((np.abs(null_distributions[:, lag_idx]) >= np.abs(observed_acfs[lag_idx])).mean())
    if p == 0:
        p = 1.0 / (N_PERMUTATIONS + 1)  # upper bound
    raw_p_values[lag_idx] = p

# Bonferroni correction for 10 simultaneous tests
bonferroni_p_values = np.minimum(raw_p_values * MAX_LAG, 1.0)

print("\n  Per-lag results (Bonferroni-corrected for 10 tests):")
for lag_idx in range(MAX_LAG):
    sig = "***" if bonferroni_p_values[lag_idx] < 0.001 else \
          "**" if bonferroni_p_values[lag_idx] < 0.01 else \
          "*" if bonferroni_p_values[lag_idx] < 0.05 else "n.s."
    print(f"    Lag {lag_idx+1}: ACF = {observed_acfs[lag_idx]:+.4f}, "
          f"raw p = {raw_p_values[lag_idx]:.4f}, "
          f"corrected p = {bonferroni_p_values[lag_idx]:.4f} {sig}")

# Keep lag-1 values for backward compatibility
observed_lag1 = observed_acfs[0]
p_value = bonferroni_p_values[0]
null_lag1s = null_distributions[:, 0]

n_significant = int((bonferroni_p_values < 0.05).sum())
print(f"\n  {n_significant}/{MAX_LAG} lags significant after Bonferroni correction")

# ---------- save ----------------------------------------------------------
results = {
    "n_writers": len(writer_sequences),
    "n_writers_permutation_test": len(subsample_wids),
    "acf_genuine_mean": [round(float(x), 4) for x in mean_genuine_acf],
    "acf_shuffled_mean": [round(float(x), 4) for x in mean_shuffled_acf],
    "marginal_entropy_mean": round(float(genuine_marginal.mean()), 4),
    "conditional_entropy_mean": round(float(genuine_conditional.mean()), 4),
    "temporal_redundancy_genuine": round(float(temporal_redundancy.mean()), 4),
    "temporal_redundancy_shuffled": round(float(shuffled_redundancy.mean()), 4),
    "delta_entropy": round(float(temporal_redundancy.mean() - shuffled_redundancy.mean()), 4),
    "permutation_test": {
        "observed_lag1_acf": round(float(observed_lag1), 4),
        "null_mean": round(float(null_lag1s.mean()), 4),
        "null_std": round(float(null_lag1s.std()), 4),
        "p_value_lag1_corrected": round(float(bonferroni_p_values[0]), 4),
        "significant_lag1": bool(bonferroni_p_values[0] < 0.05),
        "n_permutations": N_PERMUTATIONS,
        "correction": "Bonferroni",
        "n_tests": MAX_LAG,
        "per_lag_raw_p": [round(float(p), 4) for p in raw_p_values],
        "per_lag_corrected_p": [round(float(p), 4) for p in bonferroni_p_values],
        "n_significant_lags": n_significant,
    },
    "interpretation": (
        f"Genuine sequences show temporal redundancy of {temporal_redundancy.mean():.2f} bits "
        f"vs {shuffled_redundancy.mean():.2f} bits for shuffled (iid). "
        f"Lag-1 ACF = {observed_lag1:.3f} (Bonferroni-corrected p = {bonferroni_p_values[0]:.4f}), "
        f"{n_significant}/{MAX_LAG} lags significant after Bonferroni correction. "
        "Confirms that iid retype attacks destroy detectable temporal structure."
    ),
}

out = Path(__file__).parent / "temporal_structure_results.json"
with open(out, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out}")

print("\n" + "=" * 70)
print("KEY NUMBERS FOR PAPER UPDATES")
print("=" * 70)
print(f"  Paper 09 C4: Temporal redundancy = {temporal_redundancy.mean():.2f} bits")
print(f"  Paper 09 C4: Lag-1 ACF = {observed_lag1:.3f} (Bonferroni p = {bonferroni_p_values[0]:.4f})")
print(f"  Papers 01/05: {n_significant}/{MAX_LAG} lags significant after Bonferroni correction")
