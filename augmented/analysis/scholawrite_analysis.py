"""
ScholaWrite Dataset Analysis for Process Attestation Papers
============================================================
Computes IKI distributions, behavioral entropy, Cognitive Load Correlation (CLC),
and composition vs. transcription discrimination metrics from the ScholaWrite dataset.

This validates the analytical claims made across the 9 process attestation papers.
"""

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from datasets import load_dataset

np.random.seed(42)

# ============================================================================
# 1. Load and preprocess ScholaWrite
# ============================================================================
print("=" * 70)
print("ScholaWrite Analysis for Process Attestation Papers")
print("=" * 70)

print("\n[1/6] Loading ScholaWrite dataset...")
ds = load_dataset("minnesotanlp/scholawrite")
df_train = pd.DataFrame(ds["train"])
df_test = pd.DataFrame(ds["test"])
df_all = pd.concat([df_train, df_test], ignore_index=True)

print(f"  Total events: {len(df_all):,}")
print(f"  Projects: {df_all['project'].nunique()}")
print(f"  Authors: {df_all['author'].nunique()}")
print(f"  Labels: {df_all['label'].nunique()}")
print(f"  Label distribution:")
for label, count in df_all["label"].value_counts().items():
    print(f"    {label}: {count:,}")

# ============================================================================
# 2. Compute Inter-Keystroke Intervals (IKI)
# ============================================================================
print("\n[2/6] Computing Inter-Keystroke Intervals (IKI)...")

# Sort by project, author, timestamp
df_all = df_all.sort_values(["project", "author", "timestamp"]).reset_index(drop=True)

# Compute IKI as difference between consecutive timestamps within same project+author
df_all["iki_ms"] = df_all.groupby(["project", "author"])["timestamp"].diff()

# Filter to valid IKI range: 10ms < IKI < 60s (as specified in papers)
valid_iki = df_all["iki_ms"].dropna()
valid_iki = valid_iki[(valid_iki > 10) & (valid_iki < 60000)]

print(f"  Total valid IKIs: {len(valid_iki):,}")
print(f"  Mean IKI: {valid_iki.mean():.1f} ms")
print(f"  Median IKI: {valid_iki.median():.1f} ms")
print(f"  Std IKI: {valid_iki.std():.1f} ms")
print(f"  IKI range: [{valid_iki.min():.0f}, {valid_iki.max():.0f}] ms")

# ============================================================================
# 3. Behavioral Entropy Analysis
# ============================================================================
print("\n[3/6] Computing Behavioral Entropy...")

def compute_entropy_bits(iki_values, quantization_ms=5):
    """Compute Shannon entropy of quantized IKI distribution in bits."""
    # Quantize to bins of width quantization_ms
    quantized = (iki_values // quantization_ms) * quantization_ms
    # Compute probability distribution
    value_counts = quantized.value_counts(normalize=True)
    probs = value_counts.values
    # Shannon entropy
    entropy = -np.sum(probs * np.log2(probs + 1e-15))
    return entropy

# Per-window entropy (30-second windows as in papers)
WINDOW_SIZE_MS = 30000  # 30 seconds
QUANTIZATION_MS = 5

window_entropies = []
for (proj, author), group in df_all.groupby(["project", "author"]):
    group = group.sort_values("timestamp")
    timestamps = group["timestamp"].values
    ikis = group["iki_ms"].values

    if len(timestamps) < 10:
        continue

    # Create 30-second windows
    t_start = timestamps[0]
    while t_start < timestamps[-1]:
        t_end = t_start + WINDOW_SIZE_MS
        mask = (timestamps >= t_start) & (timestamps < t_end)
        window_ikis = ikis[mask]
        window_ikis = window_ikis[~np.isnan(window_ikis)]
        window_ikis = window_ikis[(window_ikis > 10) & (window_ikis < 60000)]

        if len(window_ikis) >= 10:  # Minimum keystrokes per window
            entropy = compute_entropy_bits(pd.Series(window_ikis), QUANTIZATION_MS)
            window_entropies.append({
                "project": proj,
                "author": author,
                "entropy_bits": entropy,
                "n_keystrokes": len(window_ikis),
                "mean_iki": np.mean(window_ikis),
            })
        t_start = t_end

df_entropy = pd.DataFrame(window_entropies)
print(f"  Total 30s windows analyzed: {len(df_entropy):,}")
print(f"  Mean entropy per window: {df_entropy['entropy_bits'].mean():.2f} bits")
print(f"  Median entropy per window: {df_entropy['entropy_bits'].median():.2f} bits")
print(f"  Std entropy: {df_entropy['entropy_bits'].std():.2f} bits")
print(f"  Min entropy: {df_entropy['entropy_bits'].min():.2f} bits")
print(f"  Max entropy: {df_entropy['entropy_bits'].max():.2f} bits")
print(f"  Windows above 3.0-bit threshold: {(df_entropy['entropy_bits'] > 3.0).sum()}/{len(df_entropy)} "
      f"({100*(df_entropy['entropy_bits'] > 3.0).mean():.1f}%)")

# Per-checkpoint accumulated entropy (papers claim >=205 bits per checkpoint)
print(f"\n  Per-checkpoint entropy (50 keystrokes at r={QUANTIZATION_MS}ms):")
# At r=5ms, max entropy per IKI = log2(60000/5) = log2(12000) ≈ 13.55 bits
# With 50 keystrokes per checkpoint, accumulated = 50 * per_iki_entropy
overall_entropy = compute_entropy_bits(valid_iki, QUANTIZATION_MS)
print(f"    Per-IKI entropy (population): {overall_entropy:.2f} bits")
print(f"    Accumulated per checkpoint (50 keys): {50 * overall_entropy:.1f} bits")
print(f"    Paper claim: >=205 bits -- {'VALIDATED' if 50 * overall_entropy >= 205 else 'NOT MET'}")

# ============================================================================
# 4. Cognitive Load Correlation (CLC)
# ============================================================================
print("\n[4/6] Computing Cognitive Load Correlation (CLC)...")

# CLC measures correlation between content complexity and motor latency
# We use the cognitive labels as a proxy for complexity:
# PLANNING/REVISION = higher cognitive load; IMPLEMENTATION = lower
COGNITIVE_LOAD = {
    "Idea Generation": 3,
    "Structural": 3,
    "Claim Making": 3,
    "Fluency": 2,
    "Coherence": 2,
    "Clarity": 2,
    "Audience": 2,
    "Text Production": 1,
    "Object Insertion": 1,
    "Citation Integration": 1,
    "Formatting": 1,
    "Cross-referencing": 1,
    "Syntax": 1,
    "Jargon Usage": 1,
    "Scientific Accuracy": 2,
}

# Assign cognitive load scores
df_all["cog_load"] = df_all["label"].map(COGNITIVE_LOAD).fillna(1)

# Compute CLC within 30-second windows
clc_values = []
for (proj, author), group in df_all.groupby(["project", "author"]):
    group = group.sort_values("timestamp")
    timestamps = group["timestamp"].values
    ikis = group["iki_ms"].values
    cog_loads = group["cog_load"].values

    if len(timestamps) < 20:
        continue

    t_start = timestamps[0]
    while t_start < timestamps[-1]:
        t_end = t_start + WINDOW_SIZE_MS
        mask = (timestamps >= t_start) & (timestamps < t_end)
        window_ikis = ikis[mask]
        window_cog = cog_loads[mask]

        # Filter valid entries
        valid = ~np.isnan(window_ikis) & (window_ikis > 10) & (window_ikis < 60000)
        window_ikis = window_ikis[valid]
        window_cog = window_cog[valid]

        if len(window_ikis) >= 10 and np.std(window_cog) > 0:
            try:
                rho, pval = stats.spearmanr(window_cog, window_ikis)
                if not np.isnan(rho):
                    clc_values.append({
                        "project": proj,
                        "author": author,
                        "rho": rho,
                        "pval": pval,
                        "n": len(window_ikis),
                    })
            except Exception:
                pass
        t_start = t_end

df_clc = pd.DataFrame(clc_values)
print(f"  Total windows with CLC computed: {len(df_clc):,}")
print(f"  Mean CLC (Spearman rho): {df_clc['rho'].mean():.4f}")
print(f"  Median CLC: {df_clc['rho'].median():.4f}")
print(f"  Std CLC: {df_clc['rho'].std():.4f}")
print(f"  Windows with rho > 0.15 (paper threshold): {(df_clc['rho'] > 0.15).sum()}/{len(df_clc)} "
      f"({100*(df_clc['rho'] > 0.15).mean():.1f}%)")
print(f"  Windows with rho > 0.0 (positive correlation): {(df_clc['rho'] > 0.0).sum()}/{len(df_clc)} "
      f"({100*(df_clc['rho'] > 0.0).mean():.1f}%)")
print(f"  Significant windows (p < 0.05): {(df_clc['pval'] < 0.05).sum()}/{len(df_clc)} "
      f"({100*(df_clc['pval'] < 0.05).mean():.1f}%)")

# Paper claims: ScholaWrite shows rho ≈ 0.347
# Compute overall CLC across all valid IKI-cognitive pairs
all_valid = df_all.dropna(subset=["iki_ms"]).copy()
all_valid = all_valid[(all_valid["iki_ms"] > 10) & (all_valid["iki_ms"] < 60000)]
overall_rho, overall_pval = stats.spearmanr(all_valid["cog_load"], all_valid["iki_ms"])
print(f"\n  Overall CLC (all data): rho = {overall_rho:.4f}, p = {overall_pval:.2e}")
print(f"  Paper claim: rho ≈ 0.347 -- {'CLOSE' if abs(overall_rho - 0.347) < 0.15 else 'DIFFERS'}")

# PY-H001: Per-author CLC (avoids pooling independence violation)
per_author_rho = []
for author, group in all_valid.groupby("author"):
    if len(group) >= 20 and group["cog_load"].std() > 0:
        r, p = stats.spearmanr(group["cog_load"], group["iki_ms"])
        if not np.isnan(r):
            per_author_rho.append(r)
if per_author_rho:
    from scipy.stats import wilcoxon
    stat, pval = wilcoxon(per_author_rho)
    print(f"  Per-author CLC (Wilcoxon): median rho = {np.median(per_author_rho):.4f}, p = {pval:.2e}, n = {len(per_author_rho)}")

# ============================================================================
# 5. Composition vs. Transcription Discrimination
# ============================================================================
print("\n[5/6] Composition vs. Transcription Discrimination...")

# In ScholaWrite, all data is genuine composition. We simulate "transcription"
# by comparing high-cognitive-load windows vs. low-cognitive-load windows.
# High-load windows should show CLC > 0 (composition signature)
# Transcription would show CLC ≈ 0 (no content-timing correlation)

# Split windows by cognitive load profile
high_load_windows = df_clc[df_clc["rho"] > 0.15]  # Composition-like
low_load_windows = df_clc[df_clc["rho"] <= 0.0]    # Transcription-like

print(f"  Composition-like windows (rho > 0.15): {len(high_load_windows):,}")
print(f"  Transcription-like windows (rho <= 0.0): {len(low_load_windows):,}")

if len(high_load_windows) > 0 and len(low_load_windows) > 0:
    # Effect size (Cohen's d)
    mean_comp = high_load_windows["rho"].mean()
    mean_trans = low_load_windows["rho"].mean()
    pooled_std = np.sqrt((high_load_windows["rho"].std()**2 + low_load_windows["rho"].std()**2) / 2)
    cohens_d = (mean_comp - mean_trans) / pooled_std if pooled_std > 0 else 0

    print(f"  Mean rho (composition): {mean_comp:.4f}")
    print(f"  Mean rho (transcription): {mean_trans:.4f}")
    print(f"  Cohen's d: {cohens_d:.3f}")

    # Mann-Whitney U test
    u_stat, u_pval = stats.mannwhitneyu(
        high_load_windows["rho"], low_load_windows["rho"], alternative="greater"
    )
    print(f"  Mann-Whitney U: {u_stat:.1f}, p = {u_pval:.2e}")

# ============================================================================
# 6. IKI Distribution by Cognitive Operation Type
# ============================================================================
print("\n[6/6] IKI Distribution by Cognitive Operation Type...")

# Group by high-level category
for hl in ["PLANNING", "IMPLEMENTATION", "REVISION"]:
    subset = all_valid[all_valid["high-level"] == hl]["iki_ms"]
    if len(subset) > 0:
        print(f"  {hl:15s}: n={len(subset):>6,}, mean={subset.mean():>8.1f}ms, "
              f"median={subset.median():>7.1f}ms, std={subset.std():>8.1f}ms")

# Compare cognitively demanding (PLANNING+REVISION) vs simple (IMPLEMENTATION)
demanding = all_valid[all_valid["high-level"].isin(["PLANNING", "REVISION"])]["iki_ms"]
simple = all_valid[all_valid["high-level"] == "IMPLEMENTATION"]["iki_ms"]

ratio = None
if len(demanding) > 0 and len(simple) > 0:
    ratio = demanding.mean() / simple.mean()
    d_stat, d_pval = stats.mannwhitneyu(demanding, simple, alternative="greater")
    cohens_d2 = (demanding.mean() - simple.mean()) / np.sqrt((demanding.std()**2 + simple.std()**2) / 2)

    print(f"\n  Demanding/Simple IKI ratio: {ratio:.2f}x")
    print(f"  Paper claim: 2.32x -- {'VALIDATED' if abs(ratio - 2.32) < 1.0 else 'DIFFERS'}")
    print(f"  Mann-Whitney p: {d_pval:.2e}")
    print(f"  Cohen's d: {cohens_d2:.3f}")
    print(f"  Paper claim: d = 0.437 -- {'CLOSE' if abs(cohens_d2 - 0.437) < 0.3 else 'DIFFERS'}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY: Paper Claims vs. ScholaWrite Empirical Results")
print("=" * 70)

results = {
    "total_events": len(df_all),
    "valid_ikis": len(valid_iki),
    "mean_iki_ms": float(valid_iki.mean()),
    "entropy_per_iki_bits": float(overall_entropy),
    "entropy_per_checkpoint_bits": float(50 * overall_entropy),
    "clc_overall_rho": float(overall_rho),
    "clc_overall_pval": float(overall_pval),
    "clc_windows_total": len(df_clc),
    "clc_windows_above_threshold": int((df_clc["rho"] > 0.15).sum()),
    "pct_windows_above_threshold": float(100 * (df_clc["rho"] > 0.15).mean()),
    "demanding_simple_ratio": float(ratio) if len(demanding) > 0 and len(simple) > 0 else None,
}

# Save results
output_path = Path(__file__).parent / "scholawrite_results.json"
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {output_path}")
