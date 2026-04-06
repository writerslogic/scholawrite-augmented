"""
KLiCKe Corpus Analysis for Process Attestation Papers
======================================================
Computes IKI distributions, behavioral entropy, Cognitive Load Correlation (CLC),
and composition vs. transcription discrimination metrics from the KLiCKe corpus
(~5,000 writers, argumentative essays with keystroke logs).

This provides a much larger-N validation than ScholaWrite (10 participants).
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

np.random.seed(42)

# ============================================================================
# 1. Load and preprocess KLiCKe keystroke logs
# ============================================================================
print("=" * 70)
print("KLiCKe Corpus Analysis for Process Attestation Papers")
print("=" * 70)

CSV_DIR = Path(__file__).parent / "klicke" / "Files" / "WritingTask" / "WritingTask" / "keystrokelogs" / "csv"
SCORES_FILE = Path(__file__).parent / "klicke" / "Files" / "WritingTask" / "WritingTask" / "holistic_scores.csv"

print(f"\n[1/7] Loading KLiCKe keystroke logs from {CSV_DIR}...")

all_rows = []
csv_files = sorted(CSV_DIR.glob("*.csv"))
n_files = len(csv_files)
print(f"  Found {n_files} writer files")

for i, f in enumerate(csv_files):
    if (i + 1) % 500 == 0:
        print(f"  Loading file {i+1}/{n_files}...")
    try:
        df = pd.read_csv(f)
        writer_id = f.stem
        df["writer_id"] = writer_id
        all_rows.append(df)
    except Exception as e:
        pass

df_all = pd.concat(all_rows, ignore_index=True)
print(f"  Total events loaded: {len(df_all):,}")
print(f"  Writers: {df_all['writer_id'].nunique():,}")
print(f"  Columns: {list(df_all.columns)}")

# Activity distribution
print(f"\n  Activity distribution:")
for act, count in df_all["Activity"].value_counts().items():
    print(f"    {act}: {count:,} ({100*count/len(df_all):.1f}%)")

# ============================================================================
# 2. Compute Inter-Keystroke Intervals (IKI)
# ============================================================================
print(f"\n[2/7] Computing Inter-Keystroke Intervals (IKI)...")

# IKI = DownTime[i+1] - DownTime[i] within each writer
df_all = df_all.sort_values(["writer_id", "DownTime"]).reset_index(drop=True)
df_all["iki_ms"] = df_all.groupby("writer_id")["DownTime"].diff()

# Explicit NaN handling: drop NaN values before any filtering (PY-C001)
IKI_LOWER_BOUND_MS = 10
IKI_UPPER_BOUND_MS = 60000

valid_iki = df_all["iki_ms"].dropna()
assert not valid_iki.isna().any(), "NaN values remain after dropna()"
valid_iki = valid_iki[(valid_iki > IKI_LOWER_BOUND_MS) & (valid_iki < IKI_UPPER_BOUND_MS)]
assert not valid_iki.isna().any(), "NaN values after IKI bounds filtering"
assert (valid_iki > IKI_LOWER_BOUND_MS).all(), "IKI below lower bound after filtering"
assert (valid_iki < IKI_UPPER_BOUND_MS).all(), "IKI above upper bound after filtering"

print(f"  Total valid IKIs: {len(valid_iki):,}")
print(f"  Mean IKI: {valid_iki.mean():.1f} ms")
print(f"  Median IKI: {valid_iki.median():.1f} ms")
print(f"  Std IKI: {valid_iki.std():.1f} ms")
print(f"  IKI range: [{valid_iki.min():.0f}, {valid_iki.max():.0f}] ms")

# Per-writer IKI statistics: consistent NaN handling and bounds (PY-C001)
writer_iki_stats = df_all.dropna(subset=["iki_ms"]).copy()
writer_iki_stats = writer_iki_stats[
    (writer_iki_stats["iki_ms"] > IKI_LOWER_BOUND_MS) & (writer_iki_stats["iki_ms"] < IKI_UPPER_BOUND_MS)
]
assert not writer_iki_stats["iki_ms"].isna().any(), "NaN in writer_iki_stats after filtering"
per_writer = writer_iki_stats.groupby("writer_id")["iki_ms"].agg(["mean", "median", "std", "count"])
print(f"\n  Per-writer IKI means: mean={per_writer['mean'].mean():.1f}, median={per_writer['mean'].median():.1f}")
print(f"  Per-writer IKI medians: mean={per_writer['median'].mean():.1f}, median={per_writer['median'].median():.1f}")

# ============================================================================
# 3. Behavioral Entropy Analysis
# ============================================================================
print(f"\n[3/7] Computing Behavioral Entropy...")

def compute_entropy_bits(iki_values, quantization_ms=5):
    """Compute Shannon entropy of quantized IKI distribution in bits."""
    quantized = (iki_values // quantization_ms) * quantization_ms
    value_counts = quantized.value_counts(normalize=True)
    probs = value_counts.values
    entropy = -np.sum(probs * np.log2(probs + 1e-15))
    return entropy

QUANTIZATION_MS = 5
WINDOW_SIZE_MS = 30000  # 30 seconds

# Population-level entropy
overall_entropy = compute_entropy_bits(valid_iki, QUANTIZATION_MS)
print(f"  Per-IKI entropy (population): {overall_entropy:.2f} bits")
print(f"  Accumulated per checkpoint (50 keys): {50 * overall_entropy:.1f} bits")
print(f"  Paper claim: >=205 bits -- {'VALIDATED' if 50 * overall_entropy >= 205 else 'NOT MET'}")

# Per-writer entropy
writer_entropies = []
for wid, group in writer_iki_stats.groupby("writer_id"):
    if len(group) >= 30:
        ent = compute_entropy_bits(group["iki_ms"], QUANTIZATION_MS)
        writer_entropies.append({"writer_id": wid, "entropy_bits": ent, "n": len(group)})

df_went = pd.DataFrame(writer_entropies)
print(f"\n  Per-writer entropy (n={len(df_went)} writers with >=30 keystrokes):")
print(f"    Mean: {df_went['entropy_bits'].mean():.2f} bits")
print(f"    Median: {df_went['entropy_bits'].median():.2f} bits")
print(f"    Std: {df_went['entropy_bits'].std():.2f} bits")
print(f"    Min: {df_went['entropy_bits'].min():.2f}, Max: {df_went['entropy_bits'].max():.2f}")
print(f"    Writers above 3.0 bits: {(df_went['entropy_bits'] > 3.0).sum()}/{len(df_went)} "
      f"({100*(df_went['entropy_bits'] > 3.0).mean():.1f}%)")

# 30-second window entropy
window_entropies = []
for wid, group in writer_iki_stats.groupby("writer_id"):
    group = group.sort_values("DownTime")
    timestamps = group["DownTime"].values
    ikis = group["iki_ms"].values

    if len(timestamps) < 10:
        continue

    t_start = timestamps[0]
    while t_start < timestamps[-1]:
        t_end = t_start + WINDOW_SIZE_MS
        mask = (timestamps >= t_start) & (timestamps < t_end)
        window_ikis = ikis[mask]
        window_ikis = window_ikis[~np.isnan(window_ikis)]
        window_ikis = window_ikis[(window_ikis > 10) & (window_ikis < 60000)]

        if len(window_ikis) >= 10:
            entropy = compute_entropy_bits(pd.Series(window_ikis), QUANTIZATION_MS)
            window_entropies.append({
                "writer_id": wid,
                "entropy_bits": entropy,
                "n_keystrokes": len(window_ikis),
            })
        t_start = t_end

df_winent = pd.DataFrame(window_entropies)
print(f"\n  30s window entropy (n={len(df_winent):,} windows):")
print(f"    Mean: {df_winent['entropy_bits'].mean():.2f} bits")
print(f"    Median: {df_winent['entropy_bits'].median():.2f} bits")
print(f"    Windows above 3.0-bit threshold: {(df_winent['entropy_bits'] > 3.0).sum()}/{len(df_winent)} "
      f"({100*(df_winent['entropy_bits'] > 3.0).mean():.1f}%)")

# ============================================================================
# 4. Cognitive Load Correlation (CLC)
# ============================================================================
print(f"\n[4/7] Computing Cognitive Load Correlation (CLC)...")

# Map Activity to cognitive load levels:
# Nonproduction (pausing/planning) = 3 (high cognitive load)
# Remove/Cut (revision) = 2 (medium)
# Input (text production) = 1 (low)
COGNITIVE_LOAD = {
    "Nonproduction": 3,
    "Remove/Cut": 2,
    "Input": 1,
}

df_all["cog_load"] = df_all["Activity"].map(COGNITIVE_LOAD).fillna(1)

# Overall CLC
all_valid = df_all.dropna(subset=["iki_ms"]).copy()
all_valid = all_valid[(all_valid["iki_ms"] > 10) & (all_valid["iki_ms"] < 60000)]
# Filter to events with known cognitive load
all_valid_clc = all_valid[all_valid["cog_load"].notna()]

overall_rho, overall_pval = stats.spearmanr(all_valid_clc["cog_load"], all_valid_clc["iki_ms"])
print(f"  Overall CLC (all data): rho = {overall_rho:.4f}, p = {overall_pval:.2e}")

# Per-writer CLC
writer_clcs = []
for wid, group in all_valid_clc.groupby("writer_id"):
    if len(group) >= 30 and group["cog_load"].std() > 0:
        try:
            rho, pval = stats.spearmanr(group["cog_load"], group["iki_ms"])
            if not np.isnan(rho):
                writer_clcs.append({"writer_id": wid, "rho": rho, "pval": pval, "n": len(group)})
        except Exception:
            pass

df_wclc = pd.DataFrame(writer_clcs)
print(f"\n  Per-writer CLC (n={len(df_wclc)} writers):")
print(f"    Mean rho: {df_wclc['rho'].mean():.4f}")
print(f"    Median rho: {df_wclc['rho'].median():.4f}")
print(f"    Std rho: {df_wclc['rho'].std():.4f}")
print(f"    Writers with rho > 0.15: {(df_wclc['rho'] > 0.15).sum()}/{len(df_wclc)} "
      f"({100*(df_wclc['rho'] > 0.15).mean():.1f}%)")
print(f"    Writers with rho > 0.0 (positive): {(df_wclc['rho'] > 0.0).sum()}/{len(df_wclc)} "
      f"({100*(df_wclc['rho'] > 0.0).mean():.1f}%)")
print(f"    Writers with significant CLC (p < 0.05): {(df_wclc['pval'] < 0.05).sum()}/{len(df_wclc)} "
      f"({100*(df_wclc['pval'] < 0.05).mean():.1f}%)")

# Windowed CLC (30-second windows)
print(f"\n  Windowed CLC (30s windows):")
clc_values = []
for wid, group in all_valid_clc.groupby("writer_id"):
    group = group.sort_values("DownTime")
    timestamps = group["DownTime"].values
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

        valid = ~np.isnan(window_ikis) & (window_ikis > 10) & (window_ikis < 60000)
        window_ikis = window_ikis[valid]
        window_cog = window_cog[valid]

        if len(window_ikis) >= 10 and np.std(window_cog) > 0:
            try:
                rho, pval = stats.spearmanr(window_cog, window_ikis)
                if not np.isnan(rho):
                    clc_values.append({"writer_id": wid, "rho": rho, "pval": pval, "n": len(window_ikis)})
            except Exception:
                pass
        t_start = t_end

df_clc = pd.DataFrame(clc_values)
print(f"    Total windows: {len(df_clc):,}")
print(f"    Mean rho: {df_clc['rho'].mean():.4f}")
print(f"    Median rho: {df_clc['rho'].median():.4f}")
print(f"    Windows with rho > 0.15: {(df_clc['rho'] > 0.15).sum()}/{len(df_clc)} "
      f"({100*(df_clc['rho'] > 0.15).mean():.1f}%)")

# ============================================================================
# 5. IKI by Activity Type (Composition vs Transcription proxy)
# ============================================================================
print(f"\n[5/7] IKI Distribution by Activity Type...")

for act in ["Input", "Remove/Cut", "Nonproduction"]:
    subset = all_valid[all_valid["Activity"] == act]["iki_ms"]
    if len(subset) > 0:
        print(f"  {act:20s}: n={len(subset):>9,}, mean={subset.mean():>8.1f}ms, "
              f"median={subset.median():>7.1f}ms, std={subset.std():>8.1f}ms")

# Demanding (Nonproduction + Remove/Cut) vs Simple (Input)
demanding = all_valid[all_valid["Activity"].isin(["Nonproduction", "Remove/Cut"])]["iki_ms"]
simple = all_valid[all_valid["Activity"] == "Input"]["iki_ms"]

ratio = None
cohens_d = None
if len(demanding) > 0 and len(simple) > 0:
    ratio = demanding.mean() / simple.mean()
    d_stat, d_pval = stats.mannwhitneyu(demanding, simple, alternative="greater")
    cohens_d = (demanding.mean() - simple.mean()) / np.sqrt((demanding.std()**2 + simple.std()**2) / 2)

    print(f"\n  Demanding/Simple IKI ratio: {ratio:.2f}x")
    print(f"  Mann-Whitney p: {d_pval:.2e}")
    print(f"  Cohen's d: {cohens_d:.3f}")

# ============================================================================
# 6. Composition vs. Transcription Window Discrimination
# ============================================================================
print(f"\n[6/7] Composition vs. Transcription Discrimination...")

high_load_windows = df_clc[df_clc["rho"] > 0.15]
low_load_windows = df_clc[df_clc["rho"] <= 0.0]

print(f"  Composition-like windows (rho > 0.15): {len(high_load_windows):,}")
print(f"  Transcription-like windows (rho <= 0.0): {len(low_load_windows):,}")

if len(high_load_windows) > 0 and len(low_load_windows) > 0:
    mean_comp = high_load_windows["rho"].mean()
    mean_trans = low_load_windows["rho"].mean()
    pooled_std = np.sqrt((high_load_windows["rho"].std()**2 + low_load_windows["rho"].std()**2) / 2)
    cohens_d_disc = (mean_comp - mean_trans) / pooled_std if pooled_std > 0 else 0

    print(f"  Mean rho (composition): {mean_comp:.4f}")
    print(f"  Mean rho (transcription): {mean_trans:.4f}")
    print(f"  Cohen's d (discrimination): {cohens_d_disc:.3f}")

    u_stat, u_pval = stats.mannwhitneyu(
        high_load_windows["rho"], low_load_windows["rho"], alternative="greater"
    )
    print(f"  Mann-Whitney U: {u_stat:.1f}, p = {u_pval:.2e}")

# ============================================================================
# 7. Quality Score Correlation
# ============================================================================
print(f"\n[7/7] Quality Score Correlation with Writing Behavior...")

scores = pd.read_csv(SCORES_FILE)
scores["writer_id"] = scores["ID"].astype(str)

# Merge with per-writer metrics
per_writer_metrics = writer_iki_stats.groupby("writer_id").agg(
    mean_iki=("iki_ms", "mean"),
    median_iki=("iki_ms", "median"),
    std_iki=("iki_ms", "std"),
    n_events=("iki_ms", "count"),
).reset_index()

# Merge CLC
if len(df_wclc) > 0:
    per_writer_metrics = per_writer_metrics.merge(
        df_wclc[["writer_id", "rho"]].rename(columns={"rho": "clc_rho"}),
        on="writer_id", how="left"
    )

# Merge entropy
if len(df_went) > 0:
    per_writer_metrics = per_writer_metrics.merge(
        df_went[["writer_id", "entropy_bits"]],
        on="writer_id", how="left"
    )

# Merge scores
merged = per_writer_metrics.merge(scores[["writer_id", "Score"]], on="writer_id", how="inner")
print(f"  Writers with both metrics and scores: {len(merged):,}")

if len(merged) > 10:
    for col in ["mean_iki", "std_iki", "clc_rho", "entropy_bits"]:
        if col in merged.columns:
            valid = merged.dropna(subset=[col, "Score"])
            if len(valid) > 10:
                rho, pval = stats.spearmanr(valid[col], valid["Score"])
                print(f"  {col:15s} vs Score: rho = {rho:.4f}, p = {pval:.2e} (n={len(valid)})")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY: KLiCKe Corpus Results for Paper Claims")
print("=" * 70)

results = {
    "dataset": "KLiCKe Corpus",
    "n_writers": int(df_all["writer_id"].nunique()),
    "total_events": len(df_all),
    "valid_ikis": len(valid_iki),
    "mean_iki_ms": float(valid_iki.mean()),
    "median_iki_ms": float(valid_iki.median()),
    "std_iki_ms": float(valid_iki.std()),
    "entropy_per_iki_bits": float(overall_entropy),
    "entropy_per_checkpoint_bits": float(50 * overall_entropy),
    "entropy_claim_205_bits": "VALIDATED" if 50 * overall_entropy >= 205 else "NOT MET",
    "clc_overall_rho": float(overall_rho),
    "clc_overall_pval": float(overall_pval),
    "clc_per_writer_mean_rho": float(df_wclc["rho"].mean()) if len(df_wclc) > 0 else None,
    "clc_per_writer_median_rho": float(df_wclc["rho"].median()) if len(df_wclc) > 0 else None,
    "pct_writers_positive_clc": float(100 * (df_wclc["rho"] > 0.0).mean()) if len(df_wclc) > 0 else None,
    "pct_writers_significant_clc": float(100 * (df_wclc["pval"] < 0.05).mean()) if len(df_wclc) > 0 else None,
    "demanding_simple_ratio": float(ratio) if len(demanding) > 0 and len(simple) > 0 else None,
    "cohens_d_activity": float(cohens_d) if len(demanding) > 0 and len(simple) > 0 else None,
}

print(f"\n  Key Findings:")
print(f"  - N = {results['n_writers']:,} writers, {results['valid_ikis']:,} valid IKIs")
print(f"  - Mean IKI: {results['mean_iki_ms']:.1f} ms (median: {results['median_iki_ms']:.1f} ms)")
print(f"  - Entropy per IKI: {results['entropy_per_iki_bits']:.2f} bits")
print(f"  - Entropy per checkpoint (50 keys): {results['entropy_per_checkpoint_bits']:.1f} bits  [{results['entropy_claim_205_bits']}]")
print(f"  - Overall CLC rho: {results['clc_overall_rho']:.4f}")
if results.get('clc_per_writer_mean_rho'):
    print(f"  - Per-writer CLC rho: mean={results['clc_per_writer_mean_rho']:.4f}, median={results['clc_per_writer_median_rho']:.4f}")
if results.get('pct_writers_positive_clc'):
    print(f"  - Writers with positive CLC: {results['pct_writers_positive_clc']:.1f}%")
if results.get('demanding_simple_ratio'):
    print(f"  - Demanding/Simple IKI ratio: {results['demanding_simple_ratio']:.2f}x")
if results.get('cohens_d_activity'):
    print(f"  - Cohen's d (activity type): {results['cohens_d_activity']:.3f}")

# Save results
output_path = Path(__file__).parent / "klicke_results.json"
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {output_path}")
