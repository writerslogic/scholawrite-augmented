"""
Demographic Subgroup Analysis on KLiCKE Corpus
================================================
Tests whether cross-domain independence and CLC results hold across
demographic populations (native vs non-native, age groups, writing strength).

Output: demographic_results.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

np.random.seed(42)

# ---------- paths -----------------------------------------------------------
BASE = Path(__file__).parent
CSV_DIR = BASE / "klicke" / "Files" / "WritingTask" / "WritingTask" / "keystrokelogs" / "csv"
DEMO_PATH = BASE / "klicke" / "Files" / "demographic_info.csv"
if not DEMO_PATH.exists():
    raise FileNotFoundError(f"Demographics file not found: {DEMO_PATH}")
QUANTIZATION_MS = 5
MIN_EVENTS = 50

print("=" * 70)
print("Demographic Subgroup Analysis")
print("=" * 70)

# ---------- load demographics -----------------------------------------------
print("\n[1/5] Loading demographics...")
demo = pd.read_csv(DEMO_PATH)

# Parse nativeness to binary
demo["is_native"] = demo["Nativeness"].str.contains("native English speaker", case=False, na=False)

# Parse writing strength to numeric
def parse_strength(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if s.startswith("1"):
        return 1
    if s.startswith("5"):
        return 5
    try:
        return int(s)
    except ValueError:
        return np.nan

demo["strength_num"] = demo["Writing_Strength"].apply(parse_strength)

# Age groups
demo["age_group"] = pd.cut(
    demo["Age"],
    bins=[0, 25, 40, 200],
    labels=["<25", "25-40", "40+"],
    right=True,
)

# Strength groups
demo["strength_group"] = demo["strength_num"].map(
    lambda x: "low" if x in (1, 2) else ("high" if x in (4, 5) else "mid") if not np.isnan(x) else np.nan
)

# ID as string for joining
demo["writer_id"] = demo["ID"].astype(str)

print(f"  Total demographics: {len(demo)}")
print(f"  Native: {demo['is_native'].sum()}, Non-native: {(~demo['is_native']).sum()}")
print(f"  Age groups: {demo['age_group'].value_counts().to_dict()}")
print(f"  Strength groups: {demo['strength_group'].value_counts().to_dict()}")

# ---------- compute per-writer features -------------------------------------
print("\n[2/5] Computing per-writer features...")

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
print(f"  Writers with complete features: {len(df_feat)}")

# ---------- join features with demographics ---------------------------------
print("\n[3/5] Joining features with demographics...")
n_before = len(df_feat)
merged = df_feat.merge(demo[["writer_id", "is_native", "Age", "age_group", "strength_num", "strength_group"]],
                       on="writer_id", how="inner")
n_after = len(merged)
n_dropped = n_before - n_after
print(f"  Matched writers: {n_after}")
print(f"  Unmatched feature writers: {n_dropped}")
if n_dropped > 0:
    print(f"  WARNING: {n_dropped} writers ({100*n_dropped/n_before:.1f}%) dropped due to missing demographics")

# ---------- subgroup analysis -----------------------------------------------
print("\n[4/5] Running subgroup analyses...")

def subgroup_stats(sub_df, label):
    """Compute standard statistics for a subgroup."""
    n = len(sub_df)
    if n < 5:
        return {"n": n, "skipped": True, "reason": "too few writers"}

    result = {
        "n": n,
        "entropy_mean": round(float(sub_df["entropy"].mean()), 4),
        "entropy_std": round(float(sub_df["entropy"].std()), 4),
        "clc_median": round(float(sub_df["clc_rho"].median()), 4),
        "clc_mean": round(float(sub_df["clc_rho"].mean()), 4),
        "clc_positive_pct": round(float((sub_df["clc_rho"] > 0).mean() * 100), 1),
        "pause_freq_mean": round(float(sub_df["pause_freq"].mean()), 4),
        "iki_log_var_mean": round(float(sub_df["iki_log_var"].mean()), 4),
    }

    # Correlations within subgroup
    if n >= 10:
        rho_ec, p_ec = stats.spearmanr(sub_df["entropy"], sub_df["clc_rho"])
        rho_ep, p_ep = stats.spearmanr(sub_df["entropy"], sub_df["pause_freq"])
        result["corr_entropy_clc"] = {"rho": round(float(rho_ec), 4), "p": float(f"{p_ec:.4e}")}
        result["corr_entropy_pause"] = {"rho": round(float(rho_ep), 4), "p": float(f"{p_ep:.4e}")}
        result["independence_holds"] = abs(rho_ec) < 0.1
    else:
        result["corr_entropy_clc"] = None
        result["corr_entropy_pause"] = None
        result["independence_holds"] = None

    return result


def mann_whitney_compare(group_a, group_b, feature):
    """Mann-Whitney U between two groups on a feature."""
    a = group_a[feature].dropna()
    b = group_b[feature].dropna()
    if len(a) < 5 or len(b) < 5:
        return {"U": None, "p": None, "skipped": True}
    u_stat, p_val = stats.mannwhitneyu(a, b, alternative="two-sided")
    # Effect size: rank-biserial r = 1 - 2U/(n1*n2)
    r_effect = 1 - (2 * u_stat) / (len(a) * len(b))
    return {
        "U": float(u_stat),
        "p": float(f"{p_val:.4e}"),
        "effect_size_r": round(float(r_effect), 4),
        "n_a": len(a),
        "n_b": len(b),
    }


results = {"n_total_matched": len(merged), "subgroups": {}}

# --- 1. Native vs Non-native ---
print("\n  --- Native vs Non-native ---")
native = merged[merged["is_native"]]
nonnative = merged[~merged["is_native"]]

native_stats = subgroup_stats(native, "native")
nonnative_stats = subgroup_stats(nonnative, "non-native")

print(f"    Native (n={native_stats['n']}):     entropy={native_stats.get('entropy_mean','N/A')}, "
      f"CLC median={native_stats.get('clc_median','N/A')}, CLC+={native_stats.get('clc_positive_pct','N/A')}%")
print(f"    Non-native (n={nonnative_stats['n']}): entropy={nonnative_stats.get('entropy_mean','N/A')}, "
      f"CLC median={nonnative_stats.get('clc_median','N/A')}, CLC+={nonnative_stats.get('clc_positive_pct','N/A')}%")

nativeness_comparisons = {}
for feat in ["entropy", "clc_rho", "pause_freq", "iki_log_var"]:
    mw = mann_whitney_compare(native, nonnative, feat)
    nativeness_comparisons[feat] = mw
    print(f"    MW-U {feat}: p={mw.get('p','N/A')}, effect_r={mw.get('effect_size_r','N/A')}")

if native_stats.get("corr_entropy_clc"):
    print(f"    Native entropy-CLC rho={native_stats['corr_entropy_clc']['rho']}, independence={'HOLDS' if native_stats['independence_holds'] else 'FAILS'}")
if nonnative_stats.get("corr_entropy_clc"):
    print(f"    Non-native entropy-CLC rho={nonnative_stats['corr_entropy_clc']['rho']}, independence={'HOLDS' if nonnative_stats['independence_holds'] else 'FAILS'}")

results["subgroups"]["nativeness"] = {
    "native": native_stats,
    "non_native": nonnative_stats,
    "between_group_tests": nativeness_comparisons,
}

# --- 2. Age groups ---
print("\n  --- Age Groups ---")
age_results = {}
age_groups_data = {}
for ag in ["<25", "25-40", "40+"]:
    sub = merged[merged["age_group"] == ag]
    s = subgroup_stats(sub, ag)
    age_results[ag] = s
    age_groups_data[ag] = sub
    print(f"    {ag:5s} (n={s['n']}): entropy={s.get('entropy_mean','N/A')}, "
          f"CLC median={s.get('clc_median','N/A')}, independence={'HOLDS' if s.get('independence_holds') else 'FAILS' if s.get('independence_holds') is not None else 'N/A'}")

# Pairwise MW-U between age groups
age_comparisons = {}
age_labels = ["<25", "25-40", "40+"]
for i in range(len(age_labels)):
    for j in range(i + 1, len(age_labels)):
        a, b = age_labels[i], age_labels[j]
        key = f"{a}_vs_{b}"
        age_comparisons[key] = {}
        for feat in ["entropy", "clc_rho", "pause_freq"]:
            mw = mann_whitney_compare(age_groups_data[a], age_groups_data[b], feat)
            age_comparisons[key][feat] = mw

results["subgroups"]["age_groups"] = {
    "groups": age_results,
    "between_group_tests": age_comparisons,
}

# --- 3. Writing Strength ---
print("\n  --- Writing Strength ---")
low_strength = merged[merged["strength_group"] == "low"]
high_strength = merged[merged["strength_group"] == "high"]

low_stats = subgroup_stats(low_strength, "low_strength")
high_stats = subgroup_stats(high_strength, "high_strength")

print(f"    Low 1-2 (n={low_stats['n']}):  entropy={low_stats.get('entropy_mean','N/A')}, "
      f"CLC median={low_stats.get('clc_median','N/A')}, independence={'HOLDS' if low_stats.get('independence_holds') else 'FAILS' if low_stats.get('independence_holds') is not None else 'N/A'}")
print(f"    High 4-5 (n={high_stats['n']}): entropy={high_stats.get('entropy_mean','N/A')}, "
      f"CLC median={high_stats.get('clc_median','N/A')}, independence={'HOLDS' if high_stats.get('independence_holds') else 'FAILS' if high_stats.get('independence_holds') is not None else 'N/A'}")

strength_comparisons = {}
for feat in ["entropy", "clc_rho", "pause_freq", "iki_log_var"]:
    mw = mann_whitney_compare(low_strength, high_strength, feat)
    strength_comparisons[feat] = mw
    print(f"    MW-U {feat}: p={mw.get('p','N/A')}, effect_r={mw.get('effect_size_r','N/A')}")

results["subgroups"]["writing_strength"] = {
    "low_1_2": low_stats,
    "high_4_5": high_stats,
    "between_group_tests": strength_comparisons,
}

# ---------- summary ---------------------------------------------------------
print("\n[5/5] Summary & interpretation...")

# Collect independence results
independence_summary = {}
for label, s in [("native", native_stats), ("non_native", nonnative_stats)]:
    if s.get("corr_entropy_clc"):
        independence_summary[label] = {
            "rho": s["corr_entropy_clc"]["rho"],
            "holds": s["independence_holds"],
        }

for ag in age_labels:
    s = age_results[ag]
    if s.get("corr_entropy_clc"):
        independence_summary[f"age_{ag}"] = {
            "rho": s["corr_entropy_clc"]["rho"],
            "holds": s["independence_holds"],
        }

for label, s in [("strength_low", low_stats), ("strength_high", high_stats)]:
    if s.get("corr_entropy_clc"):
        independence_summary[label] = {
            "rho": s["corr_entropy_clc"]["rho"],
            "holds": s["independence_holds"],
        }

all_hold = all(v["holds"] for v in independence_summary.values() if v["holds"] is not None)
results["independence_across_subgroups"] = independence_summary
results["all_independence_holds"] = all_hold

# Key accessibility finding: non-native CLC
if nonnative_stats.get("clc_median") is not None and native_stats.get("clc_median") is not None:
    clc_diff = nonnative_stats["clc_median"] - native_stats["clc_median"]
    results["accessibility_finding"] = {
        "non_native_clc_median": nonnative_stats["clc_median"],
        "native_clc_median": native_stats["clc_median"],
        "clc_median_difference": round(clc_diff, 4),
        "non_native_clc_positive_pct": nonnative_stats["clc_positive_pct"],
        "native_clc_positive_pct": native_stats["clc_positive_pct"],
        "mann_whitney_clc_p": nativeness_comparisons["clc_rho"]["p"],
        "mann_whitney_clc_effect_r": nativeness_comparisons["clc_rho"]["effect_size_r"],
        "interpretation": (
            "Non-native speakers show DIFFERENT CLC patterns"
            if nativeness_comparisons["clc_rho"]["p"] is not None and float(nativeness_comparisons["clc_rho"]["p"]) < 0.05
            else "No significant CLC difference between native and non-native speakers"
        ),
    }

print(f"\n  Independence holds across ALL subgroups: {all_hold}")
print(f"  Independence per subgroup: { {k: v['holds'] for k, v in independence_summary.items()} }")

if "accessibility_finding" in results:
    af = results["accessibility_finding"]
    print(f"\n  ACCESSIBILITY FINDING:")
    print(f"    Native CLC median:     {af['native_clc_median']}")
    print(f"    Non-native CLC median: {af['non_native_clc_median']}")
    print(f"    Difference:            {af['clc_median_difference']}")
    print(f"    MW-U p-value:          {af['mann_whitney_clc_p']}")
    print(f"    Effect size r:         {af['mann_whitney_clc_effect_r']}")
    print(f"    => {af['interpretation']}")

# ---------- save ------------------------------------------------------------
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return super().default(obj)

out = BASE / "demographic_results.json"
with open(out, "w") as f:
    json.dump(results, f, indent=2, cls=NumpyEncoder)
print(f"\nResults saved to {out}")
