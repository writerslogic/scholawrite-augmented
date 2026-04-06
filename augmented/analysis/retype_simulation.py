"""
Retype Attack Simulation (Script 2)
=====================================
Simulates four adversarial retype strategies against genuine KLiCKe sessions:
  (a) Constant-rate:        fixed IKI = writer's mean
  (b) Distribution-matched: IKI sampled iid from writer's own marginal
  (c) Cross-writer:         IKI sampled from a different writer's distribution
  (d) Markov-chain:         first-order Markov chain preserving lag-1 autocorrelation

For each genuine and forged session, computes:
  - Shannon entropy (quantised IKI)
  - CLC proxy (correlation between IKI and activity sequence)
  - Temporal autocorrelation (lag-1)

Following adversarial stylometry and keystroke biometric authentication
methodology (Killourhy & Maxion 2009; Monaco et al. 2017), we conduct a
white-box adaptive attack analysis where the attacker has full knowledge of
the target's statistical profile.

Output: retype_simulation_results.json (+ per-writer features for Scripts 3-5)
"""

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ---------- config --------------------------------------------------------
CSV_DIR = Path(__file__).parent / "klicke" / "Files" / "WritingTask" / "WritingTask" / "keystrokelogs" / "csv"
QUANTIZATION_MS = 5
MIN_EVENTS = 100       # per writer
MAX_WRITERS = 500      # subsample for tractability
SESSION_LEN = 200      # keystrokes per simulated session
N_FORGED_PER_WRITER = 4  # one per attack strategy
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

COGNITIVE_LOAD = {"Nonproduction": 3, "Remove/Cut": 2, "Input": 1}

print("=" * 70)
print("Script 2: Retype Attack Simulation")
print("=" * 70)

# ---------- helpers -------------------------------------------------------

def entropy_bits(ikis, q=QUANTIZATION_MS):
    quantized = (ikis // q) * q
    _, counts = np.unique(quantized, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs + 1e-15))


def autocorr_lag1(ikis):
    if len(ikis) < 3:
        return 0.0
    x = ikis - ikis.mean()
    c0 = np.dot(x, x)
    if c0 == 0:
        return 0.0
    return np.dot(x[:-1], x[1:]) / c0


def clc_proxy(ikis, activities):
    """Spearman correlation between cognitive load and IKI."""
    cog = np.array([COGNITIVE_LOAD.get(a, 1) for a in activities], dtype=float)
    if len(ikis) < 10 or np.std(cog) == 0:
        return 0.0
    rho, _ = stats.spearmanr(cog, ikis)
    return 0.0 if np.isnan(rho) else rho


def markov_retype(ikis, n_out, n_bins=50):
    """Generate synthetic IKIs via a first-order Markov chain that preserves
    lag-1 autocorrelation from the genuine sequence.

    Discretizes IKIs into `n_bins` equal-frequency bins, builds a transition
    matrix from consecutive bin pairs, then samples a new sequence and maps
    back to continuous IKIs (uniform sample within each bin).
    """
    # Discretize into equal-frequency bins
    bin_edges = np.percentile(ikis, np.linspace(0, 100, n_bins + 1))
    bin_edges[0] -= 1  # ensure min value is included
    bin_edges[-1] += 1
    bin_indices = np.digitize(ikis, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    # Build transition matrix
    trans_counts = np.zeros((n_bins, n_bins), dtype=float)
    for i in range(len(bin_indices) - 1):
        trans_counts[bin_indices[i], bin_indices[i + 1]] += 1

    # Add small smoothing to avoid zero-row issues
    trans_counts += 0.01
    row_sums = trans_counts.sum(axis=1, keepdims=True)
    trans_probs = trans_counts / row_sums

    # Sample chain
    synthetic_bins = np.zeros(n_out, dtype=int)
    # Use empirical bin frequencies (not transition row-sums) for initial state
    init_probs = np.bincount(bin_indices, minlength=n_bins).astype(float)
    init_probs /= init_probs.sum()
    synthetic_bins[0] = np.random.choice(n_bins, p=init_probs)
    for i in range(1, n_out):
        synthetic_bins[i] = np.random.choice(n_bins, p=trans_probs[synthetic_bins[i - 1]])

    # Map back to continuous IKIs (uniform within bin)
    synthetic_ikis = np.zeros(n_out, dtype=float)
    for i in range(n_out):
        lo = bin_edges[synthetic_bins[i]]
        hi = bin_edges[synthetic_bins[i] + 1]
        synthetic_ikis[i] = np.random.uniform(lo, hi)

    return np.clip(synthetic_ikis, 10, 60000)


def extract_features(ikis, activities=None):
    """Compute feature vector for a session."""
    return {
        "entropy": float(entropy_bits(ikis)),
        "autocorr_lag1": float(autocorr_lag1(ikis)),
        "iki_mean": float(ikis.mean()),
        "iki_std": float(ikis.std()),
        "iki_log_var": float(np.log(ikis.var() + 1)),
        "pause_freq": float((ikis > 2000).mean()),
        "clc_proxy": float(clc_proxy(ikis, activities)) if activities is not None else 0.0,
    }


# ---------- load writer data ---------------------------------------------
print("\n[1/4] Loading writer sessions...")

writer_data = {}  # writer_id -> {ikis, activities}
csv_files = sorted(CSV_DIR.glob("*.csv"))

for f in csv_files:
    try:
        df = pd.read_csv(f)
    except Exception:
        continue
    df = df.sort_values("DownTime").reset_index(drop=True)
    iki = df["DownTime"].diff().values[1:]
    act = df["Activity"].values[1:]
    valid = (iki > 10) & (iki < 60000)
    iki = iki[valid]
    act = act[valid]
    if len(iki) >= MIN_EVENTS:
        writer_data[f.stem] = {"ikis": iki.astype(float), "activities": act}

print(f"  Eligible writers (>={MIN_EVENTS} events): {len(writer_data)}")

# Subsample for tractability
all_wids = sorted(writer_data.keys())
if len(all_wids) > MAX_WRITERS:
    sampled_wids = sorted(random.sample(all_wids, MAX_WRITERS))
else:
    sampled_wids = all_wids
print(f"  Using {len(sampled_wids)} writers for simulation")

# ---------- generate genuine + forged sessions ----------------------------
print("\n[2/4] Generating genuine and forged sessions...")

records = []  # each: {writer_id, session_type, features...}

# Pool of all writer IKIs for cross-writer attacks
other_wids = list(writer_data.keys())

for idx, wid in enumerate(sampled_wids):
    if (idx + 1) % 100 == 0:
        print(f"  {idx+1}/{len(sampled_wids)} writers...")

    w = writer_data[wid]
    ikis_full = w["ikis"]
    acts_full = w["activities"]

    # --- Genuine session: random contiguous window ---
    if len(ikis_full) > SESSION_LEN:
        start = random.randint(0, len(ikis_full) - SESSION_LEN)
        g_ikis = ikis_full[start : start + SESSION_LEN]
        g_acts = acts_full[start : start + SESSION_LEN]
    else:
        g_ikis = ikis_full
        g_acts = acts_full

    feat_g = extract_features(g_ikis, g_acts)
    feat_g["writer_id"] = wid
    feat_g["session_type"] = "genuine"
    feat_g["label"] = 0  # 0 = genuine
    records.append(feat_g)

    # --- Attack (a): Constant-rate ---
    # NOTE: All forged sessions have CLC=0 by construction because there is no
    # real cognitive process generating the IKIs. The CLC proxy requires a
    # correlation between IKI and cognitive-load activity labels, which only
    # exist in genuine writing sessions. This means CLC is a "free" discriminator
    # that inflates classifier performance; detection claims should be validated
    # with CLC excluded as well.
    const_ikis = np.full(SESSION_LEN, ikis_full.mean())
    feat_a = extract_features(const_ikis)
    feat_a["writer_id"] = wid
    feat_a["session_type"] = "attack_constant"
    feat_a["label"] = 1
    records.append(feat_a)

    # --- Attack (b): Distribution-matched (iid from marginal) ---
    iid_ikis = np.random.choice(ikis_full, size=SESSION_LEN, replace=True)
    feat_b = extract_features(iid_ikis)
    feat_b["writer_id"] = wid
    feat_b["session_type"] = "attack_iid"
    feat_b["label"] = 1
    records.append(feat_b)

    # --- Attack (c): Cross-writer ---
    donor = random.choice([w2 for w2 in other_wids if w2 != wid])
    donor_ikis = writer_data[donor]["ikis"]
    cross_ikis = np.random.choice(donor_ikis, size=SESSION_LEN, replace=True)
    feat_c = extract_features(cross_ikis)
    feat_c["writer_id"] = wid
    feat_c["session_type"] = "attack_cross"
    feat_c["label"] = 1
    records.append(feat_c)

    # --- Attack (d): Markov-chain retype (preserves lag-1 autocorrelation) ---
    # Most sophisticated attack: builds a first-order Markov chain from the
    # genuine writer's IKI sequence to preserve temporal structure (lag-1 ACF).
    # Directly tests whether higher-order temporal features are needed for
    # detection, as claimed in Papers 05 and 09.
    markov_ikis = markov_retype(ikis_full, SESSION_LEN)
    feat_d = extract_features(markov_ikis)
    feat_d["writer_id"] = wid
    feat_d["session_type"] = "attack_markov"
    feat_d["label"] = 1
    feat_d["acf1_genuine"] = float(autocorr_lag1(g_ikis))
    feat_d["acf1_synthetic"] = float(autocorr_lag1(markov_ikis))
    records.append(feat_d)

df_sessions = pd.DataFrame(records)
print(f"  Total sessions: {len(df_sessions)}")
print(f"    Genuine: {(df_sessions['label'] == 0).sum()}")
print(f"    Forged:  {(df_sessions['label'] == 1).sum()}")

# PY-M002: Validate Markov chain lag-1 ACF preservation
markov_rows = df_sessions[df_sessions["session_type"] == "attack_markov"]
if "acf1_genuine" in markov_rows.columns and "acf1_synthetic" in markov_rows.columns:
    acf_genuine = markov_rows["acf1_genuine"].dropna()
    acf_synthetic = markov_rows["acf1_synthetic"].dropna()
    print(f"\n  Markov lag-1 ACF validation:")
    print(f"    Genuine   mean={acf_genuine.mean():.4f}, std={acf_genuine.std():.4f}")
    print(f"    Synthetic mean={acf_synthetic.mean():.4f}, std={acf_synthetic.std():.4f}")
    acf_diff = (acf_genuine - acf_synthetic.values).abs().mean()
    print(f"    Mean |genuine - synthetic| ACF1: {acf_diff:.4f}")

# ---------- detection analysis (threshold-based) --------------------------
print("\n[3/4] Threshold-based detection analysis...")

feature_cols = ["entropy", "autocorr_lag1", "iki_std", "iki_log_var", "pause_freq", "clc_proxy"]

genuine = df_sessions[df_sessions["label"] == 0]
attack_types = ["attack_constant", "attack_iid", "attack_cross", "attack_markov"]

detection_results = {}
# PY-H002: CLC proxy is always 0 in forged sessions (no real cognitive process),
# which inflates classifier AUC. Compute metrics both with and without CLC.
detection_results_without_clc = {}
feature_cols_no_clc = [f for f in feature_cols if f != "clc_proxy"]

for attack in attack_types:
    forged = df_sessions[df_sessions["session_type"] == attack]

    per_feature = {}
    for feat in feature_cols:
        g_vals = genuine[feat].values
        f_vals = forged[feat].values

        # Effect size
        pooled_std = np.sqrt((g_vals.std() ** 2 + f_vals.std() ** 2) / 2)
        cohens_d = abs(g_vals.mean() - f_vals.mean()) / pooled_std if pooled_std > 0 else 0

        # Mann-Whitney separability
        try:
            u_stat, u_pval = stats.mannwhitneyu(g_vals, f_vals, alternative="two-sided")
            auc_u = u_stat / (len(g_vals) * len(f_vals))  # AUC from U statistic
        except Exception:
            auc_u, u_pval = 0.5, 1.0

        per_feature[feat] = {
            "genuine_mean": round(float(g_vals.mean()), 4),
            "forged_mean": round(float(f_vals.mean()), 4),
            "cohens_d": round(cohens_d, 4),
            "auc_mann_whitney": round(max(auc_u, 1 - auc_u), 4),  # ensure > 0.5
        }

    detection_results[attack] = per_feature
    detection_results_without_clc[attack] = {
        k: v for k, v in per_feature.items() if k != "clc_proxy"
    }
    best_feat = max(per_feature, key=lambda k: per_feature[k]["cohens_d"])
    best_feat_no_clc = max(
        (k for k in per_feature if k != "clc_proxy"),
        key=lambda k: per_feature[k]["cohens_d"],
    )
    print(f"\n  {attack}:")
    print(f"    Best single feature: {best_feat} (d={per_feature[best_feat]['cohens_d']:.2f})")
    print(f"    Best without CLC:    {best_feat_no_clc} (d={per_feature[best_feat_no_clc]['cohens_d']:.2f})")
    for feat in feature_cols:
        d = per_feature[feat]
        print(f"    {feat:18s}: d={d['cohens_d']:.2f}, AUC={d['auc_mann_whitney']:.3f}")

# ---------- save session-level data for downstream scripts ----------------
print("\n[4/4] Saving results...")

# Save session features (for Scripts 3-5)
session_csv = Path(__file__).parent / "retype_sessions.csv"
df_sessions.to_csv(session_csv, index=False)
print(f"  Session features: {session_csv}")

# Summary JSON
summary = {
    "n_writers": len(sampled_wids),
    "session_length": SESSION_LEN,
    "total_sessions": len(df_sessions),
    "genuine_sessions": int((df_sessions["label"] == 0).sum()),
    "forged_sessions": int((df_sessions["label"] == 1).sum()),
    "detection_by_attack_type": detection_results,
    "detection_by_attack_type_without_clc": detection_results_without_clc,
    "methodology": (
        "White-box adaptive attack analysis following Killourhy & Maxion (2009). "
        "Four attack strategies of increasing sophistication: constant-rate replay, "
        "distribution-matched iid sampling, cross-writer transfer, and Markov-chain "
        "retype preserving lag-1 autocorrelation."
    ),
    "limitations": {
        "clc_zero_in_forged": (
            "All forged sessions have CLC proxy = 0 by construction because synthetic "
            "IKI sequences lack the cognitive-load activity labels present in genuine "
            "writing. This makes CLC a trivially discriminative feature. Detection "
            "results should be validated with CLC excluded to assess robustness."
        ),
    },
}

out = Path(__file__).parent / "retype_simulation_results.json"
with open(out, "w") as f:
    json.dump(summary, f, indent=2)
print(f"  Summary: {out}")
