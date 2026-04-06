"""
ROC Classifier (Script 3)
==========================
Trains a logistic regression classifier to distinguish genuine from forged
sessions using features from Script 2. Uses leave-one-writer-out CV to avoid
identity leakage.

Replaces:
  - Paper 06 "projected >90% accuracy" with actual AUC
  - Paper 02 Proposition 4 detection power with ROC AUC
  - Paper 09 "5.8% FAR" with measured FAR at operating points

Output: roc_classifier_results.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

print("=" * 70)
print("Script 3: ROC Classifier (Leave-One-Writer-Out CV)")
print("=" * 70)

# ---------- load session data from Script 2 -------------------------------
session_csv = Path(__file__).parent / "retype_sessions.csv"
if not session_csv.exists():
    raise FileNotFoundError(f"Run retype_simulation.py first: {session_csv}")

df = pd.read_csv(session_csv)
print(f"\n[1/4] Loaded {len(df)} sessions ({(df['label']==0).sum()} genuine, {(df['label']==1).sum()} forged)")

feature_cols = ["entropy", "autocorr_lag1", "iki_mean", "iki_std", "iki_log_var", "pause_freq", "clc_proxy"]
attack_types = ["attack_constant", "attack_iid", "attack_cross", "attack_markov"]

# ---------- per-attack-type evaluation ------------------------------------
all_results = {}
# Collect per-fold coefficients for the all_attacks evaluation to average later
all_attacks_fold_coefs = []

for attack in attack_types + ["all_attacks"]:
    print(f"\n[2/4] Evaluating: {attack}")

    if attack == "all_attacks":
        df_eval = df.copy()
    else:
        df_eval = pd.concat([df[df["session_type"] == "genuine"], df[df["session_type"] == attack]])

    writers = df_eval["writer_id"].unique()
    y_true_all = []
    y_score_all = []
    y_pred_all = []
    fold_coefs = []

    # Leave-one-writer-out CV
    for w in writers:
        test = df_eval[df_eval["writer_id"] == w]
        train = df_eval[df_eval["writer_id"] != w]

        if len(test) == 0 or len(train[train["label"] == 0]) == 0 or len(train[train["label"] == 1]) == 0:
            continue

        X_train = train[feature_cols].values
        y_train = train["label"].values
        X_test = test[feature_cols].values
        y_test = test["label"].values

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train_s, y_train)

        y_score = clf.predict_proba(X_test_s)[:, 1]
        y_pred = clf.predict(X_test_s)

        y_true_all.extend(y_test.tolist())
        y_score_all.extend(y_score.tolist())
        y_pred_all.extend(y_pred.tolist())

        # Collect coefficients from this fold
        fold_coefs.append(clf.coef_[0].copy())

    # Store all_attacks fold coefficients for feature importance
    if attack == "all_attacks":
        all_attacks_fold_coefs = fold_coefs

    y_true_all = np.array(y_true_all)
    y_score_all = np.array(y_score_all)
    y_pred_all = np.array(y_pred_all)

    # Metrics
    auc = roc_auc_score(y_true_all, y_score_all)
    fpr, tpr, thresholds = roc_curve(y_true_all, y_score_all)

    # Operating points
    # EER (where FPR ≈ FNR)
    fnr = 1 - tpr
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2

    # FAR at specific FRR targets
    operating_points = {}
    for target_frr in [0.01, 0.05, 0.10]:
        idx = np.argmin(np.abs(fnr - target_frr))
        operating_points[f"FAR_at_{int(target_frr*100)}pct_FRR"] = round(float(fpr[idx]), 4)
        operating_points[f"FRR_at_{int(target_frr*100)}pct_FRR"] = round(float(fnr[idx]), 4)

    # Confusion matrix at default threshold (0.5)
    tp = int(((y_true_all == 1) & (y_pred_all == 1)).sum())
    fp = int(((y_true_all == 0) & (y_pred_all == 1)).sum())
    tn = int(((y_true_all == 0) & (y_pred_all == 0)).sum())
    fn = int(((y_true_all == 1) & (y_pred_all == 0)).sum())
    accuracy = (tp + tn) / len(y_true_all) if len(y_true_all) > 0 else 0
    far = fp / (fp + tn) if (fp + tn) > 0 else 0
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0

    result = {
        "n_sessions": len(df_eval),
        "n_genuine": int((df_eval["label"] == 0).sum()),
        "n_forged": int((df_eval["label"] == 1).sum()),
        "auc": round(auc, 4),
        "eer": round(eer, 4),
        "accuracy_at_05": round(accuracy, 4),
        "far_at_05": round(far, 4),
        "frr_at_05": round(frr, 4),
        "operating_points": operating_points,
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }
    all_results[attack] = result

    print(f"    AUC:      {auc:.4f}")
    print(f"    EER:      {eer:.4f}")
    print(f"    Accuracy: {accuracy:.4f}")
    print(f"    FAR@5%FRR: {operating_points.get('FAR_at_5pct_FRR', 'N/A')}")

# ---------- CLC-excluded evaluation (robustness check) --------------------
# Forged sessions have CLC=0 by construction, making it a trivially
# discriminative feature. Re-evaluate all_attacks without clc_proxy.
print("\n[3/6] CLC-excluded evaluation (robustness check)...")

feature_cols_no_clc = [f for f in feature_cols if f != "clc_proxy"]
df_eval_nc = df.copy()
writers_nc = df_eval_nc["writer_id"].unique()
y_true_nc, y_score_nc = [], []

for w in writers_nc:
    test = df_eval_nc[df_eval_nc["writer_id"] == w]
    train = df_eval_nc[df_eval_nc["writer_id"] != w]
    if len(test) == 0 or len(train[train["label"] == 0]) == 0 or len(train[train["label"] == 1]) == 0:
        continue
    X_train = train[feature_cols_no_clc].values
    y_train = train["label"].values
    X_test = test[feature_cols_no_clc].values
    y_test = test["label"].values
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train_s, y_train)
    y_score_nc.extend(clf.predict_proba(X_test_s)[:, 1].tolist())
    y_true_nc.extend(y_test.tolist())

y_true_nc = np.array(y_true_nc)
y_score_nc = np.array(y_score_nc)
auc_nc = roc_auc_score(y_true_nc, y_score_nc)
fpr_nc, tpr_nc, _ = roc_curve(y_true_nc, y_score_nc)
fnr_nc = 1 - tpr_nc
eer_idx_nc = np.argmin(np.abs(fpr_nc - fnr_nc))
eer_nc = (fpr_nc[eer_idx_nc] + fnr_nc[eer_idx_nc]) / 2

all_results["all_attacks_no_clc"] = {
    "auc": round(auc_nc, 4),
    "eer": round(eer_nc, 4),
    "features_used": feature_cols_no_clc,
    "note": "CLC excluded because forged sessions have CLC=0 by construction",
}
print(f"    AUC (no CLC):  {auc_nc:.4f}")
print(f"    EER (no CLC):  {eer_nc:.4f}")
print(f"    Delta AUC:     {auc_nc - all_results['all_attacks']['auc']:+.4f}")

# ---------- feature importance (averaged from CV folds) -------------------
print("\n[4/6] Feature importance (averaged across CV folds)...")

# Average the coefficients collected from each leave-one-writer-out fold
# instead of fitting on all data (which would leak test data into training)
mean_coefs = np.mean(all_attacks_fold_coefs, axis=0)
std_coefs = np.std(all_attacks_fold_coefs, axis=0)

importance = {}
for feat, coef, std in zip(feature_cols, mean_coefs, std_coefs):
    importance[feat] = round(float(abs(coef)), 4)
    importance[f"{feat}_std"] = round(float(std), 4)
    print(f"    {feat:18s}: |coef| = {abs(coef):.4f} (std = {std:.4f})")

all_results["feature_importance"] = importance

# ---------- balanced-class evaluation (500:500) ---------------------------
# Default setup has 500 genuine vs 2000 forged. Downsample to 500:500
# to verify EER is not inflated by class imbalance.
print("\n[5/6] Balanced-class evaluation (500:500)...")

genuine_df = df[df["label"] == 0]
forged_df = df[df["label"] == 1]
# Stratified downsample: equal forged from each attack type
forged_per_type = forged_df.groupby("session_type").apply(
    lambda x: x.sample(n=min(125, len(x)), random_state=42),
    include_groups=False,
).reset_index(level=0, drop=True)
df_bal = pd.concat([genuine_df, forged_per_type])

writers_bal = df_bal["writer_id"].unique()
y_true_bal, y_score_bal = [], []

for w in writers_bal:
    test = df_bal[df_bal["writer_id"] == w]
    train = df_bal[df_bal["writer_id"] != w]
    if len(test) == 0 or len(train[train["label"] == 0]) == 0 or len(train[train["label"] == 1]) == 0:
        continue
    X_train = train[feature_cols].values
    y_train = train["label"].values
    X_test = test[feature_cols].values
    y_test = test["label"].values
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train_s, y_train)
    y_score_bal.extend(clf.predict_proba(X_test_s)[:, 1].tolist())
    y_true_bal.extend(y_test.tolist())

y_true_bal = np.array(y_true_bal)
y_score_bal = np.array(y_score_bal)
auc_bal = roc_auc_score(y_true_bal, y_score_bal)
fpr_bal, tpr_bal, _ = roc_curve(y_true_bal, y_score_bal)
fnr_bal = 1 - tpr_bal
eer_idx_bal = np.argmin(np.abs(fpr_bal - fnr_bal))
eer_bal = (fpr_bal[eer_idx_bal] + fnr_bal[eer_idx_bal]) / 2

all_results["all_attacks_balanced"] = {
    "n_genuine": int((df_bal["label"] == 0).sum()),
    "n_forged": int((df_bal["label"] == 1).sum()),
    "auc": round(auc_bal, 4),
    "eer": round(eer_bal, 4),
    "note": "Balanced 500:500 downsample (125 per attack type)",
}
print(f"    Balanced: {(df_bal['label']==0).sum()} genuine, {(df_bal['label']==1).sum()} forged")
print(f"    AUC (balanced): {auc_bal:.4f}")
print(f"    EER (balanced): {eer_bal:.4f}")
print(f"    Delta AUC:      {auc_bal - all_results['all_attacks']['auc']:+.4f}")

# ---------- save ----------------------------------------------------------
print("\n[6/6] Saving results...")

out = Path(__file__).parent / "roc_classifier_results.json"
with open(out, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"  Results saved to {out}")

# Summary for paper updates
print("\n" + "=" * 70)
print("KEY NUMBERS FOR PAPER UPDATES")
print("=" * 70)
all_atk = all_results["all_attacks"]
print(f"  Paper 06: Replace 'projected >90%' with AUC = {all_atk['auc']:.3f}")
print(f"  Paper 02: Detection power AUC = {all_atk['auc']:.3f}")
print(f"  Paper 09: FAR = {all_atk['far_at_05']:.3f} at default threshold")
print(f"  Paper 09: FAR@5%FRR = {all_atk['operating_points'].get('FAR_at_5pct_FRR', 'N/A')}")
print(f"  EER: {all_atk['eer']:.3f}")
