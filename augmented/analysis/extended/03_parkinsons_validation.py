"""Validates that process signals detect Parkinson's disease (a known cognitive impairment).

If signals discriminate PD patients from healthy controls, they're measuring real cognitive
processes. Paper relevance: provides clinical-grade validation independent of AI detection task.

Usage:
    uv run python analysis/extended/03_parkinsons_validation.py \\
        --data-dir data/ -o results/parkinsons_validation.json
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scholawrite.datasets import entropy_bits, lag1_autocorr


def _out(*args, **kwargs):
    sys.stdout.write(" ".join(str(a) for a in args) + kwargs.get("end", "\n"))


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _std(values):
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return (sum((v - m) ** 2 for v in values) / (len(values) - 1)) ** 0.5


def _cohens_d(a, b):
    if not a or not b:
        return 0.0
    na, nb = len(a), len(b)
    if na + nb < 4:
        return 0.0
    ma, mb = _mean(a), _mean(b)
    pooled_var = (
        (na - 1) * _std(a) ** 2 + (nb - 1) * _std(b) ** 2
    ) / (na + nb - 2)
    pooled_std = pooled_var ** 0.5
    return (ma - mb) / pooled_std if pooled_std > 1e-12 else 0.0


def _ks_2sample(a, b):
    """Two-sample KS statistic and approximate p-value."""
    sa = sorted(a)
    sb = sorted(b)
    na, nb = len(sa), len(sb)
    if na == 0 or nb == 0:
        return 0.0, 1.0

    all_vals = sorted(set(sa + sb))
    max_diff = 0.0
    ia = ib = 0
    for v in all_vals:
        while ia < na and sa[ia] <= v:
            ia += 1
        while ib < nb and sb[ib] <= v:
            ib += 1
        diff = abs(ia / na - ib / nb)
        if diff > max_diff:
            max_diff = diff

    # Kolmogorov distribution approximation
    n_eff = (na * nb) / (na + nb)
    z = max_diff * math.sqrt(n_eff)
    if z < 1e-12:
        p = 1.0
    else:
        # Two-tail approximation via series
        p = 2.0 * sum(
            ((-1) ** (k - 1)) * math.exp(-2 * k * k * z * z)
            for k in range(1, 20)
        )
        p = max(0.0, min(1.0, p))

    return max_diff, p


def _auc_mann_whitney(pos_vals, neg_vals):
    """AUC via Mann-Whitney U statistic (positive class has higher values = PD)."""
    if not pos_vals or not neg_vals:
        return 0.5
    u = sum(
        (1.0 if p > n else 0.5 if p == n else 0.0)
        for p in pos_vals
        for n in neg_vals
    )
    return u / (len(pos_vals) * len(neg_vals))


def _load_pd_labels(tappy_dir: Path) -> dict[str, bool]:
    """Return {UserID: is_pd} from Archived-users.zip or extracted CSVs."""
    pd_map: dict[str, bool] = {}

    zip_path = tappy_dir / "Archived-users.zip"
    if zip_path.exists():
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                for name in zf.namelist():
                    if not name.endswith(".txt"):
                        continue
                    # Files are "User_USERID.txt" with "Key: Value" lines
                    uid = Path(name).stem.replace("User_", "")
                    with zf.open(name) as raw:
                        text = raw.read().decode("utf-8", errors="replace")
                    fields: dict[str, str] = {}
                    for line in text.splitlines():
                        if ": " in line:
                            k, v = line.split(": ", 1)
                            fields[k.strip()] = v.strip()
                    pd_val = fields.get("Parkinsons", "")
                    if uid and pd_val:
                        pd_map[uid] = pd_val.lower() == "true"
        except (zipfile.BadZipFile, KeyError, UnicodeDecodeError):
            pass

    # Also scan any already-extracted CSVs
    for csv_path in tappy_dir.rglob("*.csv"):
        try:
            with open(csv_path, encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                fields = reader.fieldnames or []
                if not any("arkinsons" in fld for fld in fields):
                    continue
                for row in reader:
                    uid = (row.get("UserID") or row.get("userid") or "").strip()
                    pd_val = (row.get("Parkinsons") or row.get("parkinsons") or "").strip()
                    if uid and pd_val:
                        pd_map[uid] = pd_val.lower() in ("true", "1", "yes")
        except OSError:
            continue

    return pd_map


def _load_tappy_ikis(tappy_dir: Path) -> dict[str, list[float]]:
    """Return {UserID: [iki_ms, ...]} from Tappy keystroke files."""
    users: dict[str, list[float]] = {}

    txt_files = list(tappy_dir.rglob("*.txt"))
    if not txt_files:
        for zp in sorted(tappy_dir.rglob("*.zip")):
            if "user" in zp.name.lower() or "archived" in zp.name.lower():
                continue
            try:
                with zipfile.ZipFile(zp, "r") as zf:
                    zf.extractall(zp.parent)
            except zipfile.BadZipFile:
                continue
        txt_files = list(tappy_dir.rglob("*.txt"))

    for txt_path in txt_files:
        if txt_path.name.startswith("."):
            continue
        try:
            with open(txt_path, encoding="utf-8", errors="replace") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) < 7:
                        continue
                    uid = parts[0].strip()
                    if not uid:
                        continue
                    try:
                        hold = float(parts[4])
                        flight = float(parts[6])
                    except (ValueError, IndexError):
                        continue
                    iki = hold + flight
                    if 0 < iki < 30_000:
                        users.setdefault(uid, []).append(iki)
        except OSError:
            continue

    return users


def _compute_user_signals(iki_values: list[float]) -> dict[str, float | None]:
    filtered = [v for v in iki_values if 0 < v < 30_000]
    if len(filtered) < 5:
        return {}
    return {
        "mean_iki": _mean(filtered),
        "std_iki": _std(filtered),
        "entropy": entropy_bits(filtered),
        "lag1_autocorr": lag1_autocorr(filtered),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data", help="Root data directory")
    parser.add_argument("-o", "--output", default="results/parkinsons_validation.json")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    tappy_dir = data_dir / "tappy_parkinsons" / "tappy-keystroke-data-1.0.0"
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    _out("Loading PD labels from Archived-users.zip...")
    if not tappy_dir.exists():
        _out(f"WARNING: Tappy data dir not found at {tappy_dir}")
        pd_map: dict[str, bool] = {}
    else:
        pd_map = _load_pd_labels(tappy_dir)
    _out(f"  Labels loaded: {len(pd_map)} users "
         f"({sum(pd_map.values())} PD, {sum(not v for v in pd_map.values())} healthy)")

    _out("Loading Tappy keystroke files...")
    if tappy_dir.exists():
        user_ikis = _load_tappy_ikis(tappy_dir)
    else:
        user_ikis = {}
    _out(f"  Keystroke data for {len(user_ikis)} users")

    # Compute per-user signals, split by PD status
    pd_signals: dict[str, list[float]] = {
        "mean_iki": [], "std_iki": [], "entropy": [], "lag1_autocorr": []
    }
    healthy_signals: dict[str, list[float]] = {
        "mean_iki": [], "std_iki": [], "entropy": [], "lag1_autocorr": []
    }

    n_pd = 0
    n_healthy = 0
    n_skipped = 0

    for uid, ikis in user_ikis.items():
        sigs = _compute_user_signals(ikis)
        if not sigs:
            n_skipped += 1
            continue
        is_pd = pd_map.get(uid)
        if is_pd is None:
            n_skipped += 1
            continue
        target = pd_signals if is_pd else healthy_signals
        for sig_name in ("mean_iki", "std_iki", "entropy", "lag1_autocorr"):
            val = sigs.get(sig_name)
            if val is not None:
                target[sig_name].append(val)
        if is_pd:
            n_pd += 1
        else:
            n_healthy += 1

    _out(f"  PD users with data: {n_pd}, healthy: {n_healthy}, skipped: {n_skipped}")

    if n_pd == 0 or n_healthy == 0:
        _out("WARNING: insufficient labeled data for comparison.")

    signal_results: dict[str, dict] = {}
    for sig_name in ("mean_iki", "std_iki", "entropy", "lag1_autocorr"):
        pd_vals = pd_signals[sig_name]
        healthy_vals = healthy_signals[sig_name]

        if len(pd_vals) < 2 or len(healthy_vals) < 2:
            signal_results[sig_name] = {
                "ks_statistic": None,
                "ks_pvalue": None,
                "cohens_d": None,
                "auc_pd_classification": None,
                "n_pd": len(pd_vals),
                "n_healthy": len(healthy_vals),
            }
            _out(f"  {sig_name}: insufficient data (PD={len(pd_vals)}, H={len(healthy_vals)})")
            continue

        ks_stat, ks_p = _ks_2sample(pd_vals, healthy_vals)
        cd = _cohens_d(pd_vals, healthy_vals)
        # AUC: higher signal value = more likely PD
        auc = _auc_mann_whitney(pd_vals, healthy_vals)

        signal_results[sig_name] = {
            "ks_statistic": round(ks_stat, 4),
            "ks_pvalue": round(ks_p, 6),
            "cohens_d": round(cd, 4),
            "auc_pd_classification": round(auc, 4),
            "n_pd": len(pd_vals),
            "n_healthy": len(healthy_vals),
        }
        _out(f"  {sig_name}: KS={ks_stat:.3f} (p={ks_p:.4f}), d={cd:.3f}, AUC={auc:.3f}")

    result = {
        "n_pd": n_pd,
        "n_healthy": n_healthy,
        "n_skipped_no_label_or_data": n_skipped,
        "signals": signal_results,
    }

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    _out(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
