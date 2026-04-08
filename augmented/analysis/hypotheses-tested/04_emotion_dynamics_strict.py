"""Emotion dynamics validation with minimum user threshold per emotion.

HYPOTHESIS: Part of spectrum test. Keystroke signals for emotional state detection (state-intrinsic).
STATUS: Tested and falsified. See SPECTRUM_HYPOTHESIS_LOG.md

Filters to only emotions with ≥50 users per emotion. Original data is sparse
(Angry: 23 users only), which leads to unreliable estimates. This strict filter
ensures adequate statistical power for pairwise comparisons.

Usage:
    uv run python hypotheses-tested/04_emotion_dynamics_strict.py \\
        --data-dir data/ -o results/emotion_dynamics_strict.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
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


def _ks_2sample(a, b):
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

    n_eff = (na * nb) / (na + nb)
    z = max_diff * math.sqrt(n_eff)
    if z < 1e-12:
        p = 1.0
    else:
        p = 2.0 * sum(
            ((-1) ** (k - 1)) * math.exp(-2 * k * k * z * z)
            for k in range(1, 20)
        )
        p = max(0.0, min(1.0, p))

    return max_diff, p


def _cohens_d(a, b):
    if not a or not b:
        return 0.0
    na, nb = len(a), len(b)
    if na + nb < 4:
        return 0.0
    pooled_var = (
        (na - 1) * _std(a) ** 2 + (nb - 1) * _std(b) ** 2
    ) / (na + nb - 2)
    pooled_std = pooled_var ** 0.5
    return (_mean(a) - _mean(b)) / pooled_std if pooled_std > 1e-12 else 0.0


EMOTION_LABELS = {
    "N": "Neutral",
    "H": "Happy",
    "A": "Angry",
    "S": "Sad",
    "C": "Calm",
}


def _load_emosurv(data_dir: Path) -> dict[str, dict[str, list[float]]]:
    """Return {emotion_code: {user_id: [iki_ms, ...]}} from EmoSurv free-text CSV.

    Searches for any CSV under data_dir/emosurv/ containing D1D2 and emotionIndex columns.
    """
    emosurv_dir = data_dir / "emosurv"
    if not emosurv_dir.exists():
        return {}

    csv_files = list(emosurv_dir.rglob("*.csv"))
    free_csvs = [f for f in csv_files if "free" in f.name.lower()] or csv_files

    # emotion_code -> user_id -> [iki values]
    data: dict[str, dict[str, list[float]]] = {}

    for csv_path in free_csvs:
        try:
            with open(csv_path, newline="", encoding="utf-8-sig", errors="replace") as f:
                reader = csv.DictReader(f, delimiter=";")
                for row in reader:
                    uid = (
                        row.get("userid")
                        or row.get("_id")
                        or row.get("User_Id")
                        or row.get("user_id")
                        or ""
                    ).strip()
                    emotion = (
                        row.get("emotionIndex")
                        or row.get("EmotionIndex")
                        or row.get("emotion")
                        or ""
                    ).strip().upper()

                    iki_raw = row.get("D1D2") or row.get("d1d2") or ""
                    if not uid or not emotion or not iki_raw:
                        continue
                    if emotion not in EMOTION_LABELS:
                        continue

                    try:
                        val = float(iki_raw.replace(",", "."))
                    except (ValueError, AttributeError):
                        continue

                    if not (0 < val < 30_000):
                        continue

                    data.setdefault(emotion, {}).setdefault(uid, []).append(val)
        except OSError:
            continue

    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data", help="Root data directory")
    parser.add_argument("-o", "--output", default="results/emotion_dynamics_strict.json")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    _out("Loading EmoSurv free-text typing data...")
    raw = _load_emosurv(data_dir)

    if not raw:
        _out(f"WARNING: No EmoSurv data found in {data_dir / 'emosurv'}")

    MIN_USERS_PER_EMOTION = 50
    _out(f"Requiring minimum {MIN_USERS_PER_EMOTION} users per emotion for adequate power...")

    # Count users per emotion
    n_users_per_emotion = {code: len(user_dict) for code, user_dict in raw.items()}
    qualified_emotions = {code for code, n in n_users_per_emotion.items() if n >= MIN_USERS_PER_EMOTION}

    _out(f"  Before filtering: {len(raw)} emotions total")
    for code in sorted(raw.keys()):
        label = EMOTION_LABELS.get(code, code)
        _out(f"    {label} ({code}): {n_users_per_emotion[code]} users")
    _out(f"  After filtering: {len(qualified_emotions)} emotions with >= {MIN_USERS_PER_EMOTION} users")

    if len(qualified_emotions) < 2:
        _out(f"WARNING: insufficient emotions after filtering (need >=2 for pairwise comparison)")
        json.dump({
            "n_emotions_total": len(raw),
            "n_emotions_qualified": len(qualified_emotions),
            "minimum_users_per_emotion": MIN_USERS_PER_EMOTION,
            "note": "Hypothesis test (spectrum): emotion state detection. Insufficient data after strict filtering.",
            "emotions": {},
        }, open(args.output, "w"), indent=2)
        return

    # Per-emotion aggregate: pool all IKI values across users, count distinct users
    emotion_stats: dict[str, dict] = {}
    # Also store per-user per-emotion std_iki list for arousal hypothesis
    emotion_user_stds: dict[str, list[float]] = {}

    for code in sorted(qualified_emotions):
        user_dict = raw[code]
        label = EMOTION_LABELS.get(code, code)
        all_ikis: list[float] = []
        user_stds: list[float] = []
        for uid, ikis in user_dict.items():
            all_ikis.extend(ikis)
            if len(ikis) >= 2:
                user_stds.append(_std(ikis))

        n_users = len(user_dict)
        mean_iki = _mean(all_ikis)
        std_iki = _std(all_ikis)
        ent = entropy_bits(all_ikis)
        ac = lag1_autocorr(all_ikis)

        emotion_stats[label] = {
            "emotion_code": code,
            "n_users": n_users,
            "n_iki_observations": len(all_ikis),
            "mean_iki": round(mean_iki, 2),
            "std_iki": round(std_iki, 2),
            "entropy": round(ent, 4),
            "lag1_autocorr": round(ac, 4) if ac is not None else None,
        }
        emotion_user_stds[code] = user_stds

        _out(f"  {label} ({code}): n_users={n_users}, "
             f"mean_iki={mean_iki:.1f}ms, std={std_iki:.1f}ms, entropy={ent:.3f}b")

    # Pairwise KS tests on pooled mean IKI per user (one value per user per emotion)
    # Use per-user mean IKI as the unit for pairwise comparisons
    emotion_user_means: dict[str, list[float]] = {}
    for code in qualified_emotions:
        user_dict = raw[code]
        label = EMOTION_LABELS.get(code, code)
        emotion_user_means[label] = [_mean(ikis) for ikis in user_dict.values() if ikis]

    pairwise_ks: dict[str, dict] = {}
    labels_present = sorted(emotion_user_means.keys())
    for i, la in enumerate(labels_present):
        for lb in labels_present[i + 1:]:
            vals_a = emotion_user_means[la]
            vals_b = emotion_user_means[lb]
            if len(vals_a) < 2 or len(vals_b) < 2:
                continue
            ks, p = _ks_2sample(vals_a, vals_b)
            cd = _cohens_d(vals_a, vals_b)
            pair_key = f"{la}_vs_{lb}"
            pairwise_ks[pair_key] = {
                "ks_statistic": round(ks, 4),
                "ks_pvalue": round(p, 6),
                "cohens_d": round(cd, 4),
            }
            _out(f"  {pair_key}: KS={ks:.3f} (p={p:.4f}), d={cd:.3f}")

    # Arousal hypothesis: Angry std_iki > Neutral std_iki
    # Use pooled per-user std_iki values
    angry_stds = emotion_user_stds.get("A", [])
    neutral_stds = emotion_user_stds.get("N", [])
    angry_mean_std = _mean(angry_stds) if angry_stds else None
    neutral_mean_std = _mean(neutral_stds) if neutral_stds else None

    if angry_mean_std is not None and neutral_mean_std is not None:
        arousal_confirmed = angry_mean_std > neutral_mean_std
        arousal_ratio = round(angry_mean_std / neutral_mean_std, 4) if neutral_mean_std > 1e-12 else None
    else:
        arousal_confirmed = None
        arousal_ratio = None

    angry_s = f"{angry_mean_std:.2f}" if angry_mean_std is not None else "N/A"
    neutral_s = f"{neutral_mean_std:.2f}" if neutral_mean_std is not None else "N/A"
    _out(f"\nArousal hypothesis (Angry std > Neutral std): "
         f"Angry={angry_s}, Neutral={neutral_s}, confirmed={arousal_confirmed}")

    result = {
        "n_emotions_total": len(raw),
        "n_emotions_qualified": len(qualified_emotions),
        "minimum_users_per_emotion": MIN_USERS_PER_EMOTION,
        "hypothesis": "Emotion state detection (spectrum test)",
        "emotions": emotion_stats,
        "pairwise_ks": pairwise_ks,
        "arousal_hypothesis": {
            "confirmed": arousal_confirmed,
            "angry_mean_std_iki": round(angry_mean_std, 2) if angry_mean_std is not None else None,
            "neutral_mean_std_iki": round(neutral_mean_std, 2) if neutral_mean_std is not None else None,
            "angry_std_vs_neutral_std": arousal_ratio,
        },
    }

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    _out(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
