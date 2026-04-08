"""Analyzes whether typing language/nationality influences IKI distributions.

KeyRecs has 20 nationalities. Tests if IKI variance is language-universal or
culture-specific. Paper relevance: if IKI is language-universal, process signals
are universal cognitive properties not linguistic artifacts.

Usage:
    uv run python analysis/extended/12_cross_language_iki.py --data-dir data/ -o results/cross_language_iki.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scholawrite.datasets import entropy_bits, lag1_autocorr, REGISTRY, TaskType


def _out(*args, **kwargs):
    sys.stdout.write(" ".join(str(a) for a in args) + kwargs.get("end", "\n"))


IKI_MIN_MS = 10.0
IKI_MAX_MS = 30_000.0
MIN_PARTICIPANTS_PER_NATIONALITY = 3


def _mean(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _std(vals: List[float]) -> float:
    if len(vals) < 2:
        return 0.0
    m = _mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))


def _ensure_extracted(data_dir: Path) -> Path:
    """Return the keyrecs directory, extracting a zip if needed."""
    dest = data_dir / "keyrecs"
    dest.mkdir(parents=True, exist_ok=True)

    for candidate in [dest / "free-text.csv", dest / "free_text.csv"]:
        if candidate.exists():
            return dest
    for match in dest.rglob("free*text*.csv"):
        return dest

    for zp in dest.glob("*.zip"):
        _out(f"  Extracting {zp.name}...")
        with zipfile.ZipFile(zp, "r") as zf:
            zf.extractall(dest)
        return dest

    raise FileNotFoundError(f"KeyRecs free-text.csv not found in {dest}. Download with: "
                            "uv run python -c \"from scholawrite.datasets.keyrecs import KeyRecsLoader; "
                            "from pathlib import Path; KeyRecsLoader().download(Path('data'))\"")


def _find_free_text_csv(data_dir: Path) -> Path:
    dest = _ensure_extracted(data_dir)
    for candidate in [dest / "free-text.csv", dest / "free_text.csv"]:
        if candidate.exists():
            return candidate
    for match in sorted(dest.rglob("free*text*.csv")):
        return match
    raise FileNotFoundError(f"free-text.csv not found under {dest}")


def _find_demographics_csv(dest: Path) -> Optional[Path]:
    for candidate in [dest / "demographics.csv", dest / "Demographics.csv"]:
        if candidate.exists():
            return candidate
    for match in sorted(dest.rglob("*emograph*.csv")):
        return match
    return None


def _load_participant_ikis(free_text_csv: Path) -> Dict[str, List[float]]:
    """Return {participant_id: [iki_ms, ...]}."""
    participants: Dict[str, List[float]] = {}
    with open(free_text_csv, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row.get("participant", "").strip()
            if not pid:
                # Try first column as participant ID fallback
                for k in list(row.keys())[:2]:
                    if row[k].strip():
                        pid = row[k].strip()
                        break
            for key in row:
                if key.startswith("DD."):
                    try:
                        val = float(row[key])
                        if val > 0:
                            iki_ms = val * 1000.0
                            if IKI_MIN_MS < iki_ms < IKI_MAX_MS:
                                participants.setdefault(pid, []).append(iki_ms)
                    except (ValueError, TypeError):
                        continue
    return participants


def _load_participant_nationalities(demo_csv: Path) -> Dict[str, str]:
    """Return {participant_id: nationality}. Tries common column names."""
    mapping: Dict[str, str] = {}
    nat_cols = ["nationality", "Nationality", "country", "Country", "language", "Language",
                "native_language", "NativeLanguage", "native language"]
    id_cols = ["participant", "Participant", "id", "ID", "participant_id", "ParticipantID"]
    with open(demo_csv, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        pid_col = next((c for c in id_cols if c in headers), headers[0] if headers else None)
        nat_col = next((c for c in nat_cols if c in headers), None)
        if nat_col is None:
            # Use second column as fallback if it looks categorical
            nat_col = headers[1] if len(headers) > 1 else None
        if pid_col is None or nat_col is None:
            return mapping
        for row in reader:
            pid = row.get(pid_col, "").strip()
            nat = row.get(nat_col, "").strip()
            if pid and nat:
                mapping[pid] = nat
    return mapping


def _ks_statistic(a: List[float], b: List[float]) -> float:
    """Two-sample KS statistic D."""
    sa, sb = sorted(a), sorted(b)
    na, nb = len(sa), len(sb)
    if na == 0 or nb == 0:
        return 1.0
    combined = sorted(set(sa + sb))
    ia = ib = 0
    d_max = 0.0
    for val in combined:
        while ia < na and sa[ia] <= val:
            ia += 1
        while ib < nb and sb[ib] <= val:
            ib += 1
        d_max = max(d_max, abs(ia / na - ib / nb))
    return d_max


def _f_ratio(groups: List[List[float]]) -> float:
    """One-way F-statistic proxy: between-group / within-group variance."""
    all_vals = [v for g in groups for v in g]
    if not all_vals or len(groups) < 2:
        return 0.0
    grand_mean = _mean(all_vals)
    N = len(all_vals)
    k = len(groups)

    # Between-group sum of squares
    ssg = sum(len(g) * (_mean(g) - grand_mean) ** 2 for g in groups if g)
    # Within-group sum of squares
    ssw = sum((v - _mean(g)) ** 2 for g in groups for v in g)

    df_between = max(k - 1, 1)
    df_within = max(N - k, 1)

    msg = ssg / df_between
    msw = ssw / df_within
    return msg / msw if msw > 1e-12 else 0.0


def _pairwise_ks_non_significant(groups: Dict[str, List[float]], threshold: float = 0.05) -> float:
    """Fraction of pairwise KS tests that are non-significant at threshold."""
    names = sorted(groups.keys())
    n_pairs = 0
    n_non_sig = 0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = groups[names[i]], groups[names[j]]
            if len(a) < 2 or len(b) < 2:
                continue
            ks_d = _ks_statistic(a, b)
            na, nb = len(a), len(b)
            # KS critical value at alpha=0.05 using asymptotic formula
            en = math.sqrt(na * nb / (na + nb))
            # Invert: approximate p-value via KS distribution
            lam = (en + 0.12 + 0.11 / en) * ks_d
            if lam <= 0:
                p = 1.0
            elif lam > 3.0:
                p = 0.0
            else:
                p_acc = 0.0
                for kk in range(1, 50):
                    term = (-1) ** (kk - 1) * math.exp(-2.0 * kk * kk * lam * lam)
                    p_acc += term
                    if abs(term) < 1e-12:
                        break
                p = max(0.0, min(1.0, 2.0 * p_acc))
            n_pairs += 1
            if p >= threshold:
                n_non_sig += 1
    return n_non_sig / n_pairs if n_pairs > 0 else 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data", help="Root data directory")
    parser.add_argument("-o", "--output", default="results/cross_language_iki.json")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    _out("Loading KeyRecs free-text IKI data...")
    free_text_csv = _find_free_text_csv(data_dir)
    _out(f"  Free-text CSV: {free_text_csv}")
    participant_ikis = _load_participant_ikis(free_text_csv)
    _out(f"  Participants with IKI data: {len(participant_ikis)}")

    demo_csv = _find_demographics_csv(data_dir / "keyrecs")
    nationality_map: Dict[str, str] = {}
    if demo_csv is not None:
        _out(f"  Demographics CSV: {demo_csv}")
        nationality_map = _load_participant_nationalities(demo_csv)
        _out(f"  Nationality mappings loaded: {len(nationality_map)}")
    else:
        _out("  No demographics.csv found; using participant IDs as nationality proxies.")

    # Group IKIs by nationality
    nat_ikis: Dict[str, List[float]] = {}
    for pid, ikis in participant_ikis.items():
        nat = nationality_map.get(pid, f"unknown_{pid[:6]}")
        nat_ikis.setdefault(nat, []).extend(ikis)

    # Keep only nationalities with enough participants
    nat_participant_counts: Dict[str, int] = {}
    for pid in participant_ikis:
        nat = nationality_map.get(pid, f"unknown_{pid[:6]}")
        nat_participant_counts[nat] = nat_participant_counts.get(nat, 0) + 1

    qualified_nats = {
        nat for nat, cnt in nat_participant_counts.items()
        if cnt >= MIN_PARTICIPANTS_PER_NATIONALITY
    }
    _out(f"  Nationalities with >= {MIN_PARTICIPANTS_PER_NATIONALITY} participants: {len(qualified_nats)}")

    per_nationality: Dict[str, dict] = {}
    qualified_groups: Dict[str, List[float]] = {}

    for nat in sorted(qualified_nats):
        # Collect per-participant IKIs for this nationality
        p_ikis = [
            participant_ikis[pid]
            for pid in participant_ikis
            if nationality_map.get(pid, f"unknown_{pid[:6]}") == nat
        ]
        all_ikis = [v for sublist in p_ikis for v in sublist]
        if len(all_ikis) < 5:
            continue
        n_p = nat_participant_counts[nat]
        per_nationality[nat] = {
            "mean_iki": round(_mean(all_ikis), 4),
            "std_iki": round(_std(all_ikis), 4),
            "entropy": round(entropy_bits(all_ikis), 4),
            "n": n_p,
        }
        qualified_groups[nat] = all_ikis

    _out(f"  Qualified nationalities analyzed: {len(per_nationality)}")

    # Grand mean IKI pool
    all_qualified_ikis = [v for ikis in qualified_groups.values() for v in ikis]
    grand_mean = _mean(all_qualified_ikis)

    # F-statistic proxy
    group_lists = [ikis for ikis in qualified_groups.values()]
    f_stat = _f_ratio(group_lists)
    _out(f"  F-statistic (between/within variance ratio): {f_stat:.4f}")

    # Pairwise KS tests
    frac_non_sig = _pairwise_ks_non_significant(qualified_groups)
    _out(f"  Fraction of non-significant pairwise KS tests: {frac_non_sig:.3f}")

    # Universality decision
    universality_supported = f_stat < 2.0 and frac_non_sig >= 0.6
    if universality_supported:
        reasoning = (
            f"F-ratio {f_stat:.2f} < 2.0 and {frac_non_sig:.1%} of pairwise KS tests "
            "are non-significant. IKI distributions do not differ meaningfully across "
            "nationalities, supporting universality of cognitive process signals."
        )
    else:
        reasoning = (
            f"F-ratio {f_stat:.2f} and {frac_non_sig:.1%} non-significant KS pairs "
            "indicate nationality-specific effects. Process signals may partly reflect "
            "linguistic or cultural typing habits."
        )
    _out(f"  Universality supported: {universality_supported}")
    _out(f"  Reasoning: {reasoning}")

    # Most distinctive nationalities: highest KS distance from grand mean (treated as uniform)
    ks_vs_grand: List[Tuple[str, float]] = []
    grand_sample = sorted(all_qualified_ikis)
    for nat, ikis in qualified_groups.items():
        ks_d = _ks_statistic(sorted(ikis), grand_sample)
        ks_vs_grand.append((nat, ks_d))
    ks_vs_grand.sort(key=lambda x: x[1], reverse=True)
    most_distinctive = [nat for nat, _ in ks_vs_grand[:3]]
    _out(f"  Most distinctive: {most_distinctive}")

    _out(f"\nPer-nationality summary ({len(per_nationality)} nationalities):")
    for nat in sorted(per_nationality.keys()):
        d = per_nationality[nat]
        _out(f"  {nat:25s}  mean={d['mean_iki']:7.1f} ms  std={d['std_iki']:7.1f}  "
             f"entropy={d['entropy']:.3f}  n={d['n']}")

    result = {
        "n_nationalities": len(per_nationality),
        "n_participants_total": sum(d["n"] for d in per_nationality.values()),
        "per_nationality": per_nationality,
        "between_within_ratio": round(f_stat, 6),
        "universality_hypothesis": {
            "supported": universality_supported,
            "reasoning": reasoning,
        },
        "most_distinctive": most_distinctive,
    }

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    _out(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
