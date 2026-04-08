"""Compares 5 synthetic keystroke synthesizer methods against real human composition.

Tests which forgery method is closest to human writing, connecting to adversarial
tier ranking. Paper relevance: characterizes the adversarial landscape empirically.

Usage:
    uv run python analysis/extended/02_synthesizer_comparison.py \\
        --data-dir data/ -o results/synthesizer_comparison.json
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scholawrite.datasets import entropy_bits, lag1_autocorr, load_by_task_type, TaskType
from scholawrite.validation import compare_distributions


def _out(*args, **kwargs):
    sys.stdout.write(" ".join(str(a) for a in args) + kwargs.get("end", "\n"))


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _parse_arff(path: Path) -> tuple[str, list[float]]:
    """Parse an ARFF file and return (synthesizer_name, iki_values_ms).

    Synthesizer name is inferred from the filename stem: Gaussian, Histogram,
    LCBM, NonStationary, Uniform, Average, or HUMAN.
    IKI values are taken from DD.* or FT.* columns; values < 1.0 are scaled to ms.
    """
    stem = path.stem.upper()
    synth_name = "UNKNOWN"
    for candidate in ("GAUSSIAN", "HISTOGRAM", "LCBM", "NONSTATIONARY", "UNIFORM", "AVERAGE", "HUMAN"):
        if candidate in stem:
            synth_name = candidate
            break

    attributes: list[str] = []
    data_started = False
    iki_ms: list[float] = []

    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("%"):
                    continue
                upper = line.upper()
                if upper.startswith("@ATTRIBUTE"):
                    parts = line.split()
                    if len(parts) >= 2:
                        attributes.append(parts[1])
                elif upper.startswith("@DATA"):
                    data_started = True
                    continue

                if not data_started:
                    continue

                values = line.split(",")
                for idx, raw in enumerate(values):
                    if idx >= len(attributes):
                        break
                    attr = attributes[idx].upper()
                    if not ("DD" in attr or "FT" in attr or "FLIGHT" in attr):
                        continue
                    try:
                        val = float(raw)
                    except (ValueError, TypeError):
                        continue
                    if val < 1.0:
                        val *= 1000.0
                    if 0 < val < 30_000:
                        iki_ms.append(val)
    except OSError:
        pass

    return synth_name, iki_ms


def _collect_arff_files(data_dir: Path) -> list[Path]:
    """Find all ARFF files under data_dir/synthetic_liveness/, unpacking zips if needed."""
    dest = data_dir / "synthetic_liveness"
    arff_files = list(dest.rglob("*.arff"))

    if not arff_files:
        for zp in sorted(dest.rglob("*.zip")):
            try:
                with zipfile.ZipFile(zp, "r") as zf:
                    zf.extractall(dest)
            except zipfile.BadZipFile:
                continue
        arff_files = list(dest.rglob("*.arff"))

    return arff_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data", help="Root data directory")
    parser.add_argument("-o", "--output", default="results/synthesizer_comparison.json")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load KLiCKe as human baseline (composition task)
    _out("Loading KLiCKe composition checkpoints as human baseline...")
    human_records = load_by_task_type(data_dir, [TaskType.COMPOSITION]).get("klicke", [])
    if not human_records:
        _out("WARNING: KLiCKe data not found; human baseline will be empty.")

    human_mean_ikis = [r.mean_iki_ms for r in human_records if r.mean_iki_ms > 0]
    _out(f"  Human baseline: {len(human_mean_ikis)} sessions")

    human_baseline = {
        "n_sessions": len(human_mean_ikis),
        "mean_iki": round(_mean(human_mean_ikis), 2) if human_mean_ikis else None,
        "entropy": round(entropy_bits(human_mean_ikis), 4) if human_mean_ikis else None,
        "lag1_autocorr": (
            round(lag1_autocorr(human_mean_ikis), 4)
            if human_mean_ikis and lag1_autocorr(human_mean_ikis) is not None
            else None
        ),
    }

    # Load ARFF files from synthetic_liveness
    _out("\nScanning synthetic_liveness ARFF files...")
    arff_path = data_dir / "synthetic_liveness"
    if not arff_path.exists():
        _out(f"WARNING: {arff_path} does not exist; no synthesizer data available.")
        arff_files = []
    else:
        arff_files = _collect_arff_files(data_dir)
        _out(f"  Found {len(arff_files)} ARFF files")

    # Aggregate IKI values per synthesizer name
    synth_ikis: dict[str, list[float]] = {}
    for arff_file in arff_files:
        name, ikis = _parse_arff(arff_file)
        synth_ikis.setdefault(name, []).extend(ikis)

    _out(f"  Synthesizers found: {sorted(synth_ikis.keys())}")

    # Compute stats per synthesizer and compare against human baseline
    synthesizer_results: dict[str, dict] = {}
    for synth_name, ikis in sorted(synth_ikis.items()):
        if len(ikis) < 5:
            _out(f"  Skipping {synth_name}: only {len(ikis)} IKI values")
            continue

        mean_iki = _mean(ikis)
        ent = entropy_bits(ikis)
        ac = lag1_autocorr(ikis)

        if human_mean_ikis and len(human_mean_ikis) >= 2:
            cmp = compare_distributions(human_mean_ikis, [mean_iki] if len(ikis) < 2 else
                                        # Use per-file mean IKIs for the distribution comparison
                                        ikis[:len(human_mean_ikis)],
                                        synth_name)
            ks_stat = round(cmp.ks_statistic, 4)
            ks_pvalue = round(cmp.ks_pvalue, 6)
            cohens_d = round(cmp.cohens_d, 4)
            wasserstein = round(cmp.wasserstein_distance, 4)
        else:
            ks_stat = None
            ks_pvalue = None
            cohens_d = None
            wasserstein = None

        synthesizer_results[synth_name] = {
            "n_iki_values": len(ikis),
            "mean_iki": round(mean_iki, 2),
            "entropy": round(ent, 4),
            "lag1_autocorr": round(ac, 4) if ac is not None else None,
            "ks_statistic": ks_stat,
            "ks_pvalue": ks_pvalue,
            "cohens_d": cohens_d,
            "wasserstein": wasserstein,
        }

        _out(f"  {synth_name}: n={len(ikis)}, mean_iki={mean_iki:.1f}ms, "
             f"entropy={ent:.3f}b, KS={ks_stat}")

    # Rank synthesizers by KS statistic (lower = closer to human)
    ranked = sorted(
        [(name, info["ks_statistic"]) for name, info in synthesizer_results.items()
         if info["ks_statistic"] is not None],
        key=lambda x: x[1],
    )
    ranking = [name for name, _ in ranked]

    _out("\nSynthesizer ranking (closest to human first):")
    for rank, (name, ks) in enumerate(ranked, 1):
        _out(f"  {rank}. {name}: KS={ks:.4f}")

    result = {
        "synthesizers": synthesizer_results,
        "ranking": ranking,
        "human_baseline": human_baseline,
    }

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    _out(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
