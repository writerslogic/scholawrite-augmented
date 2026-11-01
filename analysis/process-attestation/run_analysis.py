#!/usr/bin/env python3
"""
Cognitive Process Signatures in Keystroke Data: Empirical Validation
====================================================================
Single-script pipeline for exploratory timing statistics in ScholaWrite revision
events and the KLiCKe keystroke corpus, with a simulated retype attack and a
leave-one-writer-out classifier. Results are descriptive; they do not by
themselves validate cognitive process signatures.

ScholaWrite (HuggingFace) records text-change events, so its intervals are
inter-event intervals (mean gap of several seconds), not inter-keystroke
intervals. KLiCKe (Kaggle) is keystroke-level. Entropy values are plug-in
estimates on 5 ms-quantized intervals and shift by one bit per halving or
doubling of the bin width.

Pipeline stages:
  1. Environment check (auto-download KLiCKe if missing)
  2. ScholaWrite analysis (HuggingFace; always runs)
  3. KLiCKe corpus analysis (compute per-writer features + windowed data)
  4. Adversarial retype simulation (4 attack strategies)
  5. ROC classification (leave-one-writer-out CV)
  6. Effect sizes and figures (reuses Stage 3 pre-computed features)
  7. Summary report

Usage:
  pip install -r requirements.txt
  python run_analysis.py

Author: David Condrey
"""

import json
import logging
import random
import sys
import time
from pathlib import Path

# Suppress HTTP noise before importing HuggingFace libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

# Configure logging to stdout with bare format
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
QUANTIZATION_MS = 5
WINDOW_SIZE_MS = 30000
IKI_MIN = 10
IKI_MAX = 60000


# ============================================================================
# Shared Utilities
# ============================================================================

def compute_entropy_bits(iki_values, q=QUANTIZATION_MS):
    """Shannon entropy of IKI distribution quantized to q-ms bins, in bits."""
    import numpy as np
    quantized = (iki_values // q) * q
    if hasattr(quantized, 'value_counts'):
        probs = quantized.value_counts(normalize=True).values
    else:
        _, counts = np.unique(quantized, return_counts=True)
        probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs + 1e-15))


def autocorr_lag1(ikis):
    """Lag-1 autocorrelation of IKI sequence."""
    import numpy as np
    if len(ikis) < 3:
        return 0.0
    x = ikis - ikis.mean()
    c0 = np.dot(x, x)
    if c0 == 0:
        return 0.0
    return float(np.dot(x[:-1], x[1:]) / c0)


def _fast_spearman(x, y):
    """Fast Spearman rho with proper tie-averaged ranking.
    Returns rho only (no p-value). For inner loops where p is not needed."""
    import numpy as np
    n = len(x)
    if n < 3:
        return np.nan

    def _rank_with_ties(arr):
        sorter = arr.argsort()
        ranks = np.empty(n, dtype=float)
        ranks[sorter] = np.arange(1, n + 1, dtype=float)
        sorted_arr = arr[sorter]
        i = 0
        while i < n:
            j = i
            while j < n and sorted_arr[j] == sorted_arr[i]:
                j += 1
            if j > i + 1:
                avg_rank = (i + 1 + j) / 2.0
                for k in range(i, j):
                    ranks[sorter[k]] = avg_rank
            i = j
        return ranks

    rx = _rank_with_ties(np.asarray(x, dtype=float))
    ry = _rank_with_ties(np.asarray(y, dtype=float))
    dx, dy = rx - rx.mean(), ry - ry.mean()
    denom = np.sqrt(np.dot(dx, dx) * np.dot(dy, dy))
    if denom == 0:
        return np.nan
    return np.dot(dx, dy) / denom


def clc_spearman(cog_loads, ikis):
    """Spearman correlation between cognitive load levels and IKIs.
    Returns (rho, pval) or (nan, nan) on failure."""
    import numpy as np
    from scipy import stats
    if len(ikis) < 10 or np.std(cog_loads) == 0:
        return np.nan, np.nan
    try:
        rho, pval = stats.spearmanr(cog_loads, ikis)
        if np.isnan(rho):
            return np.nan, np.nan
        return float(rho), float(pval)
    except Exception:
        return np.nan, np.nan


def filter_valid_iki(iki_series):
    """Filter IKIs to valid range, dropping NaN."""
    valid = iki_series.dropna()
    return valid[(valid > IKI_MIN) & (valid < IKI_MAX)]


def cohens_d(a, b):
    """Pooled-variance Cohen's d."""
    import numpy as np
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    pooled = np.sqrt(((na - 1) * a.std(ddof=1)**2 + (nb - 1) * b.std(ddof=1)**2) / (na + nb - 2))
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


def glass_delta(a, b):
    """Glass's delta (using b's std as denominator)."""
    s = b.std(ddof=1)
    return float((a.mean() - b.mean()) / s) if s > 0 else 0.0


def cliffs_delta(a, b):
    """Cliff's delta effect size."""
    from scipy import stats
    u_stat, _ = stats.mannwhitneyu(a, b, alternative="two-sided")
    return float((2 * u_stat) / (len(a) * len(b)) - 1)


def progress(i, total, every=500, label=""):
    """Print progress every N iterations."""
    if (i + 1) % every == 0 or (i + 1) == total:
        log.info(f"    {label}{i + 1}/{total} processed...")
        sys.stdout.flush()


# ============================================================================
# STAGE 1: Environment Check
# ============================================================================

def _try_download_klicke_opendatasets():
    """Attempt to download KLiCKe data from Kaggle using opendatasets."""
    try:
        import opendatasets as od
    except ImportError:
        return False
    log.info("\n  Attempting KLiCKe download via opendatasets (Kaggle)...")
    log.info("  You will be prompted for your Kaggle username and API key.")
    log.info("  (Find your API key at https://www.kaggle.com/settings -> API -> Create New Token)")
    try:
        od.download(
            "https://www.kaggle.com/competitions/linking-writing-processes-to-writing-quality",
            data_dir=str(SCRIPT_DIR),
        )
        return True
    except Exception as e:
        log.warning(f"  opendatasets download failed: {e}")
        return False


def _try_download_klicke_kaggle_api():
    """Attempt to download KLiCKe data from Kaggle using the kaggle CLI/API."""
    try:
        import kaggle  # noqa: F401
    except ImportError:
        return False
    log.info("\n  Attempting KLiCKe download via kaggle API...")
    import subprocess
    try:
        subprocess.check_call([
            sys.executable, "-m", "kaggle", "competitions", "download",
            "-c", "linking-writing-processes-to-writing-quality",
            "-p", str(SCRIPT_DIR),
        ])
        zip_path = SCRIPT_DIR / "linking-writing-processes-to-writing-quality.zip"
        if zip_path.exists():
            import zipfile
            dest = SCRIPT_DIR / "linking-writing-processes-to-writing-quality"
            dest.mkdir(exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(dest)
            zip_path.unlink()
        return True
    except Exception as e:
        log.warning(f"  kaggle API download failed: {e}")
        return False


def _detect_klicke_format():
    """Detect which KLiCKe data format is available.
    Returns (format, has_klicke) where format is 'local', 'kaggle', or None."""
    local_csv_dir = (SCRIPT_DIR / "klicke" / "Files" / "WritingTask"
                     / "WritingTask" / "keystrokelogs" / "csv")
    if local_csv_dir.is_dir() and any(local_csv_dir.glob("*.csv")):
        return "local", True

    kaggle_dir = SCRIPT_DIR / "linking-writing-processes-to-writing-quality"
    if kaggle_dir.is_dir() and (kaggle_dir / "train_logs.csv").exists():
        return "kaggle", True

    if (SCRIPT_DIR / "train_logs.csv").exists():
        return "kaggle", True

    return None, False


def stage_environment_check():
    """Verify Python version, required packages, and data availability."""
    log.info("=" * 72)
    log.info("STAGE 1: Environment Check")
    log.info("=" * 72)

    v = sys.version_info
    log.info(f"\n  Python version: {v.major}.{v.minor}.{v.micro}")
    if v < (3, 8):
        log.error("  ERROR: Python >= 3.8 required.")
        sys.exit(1)
    log.info("  Python version check: PASSED")

    required = {
        "numpy": "numpy", "scipy": "scipy", "pandas": "pandas",
        "matplotlib": "matplotlib", "sklearn": "scikit-learn",
        "datasets": "datasets",
    }
    missing = []
    for module, pip_name in required.items():
        try:
            mod = __import__(module)
            version = getattr(mod, "__version__", "unknown")
            log.info(f"  {pip_name:20s} {version}")
        except ImportError:
            log.info(f"  {pip_name:20s} MISSING")
            missing.append(pip_name)
    if missing:
        log.error(f"\n  ERROR: Missing packages: {', '.join(missing)}")
        log.error(f"  Install with: pip install {' '.join(missing)}")
        sys.exit(1)
    log.info("  Package check: PASSED")

    for module, pip_name in [("opendatasets", "opendatasets")]:
        try:
            mod = __import__(module)
            version = getattr(mod, "__version__", "unknown")
            log.info(f"  {pip_name:20s} {version} (optional)")
        except ImportError as e:
            log.info(f"  {pip_name:20s} not importable ({e}) (optional, for Kaggle auto-download)")

    klicke_format, has_klicke = _detect_klicke_format()

    if not has_klicke:
        log.info("\n  KLiCKe data: NOT FOUND locally. Attempting auto-download...")
        if _try_download_klicke_opendatasets():
            klicke_format, has_klicke = _detect_klicke_format()
        if not has_klicke and _try_download_klicke_kaggle_api():
            klicke_format, has_klicke = _detect_klicke_format()

    if has_klicke:
        if klicke_format == "local":
            local_csv_dir = (SCRIPT_DIR / "klicke" / "Files" / "WritingTask"
                             / "WritingTask" / "keystrokelogs" / "csv")
            n_csv = len(list(local_csv_dir.glob("*.csv")))
            log.info(f"\n  KLiCKe data: FOUND ({n_csv} CSV files, local format)")
            klicke_scores = (SCRIPT_DIR / "klicke" / "Files" / "WritingTask"
                             / "WritingTask" / "holistic_scores.csv")
            log.info(f"  Holistic scores: {'FOUND' if klicke_scores.exists() else 'MISSING'}")
        elif klicke_format == "kaggle":
            kaggle_dir = SCRIPT_DIR / "linking-writing-processes-to-writing-quality"
            train_logs = kaggle_dir / "train_logs.csv"
            if not train_logs.exists():
                train_logs = SCRIPT_DIR / "train_logs.csv"
            with open(train_logs) as fh:
                n_lines = sum(1 for _ in fh) - 1
            log.info(f"\n  KLiCKe data: FOUND (Kaggle format, ~{n_lines:,} events in train_logs.csv)")
            train_scores = kaggle_dir / "train_scores.csv"
            if not train_scores.exists():
                train_scores = SCRIPT_DIR / "train_scores.csv"
            log.info(f"  Holistic scores: {'FOUND' if train_scores.exists() else 'MISSING'}")
    else:
        log.info("\n  KLiCKe data: NOT FOUND (auto-download was not successful)")
        log.info("  To include KLiCKe analysis, either:")
        log.info("    1. Install opendatasets (pip install opendatasets) and re-run")
        log.info("       (you will be prompted for Kaggle credentials)")
        log.info("    2. Download manually from:")
        log.info("       https://www.kaggle.com/competitions/linking-writing-processes-to-writing-quality/data")
        log.info("       Place train_logs.csv and train_scores.csv in:")
        log.info(f"       {SCRIPT_DIR / 'linking-writing-processes-to-writing-quality' / ''}")
        log.info("    3. Or place the original KLiCKe corpus at:")
        local_csv_dir = (SCRIPT_DIR / "klicke" / "Files" / "WritingTask"
                         / "WritingTask" / "keystrokelogs" / "csv")
        log.info(f"       {local_csv_dir}")
        log.info("       Download from: https://keystroke.cs.cmu.edu/ (KLiCKe corpus)")
        log.info("  Continuing with ScholaWrite-only mode.")

    return has_klicke, klicke_format


# ============================================================================
# STAGE 2: ScholaWrite Analysis
# ============================================================================

def stage_scholawrite():
    """Analyze ScholaWrite dataset from HuggingFace."""
    import numpy as np
    import pandas as pd
    from scipy import stats
    from scipy.stats import wilcoxon

    from datasets import load_dataset

    log.info("\n" + "=" * 72)
    log.info("STAGE 2: ScholaWrite Analysis")
    log.info("=" * 72)

    results = {}

    # --- 2a. Load dataset ---
    log.info("\n  [2a] Loading ScholaWrite dataset from HuggingFace...")
    ds = load_dataset("minnesotanlp/scholawrite")
    df_train = pd.DataFrame(ds["train"])
    df_test = pd.DataFrame(ds["test"])
    df_all = pd.concat([df_train, df_test], ignore_index=True)

    log.info(f"  Total events: {len(df_all):,}")
    log.info(f"  Projects: {df_all['project'].nunique()}")
    log.info(f"  Authors:  {df_all['author'].nunique()}")
    log.info(f"  Labels:   {df_all['label'].nunique()}")
    log.info(f"  Columns:  {list(df_all.columns)}")

    results["total_events"] = len(df_all)
    results["n_projects"] = int(df_all["project"].nunique())
    results["n_authors"] = int(df_all["author"].nunique())

    # --- 2b. Compute inter-event intervals ---
    log.info("\n  [2b] Computing inter-event intervals between consecutive recorded text changes...")
    df_all = df_all.sort_values(["project", "author", "timestamp"]).reset_index(drop=True)
    # ScholaWrite rows are text-change events, not keystrokes; the column keeps the
    # iki_ms name for compatibility, but the mean gap is on the order of seconds.
    df_all["iki_ms"] = df_all.groupby(["project", "author"])["timestamp"].diff()
    valid_iki = filter_valid_iki(df_all["iki_ms"])

    log.info(f"  Valid IKIs:  {len(valid_iki):,}")
    log.info(f"  Mean:   {valid_iki.mean():.1f} ms")
    log.info(f"  Median: {valid_iki.median():.1f} ms")
    log.info(f"  Std:    {valid_iki.std():.1f} ms")
    log.info(f"  Range:  [{valid_iki.min():.0f}, {valid_iki.max():.0f}] ms")

    results["valid_ikis"] = len(valid_iki)
    results["iki_mean_ms"] = float(valid_iki.mean())
    results["iki_median_ms"] = float(valid_iki.median())
    results["iki_std_ms"] = float(valid_iki.std())

    # --- 2c. Shannon entropy ---
    log.info("\n  [2c] Computing Shannon entropy of quantized IKI distributions...")

    window_entropies = []
    for (proj, author), group in df_all.groupby(["project", "author"]):
        group = group.sort_values("timestamp")
        timestamps = group["timestamp"].values
        ikis = group["iki_ms"].values
        if len(timestamps) < 10:
            continue
        t_start = timestamps[0]
        while t_start < timestamps[-1]:
            t_end = t_start + WINDOW_SIZE_MS
            mask = (timestamps >= t_start) & (timestamps < t_end)
            w_ikis = ikis[mask]
            w_ikis = w_ikis[~np.isnan(w_ikis)]
            w_ikis = w_ikis[(w_ikis > IKI_MIN) & (w_ikis < IKI_MAX)]
            if len(w_ikis) >= 10:
                window_entropies.append({
                    "project": proj, "author": author,
                    "entropy_bits": compute_entropy_bits(w_ikis),
                    "n_events": len(w_ikis),
                    "mean_iki": np.mean(w_ikis),
                })
            t_start = t_end

    df_entropy = pd.DataFrame(window_entropies)
    overall_entropy = compute_entropy_bits(valid_iki)

    log.info(f"  30s windows analyzed: {len(df_entropy):,}")
    log.info(f"  Per-window entropy:   mean={df_entropy['entropy_bits'].mean():.2f}, "
             f"median={df_entropy['entropy_bits'].median():.2f}, "
             f"std={df_entropy['entropy_bits'].std():.2f} bits")
    log.info(f"  Population per-interval entropy ({QUANTIZATION_MS} ms bins): {overall_entropy:.2f} bits")
    log.info(f"  50 x per-interval entropy (independence upper bound): {50 * overall_entropy:.1f} bits")
    log.info(f"  Windows above 3.0-bit threshold: "
             f"{(df_entropy['entropy_bits'] > 3.0).sum()}/{len(df_entropy)} "
             f"({100 * (df_entropy['entropy_bits'] > 3.0).mean():.1f}%)")

    results["interval_definition"] = "inter-event interval between recorded text changes (not keystrokes)"
    results["entropy_quantization_ms"] = QUANTIZATION_MS
    results["entropy_per_iki_bits"] = float(overall_entropy)
    results["entropy_per_checkpoint_bits"] = float(50 * overall_entropy)
    results["entropy_window_mean"] = float(df_entropy["entropy_bits"].mean())

    # --- 2d. Cognitive Load Correlation (CLC) ---
    log.info("\n  [2d] Computing Cognitive Load Correlation (CLC)...")

    # Cognitive-load level from the dataset's own high-level category (an earlier
    # label-name map used names absent from the dataset, so ~10% of events,
    # including PLANNING labels, silently received the lowest load).
    COGNITIVE_LOAD_SW = {"PLANNING": 3, "REVISION": 2, "IMPLEMENTATION": 1}
    unmapped = sorted(set(df_all["high-level"].dropna().unique()) - set(COGNITIVE_LOAD_SW))
    if unmapped or df_all["high-level"].isna().any():
        raise ValueError(f"ScholaWrite high-level categories without a load mapping: {unmapped}")
    df_all["cog_load"] = df_all["high-level"].map(COGNITIVE_LOAD_SW).astype(float)

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
            w_ikis = ikis[mask]
            w_cog = cog_loads[mask]
            valid = ~np.isnan(w_ikis) & (w_ikis > IKI_MIN) & (w_ikis < IKI_MAX)
            w_ikis = w_ikis[valid]
            w_cog = w_cog[valid]
            if len(w_ikis) >= 10:
                rho, pval = clc_spearman(w_cog, w_ikis)
                if not np.isnan(rho):
                    clc_values.append({
                        "project": proj, "author": author,
                        "rho": rho, "pval": pval, "n": len(w_ikis),
                    })
            t_start = t_end

    df_clc = pd.DataFrame(clc_values)
    log.info(f"  Windows with CLC computed: {len(df_clc):,}")
    log.info(f"  Mean CLC (Spearman rho): {df_clc['rho'].mean():.4f}")
    log.info(f"  Median CLC:              {df_clc['rho'].median():.4f}")
    log.info(f"  Std CLC:                 {df_clc['rho'].std():.4f}")
    log.info(f"  Windows with rho > 0.15: {(df_clc['rho'] > 0.15).sum()}/{len(df_clc)} "
             f"({100 * (df_clc['rho'] > 0.15).mean():.1f}%)")
    log.info(f"  Windows with rho > 0.0:  {(df_clc['rho'] > 0.0).sum()}/{len(df_clc)} "
             f"({100 * (df_clc['rho'] > 0.0).mean():.1f}%)")
    log.info(f"  Significant (p<0.05):    {(df_clc['pval'] < 0.05).sum()}/{len(df_clc)} "
             f"({100 * (df_clc['pval'] < 0.05).mean():.1f}%)")

    all_valid = df_all.dropna(subset=["iki_ms"]).copy()
    all_valid = all_valid[(all_valid["iki_ms"] > IKI_MIN) & (all_valid["iki_ms"] < IKI_MAX)]
    overall_rho, overall_pval = stats.spearmanr(all_valid["cog_load"], all_valid["iki_ms"])
    log.info(f"\n  Overall CLC (events pooled across authors; descriptive, p ignores author clustering): "
             f"rho={overall_rho:.4f}, p={overall_pval:.2e}")

    results["clc_overall_rho"] = float(overall_rho)
    results["clc_overall_pval"] = float(overall_pval)
    results["clc_window_mean_rho"] = float(df_clc["rho"].mean())
    results["clc_windows_total"] = len(df_clc)
    results["clc_windows_above_015"] = int((df_clc["rho"] > 0.15).sum())
    results["clc_pct_above_015"] = float(100 * (df_clc["rho"] > 0.15).mean())

    # Per-author CLC with Wilcoxon signed-rank test
    per_author_rho = []
    for _author, group in all_valid.groupby("author"):
        if len(group) >= 20 and group["cog_load"].std() > 0:
            r, p = stats.spearmanr(group["cog_load"], group["iki_ms"])
            if not np.isnan(r):
                per_author_rho.append(r)

    if per_author_rho:
        w_stat, w_pval = wilcoxon(per_author_rho)
        log.info(f"  Per-author CLC (Wilcoxon): median rho={np.median(per_author_rho):.4f}, "
                 f"p={w_pval:.2e}, n={len(per_author_rho)}")
        results["clc_per_author_median_rho"] = float(np.median(per_author_rho))
        results["clc_per_author_wilcoxon_p"] = float(w_pval)

    # Note: an earlier 2e step split windows by their own CLC value (rho > 0.15 vs
    # rho <= 0) and reported an effect size on rho; that contrast was tautological
    # and has been removed.

    # --- 2f. IKI by cognitive operation type ---
    log.info("\n  [2f] IKI Distribution by Cognitive Operation Type...")
    for hl in ["PLANNING", "IMPLEMENTATION", "REVISION"]:
        subset = all_valid[all_valid["high-level"] == hl]["iki_ms"]
        if len(subset) > 0:
            log.info(f"  {hl:15s}: n={len(subset):>6,}, mean={subset.mean():>8.1f}ms, "
                     f"median={subset.median():>7.1f}ms, std={subset.std():>8.1f}ms")

    demanding = all_valid[all_valid["high-level"].isin(["PLANNING", "REVISION"])]["iki_ms"]
    simple = all_valid[all_valid["high-level"] == "IMPLEMENTATION"]["iki_ms"]

    if len(demanding) > 0 and len(simple) > 0:
        ratio = demanding.mean() / simple.mean()
        d_stat, d_pval = stats.mannwhitneyu(demanding, simple, alternative="greater")
        cohens_d_hl = cohens_d(demanding, simple)

        log.info(f"\n  Demanding/Simple IKI ratio: {ratio:.2f}x")
        log.info(f"  Mann-Whitney p: {d_pval:.2e}")
        log.info(f"  Cohen's d:      {cohens_d_hl:.3f}")

        results["iki_ratio_demanding_simple"] = float(ratio)
        results["iki_cohens_d_highlevel"] = float(cohens_d_hl)

    return results


# ============================================================================
# STAGE 3: KLiCKe Analysis
# ============================================================================

COGNITIVE_LOAD_KL = {"Nonproduction": 3, "Remove/Cut": 2, "Input": 1}


def _load_klicke_data(klicke_format):
    """Load KLiCKe data from either local or Kaggle format.
    Returns (df_all, scores_file_path)."""
    import pandas as pd

    if klicke_format == "local":
        CSV_DIR = (SCRIPT_DIR / "klicke" / "Files" / "WritingTask"
                   / "WritingTask" / "keystrokelogs" / "csv")
        SCORES_FILE = (SCRIPT_DIR / "klicke" / "Files" / "WritingTask"
                       / "WritingTask" / "holistic_scores.csv")

        log.info("  Loading KLiCKe from local per-writer CSVs...")
        all_rows = []
        csv_files = sorted(CSV_DIR.glob("*.csv"))
        n_files = len(csv_files)
        log.info(f"  Found {n_files} writer files")

        n_failed = 0
        for i, f in enumerate(csv_files):
            progress(i, n_files, every=1000, label="files: ")
            try:
                try:
                    df = pd.read_csv(f)
                except UnicodeDecodeError:
                    # A few per-writer CSVs carry Windows-1252 bytes in the text
                    # column, which this analysis never reads.
                    df = pd.read_csv(f, encoding="cp1252")
                df["writer_id"] = f.stem
                all_rows.append(df)
            except Exception as e:
                n_failed += 1
                log.warning(f"  Skipping {f.name}: {type(e).__name__}: {e}")
        if not all_rows:
            raise RuntimeError(f"No KLiCKe CSV files could be read from {CSV_DIR}")
        log.info(f"  Loaded {len(all_rows)}/{n_files} writer files ({n_failed} skipped)")

        df_all = pd.concat(all_rows, ignore_index=True)
        return df_all, SCORES_FILE

    elif klicke_format == "kaggle":
        kaggle_dir = SCRIPT_DIR / "linking-writing-processes-to-writing-quality"
        train_logs = kaggle_dir / "train_logs.csv"
        if not train_logs.exists():
            train_logs = SCRIPT_DIR / "train_logs.csv"

        SCORES_FILE = kaggle_dir / "train_scores.csv"
        if not SCORES_FILE.exists():
            SCORES_FILE = SCRIPT_DIR / "train_scores.csv"

        log.info("  Loading KLiCKe from Kaggle train_logs.csv...")
        df_raw = pd.read_csv(train_logs)
        log.info(f"  Raw events loaded: {len(df_raw):,}")

        col_map = {
            "id": "writer_id", "down_time": "DownTime", "up_time": "UpTime",
            "action_time": "ActionTime", "activity": "Activity",
            "down_event": "DownEvent", "up_event": "UpEvent",
            "cursor_position": "Cursorposition", "word_count": "WordCount",
            "text_change": "TextChange", "event_id": "DownEventID",
        }
        df_all = df_raw.rename(columns=col_map)
        df_all["writer_id"] = df_all["writer_id"].astype(str)
        return df_all, SCORES_FILE
    else:
        raise ValueError(f"Unknown KLiCKe format: {klicke_format}")


def stage_klicke(klicke_format):
    """Analyze KLiCKe corpus keystroke logs.

    Returns (results, df_all, writer_valid, df_writer_features, df_window_features)
    where the last two
    are pre-computed per-writer and per-window features for Stage 6."""
    import numpy as np
    import pandas as pd
    from scipy import stats

    log.info("\n" + "=" * 72)
    log.info("STAGE 3: KLiCKe Corpus Analysis")
    log.info("=" * 72)

    results = {}

    # --- 3a. Load keystroke logs ---
    log.info("\n  [3a] Loading KLiCKe keystroke logs...")
    df_all, SCORES_FILE = _load_klicke_data(klicke_format)
    log.info(f"  Total events: {len(df_all):,}")
    log.info(f"  Writers:      {df_all['writer_id'].nunique():,}")
    log.info(f"  Columns:      {list(df_all.columns)}")

    results["n_writers"] = int(df_all["writer_id"].nunique())
    results["total_events"] = len(df_all)

    log.info("  Activity distribution:")
    for act, count in df_all["Activity"].value_counts().items():
        log.info(f"    {act}: {count:,} ({100 * count / len(df_all):.1f}%)")

    # --- 3b. Compute IKI ---
    log.info("\n  [3b] Computing IKI...")
    df_all = df_all.sort_values(["writer_id", "DownTime"]).reset_index(drop=True)
    df_all["iki_ms"] = df_all.groupby("writer_id")["DownTime"].diff()

    valid_iki = filter_valid_iki(df_all["iki_ms"])
    log.info(f"  Valid IKIs:  {len(valid_iki):,}")
    log.info(f"  Mean:   {valid_iki.mean():.1f} ms")
    log.info(f"  Median: {valid_iki.median():.1f} ms")
    log.info(f"  Std:    {valid_iki.std():.1f} ms")

    results["valid_ikis"] = len(valid_iki)
    results["iki_mean_ms"] = float(valid_iki.mean())
    results["iki_median_ms"] = float(valid_iki.median())

    # Prepare valid data with cognitive load for reuse
    df_all["cog_load"] = df_all["Activity"].map(COGNITIVE_LOAD_KL).fillna(1)
    writer_valid = df_all.dropna(subset=["iki_ms"]).copy()
    writer_valid = writer_valid[(writer_valid["iki_ms"] > IKI_MIN) & (writer_valid["iki_ms"] < IKI_MAX)]

    per_writer = writer_valid.groupby("writer_id")["iki_ms"].agg(["mean", "median", "std", "count"])
    log.info(f"  Per-writer IKI means: pop_mean={per_writer['mean'].mean():.1f}, "
             f"pop_median={per_writer['mean'].median():.1f}")

    # --- 3c. Entropy ---
    log.info("\n  [3c] Computing entropy...")
    overall_entropy = compute_entropy_bits(valid_iki)
    log.info(f"  Population per-IKI entropy ({QUANTIZATION_MS} ms bins): {overall_entropy:.2f} bits")
    log.info(f"  50 x per-IKI entropy (independence upper bound): {50 * overall_entropy:.1f} bits")

    results["entropy_per_iki_bits"] = float(overall_entropy)
    results["entropy_per_checkpoint_bits"] = float(50 * overall_entropy)

    # Per-writer entropy (also used by Stage 6)
    log.info("\n  Computing per-writer entropy and features...")
    writer_feature_records = []
    grouped = writer_valid.groupby("writer_id")
    n_writers = len(grouped)

    for i, (wid, grp) in enumerate(grouped):
        progress(i, n_writers, every=1000, label="writers: ")
        if len(grp) < 30:
            continue
        iki_vals = grp["iki_ms"]
        ent = compute_entropy_bits(iki_vals)
        rho, pval = clc_spearman(grp["cog_load"].values, grp["iki_ms"].values)
        iki_var = float(np.log(iki_vals.var(ddof=0) + 1))
        pause_freq = float((iki_vals > 2000).mean())

        writer_feature_records.append({
            "writer_id": wid, "entropy": ent,
            "clc_rho": rho, "clc_pval": pval if not np.isnan(pval) else 1.0,
            "iki_log_var": iki_var, "pause_freq": pause_freq,
            "n": len(grp),
        })

    df_writer_features = pd.DataFrame(writer_feature_records)
    df_went = df_writer_features[["writer_id", "entropy", "n"]].rename(
        columns={"entropy": "entropy_bits"})
    df_wclc = df_writer_features[["writer_id", "clc_rho", "clc_pval", "n"]].rename(
        columns={"clc_rho": "rho", "clc_pval": "pval"}).dropna(subset=["rho"])

    log.info(f"\n  Per-writer entropy ({len(df_went)} writers with >=30 keystrokes):")
    log.info(f"    Mean:   {df_went['entropy_bits'].mean():.2f} bits")
    log.info(f"    Median: {df_went['entropy_bits'].median():.2f} bits")
    log.info(f"    Std:    {df_went['entropy_bits'].std():.2f} bits")
    log.info(f"    Above 3.0 bits: {(df_went['entropy_bits'] > 3.0).sum()}/{len(df_went)} "
             f"({100 * (df_went['entropy_bits'] > 3.0).mean():.1f}%)")

    # 30s window entropy
    window_entropies = []
    for i, (wid, group) in enumerate(writer_valid.groupby("writer_id")):
        progress(i, n_writers, every=1000, label="window entropy: ")
        group = group.sort_values("DownTime")
        timestamps = group["DownTime"].values
        ikis = group["iki_ms"].values
        if len(timestamps) < 10:
            continue
        t_start = timestamps[0]
        while t_start < timestamps[-1]:
            t_end = t_start + WINDOW_SIZE_MS
            mask = (timestamps >= t_start) & (timestamps < t_end)
            w_ikis = ikis[mask]
            w_ikis = w_ikis[~np.isnan(w_ikis)]
            w_ikis = w_ikis[(w_ikis > IKI_MIN) & (w_ikis < IKI_MAX)]
            if len(w_ikis) >= 10:
                window_entropies.append({
                    "writer_id": wid,
                    "entropy_bits": compute_entropy_bits(w_ikis),
                    "n_keystrokes": len(w_ikis),
                })
            t_start = t_end

    df_winent = pd.DataFrame(window_entropies)
    log.info(f"\n  30s window entropy ({len(df_winent):,} windows):")
    log.info(f"    Mean: {df_winent['entropy_bits'].mean():.2f} bits")
    log.info(f"    Above 3.0 bits: {(df_winent['entropy_bits'] > 3.0).sum()}/{len(df_winent)} "
             f"({100 * (df_winent['entropy_bits'] > 3.0).mean():.1f}%)")

    # --- 3d. CLC ---
    log.info("\n  [3d] Computing CLC...")
    overall_rho, overall_pval = stats.spearmanr(writer_valid["cog_load"], writer_valid["iki_ms"])
    log.info(f"  Overall CLC: rho={overall_rho:.4f}, p={overall_pval:.2e}")

    results["clc_overall_rho"] = float(overall_rho)

    log.info(f"\n  Per-writer CLC ({len(df_wclc)} writers):")
    log.info(f"    Mean rho:  {df_wclc['rho'].mean():.4f}")
    log.info(f"    Median rho: {df_wclc['rho'].median():.4f}")
    log.info(f"    Positive CLC: {(df_wclc['rho'] > 0.0).sum()}/{len(df_wclc)} "
             f"({100 * (df_wclc['rho'] > 0.0).mean():.1f}%)")
    log.info(f"    Significant (p<0.05): {(df_wclc['pval'] < 0.05).sum()}/{len(df_wclc)} "
             f"({100 * (df_wclc['pval'] < 0.05).mean():.1f}%)")

    results["clc_per_writer_mean_rho"] = float(df_wclc["rho"].mean())
    results["clc_per_writer_median_rho"] = float(df_wclc["rho"].median())
    results["clc_pct_positive"] = float(100 * (df_wclc["rho"] > 0.0).mean())
    results["clc_pct_significant"] = float(100 * (df_wclc["pval"] < 0.05).mean())

    # --- 3e. IKI by activity ---
    log.info("\n  [3e] IKI by activity type:")
    for act in ["Input", "Remove/Cut", "Nonproduction"]:
        subset = writer_valid[writer_valid["Activity"] == act]["iki_ms"]
        if len(subset) > 0:
            log.info(f"    {act:20s}: n={len(subset):>9,}, mean={subset.mean():>8.1f}ms, "
                     f"median={subset.median():>7.1f}ms, std={subset.std():>8.1f}ms")

    demanding = writer_valid[writer_valid["Activity"].isin(["Nonproduction", "Remove/Cut"])]["iki_ms"]
    simple = writer_valid[writer_valid["Activity"] == "Input"]["iki_ms"]
    if len(demanding) > 0 and len(simple) > 0:
        ratio = demanding.mean() / simple.mean()
        cd = cohens_d(demanding, simple)
        log.info(f"\n    Demanding/Simple ratio: {ratio:.2f}x, Cohen's d: {cd:.3f}")
        results["iki_ratio_demanding_simple"] = float(ratio)
        results["iki_cohens_d_activity"] = float(cd)

    # --- 3f. Quality score correlation ---
    if SCORES_FILE.exists():
        log.info("\n  [3f] Quality score correlation...")
        scores = pd.read_csv(SCORES_FILE, encoding="latin-1")
        scores.columns = [c.strip() for c in scores.columns]
        if "id" in scores.columns and "ID" not in scores.columns:
            scores.rename(columns={"id": "ID", "score": "Score"}, inplace=True)
        scores["writer_id"] = scores["ID"].astype(str)

        per_writer_metrics = writer_valid.groupby("writer_id").agg(
            mean_iki=("iki_ms", "mean"), std_iki=("iki_ms", "std"),
        ).reset_index()

        if len(df_wclc) > 0:
            per_writer_metrics = per_writer_metrics.merge(
                df_wclc[["writer_id", "rho"]].rename(columns={"rho": "clc_rho"}),
                on="writer_id", how="left"
            )
        if len(df_went) > 0:
            per_writer_metrics = per_writer_metrics.merge(
                df_went[["writer_id", "entropy_bits"]],
                on="writer_id", how="left"
            )

        merged = per_writer_metrics.merge(scores[["writer_id", "Score"]], on="writer_id", how="inner")
        log.info(f"    Writers with scores: {len(merged):,}")

        if len(merged) > 10:
            for col in ["mean_iki", "std_iki", "clc_rho", "entropy_bits"]:
                if col in merged.columns:
                    valid_m = merged.dropna(subset=[col, "Score"])
                    if len(valid_m) > 10:
                        rho, pval = stats.spearmanr(valid_m[col], valid_m["Score"])
                        log.info(f"    {col:15s} vs Score: rho={rho:.4f}, p={pval:.2e} (n={len(valid_m)})")

    # --- 3g. Pre-compute sliding-window composition vs transcription features for Stage 6 ---
    log.info("\n  [3g] Pre-computing composition vs transcription windows for Stage 6...")
    WINDOW_KEYS = 50
    STEP_KEYS = 25
    MIN_VALID_FOR_WINDOWS = 60

    window_feature_records = []
    writers_for_windows = writer_valid.groupby("writer_id")
    n_window_writers = len(writers_for_windows)

    for i, (wid, grp) in enumerate(writers_for_windows):
        progress(i, n_window_writers, every=1000, label="sliding windows: ")
        if len(grp) < MIN_VALID_FOR_WINDOWS:
            continue
        grp = grp.sort_values("DownTime").reset_index(drop=True)
        activities = grp["Activity"].values
        ikis = grp["iki_ms"].values
        cogs = grp["cog_load"].values

        # Vectorized: compute all window start indices
        n_events = len(grp)
        starts = list(range(0, n_events - WINDOW_KEYS + 1, STEP_KEYS))
        for start in starts:
            end = start + WINDOW_KEYS
            w_acts = activities[start:end]
            w_ikis = ikis[start:end]
            w_cogs = cogs[start:end]

            # Classify window
            n_input = np.sum(w_acts == "Input")
            n_nonprod = np.sum(w_acts == "Nonproduction")
            n_remove = np.sum(w_acts == "Remove/Cut")
            comp_frac = (n_nonprod + n_remove) / WINDOW_KEYS
            input_frac = n_input / WINDOW_KEYS

            if comp_frac >= 0.3:
                label = "composition"
            elif input_frac >= 0.8:
                label = "transcription"
            else:
                continue

            w_rho = _fast_spearman(w_cogs, w_ikis)
            w_entropy = compute_entropy_bits(w_ikis)

            window_feature_records.append({
                "writer_id": wid, "label": label,
                "clc_rho": w_rho, "entropy": w_entropy,
            })

    df_window_features = pd.DataFrame(window_feature_records).dropna()
    n_comp = (df_window_features["label"] == "composition").sum()
    n_trans = (df_window_features["label"] == "transcription").sum()
    log.info(f"  Pre-computed {len(df_window_features)} windows "
             f"(composition={n_comp}, transcription={n_trans})")

    return (results, df_all, writer_valid, df_writer_features, df_window_features)


# ============================================================================
# STAGE 4: Adversarial Retype Simulation
# ============================================================================

def stage_retype_simulation(df_all):
    """Simulate four adversarial retype strategies against genuine sessions."""
    import numpy as np
    import pandas as pd

    log.info("\n" + "=" * 72)
    log.info("STAGE 4: Adversarial Retype Simulation")
    log.info("=" * 72)

    MIN_EVENTS = 100
    MAX_WRITERS = 500
    SESSION_LEN = 200

    results = {}

    def clc_proxy_fn(ikis, activities):
        cog = np.array([COGNITIVE_LOAD_KL.get(a, 1) for a in activities], dtype=float)
        rho, _ = clc_spearman(cog, ikis)
        return 0.0 if np.isnan(rho) else rho

    def markov_retype(ikis, n_out, n_bins=20):
        bin_edges = np.percentile(ikis, np.linspace(0, 100, n_bins + 1))
        bin_edges[0] -= 1
        bin_edges[-1] += 1
        bin_indices = np.clip(np.digitize(ikis, bin_edges) - 1, 0, n_bins - 1)

        trans_counts = np.zeros((n_bins, n_bins), dtype=float)
        for j in range(len(bin_indices) - 1):
            trans_counts[bin_indices[j], bin_indices[j + 1]] += 1
        trans_counts += 0.01
        trans_probs = trans_counts / trans_counts.sum(axis=1, keepdims=True)

        init_probs = np.bincount(bin_indices, minlength=n_bins).astype(float)
        init_probs /= init_probs.sum()

        synthetic_bins = np.zeros(n_out, dtype=int)
        synthetic_bins[0] = np.random.choice(n_bins, p=init_probs)
        for j in range(1, n_out):
            synthetic_bins[j] = np.random.choice(n_bins, p=trans_probs[synthetic_bins[j - 1]])

        synthetic_ikis = np.zeros(n_out, dtype=float)
        for j in range(n_out):
            lo, hi = bin_edges[synthetic_bins[j]], bin_edges[synthetic_bins[j] + 1]
            synthetic_ikis[j] = np.random.uniform(lo, hi)
        return np.clip(synthetic_ikis, IKI_MIN, IKI_MAX)

    def extract_features(ikis, activities=None):
        return {
            "entropy": float(compute_entropy_bits(ikis)),
            "autocorr_lag1": float(autocorr_lag1(ikis)),
            "iki_mean": float(ikis.mean()),
            "iki_std": float(ikis.std()),
            "iki_log_var": float(np.log(ikis.var() + 1)),
            "pause_freq": float((ikis > 2000).mean()),
            "clc_proxy": float(clc_proxy_fn(ikis, activities)) if activities is not None else 0.0,
        }

    # --- 4a. Build writer sessions ---
    log.info("\n  [4a] Building writer sessions from loaded data...")
    writer_data = {}
    for wid, group in df_all.groupby("writer_id"):
        group = group.sort_values("DownTime").reset_index(drop=True)
        iki = group["DownTime"].diff().values[1:]
        act = group["Activity"].values[1:]
        valid = (iki > IKI_MIN) & (iki < IKI_MAX)
        iki, act = iki[valid].astype(float), act[valid]
        if len(iki) >= MIN_EVENTS:
            writer_data[str(wid)] = {"ikis": iki, "activities": act}

    log.info(f"  Eligible writers (>={MIN_EVENTS} events): {len(writer_data)}")

    all_wids = sorted(writer_data.keys())
    sampled_wids = sorted(random.sample(all_wids, min(MAX_WRITERS, len(all_wids))))
    log.info(f"  Using {len(sampled_wids)} writers for simulation")

    # --- 4b. Generate genuine + forged sessions ---
    log.info("\n  [4b] Generating genuine and forged sessions...")
    records = []

    for idx, wid in enumerate(sampled_wids):
        progress(idx, len(sampled_wids), every=100, label="writers: ")
        w = writer_data[wid]
        ikis_full, acts_full = w["ikis"], w["activities"]

        # Genuine session
        if len(ikis_full) > SESSION_LEN:
            start = random.randint(0, len(ikis_full) - SESSION_LEN)
            g_ikis = ikis_full[start:start + SESSION_LEN]
            g_acts = acts_full[start:start + SESSION_LEN]
        else:
            g_ikis, g_acts = ikis_full, acts_full

        feat = extract_features(g_ikis, g_acts)
        feat.update({"writer_id": wid, "session_type": "genuine", "label": 0})
        records.append(feat)

        # Forged sessions keep the genuine session's length and activity sequence
        # (a retyper reproduces the same edit operations), so clc_proxy is computed
        # identically for both classes instead of being fixed at 0 for forgeries.
        n_sess = len(g_ikis)

        # Attack 1: Constant-rate
        const_ikis = np.full(n_sess, np.median(ikis_full))
        feat = extract_features(const_ikis, g_acts)
        feat.update({"writer_id": wid, "session_type": "attack_constant", "label": 1})
        records.append(feat)

        # Attack 2: IID sampling
        iid_ikis = np.random.choice(ikis_full, size=n_sess, replace=True)
        feat = extract_features(iid_ikis, g_acts)
        feat.update({"writer_id": wid, "session_type": "attack_iid", "label": 1})
        records.append(feat)

        # Attack 3: Cross-writer sampling
        donor = random.choice([w2 for w2 in all_wids if w2 != wid])
        donor_ikis = writer_data[donor]["ikis"] if donor in writer_data else ikis_full
        cross_ikis = np.random.choice(donor_ikis, size=n_sess, replace=True)
        feat = extract_features(cross_ikis, g_acts)
        feat.update({"writer_id": wid, "session_type": "attack_cross", "label": 1})
        records.append(feat)

        # Attack 4: Markov-chain retype
        markov_ikis = markov_retype(ikis_full, n_sess)
        feat = extract_features(markov_ikis, g_acts)
        feat.update({"writer_id": wid, "session_type": "attack_markov", "label": 1})
        records.append(feat)

    df_sessions = pd.DataFrame(records)
    log.info(f"\n  Total sessions: {len(df_sessions)}")
    log.info(f"    Genuine: {(df_sessions['label'] == 0).sum()}")
    log.info(f"    Forged:  {(df_sessions['label'] == 1).sum()}")

    # --- 4c. Feature distributions ---
    log.info("\n  [4c] Feature distributions by session type:")
    feature_cols = ["entropy", "autocorr_lag1", "iki_mean", "iki_std",
                    "iki_log_var", "pause_freq", "clc_proxy"]
    for stype in ["genuine", "attack_constant", "attack_iid", "attack_cross", "attack_markov"]:
        subset = df_sessions[df_sessions["session_type"] == stype]
        log.info(f"\n    {stype} (n={len(subset)}):")
        for fc in feature_cols:
            vals = subset[fc]
            log.info(f"      {fc:18s}: mean={vals.mean():>10.4f}, std={vals.std():>10.4f}")

    # Save session features to CSV
    session_csv = SCRIPT_DIR / "retype_sessions.csv"
    df_sessions.to_csv(session_csv, index=False)
    log.info(f"\n  Session features saved to {session_csv}")

    results["n_writers"] = len(sampled_wids)
    results["total_sessions"] = len(df_sessions)
    results["genuine_sessions"] = int((df_sessions["label"] == 0).sum())
    results["forged_sessions"] = int((df_sessions["label"] == 1).sum())

    return results, df_sessions


# ============================================================================
# STAGE 5: ROC Classification
# ============================================================================

def stage_roc_classification(df_sessions):
    """Leave-one-writer-out logistic regression classification.
    Receives df_sessions directly from Stage 4 (no CSV read).

    AUC, EER, and operating points are computed on the pooled out-of-fold scores
    of the per-writer models. Each attack is evaluated twice: with the full
    feature set and without clc_proxy (the no-CLC control)."""
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.preprocessing import StandardScaler

    log.info("\n" + "=" * 72)
    log.info("STAGE 5: ROC Classification (Leave-One-Writer-Out CV)")
    log.info("=" * 72)

    df = df_sessions
    log.info(f"\n  Loaded {len(df)} sessions ({(df['label'] == 0).sum()} genuine, "
             f"{(df['label'] == 1).sum()} forged)")

    feature_cols = ["entropy", "autocorr_lag1", "iki_mean", "iki_std",
                    "iki_log_var", "pause_freq", "clc_proxy"]
    feature_cols_no_clc = [c for c in feature_cols if c != "clc_proxy"]
    attack_types = ["attack_constant", "attack_iid", "attack_cross", "attack_markov"]

    def evaluate(df_eval, cols):
        """Pooled out-of-fold scores from leave-one-writer-out logistic regression."""
        writers = df_eval["writer_id"].unique()
        y_true_all, y_score_all, y_pred_all = [], [], []
        fold_coefs = []

        for w in writers:
            test = df_eval[df_eval["writer_id"] == w]
            train = df_eval[df_eval["writer_id"] != w]
            if (len(test) == 0 or len(train[train["label"] == 0]) == 0
                    or len(train[train["label"] == 1]) == 0):
                continue

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(train[cols].values)
            X_test_s = scaler.transform(test[cols].values)

            clf = LogisticRegression(max_iter=1000, random_state=42)
            clf.fit(X_train_s, train["label"].values)

            y_true_all.extend(test["label"].values.tolist())
            y_score_all.extend(clf.predict_proba(X_test_s)[:, 1].tolist())
            y_pred_all.extend(clf.predict(X_test_s).tolist())
            fold_coefs.append(clf.coef_[0].copy())

        y_true_all = np.array(y_true_all)
        y_score_all = np.array(y_score_all)
        y_pred_all = np.array(y_pred_all)

        auc = roc_auc_score(y_true_all, y_score_all)
        fpr, tpr, _thresholds = roc_curve(y_true_all, y_score_all)
        fnr = 1 - tpr
        eer_idx = np.argmin(np.abs(fpr - fnr))
        eer = (fpr[eer_idx] + fnr[eer_idx]) / 2

        # Label 1 = forged, so fnr is the false-acceptance rate (forgery passed as
        # genuine) and fpr is the false-rejection rate (genuine flagged). Report the
        # FRR at the first ROC point whose FAR is at or below the target.
        operating_points = {}
        for target_far in [0.01, 0.05, 0.10]:
            ok = np.where(fnr <= target_far)[0]
            idx = int(ok[0]) if len(ok) else len(fpr) - 1
            operating_points[f"FRR_at_{int(target_far * 100)}pct_FAR"] = round(float(fpr[idx]), 4)

        # FPR-targeted operating points: the empirical FPR grid has resolution
        # 1/n_negatives, so only targets at or above that resolution are reported,
        # using the highest TPR whose FPR does not exceed the target.
        n_neg = int((y_true_all == 0).sum())
        fpr_targets = {}
        for target_fpr in [0.01, 0.05, 0.10]:
            if n_neg == 0 or target_fpr < 1.0 / n_neg:
                continue
            ok = np.where(fpr <= target_fpr)[0]
            idx = ok[np.argmax(tpr[ok])]
            fpr_targets[f"TPR_at_{target_fpr:.0%}_FPR"] = round(float(tpr[idx]), 4)
            fpr_targets[f"FPR_actual_at_{target_fpr:.0%}_FPR"] = round(float(fpr[idx]), 4)

        tp = int(((y_true_all == 1) & (y_pred_all == 1)).sum())
        fp = int(((y_true_all == 0) & (y_pred_all == 1)).sum())
        tn = int(((y_true_all == 0) & (y_pred_all == 0)).sum())
        fn = int(((y_true_all == 1) & (y_pred_all == 0)).sum())
        accuracy = (tp + tn) / len(y_true_all) if len(y_true_all) > 0 else 0
        majority_baseline = max(int((y_true_all == 1).sum()), n_neg) / len(y_true_all) if len(y_true_all) else 0

        result = {
            "features": list(cols),
            "n_genuine": n_neg,
            "n_forged": int((y_true_all == 1).sum()),
            "auc": round(float(auc), 4),
            "eer": round(float(eer), 4),
            "accuracy_at_0_5": round(float(accuracy), 4),
            "majority_class_accuracy": round(float(majority_baseline), 4),
            "operating_points": operating_points,
            "fpr_operating_points": fpr_targets,
            "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        }
        return result, fold_coefs

    results = {}

    for attack in attack_types + ["all_attacks"]:
        log.info(f"\n  Evaluating: {attack}")

        if attack == "all_attacks":
            df_eval = df.copy()
        else:
            df_eval = pd.concat([df[df["session_type"] == "genuine"],
                                 df[df["session_type"] == attack]])

        result, fold_coefs = evaluate(df_eval, feature_cols)
        result_no_clc, _ = evaluate(df_eval, feature_cols_no_clc)
        result["no_clc"] = {k: result_no_clc[k] for k in ("auc", "eer", "accuracy_at_0_5")}
        results[attack] = result

        log.info(f"    AUC:      {result['auc']:.4f}   (without clc_proxy: {result_no_clc['auc']:.4f})")
        log.info(f"    EER:      {result['eer']:.4f}   (without clc_proxy: {result_no_clc['eer']:.4f})")
        log.info(f"    Accuracy@0.5: {result['accuracy_at_0_5']:.4f} "
                 f"(majority-class baseline {result['majority_class_accuracy']:.4f})")
        log.info(f"    FRR@5%FAR: {result['operating_points'].get('FRR_at_5pct_FAR', 'N/A')}")
        if result["fpr_operating_points"]:
            log.info("    FPR-targeted operating points (highest TPR at or below target FPR):")
            for target_fpr in [0.01, 0.05, 0.10]:
                tpr_key = f"TPR_at_{target_fpr:.0%}_FPR"
                if tpr_key in result["fpr_operating_points"]:
                    log.info(f"      TPR @ {target_fpr:.0%} FPR: "
                             f"{result['fpr_operating_points'][tpr_key]:.4f}")

        if attack == "all_attacks" and fold_coefs:
            mean_coefs = np.mean(fold_coefs, axis=0)
            std_coefs = np.std(fold_coefs, axis=0)
            log.info("\n  Feature importance (averaged |coef| across CV folds):")
            importance = {}
            for feat, coef, std in zip(feature_cols, mean_coefs, std_coefs):
                log.info(f"    {feat:18s}: |coef|={abs(coef):.4f} (std={std:.4f})")
                importance[feat] = round(float(abs(coef)), 4)
            results["feature_importance"] = importance

    return results


# ============================================================================
# STAGE 6: Effect Sizes and Figures
# ============================================================================

def stage_effect_sizes_and_figures(df_writer_features, df_window_features):
    """Compute effect sizes and generate figures from pre-computed features.

    Args:
        df_writer_features: Per-writer features from Stage 3 (entropy, clc_rho,
            iki_log_var, pause_freq).
        df_window_features: Per-window composition/transcription features from
            Stage 3 (label, clc_rho, entropy).
    """
    import matplotlib
    import numpy as np
    from scipy import stats
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    log.info("\n" + "=" * 72)
    log.info("STAGE 6: Effect Sizes and Figures")
    log.info("=" * 72)

    fig_dir = SCRIPT_DIR / "figures"
    fig_dir.mkdir(exist_ok=True)

    plt.rcParams.update({
        "font.family": "serif", "font.size": 10,
        "axes.titlesize": 12, "axes.labelsize": 11,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 9, "figure.dpi": 300,
        "savefig.dpi": 300, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False,
    })

    results = {}

    # Use pre-computed data directly
    df_feat = df_writer_features.dropna(subset=["entropy", "clc_rho", "iki_log_var", "pause_freq"])
    df_windows = df_window_features
    n_comp = (df_windows["label"] == "composition").sum()
    n_trans = (df_windows["label"] == "transcription").sum()
    log.info("\n  [6a] Using pre-computed features from Stage 3")
    log.info(f"  Writers with complete features: {len(df_feat)}")
    log.info(f"  Windows: {len(df_windows)} (composition={n_comp}, transcription={n_trans})")

    # --- 6b. Figure 1: CLC Distribution ---
    log.info("\n  [6b] Figure 1: Per-Writer CLC Distribution...")
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    clc_vals = df_feat["clc_rho"].values

    bins = np.linspace(clc_vals.min() - 0.02, clc_vals.max() + 0.02, 60)
    n_hist, bin_edges, patches = ax.hist(
        clc_vals, bins=bins, color="#4878CF", edgecolor="white", linewidth=0.3, alpha=0.85
    )
    for patch, left_edge in zip(patches, bin_edges[:-1]):
        if left_edge + (bin_edges[1] - bin_edges[0]) / 2 < 0:
            patch.set_facecolor("#E24A33")
            patch.set_alpha(0.7)
    ax.axvline(0.15, color="black", linestyle="--", linewidth=1.0, label=r"$\tau = 0.15$")
    ax.axvline(0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xlabel(r"Per-Writer CLC (Spearman $\rho$)")
    ax.set_ylabel("Number of Writers")
    ax.set_title("Per-Writer Cognitive-Load Correlation Distribution")
    ax.legend(frameon=False)
    fig.savefig(fig_dir / "fig_clc_distribution.pdf", metadata={"CreationDate": None})
    fig.savefig(fig_dir / "fig_clc_distribution.png")
    plt.close(fig)
    log.info("    Saved figures/fig_clc_distribution.{pdf,png}")

    # --- 6c. Figure 2: Correlation Matrix ---
    log.info("\n  [6c] Figure 2: Correlation Matrix Heatmap...")
    features = ["entropy", "clc_rho", "iki_log_var", "pause_freq"]
    labels = ["Entropy", "CLC", r"IKI $\log$ Var", "Pause Freq"]
    n_f = len(features)
    corr_mat = np.ones((n_f, n_f))
    for i in range(n_f):
        for j in range(n_f):
            if i != j:
                rho, _ = stats.spearmanr(df_feat[features[i]], df_feat[features[j]])
                corr_mat[i, j] = rho

    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    cax = ax.imshow(corr_mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    for i in range(n_f):
        for j in range(n_f):
            color = "white" if abs(corr_mat[i, j]) > 0.5 else "black"
            ax.text(j, i, f"{corr_mat[i, j]:.2f}", ha="center", va="center",
                    fontsize=10, color=color, fontweight="bold")
    ax.set_xticks(range(n_f))
    ax.set_yticks(range(n_f))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title(f"Cross-Domain Correlation Matrix (N={len(df_feat):,} Writers)", pad=12)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04, label=r"Spearman $\rho$")
    fig.savefig(fig_dir / "fig_correlation_matrix.pdf", metadata={"CreationDate": None})
    fig.savefig(fig_dir / "fig_correlation_matrix.png")
    plt.close(fig)
    log.info("    Saved figures/fig_correlation_matrix.{pdf,png}")

    # --- 6d. Figure 3: Composition vs Transcription ---
    log.info("\n  [6d] Figure 3: Composition vs Transcription CLC...")
    comp = df_windows[df_windows["label"] == "composition"]["clc_rho"].values
    trans = df_windows[df_windows["label"] == "transcription"]["clc_rho"].values
    comp_ent = df_windows[df_windows["label"] == "composition"]["entropy"].values
    trans_ent = df_windows[df_windows["label"] == "transcription"]["entropy"].values

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    bins_c = np.linspace(-0.8, 0.8, 50)
    axes[0].hist(comp, bins=bins_c, alpha=0.65, color="#4878CF", label="Composition",
                 density=True, edgecolor="white", linewidth=0.3)
    axes[0].hist(trans, bins=bins_c, alpha=0.65, color="#E24A33", label="Transcription",
                 density=True, edgecolor="white", linewidth=0.3)
    axes[0].set_xlabel(r"Window CLC (Spearman $\rho$)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("A. CLC Distributions by Mode")
    axes[0].legend(frameon=False)

    bins_e = np.linspace(
        min(comp_ent.min(), trans_ent.min()) - 0.2,
        max(comp_ent.max(), trans_ent.max()) + 0.2, 50
    )
    axes[1].hist(comp_ent, bins=bins_e, alpha=0.65, color="#4878CF", label="Composition",
                 density=True, edgecolor="white", linewidth=0.3)
    axes[1].hist(trans_ent, bins=bins_e, alpha=0.65, color="#E24A33", label="Transcription",
                 density=True, edgecolor="white", linewidth=0.3)
    axes[1].set_xlabel("Window Entropy (bits)")
    axes[1].set_ylabel("Density")
    axes[1].set_title("B. Entropy Distributions by Mode")
    axes[1].legend(frameon=False)

    fig.suptitle("Composition vs Transcription-like Windows", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_clc_composition_vs_transcription.pdf", metadata={"CreationDate": None})
    fig.savefig(fig_dir / "fig_clc_composition_vs_transcription.png")
    plt.close(fig)
    log.info("    Saved figures/fig_clc_composition_vs_transcription.{pdf,png}")

    # --- 6e. Effect sizes ---
    log.info("\n  [6e] Computing effect sizes...")

    effects = {
        "n_composition_windows": int(len(comp)),
        "n_transcription_windows": int(len(trans)),
    }

    for metric_name, comp_vals, trans_vals in [
        ("entropy", comp_ent, trans_ent),
        ("clc", comp, trans),
    ]:
        cd = cohens_d(comp_vals, trans_vals)
        gd = glass_delta(comp_vals, trans_vals)
        cld = cliffs_delta(comp_vals, trans_vals)
        mw_p = stats.mannwhitneyu(comp_vals, trans_vals).pvalue

        magnitude = ("large" if abs(cd) >= 0.8 else "medium" if abs(cd) >= 0.5
                      else "small" if abs(cd) >= 0.2 else "negligible")

        effects[metric_name] = {
            "composition_mean": round(float(comp_vals.mean()), 4),
            "composition_sd": round(float(comp_vals.std(ddof=1)), 4),
            "transcription_mean": round(float(trans_vals.mean()), 4),
            "transcription_sd": round(float(trans_vals.std(ddof=1)), 4),
            "cohens_d": round(float(cd), 4),
            "glass_delta": round(float(gd), 4),
            "cliffs_delta": round(float(cld), 4),
            "mann_whitney_p": float(f"{mw_p:.4e}"),
            "magnitude": magnitude,
        }

        log.info(f"\n  {metric_name.upper()}:")
        log.info(f"    Cohen's d:  {cd:.4f} ({magnitude})")
        log.info(f"    Glass's D:  {gd:.4f}")
        log.info(f"    Cliff's d:  {cld:.4f}")
        log.info(f"    Mann-Whitney p: {mw_p:.4e}")

    # Window labels and the CLC variable are both functions of the activity column,
    # windows overlap by 50%, and windows are nested within writers, so the
    # window-level tests above overstate independence. Writer-level paired
    # contrast: per writer, mean CLC over composition windows minus mean CLC over
    # transcription windows (writers with both window types).
    if {"writer_id", "label", "clc_rho"}.issubset(df_window_features.columns):
        per_writer = (df_window_features.groupby(["writer_id", "label"])["clc_rho"]
                      .mean().unstack("label").dropna(subset=["composition", "transcription"]))
        diffs = (per_writer["composition"] - per_writer["transcription"]).values
        if len(diffs) >= 10:
            w_stat, w_p = stats.wilcoxon(diffs)
            effects["clc_writer_level"] = {
                "n_writers_with_both_modes": int(len(diffs)),
                "mean_difference_composition_minus_transcription": round(float(diffs.mean()), 4),
                "median_difference": round(float(np.median(diffs)), 4),
                "sd_difference": round(float(diffs.std(ddof=1)), 4),
                "paired_cohens_d": round(float(diffs.mean() / diffs.std(ddof=1)), 4) if diffs.std(ddof=1) > 0 else 0.0,
                "pct_writers_negative": round(float(100 * (diffs < 0).mean()), 2),
                "wilcoxon_p": float(f"{w_p:.4e}"),
                "note": "labels and CLC both derive from the activity column; the contrast is partly definitional",
            }
            log.info("\n  CLC writer-level paired contrast (composition - transcription):")
            log.info(f"    writers: {len(diffs)}, mean diff: {diffs.mean():.4f}, "
                     f"paired d: {effects['clc_writer_level']['paired_cohens_d']:.4f}, "
                     f"Wilcoxon p: {w_p:.2e}")

    results["effect_sizes"] = effects
    return results


# ============================================================================
# STAGE 7: Summary Report
# ============================================================================

def stage_summary(all_results):
    """Print consolidated summary and save results."""
    log.info("\n" + "=" * 72)
    log.info("STAGE 7: Summary Report")
    log.info("=" * 72)

    log.info("\n  Paper Claims vs. Empirical Results:")
    log.info("  " + "-" * 68)

    sw = all_results.get("scholawrite", {})
    kl = all_results.get("klicke", {})
    roc = all_results.get("roc", {})
    eff = all_results.get("effects", {}).get("effect_sizes", {})

    # ScholaWrite entropy
    sw_ent = sw.get("entropy_per_iki_bits")
    if sw_ent is not None:
        log.info(f"  ScholaWrite inter-event interval entropy: {sw_ent:.2f} bits "
                 f"({sw.get('entropy_quantization_ms', QUANTIZATION_MS)} ms bins; text-change events, not keystrokes)")

    # KLiCKe CLC Cohen's d (signed)
    if "clc" in eff:
        clc_d = eff["clc"].get("cohens_d")
        if clc_d is not None:
            direction = ("transcription-like windows have HIGHER CLC" if clc_d < 0
                         else "composition-like windows have higher CLC")
            log.info(f"  KLiCKe CLC Cohen's d:     {clc_d:.4f} (composition minus transcription; "
                     f"{direction}; {eff['clc'].get('magnitude', '')})")

    if "entropy" in eff:
        ent_d = eff["entropy"].get("cohens_d")
        if ent_d is not None:
            log.info(f"  KLiCKe entropy Cohen's d: {ent_d:.4f} ({eff['entropy'].get('magnitude', '')})")

    # ROC AUC
    all_atk = roc.get("all_attacks", {})
    if all_atk:
        auc_val = all_atk.get("auc")
        if auc_val is not None:
            log.info(f"  Classification AUC:       {auc_val:.4f} "
                     f"(without clc_proxy: {all_atk.get('no_clc', {}).get('auc', float('nan')):.4f})")
        eer_val = all_atk.get("eer")
        if eer_val is not None:
            log.info(f"  EER:                      {eer_val:.4f}")

    # Feature importance
    fi = roc.get("feature_importance")
    if fi:
        log.info("\n  Feature Importance (|coef|, logistic regression):")
        for feat in sorted(fi, key=fi.get, reverse=True):
            log.info(f"    {feat:18s}: {fi[feat]:.4f}")

    # KLiCKe population stats
    if kl:
        log.info("\n  KLiCKe Population Statistics:")
        log.info(f"    Writers:          {kl.get('n_writers', 'N/A'):,}")
        log.info(f"    Valid IKIs:       {kl.get('valid_ikis', 'N/A'):,}")
        log.info(f"    Mean IKI:         {kl.get('iki_mean_ms', 0):.1f} ms")
        log.info(f"    Entropy/IKI:      {kl.get('entropy_per_iki_bits', 0):.2f} bits")
        log.info(f"    Per-writer CLC:   mean rho={kl.get('clc_per_writer_mean_rho', 0):.4f}")
        log.info(f"    Positive CLC:     {kl.get('clc_pct_positive', 0):.1f}%")

    # Save results
    output_path = SCRIPT_DIR / "analysis_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log.info(f"\n  Results saved to {output_path}")

    log.info("\n" + "=" * 72)
    log.info("ANALYSIS COMPLETE")
    log.info("=" * 72)


def _recompute_features_from_raw(df_klicke, writer_valid):
    """Recompute per-writer and per-window features from raw data (slow path).
    Used with --recompute to independently verify Stage 3 pre-computed results."""
    import numpy as np
    import pandas as pd
    from scipy import stats

    COGNITIVE_LOAD = {"Nonproduction": 3, "Remove/Cut": 2, "Input": 1}
    WINDOW_KEYS, STEP_KEYS = 50, 25

    log.info("\n  Recomputing per-writer features from raw data...")
    writer_records = []
    for i, (wid, grp) in enumerate(writer_valid.groupby("writer_id")):
        progress(i, writer_valid["writer_id"].nunique(), every=1000, label="recompute writers: ")
        if len(grp) < 50:
            continue
        iki = grp["iki_ms"]
        ent = compute_entropy_bits(iki)
        cog = grp["Activity"].map(COGNITIVE_LOAD).fillna(1).values
        rho, _ = stats.spearmanr(cog, iki.values) if np.std(cog) > 0 else (np.nan, np.nan)
        writer_records.append({
            "writer_id": wid, "entropy": ent, "clc_rho": rho,
            "iki_log_var": float(np.log(iki.var(ddof=0) + 1)),
            "pause_freq": float((iki > 2000).mean()),
        })
    df_wf = pd.DataFrame(writer_records).dropna()

    log.info("\n  Recomputing sliding windows from raw data...")
    window_records = []
    for i, (wid, grp) in enumerate(writer_valid.groupby("writer_id")):
        progress(i, writer_valid["writer_id"].nunique(), every=1000, label="recompute windows: ")
        grp = grp.sort_values("DownTime").reset_index(drop=True)
        if len(grp) < 60:
            continue
        activities = grp["Activity"].values
        ikis = grp["iki_ms"].values
        cogs = grp["Activity"].map(COGNITIVE_LOAD).fillna(1).values
        for start in range(0, len(grp) - WINDOW_KEYS + 1, STEP_KEYS):
            end = start + WINDOW_KEYS
            w_acts = activities[start:end]
            n_input = np.sum(w_acts == "Input")
            n_nonprod = np.sum(w_acts == "Nonproduction")
            n_remove = np.sum(w_acts == "Remove/Cut")
            comp_frac = (n_nonprod + n_remove) / WINDOW_KEYS
            if comp_frac >= 0.3:
                label = "composition"
            elif n_input / WINDOW_KEYS >= 0.8:
                label = "transcription"
            else:
                continue
            w_rho, _ = stats.spearmanr(cogs[start:end], ikis[start:end])
            w_ent = compute_entropy_bits(ikis[start:end])
            window_records.append({"writer_id": wid, "label": label,
                                   "clc_rho": w_rho, "entropy": w_ent})
    df_ww = pd.DataFrame(window_records).dropna()
    log.info(f"  Recomputed: {len(df_wf)} writers, {len(df_ww)} windows")
    return df_wf, df_ww


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse

    import numpy as np

    parser = argparse.ArgumentParser(description="Process attestation analysis pipeline")
    parser.add_argument("--recompute", action="store_true",
                        help="Recompute all features from raw data in Stage 6 "
                             "instead of using pre-computed results from Stage 3. "
                             "Slower but independently verifiable.")
    args = parser.parse_args()

    np.random.seed(42)
    random.seed(42)

    all_results = {}
    total_start = time.time()

    # Stage 1: Environment Check
    t0 = time.time()
    has_klicke, klicke_format = stage_environment_check()
    log.info(f"\n  Stage 1 completed in {time.time() - t0:.1f}s")

    # Stage 2: ScholaWrite Analysis (always runs)
    t0 = time.time()
    all_results["scholawrite"] = stage_scholawrite()
    log.info(f"\n  Stage 2 completed in {time.time() - t0:.1f}s")

    # Stages 3-6: KLiCKe-dependent
    if has_klicke:
        # Stage 3: KLiCKe Analysis (also pre-computes features for Stage 6)
        t0 = time.time()
        (klicke_results, df_klicke, writer_valid,
         df_writer_features, df_window_features) = stage_klicke(klicke_format)
        all_results["klicke"] = klicke_results
        log.info(f"\n  Stage 3 completed in {time.time() - t0:.1f}s")

        # Stage 4: Retype Simulation
        t0 = time.time()
        retype_results, df_sessions = stage_retype_simulation(df_klicke)
        all_results["retype"] = retype_results
        log.info(f"\n  Stage 4 completed in {time.time() - t0:.1f}s")

        # Stage 5: ROC Classification (receives df_sessions directly)
        t0 = time.time()
        all_results["roc"] = stage_roc_classification(df_sessions)
        log.info(f"\n  Stage 5 completed in {time.time() - t0:.1f}s")

        # Stage 6: Effect Sizes and Figures
        t0 = time.time()
        if args.recompute:
            log.info("\n  --recompute: Stage 6 will recompute all features from raw data")
            from_raw = _recompute_features_from_raw(df_klicke, writer_valid)
            all_results["effects"] = stage_effect_sizes_and_figures(*from_raw)
        else:
            all_results["effects"] = stage_effect_sizes_and_figures(
                df_writer_features, df_window_features)
        log.info(f"\n  Stage 6 completed in {time.time() - t0:.1f}s")
    else:
        log.info("\n  Stages 3-6 skipped (KLiCKe data not available)")
        log.info("  ScholaWrite-only results are still valid and complete.")

    # Stage 7: Summary
    stage_summary(all_results)

    total_elapsed = time.time() - total_start
    log.info(f"\n  Total elapsed time: {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
