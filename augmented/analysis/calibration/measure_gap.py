#!/usr/bin/env python
"""Measure how far the current simulator's inter-keystroke-interval (IKI) channel
is from real human typing in the KLiCKe corpus.

Both populations (KLiCKe held-out writers, simulator reference traces) are cut
into non-overlapping windows and summarised by the SAME functions
(`window_metrics`, `writer_params`), so any gap reflects the simulator, not the
estimator.

Simulator path (the one the adversarial experiments use for the latency channel):
    _generate_authentic_trace(seed, n_events)           adversarial.py (main branch)
      -> EmbodiedScholar(f"author_{seed}", initial_glucose=1.0)
      -> IrreversibleProcessEngine(author).execute(LexicalIntention(word, depth, rarity, cost))
           depth ~ U(1, 8), rarity ~ U(0.1, 0.9), cost ~ U(0.01, 0.05), rng = random.Random(seed)
      -> ExecutionEvent.latency_ms = round(author.calculate_latency(depth), 2)   causal_core.py
      -> CausalEvent.latency_ms   = round(ExecutionEvent.latency_ms, 2)
`scholawrite.adversarial` is imported when present; otherwise the function is
reproduced verbatim below (same RNG draw order) so the latency channel is identical.

Usage (from the package root `augmented/`):
    uv run --extra dev python analysis/calibration/measure_gap.py

Options added for the timing-model calibration:
    --split {heldout,train}     which KLiCKe half is compared against (default heldout, unchanged).
                                Tune on `train`; a `heldout` run is a "look" and is capped (see below).
    --sim-override KEY=VALUE    override a SimulationConfig field for the simulator side only
                                (repeatable; VALUE is parsed as JSON, else kept as a string), e.g.
                                --sim-override timing_model=klicke_ar1
    --candidate-id ID --run-timestamp TS
                                (heldout split only) append this run to the `heldout_runs` list inside
                                --out, with the pre-registered rule evaluated. At most
                                MAX_HELDOUT_LOOKS entries are accepted per file; TS is the shell's `date`.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parents[1]  # augmented/
DEFAULT_KLICKE_DIR = (
    PACKAGE_ROOT
    / "data"
    / "external"
    / "klicke"
    / "Files"
    / "WritingTask"
    / "WritingTask"
    / "keystrokelogs"
    / "csv"
)


def import_simulator(root: Path):
    """Import scholawrite.causal_core / scholawrite.embodied from `root` (the
    working tree by default) ahead of any installed copy."""
    sys.path.insert(0, str(root))
    for name in [m for m in sys.modules if m == "scholawrite" or m.startswith("scholawrite.")]:
        del sys.modules[name]
    return importlib.import_module("scholawrite.causal_core"), importlib.import_module(
        "scholawrite.embodied"
    )


def _rel(p: Path) -> str:
    p = Path(p).resolve()
    try:
        return str(p.relative_to(PACKAGE_ROOT))
    except ValueError:
        return str(p)


# ---------------------------------------------------------------------------
# Simulator reference traces
# ---------------------------------------------------------------------------

# Verbatim copy of adversarial._ACADEMIC_WORDS (main:augmented/scholawrite/adversarial.py:43-50).
_ACADEMIC_WORDS = [
    "The",
    "methodology",
    "establishes",
    "a",
    "robust",
    "baseline",
    "for",
    "comparative",
    "analysis",
    "of",
    "emerging",
    "patterns",
    "in",
    "empirical",
    "research",
    "paradigm",
    "framework",
    "this",
    "study",
    "demonstrates",
    "that",
    "underlying",
    "assumptions",
    "remain",
    "foundational",
    "to",
    "subsequent",
    "investigations",
    "however",
    "nevertheless",
    "consequently",
    "furthermore",
]


def _ported_generate_authentic_trace(seed: int, n_events: int, causal_core, embodied):
    """Verbatim port of adversarial._generate_authentic_trace
    (main:augmented/scholawrite/adversarial.py:462-495) up to the point where it
    converts ExecutionEvent -> CausalEvent. Returns the engine trace."""
    rng = random.Random(seed)
    author = embodied.EmbodiedScholar(f"author_{seed}", initial_glucose=1.0)
    engine = causal_core.IrreversibleProcessEngine(author)

    words = list(_ACADEMIC_WORDS)
    rng.shuffle(words)
    if len(words) < n_events:
        words = (words * ((n_events // len(words)) + 1))[:n_events]

    for i in range(n_events):
        word = words[i]
        depth = rng.uniform(1.0, 8.0)
        rarity = rng.uniform(0.1, 0.9)
        cost = rng.uniform(0.01, 0.05)
        engine.execute(causal_core.LexicalIntention(word, depth, rarity, cost))
    return engine.trace


def resolve_trace_generator(causal_core, embodied):
    """Return (description, fn) where fn(seed, n_events) -> (latencies_ms, glucose_at_event)."""
    try:
        adversarial = importlib.import_module("scholawrite.adversarial")
    except ImportError:

        def _fn(seed: int, n_events: int):
            trace = _ported_generate_authentic_trace(seed, n_events, causal_core, embodied)
            lat = [round(e.latency_ms, 2) for e in trace]  # CausalEvent.latency_ms rounding
            glu = [e.glucose_before for e in trace]
            return lat, glu

        return (
            "ported _generate_authentic_trace (scholawrite.adversarial absent in this package; "
            "source main:augmented/scholawrite/adversarial.py:462-495)",
            _fn,
        )

    def _fn_native(seed: int, n_events: int):
        trace = adversarial._generate_authentic_trace(seed, n_events=n_events)
        return [e.latency_ms for e in trace], [e.glucose_at_event for e in trace]

    return "scholawrite.adversarial._generate_authentic_trace", _fn_native


def parse_overrides(items) -> dict:
    """['timing_model=klicke_ar1', 'timing_offset=-0.5'] -> {'timing_model': 'klicke_ar1', 'timing_offset': -0.5}"""
    out = {}
    for item in items:
        key, sep, raw = item.partition("=")
        if not sep or not key:
            raise SystemExit(f"--sim-override expects KEY=VALUE, got {item!r}")
        try:
            out[key] = json.loads(raw)
        except json.JSONDecodeError:
            out[key] = raw
    return out


def apply_sim_overrides(causal_core, embodied, overrides: dict, base=None):
    """Replace the simulator's SimulationConfig (this process only) with `base` (default: the
    current one) plus `overrides`. Must run after `import_simulator`, which purges and re-imports
    `scholawrite.*`. Returns the effective config."""
    cfg = base if base is not None else embodied.get_sim_config()
    if overrides:
        unknown = sorted(set(overrides) - {f.name for f in dataclasses.fields(cfg)})
        if unknown:
            raise SystemExit(f"unknown SimulationConfig field(s) in --sim-override: {unknown}")
        cfg = dataclasses.replace(cfg, **overrides)
    if overrides or base is not None:
        embodied.get_sim_config = lambda: cfg
        causal_core.get_sim_config = lambda: cfg
    return cfg


def generate_simulator_sequences(
    n_traces: int, n_events: int, seed_base: int, causal_core, embodied
):
    desc, gen = resolve_trace_generator(causal_core, embodied)
    seqs, glucose_floor_idx, lengths = [], [], []
    for i in range(n_traces):
        lat, glu = gen(seed_base + i, n_events)
        seqs.append(np.asarray(lat, dtype=np.float64))
        lengths.append(len(lat))
        g = np.asarray(glu, dtype=np.float64)
        floor_hits = np.flatnonzero(g <= g.min() + 1e-9)
        glucose_floor_idx.append(int(floor_hits[0]) if g.min() <= 0.05 + 1e-9 else -1)
    return desc, seqs, lengths, glucose_floor_idx


# ---------------------------------------------------------------------------
# KLiCKe loading
# ---------------------------------------------------------------------------


def load_writer_iki(path: Path, min_iki: float, max_iki: float):
    """Return (n_events, valid_iki_ms, used_cp1252) for one writer file."""
    used_cp1252 = False
    try:
        df = pd.read_csv(path, usecols=["DownTime"], encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, usecols=["DownTime"], encoding="cp1252")
        used_cp1252 = True
    t = pd.to_numeric(df["DownTime"], errors="coerce").dropna().to_numpy(dtype=np.float64)
    t.sort(kind="stable")
    iki = np.diff(t)
    valid = iki[(iki > min_iki) & (iki < max_iki)]
    return int(len(t)), valid, used_cp1252


def filter_valid(x: np.ndarray, min_iki: float, max_iki: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x[(x > min_iki) & (x < max_iki)]


def split_writers(n_valid: dict, min_valid: int):
    """Return (eligible, train_ids, heldout_ids) from {writer_id: number of valid IKIs}.

    Eligible writers (>= min_valid valid IKIs) are sorted by id; even index -> train,
    odd index -> held-out. Shared by measure_gap, build_writer_params and fit_timing so the
    split is defined in exactly one place."""
    eligible = sorted(w for w, n in n_valid.items() if n >= min_valid)
    return eligible, eligible[0::2], eligible[1::2]


# ---------------------------------------------------------------------------
# Shared estimators (identical code path for both populations)
# ---------------------------------------------------------------------------

# Sum of squared deviations below this is treated as zero variance (a constant
# window yields ~1e-29 from rounding, which would otherwise give r = 1).
_SS_EPS = 1e-18


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return float("nan")
    ac = a - a.mean()
    bc = b - b.mean()
    ss_a = float((ac * ac).sum())
    ss_b = float((bc * bc).sum())
    if ss_a <= _SS_EPS or ss_b <= _SS_EPS:
        return float("nan")
    return float((ac * bc).sum() / math.sqrt(ss_a * ss_b))


def _count_runs(mask: np.ndarray, min_run: int) -> np.ndarray:
    """Per row: number of maximal runs of True with length >= min_run."""
    n, w = mask.shape
    padded = np.zeros((n, w + 2), dtype=np.int8)
    padded[:, 1:-1] = mask
    d = np.diff(padded, axis=1)
    starts = np.argwhere(d == 1)  # row-major: i-th start pairs with i-th end
    ends = np.argwhere(d == -1)
    lengths = ends[:, 1] - starts[:, 1]
    keep = lengths >= min_run
    return np.bincount(starts[keep, 0], minlength=n).astype(np.int64)


def _plugin_entropy_bits(bins: np.ndarray) -> np.ndarray:
    """Per row: plug-in Shannon entropy (bits) of the integer bin labels."""
    n, w = bins.shape
    s = np.sort(bins, axis=1)
    change = np.ones_like(s, dtype=bool)
    change[:, 1:] = s[:, 1:] != s[:, :-1]
    flat_idx = np.flatnonzero(change.ravel())
    lengths = np.diff(np.append(flat_idx, s.size))
    row = flat_idx // w
    p = lengths / w
    return np.bincount(row, weights=-p * np.log2(p), minlength=n)


METRICS = [
    "mean_iki_ms",
    "median_iki_ms",
    "sd_log_iki",
    "lag1_autocorr_log",
    "pause_fraction",
    "burst_count",
    "entropy_bits_5ms",
]

# Pre-registered decision rule for the calibrated simulator on held-out KLiCKe windows:
# PASS iff KS D <= DECISION_KS_MAX on every DECISION_METRICS entry simultaneously.
# The other metrics are reported but do not decide.
DECISION_METRICS = ["mean_iki_ms", "sd_log_iki", "lag1_autocorr_log", "pause_fraction"]
DECISION_KS_MAX = 0.10
MAX_HELDOUT_LOOKS = 3


def window_metrics(
    seqs, window: int, pause_ms: float, burst_ms: float, burst_min_run: int, bin_ms: float
):
    """Cut each sequence into consecutive non-overlapping windows (remainder dropped)
    and compute the per-window metrics. Returns dict metric -> 1-D array."""
    rows = []
    for x in seqs:
        n = len(x) // window
        if n:
            rows.append(np.asarray(x[: n * window], dtype=np.float64).reshape(n, window))
    if not rows:
        return {m: np.empty(0) for m in METRICS}
    xs = np.concatenate(rows)
    logs = np.log(xs)
    a, b = logs[:, :-1], logs[:, 1:]
    ac = a - a.mean(axis=1, keepdims=True)
    bc = b - b.mean(axis=1, keepdims=True)
    ss_a = (ac * ac).sum(axis=1)
    ss_b = (bc * bc).sum(axis=1)
    defined = (ss_a > _SS_EPS) & (ss_b > _SS_EPS)
    with np.errstate(invalid="ignore", divide="ignore"):
        lag1 = np.where(defined, (ac * bc).sum(axis=1) / np.sqrt(ss_a * ss_b), np.nan)
    return {
        "mean_iki_ms": xs.mean(axis=1),
        "median_iki_ms": np.median(xs, axis=1),
        "sd_log_iki": logs.std(axis=1, ddof=1),
        "lag1_autocorr_log": lag1,
        "pause_fraction": (xs > pause_ms).mean(axis=1),
        "burst_count": _count_runs(xs < burst_ms, burst_min_run).astype(np.float64),
        "entropy_bits_5ms": _plugin_entropy_bits(np.floor(xs / bin_ms).astype(np.int64)),
    }


WRITER_PARAMS = ["mu_w", "sigma_w", "rho_w", "p_w", "pause_log_median", "pause_log_sd"]


def writer_params(x: np.ndarray, pause_ms: float) -> dict:
    """Writer-level parameters of a log-IKI AR(1) + pause-mixture model."""
    L = np.log(x)
    pauses = x > pause_ms
    n_p = int(pauses.sum())
    Lp = L[pauses]
    return {
        "mu_w": float(L.mean()),
        "sigma_w": float(L.std(ddof=1)) if len(L) > 1 else float("nan"),
        "rho_w": _pearson(L[:-1], L[1:]),
        "p_w": float(pauses.mean()),
        "n_pauses": n_p,
        "pause_log_median": float(np.median(Lp)) if n_p else float("nan"),
        "pause_log_sd": float(Lp.std(ddof=1)) if n_p >= 2 else float("nan"),
    }


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def _clean(v):
    """Make a nested result JSON-safe: numpy scalars -> python, NaN -> null."""
    if isinstance(v, (np.floating, float)):
        return None if math.isnan(float(v)) else float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    return v


def quartiles(v: np.ndarray) -> dict:
    v = np.asarray(v, dtype=np.float64)
    n_nan = int(np.isnan(v).sum())
    vv = v[~np.isnan(v)]
    if len(vv) == 0:
        return {
            "n": 0,
            "n_nan": n_nan,
            "q25": None,
            "median": None,
            "q75": None,
            "iqr": None,
            "mean": None,
            "min": None,
            "max": None,
        }
    q25, q50, q75 = np.percentile(vv, [25, 50, 75])
    return {
        "n": int(len(vv)),
        "n_nan": n_nan,
        "q25": float(q25),
        "median": float(q50),
        "q75": float(q75),
        "iqr": float(q75 - q25),
        "mean": float(vv.mean()),
        "min": float(vv.min()),
        "max": float(vv.max()),
    }


def percentiles_10_50_90(v: np.ndarray) -> dict:
    v = np.asarray(v, dtype=np.float64)
    vv = v[~np.isnan(v)]
    if len(vv) == 0:
        return {"n": 0, "p10": None, "p50": None, "p90": None}
    p10, p50, p90 = np.percentile(vv, [10, 50, 90])
    return {"n": int(len(vv)), "p10": float(p10), "p50": float(p50), "p90": float(p90)}


# ---------------------------------------------------------------------------
# Held-out look log
# ---------------------------------------------------------------------------


def load_prior_heldout_runs(path: Path) -> list:
    """The `heldout_runs` already logged in `path` (empty when the file is absent)."""
    if not path.exists():
        return []
    try:
        runs = json.loads(path.read_text()).get("heldout_runs", [])
    except (json.JSONDecodeError, OSError, AttributeError) as exc:
        raise SystemExit(f"cannot read the heldout_runs log in {path}: {exc}") from exc
    if not isinstance(runs, list):
        raise SystemExit(f"{path}: heldout_runs is not a list")
    return runs


def heldout_log_entry(
    index, args, overrides, sim_cfg, metrics_out, n_klicke_windows, n_ks, n_sim_windows
) -> dict:
    ks = {m: metrics_out[m]["ks_D"] for m in METRICS}
    worst = max(ks[m] for m in DECISION_METRICS)
    return {
        "look": index,
        "candidate_id": args.candidate_id,
        "timestamp": args.run_timestamp,
        "note": args.run_note,
        "command_args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "sim_overrides": overrides,
        "timing_config": {
            k: v for k, v in dataclasses.asdict(sim_cfg).items() if k.startswith("timing_")
        },
        "latency_log_normal_sigma": sim_cfg.latency_log_normal_sigma,
        "n_klicke_windows": int(n_klicke_windows),
        "n_klicke_windows_ks": int(n_ks),
        "n_sim_windows": int(n_sim_windows),
        "ks_D": ks,
        "metrics": {
            m: {
                "klicke": {q: metrics_out[m]["klicke"][q] for q in ("q25", "median", "q75")},
                "simulator": {q: metrics_out[m]["simulator"][q] for q in ("q25", "median", "q75")},
            }
            for m in METRICS
        },
        "rule": {
            "metrics": DECISION_METRICS,
            "ks_max": DECISION_KS_MAX,
            "max_decision_D": worst,
            "pass": bool(worst <= DECISION_KS_MAX),
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--klicke-dir", type=Path, default=DEFAULT_KLICKE_DIR)
    p.add_argument(
        "--simulator-root",
        type=Path,
        default=PACKAGE_ROOT,
        help="directory containing the `scholawrite` package to measure (default: working tree)",
    )
    p.add_argument("--out", type=Path, default=SCRIPT_DIR / "gap_current_simulator.json")
    p.add_argument(
        "--writer-params-out", type=Path, default=SCRIPT_DIR / "klicke_writer_params.json"
    )
    p.add_argument("--min-iki", type=float, default=10.0, help="keep IKI > this (ms)")
    p.add_argument("--max-iki", type=float, default=60000.0, help="keep IKI < this (ms)")
    p.add_argument(
        "--min-valid", type=int, default=200, help="writers need >= this many valid IKIs"
    )
    p.add_argument("--max-heldout-writers", type=int, default=1500)
    p.add_argument("--window", type=int, default=50)
    p.add_argument("--pause-ms", type=float, default=2000.0)
    p.add_argument("--burst-ms", type=float, default=250.0)
    p.add_argument("--burst-min-run", type=int, default=3)
    p.add_argument("--bin-ms", type=float, default=5.0)
    p.add_argument("--n-traces", type=int, default=1000)
    p.add_argument("--n-events", type=int, default=200)
    p.add_argument("--sim-seed-base", type=int, default=10000)
    p.add_argument(
        "--ks-max", type=int, default=20000, help="subsample KLiCKe windows to this for KS"
    )
    p.add_argument("--seed", type=int, default=0, help="seed for writer and window subsampling")
    p.add_argument("--max-files", type=int, default=0, help="debug: load only the first N files")
    p.add_argument(
        "--split",
        choices=("heldout", "train"),
        default="heldout",
        help="KLiCKe half compared against the simulator (default heldout)",
    )
    p.add_argument(
        "--sim-override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a SimulationConfig field for the simulator side (repeatable)",
    )
    p.add_argument(
        "--candidate-id",
        default=None,
        help="log this heldout run in the `heldout_runs` list of --out",
    )
    p.add_argument(
        "--run-timestamp",
        default=None,
        help='timestamp string from the shell, e.g. "$(date -u +%%Y-%%m-%%dT%%H:%%M:%%SZ)"',
    )
    p.add_argument(
        "--run-note", default="", help="free-text reason stored with a logged heldout run"
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    t0 = time.time()
    if not args.klicke_dir.is_dir():
        print(f"KLiCKe directory not found: {args.klicke_dir}", file=sys.stderr)
        return 2
    causal_core, embodied = import_simulator(args.simulator_root)
    overrides = parse_overrides(args.sim_override)
    sim_cfg = apply_sim_overrides(causal_core, embodied, overrides)
    if args.candidate_id is not None and (args.split != "heldout" or not args.run_timestamp):
        print("--candidate-id needs --split heldout and --run-timestamp", file=sys.stderr)
        return 2
    prior_runs = load_prior_heldout_runs(args.out) if args.candidate_id is not None else []
    if len(prior_runs) >= MAX_HELDOUT_LOOKS:
        print(
            f"{args.out} already logs {len(prior_runs)} heldout runs (cap {MAX_HELDOUT_LOOKS}); refusing",
            file=sys.stderr,
        )
        return 2

    # ---- KLiCKe: load every writer ---------------------------------------
    files = sorted(args.klicke_dir.glob("*.csv"), key=lambda q: q.name)
    if args.max_files:
        files = files[: args.max_files]
    writers = {}  # writer_id -> (n_events, valid_iki)
    n_cp1252 = 0
    for i, f in enumerate(files):
        n_ev, valid, used_cp1252 = load_writer_iki(f, args.min_iki, args.max_iki)
        writers[f.stem] = (n_ev, valid)
        n_cp1252 += int(used_cp1252)
        if (i + 1) % 1000 == 0:
            print(f"  loaded {i + 1}/{len(files)} files ({time.time() - t0:.0f}s)", file=sys.stderr)
    t_load = time.time() - t0

    eligible, train_ids, heldout_ids = split_writers(
        {w: len(v) for w, (_, v) in writers.items()}, args.min_valid
    )
    split_of = dict.fromkeys(train_ids, "train")
    split_of.update(dict.fromkeys(heldout_ids, "heldout"))

    # ---- writer-level parameters (all eligible writers) --------------------
    table = []
    for w in eligible:
        n_ev, v = writers[w]
        row = {
            "writer_id": w,
            "split": split_of[w],
            "n_events": n_ev,
            "n_valid_iki": int(len(v)),
            "n_windows": int(len(v) // args.window),
        }
        row.update(writer_params(v, args.pause_ms))
        table.append(row)
    train_rows = [r for r in table if r["split"] == "train"]
    train_pct = {
        k: percentiles_10_50_90(np.array([r[k] for r in train_rows], dtype=np.float64))
        for k in WRITER_PARAMS
    }
    heldout_rows = [r for r in table if r["split"] == "heldout"]
    heldout_pct = {
        k: percentiles_10_50_90(np.array([r[k] for r in heldout_rows], dtype=np.float64))
        for k in WRITER_PARAMS
    }

    # ---- subsample of the evaluated split for the window comparison ----------
    eval_ids = heldout_ids if args.split == "heldout" else train_ids
    rng_writers = np.random.default_rng(args.seed)
    if len(eval_ids) > args.max_heldout_writers:
        pick = rng_writers.choice(len(eval_ids), size=args.max_heldout_writers, replace=False)
        eval_used = sorted(eval_ids[j] for j in pick)
    else:
        eval_used = list(eval_ids)
    klicke_seqs = [writers[w][1] for w in eval_used]

    # ---- simulator reference traces ----------------------------------------
    t1 = time.time()
    sim_desc, sim_raw, sim_lengths, floor_idx = generate_simulator_sequences(
        args.n_traces, args.n_events, args.sim_seed_base, causal_core, embodied
    )
    t_sim = time.time() - t1
    sim_seqs = [
        filter_valid(x, args.min_iki, args.max_iki) for x in sim_raw
    ]  # same filter as KLiCKe
    sim_all = np.concatenate(sim_raw)
    sim_unique_sequences = len({tuple(x.tolist()) for x in sim_raw})
    sim_dropped_by_filter = int(sum(len(a) - len(b) for a, b in zip(sim_raw, sim_seqs)))

    # ---- windows through the identical code path ---------------------------
    wm_args = (args.window, args.pause_ms, args.burst_ms, args.burst_min_run, args.bin_ms)
    km = window_metrics(klicke_seqs, *wm_args)
    sm = window_metrics(sim_seqs, *wm_args)

    rng_ks = np.random.default_rng(args.seed)
    n_k = len(km["mean_iki_ms"])
    ks_idx = (
        rng_ks.choice(n_k, size=args.ks_max, replace=False) if n_k > args.ks_max else np.arange(n_k)
    )
    metrics_out = {}
    for m in METRICS:
        k_all, s_all = km[m], sm[m]
        k_ks = k_all[ks_idx]
        k_ks = k_ks[~np.isnan(k_ks)]
        s_ks = s_all[~np.isnan(s_all)]
        if len(k_ks) and len(s_ks):
            res = ks_2samp(k_ks, s_ks)
            ks = {
                "ks_D": float(res.statistic),
                "ks_p": float(res.pvalue),
                "n_klicke": int(len(k_ks)),
                "n_sim": int(len(s_ks)),
            }
        else:
            ks = {"ks_D": None, "ks_p": None, "n_klicke": int(len(k_ks)), "n_sim": int(len(s_ks))}
        metrics_out[m] = {"klicke": quartiles(k_all), "simulator": quartiles(s_all), **ks}

    # ---- assemble ----------------------------------------------------------
    sim_glucose_floor = [g for g in floor_idx if g >= 0]
    result = {
        "meta": {
            "script": str(SCRIPT_DIR.relative_to(PACKAGE_ROOT) / "measure_gap.py"),
            "package_root": str(PACKAGE_ROOT),
            "klicke_dir": _rel(args.klicke_dir),
            "simulator_latency_path": sim_desc,
            "simulator_modules": {
                "embodied": _rel(embodied.__file__),
                "causal_core": _rel(causal_core.__file__),
            },
            "params": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "sim_config": dataclasses.asdict(sim_cfg),
            "sim_overrides": overrides,
            "iki_filter": f"{args.min_iki} < IKI < {args.max_iki} ms, applied to both populations",
            "split_rule": "eligible writers (>= min_valid valid IKIs) sorted by file name; "
            "even index -> train, odd index -> heldout",
            "sd_convention": "ddof=1 for sd_log_iki, sigma_w, pause_log_sd",
            "lag1_convention": "Pearson r between log IKI[t] and log IKI[t+1] (NaN when variance is 0)",
            "window_rule": "consecutive non-overlapping windows of `window` IKIs; remainder dropped",
            "runtime_s": {
                "load_klicke": round(t_load, 1),
                "simulate": round(t_sim, 1),
                "total": round(time.time() - t0, 1),
            },
        },
        "klicke": {
            "n_files": len(files),
            "n_files_cp1252": n_cp1252,
            "n_eligible_writers": len(eligible),
            "n_train_writers": len(train_ids),
            "n_heldout_writers": len(heldout_ids),
            "split_evaluated": args.split,
            f"n_{args.split}_writers_used": len(eval_used),
            "n_windows": int(n_k),
            "n_windows_ks": int(len(ks_idx)),
            f"valid_iki_total_{args.split}_used": int(sum(len(x) for x in klicke_seqs)),
        },
        "simulator": {
            "n_traces": args.n_traces,
            "n_events_requested": args.n_events,
            "events_per_trace": {"min": int(min(sim_lengths)), "max": int(max(sim_lengths))},
            "n_windows": int(len(sm["mean_iki_ms"])),
            "latency_ms": {
                "min": float(sim_all.min()),
                "max": float(sim_all.max()),
                "mean": float(sim_all.mean()),
                "n_unique_values": int(len(np.unique(sim_all))),
            },
            "n_unique_sequences": sim_unique_sequences,
            "n_latencies_dropped_by_iki_filter": sim_dropped_by_filter,
            "glucose_floor_reached_fraction": len(sim_glucose_floor) / max(1, args.n_traces),
            "glucose_floor_first_event_index": percentiles_10_50_90(
                np.array(sim_glucose_floor, dtype=np.float64)
            )
            if sim_glucose_floor
            else None,
        },
        "window_metrics": metrics_out,
        "writer_params_train_percentiles": train_pct,
        "writer_params_heldout_percentiles": heldout_pct,
    }

    if args.candidate_id is not None:
        result["heldout_runs"] = prior_runs + [
            heldout_log_entry(
                len(prior_runs) + 1,
                args,
                overrides,
                sim_cfg,
                metrics_out,
                n_k,
                len(ks_idx),
                len(sm["mean_iki_ms"]),
            )
        ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(_clean(result), fh, indent=2)
    with open(args.writer_params_out, "w") as fh:
        json.dump(
            _clean(
                {
                    "meta": {
                        "iki_filter": result["meta"]["iki_filter"],
                        "split_rule": result["meta"]["split_rule"],
                        "sd_convention": result["meta"]["sd_convention"],
                        "pause_definition": f"IKI > {args.pause_ms} ms",
                        "rho_w": "lag-1 Pearson autocorrelation of log IKI over the writer's whole valid sequence",
                        "n_writers": len(table),
                    },
                    "writers": table,
                }
            ),
            fh,
            indent=1,
        )

    # ---- console summary ---------------------------------------------------
    print(f"\nSimulator latency path: {sim_desc}")
    print(
        f"KLiCKe: {len(files)} files, {len(eligible)} eligible, train={len(train_ids)}, "
        f"heldout={len(heldout_ids)}; compared against {args.split} (used {len(eval_used)}), "
        f"windows={n_k} (KS n={len(ks_idx)})"
    )
    print(
        f"Simulator: {args.n_traces} traces x {args.n_events} events, windows={len(sm['mean_iki_ms'])}, "
        f"latency range [{sim_all.min():.2f}, {sim_all.max():.2f}] ms, "
        f"unique sequences={sim_unique_sequences}"
    )
    print("\n| metric | KLiCKe median [IQR] | simulator median [IQR] | KS D |")
    print("|---|---|---|---|")
    for m in METRICS:
        r = metrics_out[m]
        k, s = r["klicke"], r["simulator"]
        print(
            f"| {m} | {k['median']:.4g} [{k['q25']:.4g}, {k['q75']:.4g}] | "
            f"{s['median']:.4g} [{s['q25']:.4g}, {s['q75']:.4g}] | {r['ks_D']:.3f} |"
        )
    print("\nTrain-writer parameter percentiles (p10 / p50 / p90):")
    for k in WRITER_PARAMS:
        p = train_pct[k]
        print(f"  {k:18s} n={p['n']:5d}  {p['p10']:.4f} / {p['p50']:.4f} / {p['p90']:.4f}")
    print(f"\nWrote {args.out}\nWrote {args.writer_params_out}")
    print(f"Runtime: load {t_load:.0f}s, simulate {t_sim:.1f}s, total {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
