"""Validation pipeline comparing real keystroke data against embodied simulation.

Addresses SYS-002/C-006: all traces are synthetic with no real human validation.
Loads checkpoint analytics from keystroke-recorder SQLite database and compares
distributions against the EmbodiedScholar simulation via KS tests, Mann-Whitney U,
effect sizes, and Wasserstein distance.
"""
from __future__ import annotations

import csv
import math
import sqlite3
from dataclasses import dataclass, field, asdict
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any, Dict, List, Optional, Tuple

from .schema import CausalEvent
from .thermodynamic import compute_entropy_production_rate
from .free_energy import compute_free_energy_trajectory
from .metrics import _f_survival

__all__ = [
    "CheckpointRecord",
    "SessionSummary",
    "SignalComparison",
    "ValidationResult",
    "load_checkpoints_from_db",
    "load_checkpoints_from_csv",
    "verify_hash_chain",
    "generate_simulation_reference",
    "compare_distributions",
    "run_validation",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CheckpointRecord:
    session_id: str
    seq: int
    mean_iki_ms: float
    median_iki_ms: float
    std_iki_ms: float
    iki_entropy_bits: float
    lag1_autocorrelation: Optional[float]
    pause_count: int
    burst_count: int
    planning_pause_count: int
    translating_burst_count: int
    revising_delete_burst_count: int
    chars_added: int
    chars_deleted: int
    revision_density: float
    wpm: float
    h_prev: str
    h_content: str
    start_time_ns: int
    end_time_ns: int
    event_count: int


@dataclass(frozen=True)
class SessionSummary:
    session_id: str
    primary_app: str
    total_keystrokes: int
    avg_wpm: float
    n_checkpoints: int
    n_active_checkpoints: int
    chain_intact: bool
    chain_links_verified: int
    duration_minutes: float


@dataclass(frozen=True)
class SignalComparison:
    signal_name: str
    real_n: int
    sim_n: int
    real_median: float
    real_iqr: Tuple[float, float]
    sim_median: float
    sim_iqr: Tuple[float, float]
    ks_statistic: float
    ks_pvalue: float
    mann_whitney_u: float
    mann_whitney_pvalue: float
    cohens_d: float
    cliffs_delta: float
    wasserstein_distance: float
    levene_statistic: float
    levene_pvalue: float


@dataclass
class ValidationResult:
    sessions: List[SessionSummary]
    signal_comparisons: List[SignalComparison]
    phase_comparison: Dict[str, Any]
    chain_integrity: Dict[str, Any]
    config_snapshot: Dict[str, Any]
    metadata: Dict[str, Any]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

_CHECKPOINT_COLS = (
    "session_id, seq, mean_iki_ms, median_iki_ms, std_iki_ms, "
    "iki_entropy_bits, lag1_autocorrelation, pause_count, burst_count, "
    "planning_pause_count, translating_burst_count, revising_delete_burst_count, "
    "chars_added, chars_deleted, revision_density, wpm, "
    "h_prev, h_content, start_time_ns, end_time_ns, event_count"
)


def load_checkpoints_from_db(
    db_path: Path,
    min_events_per_checkpoint: int = 3,
    min_active_checkpoints: int = 5,
    app_filter: Optional[str] = None,
    session_filter: Optional[str] = None,
) -> Tuple[List[CheckpointRecord], List[SessionSummary]]:
    """Load checkpoint records from keystroke-recorder SQLite database.

    Filters to sessions with substantial typing activity and checkpoints
    with non-zero metrics (95% of checkpoints are idle periods).
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Find qualifying sessions
    session_query = """
        SELECT id, primary_app, total_keystrokes, avg_wpm,
               (end_time_ns - start_time_ns) / 1e9 / 60.0 as duration_minutes
        FROM sessions
        WHERE total_keystrokes > 50
    """
    params: list = []
    if app_filter:
        session_query += " AND primary_app LIKE ?"
        params.append(f"%{app_filter}%")
    if session_filter:
        session_query += " AND id = ?"
        params.append(session_filter)

    session_rows = conn.execute(session_query, params).fetchall()

    all_checkpoints: List[CheckpointRecord] = []
    summaries: List[SessionSummary] = []

    for srow in session_rows:
        sid = srow["id"]
        cp_query = f"""
            SELECT {_CHECKPOINT_COLS}
            FROM checkpoints
            WHERE session_id = ? AND event_count >= ?
            ORDER BY seq
        """
        cp_rows = conn.execute(cp_query, (sid, min_events_per_checkpoint)).fetchall()

        active = [r for r in cp_rows if r["mean_iki_ms"] > 0]
        if len(active) < min_active_checkpoints:
            continue

        cps = [_row_to_checkpoint(r) for r in active]
        all_checkpoints.extend(cps)

        # Verify hash chain (on ALL checkpoints, not just active)
        all_cp_rows = conn.execute(
            "SELECT h_prev, h_content FROM checkpoints WHERE session_id = ? ORDER BY seq",
            (sid,),
        ).fetchall()
        chain_ok, chain_n = _verify_chain_rows(all_cp_rows)

        dur = srow["duration_minutes"] if srow["duration_minutes"] else 0.0
        summaries.append(SessionSummary(
            session_id=sid,
            primary_app=srow["primary_app"] or "unknown",
            total_keystrokes=srow["total_keystrokes"],
            avg_wpm=srow["avg_wpm"],
            n_checkpoints=len(cp_rows) + len([r for r in conn.execute(
                "SELECT 1 FROM checkpoints WHERE session_id = ?", (sid,)
            ).fetchall()]) - len(cp_rows),
            n_active_checkpoints=len(active),
            chain_intact=chain_ok,
            chain_links_verified=chain_n,
            duration_minutes=round(abs(dur), 1),
        ))

    conn.close()
    return all_checkpoints, summaries


def load_checkpoints_from_csv(csv_path: Path) -> List[CheckpointRecord]:
    """Load checkpoint records from pre-exported CSV."""
    records = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mean_iki = float(row.get("mean_iki_ms", 0))
            if mean_iki <= 0:
                continue
            records.append(CheckpointRecord(
                session_id=row["session_id"],
                seq=int(row["seq"]),
                mean_iki_ms=mean_iki,
                median_iki_ms=float(row.get("median_iki_ms", 0)),
                std_iki_ms=float(row.get("std_iki_ms", 0)),
                iki_entropy_bits=float(row.get("iki_entropy_bits", 0)),
                lag1_autocorrelation=_float_or_none(row.get("lag1_autocorrelation")),
                pause_count=int(row.get("pause_count", 0)),
                burst_count=int(row.get("burst_count", 0)),
                planning_pause_count=int(row.get("planning_pause_count", 0)),
                translating_burst_count=int(row.get("translating_burst_count", 0)),
                revising_delete_burst_count=int(row.get("revising_delete_burst_count", 0)),
                chars_added=int(row.get("chars_added", 0)),
                chars_deleted=int(row.get("chars_deleted", 0)),
                revision_density=float(row.get("revision_density", 0)),
                wpm=float(row.get("wpm", 0)),
                h_prev=row.get("h_prev", ""),
                h_content=row.get("h_content", ""),
                start_time_ns=int(row.get("start_time_ns", 0)),
                end_time_ns=int(row.get("end_time_ns", 0)),
                event_count=int(row.get("event_count", 0)),
            ))
    return records


def verify_hash_chain(
    checkpoints: List[CheckpointRecord], session_id: str
) -> Tuple[bool, int, int]:
    """Verify h_prev -> h_content chain integrity for a session.

    Returns (all_valid, valid_links, total_links).
    """
    session_cps = sorted(
        [c for c in checkpoints if c.session_id == session_id],
        key=lambda c: c.seq,
    )
    if len(session_cps) < 2:
        return True, 0, 0

    valid = 0
    total = len(session_cps) - 1
    for i in range(1, len(session_cps)):
        if session_cps[i].h_prev == session_cps[i - 1].h_content:
            valid += 1
    return valid == total, valid, total


# ---------------------------------------------------------------------------
# Simulation bridge
# ---------------------------------------------------------------------------

def generate_simulation_reference(
    n_traces: int = 200,
    n_events: int = 50,
    seed: int = 42,
) -> List[Dict[str, float]]:
    """Generate simulated checkpoint-equivalent metrics from EmbodiedScholar.

    Runs IrreversibleProcessEngine traces and extracts aggregate metrics
    comparable to keystroke-recorder checkpoint analytics.
    """
    from .adversarial import _generate_authentic_trace

    results = []
    for i in range(n_traces):
        trace = _generate_authentic_trace(seed=seed + i, n_events=n_events)
        metrics = _trace_to_checkpoint_metrics(trace)
        results.append(metrics)
    return results


def _trace_to_checkpoint_metrics(trace: List[CausalEvent]) -> Dict[str, float]:
    """Convert a CausalEvent trace to checkpoint-equivalent aggregate metrics."""
    latencies = [e.latency_ms for e in trace]
    n = len(latencies)
    if n < 3:
        return {
            "mean_iki_ms": 0.0, "median_iki_ms": 0.0, "std_iki_ms": 0.0,
            "iki_entropy_bits": 0.0, "lag1_autocorrelation": 0.0,
            "revision_density": 0.0, "wpm": 0.0,
            "planning_ratio": 0.0, "translating_ratio": 0.0, "revising_ratio": 0.0,
        }

    mean_lat = mean(latencies)
    med_lat = median(latencies)
    std_lat = stdev(latencies) if n > 1 else 0.0

    # Entropy via thermodynamic module
    entropy = compute_entropy_production_rate(trace)

    # Lag-1 autocorrelation of latencies
    lag1 = _lag1_autocorr(latencies)

    # Revision density = failure rate
    failures = sum(1 for e in trace if e.status != "success")
    rev_density = failures / n

    # Approximate WPM (assume 5 chars per word, 200ms per token average)
    total_time_ms = sum(latencies)
    wpm = (n / max(total_time_ms / 60000.0, 0.001))

    # Phase proportions from free energy trajectory
    planning_r, translating_r, revising_r = _extract_phase_proportions(trace)

    return {
        "mean_iki_ms": round(mean_lat, 2),
        "median_iki_ms": round(med_lat, 2),
        "std_iki_ms": round(std_lat, 2),
        "iki_entropy_bits": round(entropy, 4),
        "lag1_autocorrelation": round(lag1, 4),
        "revision_density": round(rev_density, 4),
        "wpm": round(wpm, 2),
        "planning_ratio": round(planning_r, 4),
        "translating_ratio": round(translating_r, 4),
        "revising_ratio": round(revising_r, 4),
    }


def _lag1_autocorr(values: List[float]) -> float:
    """Compute lag-1 autocorrelation of a series."""
    n = len(values)
    if n < 3:
        return 0.0
    mu = mean(values)
    var = sum((v - mu) ** 2 for v in values) / n
    if var < 1e-15:
        return 0.0
    cov = sum((values[i] - mu) * (values[i + 1] - mu) for i in range(n - 1)) / (n - 1)
    return cov / var


def _extract_phase_proportions(trace: List[CausalEvent]) -> Tuple[float, float, float]:
    """Extract planning/translating/revising phase proportions from a trace."""
    n = len(trace)
    if n < 6:
        return 0.33, 0.34, 0.33

    fe = compute_free_energy_trajectory(trace)
    boundaries = fe.get("phase_boundaries", [])
    if len(boundaries) == 2:
        b1, b2 = boundaries
        planning = b1 / n
        translating = (b2 - b1) / n
        revising = (n - b2) / n
        return planning, translating, revising

    return 0.33, 0.34, 0.33


# ---------------------------------------------------------------------------
# Pure-Python statistical tests
# ---------------------------------------------------------------------------

def compare_distributions(
    real_values: List[float],
    sim_values: List[float],
    signal_name: str,
) -> SignalComparison:
    """Run full statistical comparison suite on two distributions."""
    r = sorted(real_values)
    s = sorted(sim_values)

    r_med = median(r)
    s_med = median(s)
    r_iqr = _iqr(r)
    s_iqr = _iqr(s)

    ks_d, ks_p = _ks_test_2sample(r, s)
    u_stat, u_p = _mann_whitney_u(r, s)
    cd = _cohens_d(r, s)
    cliff = _cliffs_delta(r, s)
    w1 = _wasserstein_1d(r, s)
    lev_f, lev_p = _levene_test(r, s)

    return SignalComparison(
        signal_name=signal_name,
        real_n=len(r), sim_n=len(s),
        real_median=round(r_med, 4), real_iqr=(round(r_iqr[0], 4), round(r_iqr[1], 4)),
        sim_median=round(s_med, 4), sim_iqr=(round(s_iqr[0], 4), round(s_iqr[1], 4)),
        ks_statistic=round(ks_d, 4), ks_pvalue=round(ks_p, 4),
        mann_whitney_u=round(u_stat, 2), mann_whitney_pvalue=round(u_p, 4),
        cohens_d=round(cd, 4), cliffs_delta=round(cliff, 4),
        wasserstein_distance=round(w1, 4),
        levene_statistic=round(lev_f, 4), levene_pvalue=round(lev_p, 4),
    )


def _iqr(sorted_vals: List[float]) -> Tuple[float, float]:
    n = len(sorted_vals)
    if n == 0:
        return (0.0, 0.0)
    q1 = sorted_vals[n // 4]
    q3 = sorted_vals[(3 * n) // 4]
    return (q1, q3)


def _ks_test_2sample(a: List[float], b: List[float]) -> Tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov test.

    Returns (D_statistic, p_value) using the Kolmogorov limiting distribution.
    """
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return (1.0, 0.0)

    combined = sorted(set(a + b))
    d_max = 0.0
    ia = ib = 0
    for val in combined:
        while ia < na and a[ia] <= val:
            ia += 1
        while ib < nb and b[ib] <= val:
            ib += 1
        cdf_a = ia / na
        cdf_b = ib / nb
        d_max = max(d_max, abs(cdf_a - cdf_b))

    # Kolmogorov limiting distribution approximation
    en = math.sqrt(na * nb / (na + nb))
    lam = (en + 0.12 + 0.11 / en) * d_max
    p = _kolmogorov_prob(lam)
    return (d_max, max(0.0, min(1.0, p)))


def _kolmogorov_prob(lam: float) -> float:
    """P(K > lambda) for the Kolmogorov distribution via alternating series."""
    if lam <= 0:
        return 1.0
    if lam > 3.0:
        return 0.0
    p = 0.0
    for k in range(1, 100):
        term = (-1) ** (k - 1) * math.exp(-2.0 * k * k * lam * lam)
        p += term
        if abs(term) < 1e-12:
            break
    return max(0.0, min(1.0, 2.0 * p))


def _mann_whitney_u(a: List[float], b: List[float]) -> Tuple[float, float]:
    """Mann-Whitney U test with normal approximation for p-value."""
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return (0.0, 1.0)

    # Count how many b values each a value exceeds
    u = 0.0
    for va in a:
        for vb in b:
            if va > vb:
                u += 1.0
            elif va == vb:
                u += 0.5

    mu_u = na * nb / 2.0
    sigma_u = math.sqrt(na * nb * (na + nb + 1) / 12.0)
    if sigma_u < 1e-15:
        return (u, 1.0)

    z = abs(u - mu_u) / sigma_u
    # Two-tailed p-value via standard normal CDF approximation
    p = 2.0 * (1.0 - _norm_cdf(z))
    return (u, max(0.0, min(1.0, p)))


def _norm_cdf(z: float) -> float:
    """Standard normal CDF approximation (Abramowitz & Stegun)."""
    if z < -8.0:
        return 0.0
    if z > 8.0:
        return 1.0
    t = 1.0 / (1.0 + 0.2316419 * abs(z))
    d = 0.3989422804014327  # 1/sqrt(2*pi)
    poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
    p = 1.0 - d * math.exp(-0.5 * z * z) * poly
    return p if z >= 0 else 1.0 - p


def _cohens_d(a: List[float], b: List[float]) -> float:
    """Cohen's d effect size with pooled standard deviation."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    ma, mb = mean(a), mean(b)
    sa, sb = stdev(a), stdev(b)
    pooled = math.sqrt(((na - 1) * sa ** 2 + (nb - 1) * sb ** 2) / (na + nb - 2))
    if pooled < 1e-15:
        return 0.0
    return (ma - mb) / pooled


def _cliffs_delta(a: List[float], b: List[float]) -> float:
    """Cliff's delta nonparametric effect size in [-1, 1]."""
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return 0.0
    count = 0.0
    for va in a:
        for vb in b:
            if va > vb:
                count += 1.0
            elif va < vb:
                count -= 1.0
    return count / (na * nb)


def _wasserstein_1d(a: List[float], b: List[float]) -> float:
    """1D Wasserstein (Earth Mover's) distance between sorted distributions."""
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return 0.0
    sa = sorted(a)
    sb = sorted(b)
    # Interpolate both to common quantile grid
    n_pts = max(na, nb, 100)
    total = 0.0
    for i in range(n_pts):
        q = (i + 0.5) / n_pts
        va = sa[min(int(q * na), na - 1)]
        vb = sb[min(int(q * nb), nb - 1)]
        total += abs(va - vb)
    return total / n_pts


def _levene_test(a: List[float], b: List[float]) -> Tuple[float, float]:
    """Levene's test for equality of variances (median-based)."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return (0.0, 1.0)

    med_a, med_b = median(a), median(b)
    za = [abs(v - med_a) for v in a]
    zb = [abs(v - med_b) for v in b]

    mean_za, mean_zb = mean(za), mean(zb)
    grand_mean = (sum(za) + sum(zb)) / (na + nb)

    ss_between = na * (mean_za - grand_mean) ** 2 + nb * (mean_zb - grand_mean) ** 2
    ss_within = sum((z - mean_za) ** 2 for z in za) + sum((z - mean_zb) ** 2 for z in zb)

    if ss_within < 1e-15:
        return (0.0, 1.0)

    k = 2  # number of groups
    n_total = na + nb
    f_stat = (ss_between / (k - 1)) / (ss_within / (n_total - k))
    p_val = _f_survival(f_stat, k - 1, n_total - k)
    return (f_stat, p_val)


# ---------------------------------------------------------------------------
# Phase comparison
# ---------------------------------------------------------------------------

def compare_phase_proportions(
    checkpoints: List[CheckpointRecord],
    sim_metrics: List[Dict[str, float]],
) -> Dict[str, Any]:
    """Compare Flower & Hayes 3-phase proportions: real vs simulated."""
    # Real: compute from checkpoint phase counts
    total_planning = sum(c.planning_pause_count for c in checkpoints)
    total_translating = sum(c.translating_burst_count for c in checkpoints)
    total_revising = sum(c.revising_delete_burst_count for c in checkpoints)
    total_real = total_planning + total_translating + total_revising

    if total_real > 0:
        real_plan = total_planning / total_real
        real_trans = total_translating / total_real
        real_rev = total_revising / total_real
    else:
        real_plan = real_trans = real_rev = 0.33

    # Simulated: average phase proportions across traces
    sim_plan = mean([m.get("planning_ratio", 0.33) for m in sim_metrics])
    sim_trans = mean([m.get("translating_ratio", 0.34) for m in sim_metrics])
    sim_rev = mean([m.get("revising_ratio", 0.33) for m in sim_metrics])

    return {
        "real": {"planning": round(real_plan, 4), "translating": round(real_trans, 4), "revising": round(real_rev, 4)},
        "simulated": {"planning": round(sim_plan, 4), "translating": round(sim_trans, 4), "revising": round(sim_rev, 4)},
        "real_total_events": total_real,
        "sim_n_traces": len(sim_metrics),
    }


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_validation(
    db_path: Optional[Path] = None,
    checkpoint_csv: Optional[Path] = None,
    n_sim_traces: int = 200,
    n_events: int = 50,
    seed: int = 42,
    app_filter: Optional[str] = None,
    session_filter: Optional[str] = None,
) -> ValidationResult:
    """Execute the full validation pipeline."""
    from .config import snapshot_config

    # Load real data
    if checkpoint_csv:
        checkpoints = load_checkpoints_from_csv(checkpoint_csv)
        summaries: List[SessionSummary] = []
    elif db_path and db_path.exists():
        checkpoints, summaries = load_checkpoints_from_db(
            db_path, app_filter=app_filter, session_filter=session_filter,
        )
    else:
        raise FileNotFoundError(f"No data source: db_path={db_path}, csv={checkpoint_csv}")

    if not checkpoints:
        raise ValueError("No qualifying checkpoints found. Try lowering min_active_checkpoints.")

    # Generate simulation reference
    sim_metrics = generate_simulation_reference(n_sim_traces, n_events, seed)

    # Signal comparisons
    signal_map = [
        ("mean_iki_ms", [c.mean_iki_ms for c in checkpoints], [m["mean_iki_ms"] for m in sim_metrics]),
        ("iki_entropy", [c.iki_entropy_bits for c in checkpoints if c.iki_entropy_bits > 0], [m["iki_entropy_bits"] for m in sim_metrics]),
        ("lag1_autocorrelation", [c.lag1_autocorrelation for c in checkpoints if c.lag1_autocorrelation is not None], [m["lag1_autocorrelation"] for m in sim_metrics]),
        ("revision_density", [c.revision_density for c in checkpoints], [m["revision_density"] for m in sim_metrics]),
        ("wpm", [c.wpm for c in checkpoints if c.wpm > 0], [m["wpm"] for m in sim_metrics]),
    ]

    comparisons = []
    for name, real, sim in signal_map:
        if len(real) >= 5 and len(sim) >= 5:
            comparisons.append(compare_distributions(real, sim, name))

    # Phase comparison
    phase = compare_phase_proportions(checkpoints, sim_metrics)

    # Chain integrity
    chain_results = {}
    for s in summaries:
        chain_results[s.session_id] = {
            "intact": s.chain_intact,
            "links_verified": s.chain_links_verified,
        }

    config = snapshot_config() if hasattr(snapshot_config, "__call__") else {}

    return ValidationResult(
        sessions=summaries,
        signal_comparisons=comparisons,
        phase_comparison=phase,
        chain_integrity={
            "sessions_checked": len(summaries),
            "all_intact": all(s.chain_intact for s in summaries),
            "total_links_verified": sum(s.chain_links_verified for s in summaries),
            "per_session": chain_results,
        },
        config_snapshot=config,
        metadata={
            "n_checkpoints": len(checkpoints),
            "n_sessions": len(summaries),
            "n_sim_traces": n_sim_traces,
            "n_events_per_trace": n_events,
            "seed": seed,
        },
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row_to_checkpoint(row: sqlite3.Row) -> CheckpointRecord:
    return CheckpointRecord(
        session_id=row["session_id"],
        seq=row["seq"],
        mean_iki_ms=row["mean_iki_ms"] or 0.0,
        median_iki_ms=row["median_iki_ms"] or 0.0,
        std_iki_ms=row["std_iki_ms"] or 0.0,
        iki_entropy_bits=row["iki_entropy_bits"] or 0.0,
        lag1_autocorrelation=row["lag1_autocorrelation"],
        pause_count=row["pause_count"] or 0,
        burst_count=row["burst_count"] or 0,
        planning_pause_count=row["planning_pause_count"] or 0,
        translating_burst_count=row["translating_burst_count"] or 0,
        revising_delete_burst_count=row["revising_delete_burst_count"] or 0,
        chars_added=row["chars_added"] or 0,
        chars_deleted=row["chars_deleted"] or 0,
        revision_density=row["revision_density"] or 0.0,
        wpm=row["wpm"] or 0.0,
        h_prev=row["h_prev"] or "",
        h_content=row["h_content"] or "",
        start_time_ns=row["start_time_ns"] or 0,
        end_time_ns=row["end_time_ns"] or 0,
        event_count=row["event_count"] or 0,
    )


def _verify_chain_rows(rows: list) -> Tuple[bool, int]:
    if len(rows) < 2:
        return True, 0
    valid = 0
    for i in range(1, len(rows)):
        if rows[i]["h_prev"] == rows[i - 1]["h_content"]:
            valid += 1
    total = len(rows) - 1
    return valid == total, valid


def _float_or_none(val: Optional[str]) -> Optional[float]:
    if val is None or val == "":
        return None
    try:
        return float(val)
    except ValueError:
        return None
