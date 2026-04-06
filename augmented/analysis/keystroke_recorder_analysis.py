"""
Keystroke Recorder Field Data Analysis
======================================
Analyzes real-world keystroke data from ~/.local/data/keystrokes.db
to validate ScholaWrite-Augmented embodied simulation parameters.

Key challenge: the recorder captures ALL applications, so we must
filter for genuine composition sessions (sustained writing in editors)
and discard short-burst typing (terminal commands, URLs, form fields).
"""

import json
import logging
import sqlite3
from pathlib import Path

import numpy as np
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

DB_PATH = Path.home() / ".local" / "data" / "keystrokes.db"
OUTPUT_PATH = Path(__file__).parent / "keystroke_recorder_results.json"

COMPOSITION_APPS = {
    "com.apple.TextEdit",
    "com.microsoft.VSCode",
    "com.apple.Notes",
    "com.apple.dt.Xcode",
    "com.literatureandlatte.scrivener3",
    "com.sublimetext.4",
    "md.obsidian",
}

MIN_KEYSTROKES = 50
MIN_WORDS = 10

log.info("=" * 70)
log.info("Keystroke Recorder Field Data Analysis")
log.info("=" * 70)

conn = sqlite3.connect(str(DB_PATH))
conn.row_factory = sqlite3.Row

# ============================================================================
# 1. Session-level filtering
# ============================================================================
log.info("\n[1/5] Filtering composition sessions...")

all_sessions = conn.execute("""
    SELECT * FROM sessions WHERE total_keystrokes > 0
""").fetchall()

composition_sessions = conn.execute("""
    SELECT * FROM sessions
    WHERE total_keystrokes >= ? AND total_words >= ?
      AND primary_app IN ({})
""".format(",".join("?" * len(COMPOSITION_APPS))),
    [MIN_KEYSTROKES, MIN_WORDS] + list(COMPOSITION_APPS)
).fetchall()

log.info("  Total sessions with keystrokes: %d", len(all_sessions))
log.info("  Composition sessions (>=%d keys, >=%d words, editor apps): %d",
         MIN_KEYSTROKES, MIN_WORDS, len(composition_sessions))
log.info("  Short-burst sessions filtered out: %d",
         len(all_sessions) - len(composition_sessions))

total_composition_keys = sum(s["total_keystrokes"] for s in composition_sessions)
total_composition_words = sum(s["total_words"] for s in composition_sessions)
log.info("  Total composition keystrokes: %s", f"{total_composition_keys:,}")
log.info("  Total composition words: %s", f"{total_composition_words:,}")

# ============================================================================
# 2. Checkpoint-level analysis (composition sessions only)
# ============================================================================
log.info("\n[2/5] Analyzing composition checkpoints...")

session_ids = [s["id"] for s in composition_sessions]
placeholders = ",".join("?" * len(session_ids))

checkpoints = conn.execute(f"""
    SELECT c.* FROM checkpoints c
    WHERE c.session_id IN ({placeholders})
      AND c.wpm > 0
      AND c.clc_correlation IS NOT NULL
""", session_ids).fetchall()

all_checkpoints = conn.execute(f"""
    SELECT c.* FROM checkpoints c
    WHERE c.session_id IN ({placeholders})
      AND c.wpm > 0
""", session_ids).fetchall()

log.info("  Active checkpoints with CLC: %d", len(checkpoints))
log.info("  Active checkpoints (all): %d", len(all_checkpoints))

wpm = np.array([c["wpm"] for c in checkpoints])
mean_iki = np.array([c["mean_iki_ms"] for c in checkpoints])
entropy = np.array([c["iki_entropy_bits"] for c in checkpoints])
clc = np.array([c["clc_correlation"] for c in checkpoints])
planning = np.array([c["planning_pause_count"] for c in checkpoints])
translating = np.array([c["translating_burst_count"] for c in checkpoints])
revising = np.array([c["revising_delete_burst_count"] for c in checkpoints])
rev_density = np.array([c["revision_density"] for c in checkpoints])
lag1_acf = np.array([c["lag1_autocorrelation"] for c in checkpoints if c["lag1_autocorrelation"] is not None])
chars_added = np.array([c["chars_added"] for c in checkpoints])
chars_deleted = np.array([c["chars_deleted"] for c in checkpoints])

log.info("\n  WPM:     mean=%.1f, median=%.1f, std=%.1f", wpm.mean(), np.median(wpm), wpm.std())
log.info("  IKI:     mean=%.1fms, median=%.1fms, std=%.1fms", mean_iki.mean(), np.median(mean_iki), mean_iki.std())
log.info("  Entropy: mean=%.1f bits, median=%.1f bits", entropy.mean(), np.median(entropy))

# ============================================================================
# 3. CLC analysis
# ============================================================================
log.info("\n[3/5] Cognitive Load Correlation (CLC) analysis...")

log.info("  CLC:     mean=%.4f, median=%.4f", clc.mean(), np.median(clc))
log.info("           min=%.4f, max=%.4f, std=%.4f", clc.min(), clc.max(), clc.std())

pos_clc = (clc > 0).sum()
neg_clc = (clc < 0).sum()
strong_pos = (clc > 0.3).sum()
strong_neg = (clc < -0.3).sum()
log.info("  Positive CLC (>0): %d/%d (%.1f%%)", pos_clc, len(clc), 100 * pos_clc / len(clc))
log.info("  Negative CLC (<0): %d/%d (%.1f%%)", neg_clc, len(clc), 100 * neg_clc / len(clc))
log.info("  Strong positive (>0.3): %d/%d (%.1f%%)", strong_pos, len(clc), 100 * strong_pos / len(clc))
log.info("  Strong negative (<-0.3): %d/%d (%.1f%%)", strong_neg, len(clc), 100 * strong_neg / len(clc))

log.info("\n  Per-session CLC:")
session_clc = {}
for cp in checkpoints:
    sid = cp["session_id"]
    if sid not in session_clc:
        session_clc[sid] = []
    session_clc[sid].append(cp["clc_correlation"])

per_session_mean_clc = [np.mean(v) for v in session_clc.values()]
pval = None
if per_session_mean_clc:
    stat, pval = stats.wilcoxon(per_session_mean_clc)
    log.info("    Session-level mean CLC: median=%.4f", np.median(per_session_mean_clc))
    log.info("    Wilcoxon signed-rank test vs 0: W=%.1f, p=%.4f", stat, pval)
    log.info("    Sessions with positive mean CLC: %d/%d",
             sum(1 for x in per_session_mean_clc if x > 0), len(per_session_mean_clc))

# ============================================================================
# 4. Hayes-Flower phase analysis
# ============================================================================
log.info("\n[4/5] Hayes-Flower cognitive phase analysis...")

total_planning = planning.sum()
total_translating = translating.sum()
total_revising = revising.sum()
total_phases = total_planning + total_translating + total_revising

log.info("  Planning pauses:    %d (%.1f%%)", total_planning, 100 * total_planning / max(total_phases, 1))
log.info("  Translating bursts: %d (%.1f%%)", total_translating, 100 * total_translating / max(total_phases, 1))
log.info("  Revising bursts:    %d (%.1f%%)", total_revising, 100 * total_revising / max(total_phases, 1))
log.info("  Mean revision density: %.4f", rev_density.mean())
log.info("  Mean chars added/checkpoint: %.1f", chars_added.mean())
log.info("  Mean chars deleted/checkpoint: %.1f", chars_deleted.mean())
log.info("  Delete/add ratio: %.3f", chars_deleted.sum() / max(chars_added.sum(), 1))

# ============================================================================
# 5. Validation against simulation parameters
# ============================================================================
log.info("\n[5/5] Validation against ScholaWrite-Augmented simulation parameters...")

sim_baseline_latency = 139  # ms (g=1.0, d=1)

field_mean_iki = mean_iki.mean()
field_median_iki = np.median(mean_iki)

low_complexity = mean_iki[wpm > np.percentile(wpm, 75)]
high_complexity = mean_iki[wpm < np.percentile(wpm, 25)]

log.info("  Field mean IKI (all composition): %.1fms", field_mean_iki)
log.info("  Field median IKI: %.1fms", field_median_iki)
log.info("  Simulation baseline: %dms", sim_baseline_latency)
log.info("  Low-complexity checkpoints (top 25%% WPM): mean IKI = %.1fms", low_complexity.mean())
log.info("  High-complexity checkpoints (bottom 25%% WPM): mean IKI = %.1fms", high_complexity.mean())
complexity_ratio = high_complexity.mean() / low_complexity.mean()
log.info("  Complexity ratio (high/low): %.2fx", complexity_ratio)
log.info("  Simulation predicts ~4x (550/139); field shows %.2fx", complexity_ratio)

if len(lag1_acf) > 0:
    log.info("\n  Lag-1 IKI autocorrelation: mean=%.4f, std=%.4f", lag1_acf.mean(), lag1_acf.std())
    log.info("  (Positive ACF indicates serial dependency in typing rhythm)")

# ============================================================================
# Summary
# ============================================================================
log.info("\n" + "=" * 70)
log.info("SUMMARY")
log.info("=" * 70)
log.info("  Composition sessions: %d (%s keys, %s words)",
         len(composition_sessions), f"{total_composition_keys:,}", f"{total_composition_words:,}")
log.info("  Active checkpoints with CLC: %d", len(checkpoints))
log.info("  Field CLC: median=%.4f (session median=%.4f)", np.median(clc), np.median(per_session_mean_clc))
log.info("  Phase distribution: %.0f%% planning, %.0f%% translating, %.0f%% revising",
         100 * total_planning / max(total_phases, 1),
         100 * total_translating / max(total_phases, 1),
         100 * total_revising / max(total_phases, 1))
log.info("  IKI complexity ratio: %.2fx (sim predicts ~4x)", complexity_ratio)

results = {
    "data_source": "keystroke-recorder field data",
    "db_path": str(DB_PATH),
    "filtering": {
        "composition_apps": sorted(COMPOSITION_APPS),
        "min_keystrokes": MIN_KEYSTROKES,
        "min_words": MIN_WORDS,
    },
    "sessions": {
        "total_with_keystrokes": len(all_sessions),
        "composition_sessions": len(composition_sessions),
        "total_composition_keystrokes": total_composition_keys,
        "total_composition_words": total_composition_words,
    },
    "checkpoints": {
        "active_with_clc": len(checkpoints),
        "wpm": {"mean": float(wpm.mean()), "median": float(np.median(wpm)), "std": float(wpm.std())},
        "iki_ms": {"mean": float(mean_iki.mean()), "median": float(np.median(mean_iki)), "std": float(mean_iki.std())},
        "entropy_bits": {"mean": float(entropy.mean()), "median": float(np.median(entropy)), "std": float(entropy.std())},
    },
    "clc": {
        "mean": float(clc.mean()),
        "median": float(np.median(clc)),
        "std": float(clc.std()),
        "min": float(clc.min()),
        "max": float(clc.max()),
        "pct_positive": float(100 * pos_clc / len(clc)),
        "pct_strong_positive": float(100 * strong_pos / len(clc)),
        "session_median": float(np.median(per_session_mean_clc)),
        "wilcoxon_p": float(pval) if pval is not None else None,
    },
    "phases": {
        "planning_pauses": int(total_planning),
        "translating_bursts": int(total_translating),
        "revising_bursts": int(total_revising),
        "revision_density_mean": float(rev_density.mean()),
        "delete_add_ratio": float(chars_deleted.sum() / max(chars_added.sum(), 1)),
    },
    "simulation_validation": {
        "field_mean_iki_ms": float(field_mean_iki),
        "sim_baseline_iki_ms": sim_baseline_latency,
        "low_complexity_iki_ms": float(low_complexity.mean()),
        "high_complexity_iki_ms": float(high_complexity.mean()),
        "complexity_ratio": float(complexity_ratio),
        "sim_predicted_ratio": 3.96,
        "lag1_acf_mean": float(lag1_acf.mean()) if len(lag1_acf) > 0 else None,
    },
}

with open(OUTPUT_PATH, "w") as f:
    json.dump(results, f, indent=2)
log.info("\nResults saved to %s", OUTPUT_PATH)

conn.close()
