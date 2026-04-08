# Spectrum Hypothesis: Tested and Falsified

**Date**: 2026-04-07  
**Status**: FALSIFIED

---

## Hypothesis

Keystroke timing accuracy degrades monotonically along a process-to-state spectrum:

**Process-intrinsic** (high accuracy expected):
- Quality assessment
- Fatigue trajectory
- Temporal structure
- Regime switching

**State-intrinsic** (low accuracy expected):
- Parkinson's disease detection
- Emotion detection
- Writer identification
- Consciousness signals

---

## Test Method

Pre-specified ordering of 7 analyses by predicted position on spectrum. Extracted accuracy/discriminability metrics from existing results. Checked for monotonic degradation.

**Pre-specified order:**
1. Quality (ρ=0.5278)
2. Fatigue (1.0000)
3. Temporal ACF (0.1263)
4. Regime Switch (0.2264)
5. Parkinson's (0.5816)
6. Emotion (0.3000)
7. Writer ID (0.0116)

---

## Result

**Non-monotonic pattern with 3 reversals:**

```
Step 1→2: +0.4722 ↑ (EXPECTED ↓, GOT ↑)
Step 2→3: -0.8737 ↓ (expected)
Step 3→4: +0.1001 ↑ (EXPECTED ↓, GOT ↑)
Step 4→5: +0.3552 ↑ (EXPECTED ↓, GOT ↑)
Step 5→6: -0.2816 ↓ (expected)
Step 6→7: -0.2884 ↓ (expected)
```

**Interpretation**: 
- Fatigue jumps above Quality (contradicts process-level prediction)
- Temporal ACF is weakest (contradicts being process-level)
- Parkinson's performs better than Regime Switching (contradicts state-level expectation)

**Conclusion**: Spectrum hypothesis is **falsified**. No monotonic relationship between position on process-state spectrum and keystroke analysis accuracy.

---

## What This Means

The framework that motivated 6 strict-filter scripts is false. The boundary between what keystroke timing can and cannot detect does **not** follow a clean process-to-state gradient.

What actually exists:
- Some keystroke properties work well (fatigue 1.0, quality 0.53)
- Some keystroke properties work poorly (temporal ACF 0.13, emotion 0.30)
- The pattern has **no predictive relationship** to theoretical position on process-state spectrum

---

## Scripts in This Directory

- `03_parkinsons_validation_strict.py` — filter: ≥30 subjects/group
- `04_emotion_dynamics_strict.py` — filter: ≥50 users/emotion
- `06_writer_identification_strict.py` — filter: ≥1000 IKI/half
- `12_cross_language_iki_strict.py` — filter: ≥15 participants/nationality
- `13_fatigue_trajectory_validation_strict.py` — filter: ≥3 sessions, within-writer slopes
- `14_pause_burst_spectrum_strict.py` — filter: stratification by task type

**These scripts document the hypothesis-testing process.** They were created to test whether filtering data according to the spectrum model would produce clean monotonic results. The hypothesis failed, so these scripts are archived here as evidence of what was tried and why it didn't work.

---

## Lessons

1. **Pre-specification is essential.** Without writing down the ordering before analyzing, I would have rationalized the messy pattern as "confirming a complex boundary."

2. **Non-monotonicity is falsification.** A pattern with reversals doesn't support the hypothesis; it contradicts it.

3. **The failed hypothesis was protective.** Once falsified, I tried three different ways to rescue it (pragmatism, narrowness, negative results). All three paths led back to the same framework. Documenting the falsification is the honest response.

4. **Keystroke analysis has an empirical boundary, not a theoretical one.** What works and what doesn't is determined by signal-to-noise, sample size, and data homogeneity—not by fundamental limits on what behavior can reveal about mind.
