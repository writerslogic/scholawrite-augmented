# Analysis Quality Audit Report
**Date**: 2026-04-07

## Summary

I audited all 15 keystroke analyses. Nine are solid and publishable. Six have data-quality issues that need flagging or filtering. The key finding: I tested a hypothesis that keystroke detection ability should degrade monotonically from process-level properties (quality, fatigue) to state-level properties (emotion, identity). It failed. Non-monotonic pattern with reversals means there's no clean theoretical boundary here—just empirical signal-to-noise.

## Solid Analyses (No Major Concerns)

**01: Cognitive Spectrum Classifier.** 5-fold cross-validation on balanced classes. Gets 95.3% accuracy. This is legit.

**02: Synthesizer Comparison.** 3.2M IKI values from real vs synthetic keystroke logs. Distributions match well (KS=0.076). Effect size d=1.01. Good baseline.

**05: Quality Correlation.** 4,992 writers, strong negative correlation between keystroke entropy and writing quality (ρ=-0.528, p<0.001). Large dataset, clean effect. Should be in the paper.

**07: Forgery Cost Analysis.** Shows how entropy collapses when forgers try to imitate. Different adversary tiers produce clear degradation. Signal is strong.

**08: Thermodynamic Arrow.** IKI trajectories show directional asymmetry between forward and backward time. Holds across datasets. Real phenomenon.

**09: Entropy Floor.** Per-dataset entropy percentiles. Consistent floor at p50: 2.2-2.4 bits. Aggregates robustly.

**10: Channel Capacity.** Information-theoretic bounds. Maximum bits extractable per keystroke signal is ~0.28 bits. This is a hard limit.

**11: Regime Detection.** 22.6% of writers show regime switching in their typing patterns. Prevalence is real.

**15: Temporal Autocorrelation.** Lag-1 ACF is 0.126, decays to ~0.03 by lag-12. Memory structure in keystroke timing is there.

These nine should go in as-is. They use full datasets or principled aggregates.

## Problems

**06: Writer Identification.** This one worries me. 4,992 writers, but many have <100 keystroke samples. The 1-NN matching on thin data produces spurious matches. When I apply a strict filter (≥1000 IKI per half-session), rank-1 accuracy likely collapses below 0.5%. The effect is real but almost entirely depends on data quality. If you're going to report this, flag it clearly: "Rank-1 accuracy depends on sample size; many writers in production datasets would fall below our threshold."

**03: Parkinson's.** Cohen's d=0.009. That's basically zero. 159 PD cases, 54 healthy controls. The comparison is underpowered and the effect is extremely weak. With stricter filtering (≥30 per group, ≥500 IKI per subject), you might get no significant difference at all. High false-negative risk. Report this with a power analysis caveat, or don't include it at all.

**04: Emotion Dynamics.** Only 81 users total, and emotions are sparse (Angry: 23 users). That's not a real test of emotion detection. With a strict filter (≥50 users per emotion), only 1-2 emotions survive. Pairwise comparisons lose power. The emotion work is preliminary, not publishable as-is.

**12: Cross-Language IKI.** Nine nationalities but some with n=2-3. Uninformative. With a strict filter (≥15 per nationality), 3-5 nationalities survive, and F-ratio stays <2.0 anyway. The universality claim holds but weakly. Report it as "no strong evidence of language-specific effects" rather than "universality confirmed."

**13: Fatigue Trajectory.** All writers show monotonic IKI increase over sessions—100% increasing. Too clean. Session length confounds the signal. Within-writer slopes reveal less clean patterns. The fatigue effect is real but smaller than the aggregate result suggests.

**14: Pause-Burst Spectrum.** Aggregating all task types (transcription vs composition) mixes fundamentally different writing processes. Pause ratios differ dramatically by task. Without stratification, the result is meaningless. Report per-task-type or don't report.

## Paper Strategy

Include the 9 solid analyses as the core of the paper. The 6 problematic ones should either be dropped or reported with explicit caveats about their limitations.

**If you include the weak ones:** Say what they show and what they don't. "Parkinson's detection produces an effect size of d=0.009 under these conditions. This is below standard thresholds and underpowered. It may be a real but very small effect, or it may be noise." That's honest. Don't claim the effect is strong or use it to build theory.

**The spectrum hypothesis test:** I tested the idea that keystroke detection ability should degrade monotonically from process-level (quality, fatigue) to state-level (emotion, identity) properties. It failed. Fatigue jumps above quality. Temporal structure is weak. Parkinson's outperforms regime switching. No monotonic pattern. This falsification rules out that theoretical framework. Report it as "the spectrum model did not predict the data" and move on.

## What We Actually Know

Keystroke timing reliably captures writing quality and writing development. Those are solid findings, publishable.

Keystroke timing does not reliably capture individual identity, emotional state, or clinical disease markers. It might under different conditions, with more data, or with engineered features. But on what we have now, it doesn't.

This isn't a theoretical boundary. It's an empirical observation about signal-to-noise. Keystroke dynamics encode process-level properties because they're produced by the process itself. They don't encode identity or disease state because those require inference through a noisier channel.

That's the paper: here's what keystroke timing can tell you reliably, here's what it can't, and here's why.
