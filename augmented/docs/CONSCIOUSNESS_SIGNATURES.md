# Consciousness Signatures in Writing Traces

## Overview

This module measures four computational signatures *associated with* conscious,
embodied cognitive processing in writing traces. These signatures are structurally
absent in feedforward language model generation, making them useful discriminators
for human vs. machine authorship.

**Important caveat:** These measurements do not prove consciousness exists,
detect consciousness directly, or solve the hard problem of consciousness. They
measure statistical properties of writing traces that are *consistent with*
theories of conscious processing and *inconsistent with* known LLM architectures.

---

## Writing as a Dissipative Process

Human writing is thermodynamically irreversible. A writer's cognitive resources
(glucose, attention, working memory) deplete monotonically over a session. This
irreversibility produces measurable entropy: the forward-time statistics of the
writing trace differ from its reverse-time statistics.

We measure this via the **entropy production rate** (sigma), computed as the
KL divergence between forward and reverse transition probability matrices across
three channels: glucose trajectory, keystroke latency, and syntactic complexity.

- **Human traces:** sigma > 0 (dissipative, irreversible)
- **LLM traces:** sigma ~ 0 (time-symmetric, reversible)

This framing draws on Crooks' fluctuation theorem (Crooks, 1999), which
establishes that entropy production equals the log-ratio of forward to reverse
path probabilities. In our discrete approximation, we compute this ratio over
binned state transitions.

### Reference

Crooks, G. E. (1999). "Entropy production fluctuation theorem and the
nonequilibrium work relation for free energy differences." *Physical Review E*,
60(3), 2721-2726.

---

## Integrated Information in Causal Traces

Tononi's Integrated Information Theory (IIT) proposes that consciousness
corresponds to integrated information (Phi) -- the degree to which a system's
parts are informationally coupled beyond what independent parts would produce.

We adapt this to writing traces by treating the four signal channels (glucose,
latency, complexity, failure) as a 4-variable system. We compute Phi as the
minimum mutual information across all non-trivial bipartitions of these channels,
normalized by joint entropy.

- **Human traces:** High Phi (channels are causally coupled through the embodied
  cognitive state -- glucose depletion affects latency, which affects complexity,
  which affects failure rate)
- **LLM traces:** Low Phi (channels are statistically independent or trivially
  correlated)

### Limitations of the IIT Analogy

The full IIT formalism operates over the intrinsic cause-effect structure of a
system's state space, which is computationally intractable for most systems. Our
approximation uses mutual information as a proxy for integrated information and
operates over observed time series rather than the system's causal architecture.
This is a pragmatic simplification, not a faithful implementation of IIT.

### Reference

Tononi, G. (2008). "Consciousness as Integrated Information: A Provisional
Manifesto." *Biological Bulletin*, 215(3), 216-242.

---

## Temporal Binding and the Unity of Authorial Experience

Conscious experience exhibits temporal binding: information from past events
persists and influences future processing over long time scales. In writing, this
manifests as long-range correlations -- a failure at token t affects complexity at
token t+20 because the embodied state carries forward.

We measure the **Temporal Binding Index (TBI)** as the integral of mutual
information across time lags, normalized by MI at lag 1:

    TBI = sum_{k=1}^{K} MI(X_t, X_{t+k}) / MI(X_t, X_{t+1})

where X is the multivariate state vector (glucose, latency, complexity, failure).

High TBI indicates fat-tailed (power-law) MI decay -- information persists across
long time scales, consistent with a persistent cognitive state. Low TBI indicates
thin-tailed (exponential) decay -- no long-range memory, consistent with
Markovian generation.

We also fit power-law and exponential decay models to the MI curve and report
which fits better, along with the decay exponent.

### Connection to Consciousness Science

The temporal binding window is a well-studied phenomenon in consciousness
research. Wengelin (2006) demonstrated that pause patterns in writing reflect
cognitive planning that spans multiple sentences, with pause durations at
paragraph boundaries being 3-5x longer than within-sentence pauses. This
hierarchical temporal structure is a signature of conscious planning that is
absent in token-by-token generation.

### Reference

Wengelin, A. (2006). "Examining Pauses in Writing: Theory, Methods and Empirical
Data." In K. Sullivan & E. Lindgren (Eds.), *Computer Keystroke Logging and
Writing*, 107-130. Elsevier.

---

## Active Inference in Writing

Under Friston's Free Energy Principle, biological systems minimize variational
free energy -- a bound on surprise -- through active inference. A human writer
maintains an implicit generative model of their own text and iteratively reduces
prediction error.

This produces a characteristic 3-phase trajectory:

1. **Exploration** (high free energy): The writer is sampling possibilities,
   generating high prediction error as they explore the problem space.
2. **Exploitation** (decreasing free energy): The writer has found a structural
   plan and is executing it, with decreasing prediction error.
3. **Consolidation** (low free energy): The writer is polishing, with minimal
   prediction error and stable output.

We compute per-event prediction error (combining failure surprise and glucose
change rate), smooth it, and fit a 3-phase piecewise linear model. The
**trajectory score** measures how well the observed free energy profile matches
this 3-phase template relative to a flat (no-structure) model.

- **Human traces:** High trajectory score (clear 3-phase structure)
- **LLM traces:** Low trajectory score (flat or random free energy profile)

### Reference

Friston, K. (2010). "The free-energy principle: a unified brain theory?"
*Nature Reviews Neuroscience*, 11(2), 127-138.

---

## The Phenomenological Gap

Beyond the four primary metrics, we measure the **phenomenological gap**: the
divergence between objective task difficulty (syntactic complexity) and subjective
difficulty (actual failure rate).

If failures were purely determined by task demands, regressing failure rate on
syntactic complexity would explain all variance. The residual variance captures
cases where the writer struggles with "easy" words (due to fatigue, distraction,
or emotional state) or flows through "hard" ones (due to domain expertise or
heightened attention). This residual is the phenomenological gap.

- **Human traces:** High phenomenological gap (subjective experience diverges
  from objective difficulty)
- **LLM traces:** Low phenomenological gap (failure rate, if any, correlates
  mechanistically with complexity)

### Reference

Chalmers, D. J. (1995). "Facing Up to the Problem of Consciousness." *Journal
of Consciousness Studies*, 2(3), 200-219.

---

## Composite Score

The four primary metrics are combined into a single **composite consciousness
score** using equal weights:

| Metric                   | Weight | Normalization Ceiling |
|--------------------------|--------|-----------------------|
| Entropy Production       | 0.25   | 1.0                   |
| Integrated Information   | 0.25   | 1.0                   |
| Temporal Binding Index   | 0.25   | 10.0                  |
| Free Energy Score        | 0.25   | 1.0                   |

Each metric is clamped to [0, 1] by dividing by its normalization ceiling before
weighting. The composite score therefore also lies in [0, 1].

A trace is classified as **human-like** if composite > 0.4.

The phenomenological gap is reported separately as supplementary evidence but
does not enter the composite score, since it measures a different construct
(subjective-objective divergence rather than a direct consciousness correlate).

---

## Implications for the Hard Problem

These measurements operate entirely at the computational level. They detect
statistical signatures that are *associated with* theories of conscious
processing (IIT, Free Energy Principle, temporal binding) but they do not:

- Detect phenomenal consciousness (qualia, subjective experience)
- Prove that any system is or is not conscious
- Resolve the hard problem of consciousness (Chalmers, 1995)
- Distinguish between consciousness and sophisticated unconscious processing

What they *do* show is that human writing traces have structural properties
predicted by consciousness-related theories, and that current LLM architectures
do not produce these properties. This is a useful forensic signal, but it should
not be confused with a consciousness detector.

A sufficiently sophisticated LLM that accurately simulated embodied cognition
(including metabolic depletion, fatigue dynamics, and active inference) could in
principle produce traces that score highly on these metrics. The metrics measure
the *computational signature*, not the *phenomenal reality*.

---

## Limitations

1. **Proxy measures, not direct detection.** All four metrics are proxies for
   theoretical constructs. Entropy production is a proxy for thermodynamic
   irreversibility of cognition. Phi is a simplified approximation to IIT. TBI
   is a statistical summary of temporal correlations. Free energy score is a
   curve-fitting exercise. None of these is a direct measurement of the
   theoretical quantity it approximates.

2. **Discretization artifacts.** The entropy and information-theoretic
   computations use histogram-based estimators with fixed bin counts. Results
   are sensitive to bin width, especially for short traces.

3. **Small-sample instability.** All metrics require sufficient trace length to
   produce stable estimates. We require a minimum of 3 events for any
   computation and recommend at least 15-20 for meaningful results. The
   consciousness score in `compute_causal_signatures` only activates for traces
   longer than 15 events.

4. **Equal weighting is arbitrary.** The 0.25/0.25/0.25/0.25 weighting of
   the composite score has no theoretical justification. It treats all four
   metrics as equally informative, which may not reflect their actual
   discriminative power for any particular population of writers and models.

5. **No adversarial robustness guarantee.** An adversary who understands these
   metrics could construct LLM traces that score highly by simulating the
   expected statistical signatures. The metrics are useful for distinguishing
   *naive* LLM generation from human writing, not for detecting sophisticated
   adversarial forgery.

6. **Cultural and individual variation.** Writing processes vary substantially
   across individuals, languages, disciplines, and writing conditions. The
   normalization ceilings and thresholds were calibrated on English academic
   writing and may not generalize.

---

## References

- Chalmers, D. J. (1995). "Facing Up to the Problem of Consciousness." *Journal
  of Consciousness Studies*, 2(3), 200-219.
- Crooks, G. E. (1999). "Entropy production fluctuation theorem and the
  nonequilibrium work relation for free energy differences." *Physical Review E*,
  60(3), 2721-2726.
- Friston, K. (2010). "The free-energy principle: a unified brain theory?"
  *Nature Reviews Neuroscience*, 11(2), 127-138.
- Tononi, G. (2008). "Consciousness as Integrated Information: A Provisional
  Manifesto." *Biological Bulletin*, 215(3), 216-242.
- Wengelin, A. (2006). "Examining Pauses in Writing: Theory, Methods and
  Empirical Data." In K. Sullivan & E. Lindgren (Eds.), *Computer Keystroke
  Logging and Writing*, 107-130. Elsevier.
