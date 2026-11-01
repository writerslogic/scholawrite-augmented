# Cognitive Calibration: Parameter-to-Research Mapping

This document records the published findings that motivated each simulation parameter. The parameters were chosen by the author; they were not fitted to these studies, and the studies do not report the simulator's quantities.

## Methodology

The approach:

1. Identify the behavioral phenomenon a parameter is meant to evoke.
2. Cite the study that motivated it.
3. Report the simulator's own output for that parameter.

No step fits parameters to published measurements.

Parameters that are not directly measurable (e.g., internal thresholds) are derived from combinations of measurable outcomes.

## Parameter Mappings

### 1. Keystroke Latency

**Simulation formula:** `calculate_latency() = 115 + 90 * log(1 + depth) * (1.12 - glucose)`

| Condition | Simulation Output | Empirical Range | Source |
|-----------|------------------|-----------------|--------|
| Baseline (glucose=1.0, depth=1) | 122.5 ms (deterministic term; log-normal noise added at runtime) | 150-280 ms | Wengelin (2006) |
| Cognitive load (glucose=0.3, depth=5) | 247.2 ms | 400-800 ms | Wengelin (2006) |

**Citation:** Wengelin, A. (2006). Examining pauses in writing. In K. Sullivan & E. Lindgren (Eds.), *Computer keystroke logging and writing* (pp. 107-130). Elsevier.

### 2. Fatigue Onset and Glucose Depletion

**Parameters:** `glucose_depletion_rate = 0.9992`, `fatigue_divisor = 12000`

With the implemented update g' = g·0.9992^(n(1+d/10)), g crosses 0.65 after about 489 tokens at demand 1 (538 at demand 0), before intention-specific execution costs. No conversion to minutes is made.

| Metric | Simulation | Empirical | Source |
|--------|-----------|-----------|--------|
| Glucose at onset | 0.65 | N/A (modeled) | Derived |

**Citation:** Kellogg, R. T. (1987). Effects of topic knowledge on the allocation of processing time and cognitive effort to writing processes. *Memory & Cognition*, 15(3), 256-266.

### 3. Pause-Burst Ratio

**Emergent property of:** failure rate in execution traces

Writers spend 60-70% of their time pausing and 30-40% producing text. The failure modes in our causal engine (lexical starvation, syntactic collapse) generate pause-equivalent events at rates consistent with this ratio.

| Metric | Simulation | Empirical | Source |
|--------|-----------|-----------|--------|
| Pause fraction | ~0.60-0.65 | 0.60-0.70 | Barkaoui (2019) |

**Citation:** Barkaoui, K. (2019). What can L2 writers' pausing behavior tell us about their L2 writing processes? *Studies in Second Language Acquisition*, 41(3), 529-554.

### 4. Revision Frequency

**Maps to:** `repair_distance` distribution in execution traces

| Metric | Simulation | Empirical | Source |
|--------|-----------|-----------|--------|
| Revisions per 100 words | ~3 | 2-4 | Leijten & Van Waes (2013) |
| Repair cost multiplier | 2.7x | 2-3.5x | Leijten & Van Waes (2013) |

**Citation:** Leijten, M., & Van Waes, L. (2013). Keystroke logging in writing research: Using Inputlog to analyze and visualize writing processes. *Written Communication*, 30(3), 358-392.

### 5. Syntactic Complexity Decline

**Parameters:** `syntactic_collapse_base = 4.0`, `syntactic_collapse_glucose_factor = 3.5`

The syntactic collapse threshold = base + glucose * factor. As glucose depletes from 1.0 to 0.5, the threshold drops from 7.5 to 5.75, a 23% decline.

| Metric | Simulation | Empirical | Source |
|--------|-----------|-----------|--------|
| Complexity decline over 90 min | ~23% | 15-25% | Chenoweth & Hayes (2001) |

**Citation:** Chenoweth, N. A., & Hayes, J. R. (2001). Fluency in writing: Generating text in L1 and L2. *Written Communication*, 18(1), 80-98.

### 6. Type-Token Ratio (TTR) Fatigue Effect

**Parameter:** `lexical_fatigue_penalty = 0.4`

Lexical retrieval = glucose * (1 - visual_fatigue * penalty). At mid-session (visual_fatigue=0.25), lexical retrieval is reduced by 10%, matching the 8-12% TTR decline observed in extended writing sessions.

| Metric | Simulation | Empirical | Source |
|--------|-----------|-----------|--------|
| TTR decline over session | ~10% | 8-12% | Crossley & McNamara (2014) |

**Citation:** Crossley, S. A., & McNamara, D. S. (2014). Does writing development equal writing quality? A computational investigation of syntactic complexity and writing quality. *Journal of Second Language Writing*, 24, 5-22.

## Timing model (calibrated to KLiCKe, 2026-10-31)

**Scope.** KLiCKe (Tian, Crossley, and Van Waes, 2025) is a corpus of about 5,000 argumentative texts by adult writers in the United States, L1 and L2, collected with a web-based keystroke logger. It is not scholarly LaTeX. The model below is calibrated to KLiCKe keystroke timing and says nothing about other writing tasks, editors or populations.

**Selection.** `SimulationConfig.timing_model` (also in `configs/simulation_config.json`):

| value | behaviour |
|-------|-----------|
| `lognormal_iid` (default) | The formula in section 1 with log-normal noise, bit-for-bit as before, including the scholar's seeded RNG stream. Pinned by golden values (22 authors x 10 depths, with glucose depleted between calls) in `tests/test_embodied.py`. |
| `klicke_ar1` | The model below. Uses its own `random.Random`, seeded from sha256(`author_id` + `":timing:"` + `timing_seed`). It never draws from the legacy RNG. |

The default was not changed. Switching it is a separate decision.

**Model (`klicke_ar1`).** Each `EmbodiedScholar` draws one TRAIN-writer row (`mu, sigma, rho, p_pause, pause_mu, pause_sigma`) from `configs/klicke_writer_params_train.json`, with probability proportional to the writer's number of 50-IKI windows. Per event, with depth d and current glucose g:

- With probability `p_pause` the event is a pause: exp(N(`pause_mu`, `pause_sigma`)) truncated to [2000, 60000) ms. It is not modulated (`timing_pause_modulated=False`) and the AR state does not advance.
- Otherwise log(base) = `mu` + `timing_offset` + s_ar z + s_wn e. Here z is a unit-variance AR(1) with coefficient clip(`timing_rho_scale` rho, 0, 0.98) and a stationary start, e is i.i.d. N(0,1), s_ar^2 + s_wn^2 = (`timing_sigma_scale` sigma)^2, and s_ar^2 / (s_ar^2 + s_wn^2) = `timing_ar_share`. The latency is base x D(d, g) / D(1, 1) with D(d, g) = 115 + 90 ln(1 + d)(1.12 - g) (`EmbodiedScholar.deterministic_latency`), so depletion and demand still lengthen it.
- The result is floored at 50 ms and rounded to 2 decimals.

**Data and split.** KLiCKe WritingTask logs: 4,992 writers, all with at least 200 valid IKIs (10 < IKI < 60,000 ms). Sorted by id, the 2,496 even-indexed writers are TRAIN (167,115 non-overlapping 50-IKI windows) and the 2,496 odd-indexed writers are held-out. The held-out comparison uses 1,500 held-out writers drawn with seed 0 (102,147 windows), 20,000 of which are sampled for the KS statistic. The simulator side is 1,000 reference traces of 200 events (seeds 10000+i) through `_generate_authentic_trace`, 4,000 windows. The writer table is built from TRAIN writers only (`analysis/calibration/build_writer_params.py`): mu, sigma and rho come from IKIs <= 2,000 ms with pauses removed, so pauses are not counted twice. The file holds no writer ids and nothing derived from held-out writers.

**Fit (TRAIN only).** `analysis/calibration/fit_timing.py` calibrates end to end through `_generate_authentic_trace` against the 50-IKI window statistics of TRAIN writers (the estimators and filters of `measure_gap.py`), not against the whole-sequence writer table. Stage 1 is Nelder-Mead on the squared differences of window median and IQR of `mean_iki_ms`, `sd_log_iki`, `lag1_autocorr_log` and `pause_fraction`. Stage 2 is Nelder-Mead on the summed squared TRAIN KS D of the same four metrics. Fit seeds are 500000+i and 600000+i, not the reference seeds. The run takes about 3 minutes and is deterministic (a second run gave identical values). Output: `analysis/calibration/fit_timing_result.json`.

| parameter | fitted |
|-----------|--------|
| `timing_offset` | -0.8489 |
| `timing_rho_scale` | 1.9102 |
| `timing_sigma_scale` | 0.9622 |
| `timing_ar_share` | 0.5862 |

`timing_rho_scale` and `timing_ar_share` are only weakly separated by 50-IKI statistics (the stage 1 and stage 2 optima differ along that ridge).

**Where this departs from a literal reading of the specification** (each justified on TRAIN data):

1. The writer bootstrap is weighted by 50-IKI window count. On TRAIN, the window-pooled and writer-uniform populations differ by KS D 0.080 on `mean_iki_ms` and 0.050 on `pause_fraction` (0.023 `sd_log_iki`, 0.012 `lag1_autocorr_log`), because fast typists contribute more windows.
2. Pause durations are drawn from a doubly truncated normal on log IKI. An untruncated exp(N(observed log-mean, log-sd)) puts about 12% of a typical writer's pause draws below the 2,000 ms threshold. `pause_mu` and `pause_sigma` are therefore maximum-likelihood parameters of the truncated law (the observed log-mean and log-sd are stored alongside). 1,110 of 2,456 per-writer fits and the pooled fit reached a parameter bound (an exponential-like tail on the log scale); 40 writers with fewer than 20 pauses use the pooled law. Against pooled TRAIN pauses the sampler gives KS D 0.012 on log durations.
3. Stage 2 (KS polish) was added because stage 1 alone is fragile for the rule's own statistic: on four simulation samples (three independent of the reference seeds) its TRAIN KS D on `lag1_autocorr_log` was 0.105 to 0.127, above 0.10 each time. The fitted parameters give at most 0.094 on every sample. The `pause_fraction` term of the stage 1 loss uses its mean because its quartiles are coarse.
4. No drift term (`timing_drift_sd`, `timing_drift_rho`) was added: the four-parameter model satisfies the TRAIN gate on every simulation sample, so the condition for adding it was not met.

TRAIN KS D on the reference seeds was observed during development; the choice of stage 2 rests on the non-reference samples. The summary statistics of the held-out windows were visible in earlier gap files, and no model choice used them.

**Results on TRAIN** (`measure_gap.py --split train`; `*` marks the four decision metrics):

| metric | KLiCKe TRAIN median [IQR] | legacy simulator median [IQR], KS D | new simulator median [IQR], KS D |
|---|---|---|---|
| mean_iki_ms * | 281.5 [191.3, 458.2] | 260.4 [233, 291], 0.332 | 286.1 [185.5, 480], 0.029 |
| median_iki_ms | 149 [119, 190] | 173.3 [153.9, 194.6], 0.315 | 141.4 [107.6, 184], 0.087 |
| sd_log_iki * | 0.8056 [0.6526, 0.9912] | 0.8435 [0.7923, 0.8967], 0.312 | 0.8239 [0.6765, 0.974], 0.048 |
| lag1_autocorr_log * | 0.1108 [-0.03665, 0.311] | -0.01311 [-0.1069, 0.08853], 0.301 | 0.125 [0.003467, 0.2581], 0.095 |
| pause_fraction * | 0.02 [0, 0.04] | 0 [0, 0], 0.422 | 0.02 [0, 0.04], 0.091 |
| burst_count | 4 [3, 5] | 5 [4, 6], 0.167 | 4 [3, 5], 0.040 |
| entropy_bits_5ms | 4.806 [4.551, 4.999] | 4.981 [4.846, 5.094], 0.306 | 4.839 [4.463, 5.079], 0.080 |

**Pre-registered decision rule.** The new model passes iff KS D <= 0.10 on `mean_iki_ms`, `sd_log_iki`, `lag1_autocorr_log` and `pause_fraction` simultaneously, between held-out KLiCKe windows and the 1,000 reference traces. The other three metrics are reported and do not decide. Held-out looks used: 1 of at most 3, for the finalized candidate `klicke_ar1_4param_stage2_v1` (logged with timestamp and config values in `heldout_runs` of `analysis/calibration/gap_calibrated_simulator.json`; the script refuses a fourth entry).

**Results on held-out writers:**

| metric | KLiCKe held-out median [IQR] | legacy simulator median [IQR], KS D | new simulator median [IQR], KS D |
|---|---|---|---|
| mean_iki_ms * | 279.9 [188.3, 457] | 260.4 [233, 291], 0.331 | 286.1 [185.5, 480], 0.027 |
| median_iki_ms | 149 [117, 190.5] | 173.3 [153.9, 194.6], 0.325 | 141.4 [107.6, 184], 0.070 |
| sd_log_iki * | 0.8046 [0.6539, 0.9897] | 0.8435 [0.7923, 0.8967], 0.312 | 0.8239 [0.6765, 0.974], 0.049 |
| lag1_autocorr_log * | 0.1063 [-0.03899, 0.308] | -0.01311 [-0.1069, 0.08853], 0.296 | 0.125 [0.003467, 0.2581], 0.096 |
| pause_fraction * | 0.02 [0, 0.04] | 0 [0, 0], 0.423 | 0.02 [0, 0.04], 0.090 |
| burst_count | 4 [3, 5] | 5 [4, 6], 0.180 | 4 [3, 5], 0.032 |
| entropy_bits_5ms | 4.806 [4.541, 5.014] | 4.981 [4.846, 5.094], 0.312 | 4.839 [4.463, 5.079], 0.076 |

Outcome: the rule passed. The largest decision-metric D is 0.096 (`lag1_autocorr_log`); `pause_fraction` is 0.090, `sd_log_iki` 0.049 and `mean_iki_ms` 0.027. The margins to the 0.10 threshold are 0.004 and 0.010 on the two closest metrics. On TRAIN, across four simulation samples, `pause_fraction` D ranged 0.078 to 0.092 and `lag1_autocorr_log` D 0.089 to 0.094, so the pass is narrow. The model was validated on held-out KLiCKe writers by this rule, for 50-IKI window statistics of 200-event traces. The legacy column was measured before the change with the same script, seeds and held-out subsample (`gap_main_simulator.json`, kept outside the repository); its simulator side was reproduced bit-for-bit in this tree, and its KLiCKe side is identical to the one in `gap_calibrated_simulator.json`.

**What this does not establish.**

- Only 50-IKI window statistics were calibrated. Autocorrelation beyond a window is not constrained, and with `timing_rho_scale` 1.91 the AR coefficient is clipped at 0.98 for 16% of writer rows (18% window-weighted).
- Pause clustering is not modelled. Pauses are independent given the writer, so the share of pause-free windows is 0.333 simulated against 0.409 in KLiCKe. This and `lag1_autocorr_log` are the metrics closest to the limit.
- The offset absorbs the average glucose-floor modulation (about 2.2x), so a fresh scholar is fast relative to KLiCKe: the median over events 0-4 (76 ms) is below the 25th percentile of KLiCKe 50-IKI window medians (119 ms on TRAIN, 117 ms on held-out). By event index the median latency is 76 ms (events 0-4), 95 (5-9), 110 (10-14), 134 (15-20), 147 (21-49) and 146 (50-199); 16.7% of the first 30 events sit at the 50 ms floor. The rule uses 200-event traces. The `n_events` default is 30 in `_generate_authentic_trace` and in most `scripts/eval/*.py` (50 in `run_external_validation.py` and `validation.py`); in traces that short much of the trace is in the faster regime, and this was not tested.
- The comparison is distributional at the window level, against held-out writers of the same corpus (same task, same editor). It does not show that traces are indistinguishable by detectors that use other properties.

**Reproduce** (from `augmented/`):

```
uv run --extra dev python analysis/calibration/build_writer_params.py
uv run --extra dev python analysis/calibration/fit_timing.py
uv run --extra dev python analysis/calibration/measure_gap.py --split train --sim-override timing_model=klicke_ar1 --out /tmp/gap_train.json --writer-params-out /tmp/wp.json
```

A held-out run is a "look": `measure_gap.py --split heldout --candidate-id ID --run-timestamp "$(date -u +%Y-%m-%dT%H:%M:%SZ)"` appends it to `heldout_runs` of `--out`.

## Limitations and Approximations

1. **Simplified metabolic model.** Real cognitive resource depletion is not a single exponential decay. Glucose is a proxy for a complex interplay of attention, working memory, and executive function. The model does not capture micro-recoveries (e.g., brief pauses that partially restore resources).

2. **Individual differences.** The calibration targets population means. Individual writers show substantial variation in fatigue onset (30-90 min), typing speed, and revision behavior. The empirical ranges capture some but not all of this variance.

3. **Task dependency.** The cited studies used specific writing tasks (essays, L1/L2 comparisons). Academic paper writing may differ in cognitive demands, particularly for highly specialized content requiring domain expertise.

4. **Linear session assumption.** The model assumes a single continuous writing session. Real academic writing involves multiple sessions with intervening rest, which resets some but not all cognitive resources.

5. **Pause vs. failure mapping.** Barkaoui's pause-burst ratio measures temporal pauses, while our simulation models cognitive failures. These are related but not identical constructs; not all pauses indicate cognitive failure, and some failures may not manifest as measurable pauses.

## References

- Barkaoui, K. (2019). What can L2 writers' pausing behavior tell us about their L2 writing processes? *Studies in Second Language Acquisition*, 41(3), 529-554.
- Chenoweth, N. A., & Hayes, J. R. (2001). Fluency in writing: Generating text in L1 and L2. *Written Communication*, 18(1), 80-98.
- Crossley, S. A., & McNamara, D. S. (2014). Does writing development equal writing quality? A computational investigation of syntactic complexity and writing quality. *Journal of Second Language Writing*, 24, 5-22.
- Kellogg, R. T. (1987). Effects of topic knowledge on the allocation of processing time and cognitive effort to writing processes. *Memory & Cognition*, 15(3), 256-266.
- Leijten, M., & Van Waes, L. (2013). Keystroke logging in writing research: Using Inputlog to analyze and visualize writing processes. *Written Communication*, 30(3), 358-392.
- Wengelin, A. (2006). Examining pauses in writing. In K. Sullivan & E. Lindgren (Eds.), *Computer keystroke logging and writing* (pp. 107-130). Elsevier.
