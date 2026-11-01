# Process Attestation Analysis

Exploratory analysis of timing statistics in ScholaWrite revision events and in
the KLiCKe keystroke corpus, with a simulated retype attack and a
leave-one-writer-out classifier. The results are descriptive: they characterize
interval distributions and how well simple timing features separate genuine
sessions from four synthetic retype strategies. They do not validate cognitive
process signatures.

## Quick Start

```bash
pip install -r requirements.txt
python run_analysis.py
```

With uv: `uv run --no-project --with-requirements requirements.txt python run_analysis.py`.

ScholaWrite loads from Hugging Face (`minnesotanlp/scholawrite`, gated; log in
with `huggingface-cli login` first). KLiCKe is optional: place the per-writer
CSV corpus under `klicke/Files/WritingTask/WritingTask/keystrokelogs/csv/`, or
let the script download the Kaggle competition data with a Kaggle API token
(`KAGGLE_USERNAME`/`KAGGLE_KEY` or `~/.kaggle/kaggle.json`; the `opendatasets`
path needs Python < 3.13). Without KLiCKe, Stages 3-6 are skipped and only the
ScholaWrite statistics are produced. The committed results come from the local
CSV corpus (4,992 writers); the Kaggle train split has fewer writers and will
not reproduce those counts.

`--recompute` recomputes the Stage 6 features from raw data instead of reusing
the Stage 3 pre-computation (slower, independently checkable).

## Pipeline

1. **Environment check** -- packages and data availability
2. **ScholaWrite analysis** -- inter-event interval distributions between
   consecutive recorded text changes, Shannon entropy, cognitive-load
   correlation (CLC), interval contrast by high-level category
3. **KLiCKe corpus analysis** -- keystroke-level interval statistics, per-writer
   entropy and CLC, composition-like versus transcription-like windows
4. **Adversarial retype simulation** -- constant-rate, IID, cross-writer, and
   Markov-chain forgeries of each sampled writer's genuine session
5. **ROC classification** -- leave-one-writer-out logistic regression, with and
   without the CLC feature
6. **Effect sizes and figures** -- window-level and writer-level contrasts
7. **Summary report** -- values printed from `analysis_results.json`

## Definitions

- **ScholaWrite intervals** are gaps between consecutive recorded text-change
  events per author and project (mean 3.97 s, median 1.27 s); ScholaWrite
  contains no keystroke-level timing. Cognitive load is taken from the dataset's
  own high-level category: PLANNING = 3, REVISION = 2, IMPLEMENTATION = 1.
- **KLiCKe intervals** are inter-keystroke intervals. Cognitive load is
  Nonproduction = 3, Remove/Cut = 2, Input = 1. Windows of 50 keystrokes with a
  25-keystroke step are labeled composition-like (at least 30% Nonproduction
  plus Remove/Cut) or transcription-like (at least 80% Input). The labels and
  the CLC variable both derive from the activity column, so differences between
  the two window types are partly definitional; windows overlap and are nested
  within writers, so the writer-level paired contrast is the inferential
  statistic.
- **Entropy** is the plug-in Shannon entropy of intervals quantized to 5 ms
  bins; it shifts by one bit for every halving or doubling of the bin width.
- **Retype simulation**: 500 writers are sampled (seed 42), each with one
  genuine 200-event session; the four forgeries per writer reuse the genuine
  session's length and activity sequence, so every feature, including
  `clc_proxy`, is computed identically for both classes. Label 1 = forged.
  AUC, EER, and operating points use the pooled out-of-fold scores of the
  per-writer models.

## Key Results

All values are read from the committed `analysis_results.json` (run of
2026-10-31, seed 42).

| Quantity | Value | JSON key |
|----------|-------|----------|
| ScholaWrite inter-event interval entropy (55,066 intervals, 5 ms bins) | 10.44 bits | `scholawrite.entropy_per_iki_bits` |
| ScholaWrite per-author CLC (Spearman rho; 8 authors) | median 0.062, Wilcoxon p = 0.15 | `scholawrite.clc_per_author_*` |
| KLiCKe corpus | 4,992 writers, 16.9M intervals, 6.89 bits/interval | `klicke.*` |
| KLiCKe CLC, composition-like minus transcription-like windows (Cohen's d) | -1.82 (transcription-like windows have the higher CLC) | `effects.effect_sizes.clc.cohens_d` |
| KLiCKe CLC, writer-level paired contrast | 4,858 writers; mean difference -0.37; paired d -1.64; 93.9% of writers negative | `effects.effect_sizes.clc_writer_level` |
| LOWO ROC AUC, all attacks pooled (500 genuine : 2,000 forged) | 0.860 (0.850 without `clc_proxy`); EER 0.227 | `roc.all_attacks` |
| LOWO ROC AUC per attack | constant 0.998, IID 0.821, cross-writer 0.806, Markov 0.973 | `roc.attack_*` |
| TPR at 1% / 5% / 10% FPR, all attacks pooled | 0.37 / 0.50 / 0.58 | `roc.all_attacks.fpr_operating_points` |
| Accuracy at threshold 0.5, all attacks pooled | 0.843 (majority-class baseline 0.80) | `roc.all_attacks.accuracy_at_0_5` |

The CLC difference between window types runs opposite to the "writers slow down
for demanding operations" reading: within composition-like windows, the
Nonproduction events that carry the highest load value are often very fast
(median 40 ms), which produces negative within-window correlations. The
constant-rate forgery is trivially separable (zero variance); the pooled AUC
therefore overstates performance against the harder IID and cross-writer
forgeries.

## Limitations

- ScholaWrite has 8 authors; the per-author Wilcoxon test is the only
  author-level inference and it is not significant.
- Window-level tests treat overlapping, writer-nested windows as independent;
  use the writer-level contrast.
- The retype forgeries are synthetic transformations of the writers' own
  sessions, not recordings of people retyping text.
- Entropy values depend on the quantization; the per-checkpoint quantity in the
  JSON (50 x per-interval entropy) assumes independent intervals and is an
  upper bound only.

## Changes in the 2026-10-31 revision

- `clc_proxy` was fixed at 0 for every forged session and therefore encoded the
  class label; forgeries now reuse the genuine activity sequence, and every
  classifier result is reported with and without the feature.
- The ScholaWrite cognitive-load map used label names absent from the dataset
  (about 10% of events, including planning labels, silently received the lowest
  load); load now comes from the high-level category and unmapped values raise.
- A contrast that split windows by their own CLC value and then compared CLC
  between the groups was tautological and has been removed.
- Cliff's delta had the opposite sign of Cohen's d; the sliding window loop
  dropped the final full window; operating points reported FRR at a target FAR
  under the wrong name; non-UTF-8 writer CSVs were silently skipped (21 of
  4,992); figure PDFs now omit creation timestamps.
- Effect sizes use one pooled-variance helper; inline formulas were removed.
- `analysis_results.json` is committed so every number above is traceable.

## Author

David Condrey (david@writerslogic.com), Writerslogic, Inc.
