---
license: apache-2.0
task_categories:
  - text-classification
  - text-generation
language:
  - en
tags:
  - writing-process
  - revision-tracking
  - process-integrity
  - scholarly-writing
size_categories:
  - 100K<n<1M
source_datasets:
  - minnesotanlp/scholawrite
---

# ScholaWrite-Augmented

**Process integrity verification for revision-tracked scholarly writing through causal simulation and adversarial evaluation.**

*Augmented content by [David Condrey](https://github.com/writerslogic).*

The original [ScholaWrite](https://github.com/minnesotanlp/scholawrite) dataset captures scholarly writing revisions and cognitive-intention annotations. ScholaWrite-Augmented adds synthetic insertion tools, insertion-specific annotations, and adversarial process evaluation. It injects synthetic text at three sophistication levels into the revision stream, tracks boundary erosion across revisions, and provides span-level annotations with explicit ambiguity flags — creating a benchmark for *process integrity* rather than origin classification.

The augmentation is grounded in a cognitive simulation engine (`EmbodiedScholar`) that models metabolic depletion, fatigue, and syntactic demand to produce synthetic writing traces. Its parameters draw on writing-process research and timing summaries; physiological validation remains incomplete. A four-tier adversarial evaluation framework then stress-tests detection signals against adversaries of increasing knowledge, from naive to expert. The framework supports experiments on the limits of process-based detection under different assumptions about adversarial access to the observed trace.

**This dataset does NOT label origin (human vs AI).** It studies whether the observable revision process is consistent with iterative human authorship, or whether it exhibits evidence of process-inconsistent external insertion.

- **Manuscript:** *ScholaWrite-Augmented: Revision-Level Benchmarks for Hybrid Human-AI Writing Detection through Causal Process Simulation* (David Condrey, 2026). Unpublished manuscript; no venue or DOI.
- **Release:** 2026-01-29 (snapshot `d473c26d`), not rebuilt since; framework code expanded in April 2026 is not reflected in this snapshot.

---

## Core Finding

Expert forgeries can reverse the ranking of the configured process score: the archived detector comparison below reports composite AUC 0.2911 at 30 traces per class. This is a result for a fixed score direction, not a proof that detection is impossible. In the separate 50-trace, 40-event ablation, the equal-weight composite has AUC 0.3544 while standalone free energy reaches 0.7900. These synthetic-trace experiments do not establish performance on genuine human composition.

**Signal x Tier AUC matrix (n=30, events=40):**

| Detector | Naive | Statistical | RevEng | Expert |
|---|---|---|---|---|
| NCD | 0.954 | 0.989 | 0.633 | 0.971 |
| CausalCoupling | 0.368 | 0.383 | 0.361 | 0.431 |
| ConsciousnessSignatures | 0.969 | 0.893 | 0.997 | **0.291** |
| GPTZero | 0.994 | 0.909 | 0.972 | 0.969 |
| Originality.ai | 0.437 | 0.498 | 0.397 | 0.686 |

In this synthetic-trace comparison, GPTZero ranks reference and expert-forged outputs with AUC 0.9689. A separate archived four-seed real-text pilot yields AUC 0.575 for ScholaWrite-finetuned LLaMA-8B outputs. The experiments use different inputs and must not be pooled or treated as equivalent.

---

## Dataset Summary

| Statistic | Value |
|-----------|-------|
| Documents | 5 |
| Total Revisions | 126,246 |
| Injections | 237 |
| Trajectory States | COLD (50%), WARM (33%), ASSIMILATED (16%) |

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Injection Level** | Sophistication of insertion: naive (no context), topical (domain only), contextual (full local context) |
| **Trajectory State** | Boundary erosion over revisions: COLD (unedited), WARM (partial integration), ASSIMILATED (full integration) |
| **Ambiguity Flag** | Confidence in boundary determination: NONE, LOW, MEDIUM, HIGH |
| **Causal Signatures** | Process metrics: repair locality, resource coupling, biometric plausibility |

---

## Quick Start

```bash
git clone https://github.com/writerslogic/scholawrite-augmented.git
cd scholawrite-augmented/augmented
uv sync --extra dev
uv run python scripts/pipeline/build_augmented_dataset.py --help
```

### Loading the Dataset

```python
from datasets import load_dataset

dataset = load_dataset("Writerslogic/scholawrite-augmented")
for doc in dataset["train"]:
    print(doc["doc_id"], doc["text"][:100])
```

Or directly from files:

```python
import json

with open("data/augmented/documents.jsonl") as f:
    documents = [json.loads(line) for line in f]

with open("data/augmented/annotations.jsonl") as f:
    annotations = [json.loads(line) for line in f]
```

### Library Usage

```python
from scholawrite.io import load_seed, read_augmented_jsonl
from scholawrite.schema import AugmentedDocument
from scholawrite.annotations import validate_annotations
from scholawrite.metrics import span_iou, f1
from scholawrite import detect

# Simple detection
print(detect("your text here"))

# Load and validate
seed_docs = load_seed("data/seed/raw/hf_scholawrite/")  # parquet snapshot directory
docs = read_augmented_jsonl("data/augmented/documents.jsonl")  # list[AugmentedDocument]
result = validate_annotations(docs)
print(result.is_valid, len(result.errors))
```

### Reproduce Experiments

```bash
bash scripts/run_all_experiments.sh        # full reproduction pipeline
uv run python scripts/eval/run_adversarial_eval.py
uv run python scripts/eval/run_detector_comparison.py  # requires GPTZERO_API_KEY, ORIGINALITY_API_KEY
uv run python scripts/eval/run_gradient_forger.py
uv run python scripts/eval/run_signal_ablation.py
uv run python scripts/eval/run_baselines.py
```

---

## Architecture

### Embodied Simulation

The `EmbodiedScholar` engine models a stateful cognitive agent, not a prompt wrapper:

- **Metabolic tracking**: irreversible glucose depletion and visual fatigue over session time
- **Cognitive mapping**: metabolic state mapped to LLM generation parameters (temperature, top_p, presence penalty) to emulate stylistic drift
- **Syntactic demand**: cognitive load calculated from academic markers, token-granular execution

### Consciousness-Correlate Signals (four signal modules plus a composite)

| Module | Signal |
|---|---|
| `thermodynamic.py` | Irreversibility, entropy production, joint entropy across channels |
| `integrated_information.py` | Phi estimation, N_BINS sensitivity |
| `temporal_binding.py` | Autocorrelation, regime switching |
| `free_energy.py` | Prediction error, shape score, adaptation |
| `consciousness_signatures.py` | Weighted composite of six discriminative signals (causal DAG mass, causal concentration, cross-channel MI, decay type, free-energy shape, adaptation rate; weights in `_WEIGHTS`) |

### Adversarial Evaluation (4-tier hierarchy)

Defined in `adversarial.py`:

- **L0 Naive**: no adaptation, no context
- **L1 Statistical**: matches surface statistics
- **L2 Reverse-Engineered**: knows the signal formulas, targets feature distributions
- **L3 Expert**: full signal inversion with gradient-guided forgery

### Gradient Forgers (`gradient_forger.py`)

- `GradientForger`: numerical gradient descent against the composite score
- `EvolutionaryForger`: CMA-ES-style search

No convergence statistics from these forgers are preserved in `results/`; run `uv run python scripts/eval/run_gradient_forger.py -o results/gradient_forger.json` before quoting any number.

---

## Project Structure

```
augmented/
├── scholawrite/               # Core Python package
│   ├── agentic.py             # LLM orchestrator and AAA loop
│   ├── embodied.py            # Metabolic resource simulation
│   ├── causal_core.py         # Token-granular execution engine
│   ├── injection.py           # Disruption point detection
│   ├── adversarial.py         # 4-tier adversary hierarchy + AUC reporting
│   ├── detector_harness.py    # Unified Detector protocol + GPTZero/Originality clients
│   ├── gradient_forger.py     # GradientForger + EvolutionaryForger
│   ├── thermodynamic.py       # Irreversibility signals
│   ├── integrated_information.py
│   ├── temporal_binding.py
│   ├── free_energy.py
│   ├── consciousness_signatures.py
│   ├── baselines.py           # Per-class precision/recall metrics
│   ├── datasets/              # 22 external dataset loaders (500K+ checkpoints)
│   └── config.py              # Simulation thresholds
├── scripts/                   # Experiment runners (30+ scripts)
│   ├── run_all_experiments.sh # Master reproduction pipeline (9 experiments)
│   ├── run_all.sh             # Full pipeline rebuild
│   └── validate/smoke_test.py
├── analysis/
│   ├── extended/              # 15 exploratory analyses (01-15)
│   └── hypotheses-tested/     # Falsified spectrum hypothesis + 6 strict-filter scripts
├── data/
│   ├── seed/raw/hf_scholawrite/
│   ├── seed/normalized/
│   ├── augmented/
│   │   ├── documents.jsonl
│   │   ├── annotations.jsonl
│   │   ├── anomalies.jsonl
│   │   ├── failures.jsonl
│   │   ├── stats.json
│   │   └── run_manifest.json
│   ├── splits/splits.json
│   └── injections/raw/
├── configs/                   # JSON simulation configs
├── docs/                      # Protocol, thresholds, terminology, ethics
└── tests/                     # 658 passing, 15 skipped
```

---

## Data Fields

**documents.jsonl:**
- `doc_id`, `revision_id`, `revision_index`
- `text`: normalized revision text
- `timestamp`: revision timestamp (if available)
- `provenance_hash`: content hash for verification
- `before_text`: text before this revision
- `writing_intention`: expert-annotated intention (15 classes)

**annotations.jsonl:**
- `doc_id`, `revision_id`, `injection_id`
- `span_start_char`, `span_end_char`: character offsets
- `span_start_sentence`, `span_end_sentence`: sentence offsets
- `injection_level`: naive | topical | contextual
- `trajectory_state`: COLD | WARM | ASSIMILATED
- `ambiguity_flag`: NONE | LOW | MEDIUM | HIGH
- `generator_class`: weak | mid | strong
- `causal_trace`: full causal event trace

---

## External Datasets

22 loaders in `scholawrite/datasets/`, 500K+ checkpoints loaded locally:

| Dataset | Records | Use |
|---|---|---|
| KLiCKe | 4,992 sessions | Quality correlation, authorship fingerprinting |
| CMU | 20,400 sessions | Cross-dataset classification |
| Mendeley | 489K checkpoints | Simulation calibration |
| Student Fatigue | 2K sessions | Fatigue trajectory validation |
| EmoSurv | 81 users | Emotion dynamics |
| Tappy Parkinson's | varies | Clinical signal validation |

---

## Extended Analysis Suite

Fifteen exploratory analyses in `analysis/extended/` (01-15; 01-11 summarized below) test whether the process signals vary with external conditions; none establishes that the signals measure cognition.

| # | Analysis | Key Result |
|---|---|---|
| 01 | Cognitive spectrum classifier | Composition > transcription > password regime detectable |
| 02 | Synthesizer comparison | Grounds adversarial tier ranking in empirical forger proximity |
| 03 | Parkinson's comparison | Best single signal (lag-1 autocorrelation) AUC 0.58, KS p = 0.09; mean IKI AUC 0.52, entropy AUC 0.46; no reliable separation |
| 04 | Emotion dynamics | Entropy and IKI modulate systematically across 5 affect states |
| 05 | Quality correlation | Spearman r with bootstrap CI; text-only detectors cannot match |
| 06 | Authorship fingerprinting | Per-writer signal vectors form stable pairwise distance clusters |
| 07 | Adversarial channel capacity | AUC degrades predictably; capacity bounds derived |
| 08 | Simulation calibration | KS-test and Wasserstein distance vs. real KLiCKe writers |
| 09 | Entropy floor | Human 5th-percentile IKI entropy above simulation 95th percentile |
| 10 | Channel capacity | AUC connected to Shannon bounds; monotonic degradation with adversarial knowledge |
| 11 | Regime detection | Rolling IKI entropy reveals within-document composition vs. recall switches |

Run all: `for f in analysis/extended/[0-9][0-9]_*.py; do uv run python "$f"; done`

`analysis/hypotheses-tested/` archives the *spectrum hypothesis* (2026-04-07, FALSIFIED): the hypothesis predicted monotonic accuracy degradation from process-intrinsic to state-intrinsic tasks; the empirical pattern showed 3 non-monotonic reversals. Documented as required scientific practice.

---

## Real Keystroke Data

An archived recorder summary (`analysis/keystroke_recorder_results.json`) reports 14,997 keystrokes across 16 editor sessions with 103 CLC checkpoints (mean IKI 253.68 ms; session-level median CLC -0.0534, key `clc.session_median`; Wilcoxon p = 0.8552). The underlying recorder database is not distributed and was unavailable when the manuscript was audited, so these are preserved outputs, not rerun results. They do not validate the simulator; the manuscript's historical KS comparison reports the measured distributional gaps. The analysis script is `analysis/keystroke_recorder_analysis.py`.

---

## Information-Theoretic Bounds

`scripts/eval/run_info_theoretic_analysis.py` derives Fano's inequality bounds and critical trace length requirements. Analysis 10 connects empirical AUC to Shannon upper bounds.

---

## Intended Use and Non-Use

**Intended**: research on process integrity, revision-trace analysis, benchmarking discontinuity detection methods, evaluating causal signatures for process verification.

**Not intended**: punitive or automated enforcement, sole authorship determination, high-stakes decisions without human review, reverse identification of authors or institutions, surveillance of writers without consent, any framing as "AI detection."

See [docs/ETHICS.md](docs/ETHICS.md) for full ethical considerations.

---

## Documentation

| Document | Purpose |
|----------|---------|
| [docs/PROTOCOL.md](docs/PROTOCOL.md) | Augmentation protocol and pipeline steps |
| [docs/DATA_CARD.md](docs/DATA_CARD.md) | Dataset card with composition and intended use |
| [docs/TERMINOLOGY.md](docs/TERMINOLOGY.md) | Binding terminology contract |
| [docs/LABEL_TAXONOMY.md](docs/LABEL_TAXONOMY.md) | Label namespaces and disjointness rules |
| [docs/TRAJECTORIES.md](docs/TRAJECTORIES.md) | Trajectory definitions and erosion model |
| [docs/INJECTION_LEVELS.md](docs/INJECTION_LEVELS.md) | Injection sophistication level definitions |
| [docs/THRESHOLDS.md](docs/THRESHOLDS.md) | Mathematical justifications for simulation thresholds |
| [docs/COGNITIVE_CALIBRATION.md](docs/COGNITIVE_CALIBRATION.md) | Cognitive science parameter calibration |
| [docs/CONSCIOUSNESS_SIGNATURES.md](docs/CONSCIOUSNESS_SIGNATURES.md) | Consciousness-correlate signal definitions |
| [docs/ETHICS.md](docs/ETHICS.md) | Ethical considerations and misuse risks |
| [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) | Full rebuild instructions |
| [docs/CI.md](docs/CI.md) | CI pipeline configuration |
| [docs/ATTRIBUTION_AND_LICENSE.md](docs/ATTRIBUTION_AND_LICENSE.md) | Attribution, permissions, and license terms |

---

## Further Work

The current framework supports exploratory evaluation and leaves several open problems:

- **Larger-scale evaluation**: the adversarial AUC matrix was computed at n=30-50; bootstrap confidence intervals at n=100+ are needed before the result is reported as more than exploratory.
- **Formal analysis**: no impossibility result is claimed. Whether informed-adversary forgery admits a reduction to a computational hardness assumption is an open question.
- **Privileged-channel detection**: if the adversary cannot observe the raw keylog, what minimum channel bandwidth is required for reliable detection? The information-theoretic bound analysis is a starting point but not a full answer.
- **Multimodal process signals**: the current signals are keystroke-timing only. Eye-tracking, clipboard events, and pause structure could add independent channels that are harder to jointly forge.
- **Cross-language validation**: all current data is English. The cognitive calibration parameters (pause-burst ratio, IKI distributions) may not transfer to languages with different orthographic demands.
- **Dataset release refresh**: the gated Hugging Face release of 2026-01-29 (snapshot `d473c26d`) predates the April 2026 framework expansion and ships without checksums or a run manifest; the card was corrected on 2026-11-01 (UTC), and checksums, a run manifest, and a Zenodo DOI remain to be published.
- **Real human baseline**: the archived commercial-detector pilot includes four human seed texts, but process-trace evaluations use simulated reference traces. Validation of the complete detection stack on consenting human participants remains necessary.

---

## Release history

- 2026-01-29: package and gated dataset released (fork commit `cc3a6eb`; Hugging Face revision `d473c26d`).
- 2026-04-03: PR #1 merged into `minnesotanlp/scholawrite` (January version only).
- 2026-04-05 to 2026-04-08: signals, adversarial framework, external loaders, calibration, and reorganization added to the fork (`26287a1`, `ea4dc63`, `9589b7a`); no package code changes committed since, documentation corrections aside.
- 2026-04-25: `analysis/process-attestation` submitted upstream as PR #5 (open).
- 2026-10-31: manuscript prepared.

---

## Citation

For ScholaWrite-Augmented, cite David Condrey's manuscript:

```bibtex
@unpublished{condrey2026scholawriteaugmented,
  title={ScholaWrite-Augmented: Revision-Level Benchmarks for Hybrid Human-AI Writing Detection through Causal Process Simulation},
  author={Condrey, David},
  year={2026},
  note={Unpublished manuscript, Writerslogic, Inc.},
  url={https://github.com/writerslogic/scholawrite-augmented}
}
```

Also cite the original ScholaWrite dataset when using its data or derived artifacts:

```bibtex
@inproceedings{le2026scholawrite,
  title={ScholaWrite: A Dataset of End-to-End Scholarly Writing},
  author={Khanh Chi Le and Linghe Wang and Minhwa Lee and Ross Volkov and Luan Tuyen Chau and Dongyeop Kang},
  year={2026},
  booktitle={Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={34755--34788},
  publisher={Association for Computational Linguistics},
  doi={10.18653/v1/2026.acl-long.1606},
  url={https://aclanthology.org/2026.acl-long.1606/}
}
```

## License

The augmentation package declares Apache-2.0 in its [package metadata](https://github.com/writerslogic/scholawrite-augmented/blob/main/augmented/pyproject.toml); the original ScholaWrite code carries the root [MIT License](https://github.com/writerslogic/scholawrite-augmented/blob/main/LICENSE). The ScholaWrite seed dataset lists Apache-2.0 alongside additional access, redistribution, privacy, and permitted-use restrictions, which every derivative artifact inherits. See [docs/ATTRIBUTION_AND_LICENSE.md](docs/ATTRIBUTION_AND_LICENSE.md) for the recorded permission and governing data terms.
