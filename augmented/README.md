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

The original [ScholaWrite](https://github.com/minnesotanlp/scholawrite) dataset captures human writing revisions but contains no synthetic insertions, no process-level annotations, and no adversarial evaluation framework. ScholaWrite-Augmented adds all of that. It injects synthetic text at three sophistication levels into the revision stream, tracks boundary erosion across revisions, and provides span-level annotations with explicit ambiguity flags — creating a benchmark for *process integrity* rather than origin classification.

The augmentation is grounded in a cognitive simulation engine (`EmbodiedScholar`) that models metabolic depletion, fatigue, and syntactic demand to produce writing traces that are statistically calibrated against real keystroke data. A four-tier adversarial evaluation framework then stress-tests detection signals against adversaries of increasing knowledge, from naive to expert. The result is both a dataset and a proof: process-based detection has a hard boundary determined by observational privilege, not signal sophistication.

**This dataset does NOT label origin (human vs AI).** It studies whether the observable revision process is consistent with iterative human authorship, or whether it exhibits evidence of process-inconsistent external insertion.

Target venue: ACL Findings 2026.

---

## Core Finding

An expert adversary with full knowledge of the detection signals defeats all consciousness-correlate features (composite AUC 0.31 at n=50). This proves an impossibility boundary: derived features cannot detect informed adversaries. The paper frames this as the *observational privilege boundary* — process-based detection is only sound when the adversary cannot replay the observed process.

**Signal x Tier AUC matrix (n=30, events=40):**

| Detector | Naive | Statistical | RevEng | Expert |
|---|---|---|---|---|
| NCD | 0.954 | 0.989 | 0.633 | 0.971 |
| CausalCoupling | 0.368 | 0.383 | 0.361 | 0.431 |
| ConsciousnessSignatures | 0.969 | 0.893 | 0.997 | **0.291** |
| GPTZero | 0.994 | 0.909 | 0.972 | 0.969 |
| Originality.ai | 0.437 | 0.498 | 0.397 | 0.686 |

GPTZero remains robust because it evaluates product-level text features; the expert adversary targets process signals specifically.

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
uv sync
uv run python scripts/smoke_test.py
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
from scholawrite.io import load_seed
from scholawrite.schema import AugmentedDocument
from scholawrite.annotations import validate_annotations
from scholawrite.metrics import compute_span_iou, compute_f1
from scholawrite import detect

# Simple detection
print(detect("your text here"))

# Load and validate
seed_docs = load_seed("data/seed/normalized/")
errors = validate_annotations("data/augmented/annotations.jsonl")
```

### Reproduce Experiments

```bash
bash scripts/run_all_experiments.sh        # full reproduction pipeline
uv run python scripts/run_adversarial_eval.py
uv run python scripts/run_detector_comparison.py  # requires GPTZERO_API_KEY, ORIGINALITY_API_KEY
uv run python scripts/run_gradient_forger.py
uv run python scripts/run_signal_ablation.py --signal-ablation
uv run python scripts/run_baselines.py
```

---

## Architecture

### Embodied Simulation

The `EmbodiedScholar` engine models a stateful cognitive agent, not a prompt wrapper:

- **Metabolic tracking**: irreversible glucose depletion and visual fatigue over session time
- **Cognitive mapping**: metabolic state mapped to LLM generation parameters (temperature, top_p, presence penalty) to emulate stylistic drift
- **Syntactic demand**: cognitive load calculated from academic markers, token-granular execution

### Consciousness-Correlate Signals (5 modules)

| Module | Signal |
|---|---|
| `thermodynamic.py` | Irreversibility, entropy production, joint entropy across channels |
| `integrated_information.py` | Phi estimation, N_BINS sensitivity |
| `temporal_binding.py` | Autocorrelation, regime switching |
| `free_energy.py` | Prediction error, shape score, adaptation |
| `consciousness_signatures.py` | Composite signal across all five |

### Adversarial Evaluation (4-tier hierarchy)

Defined in `adversarial.py`:

- **L0 Naive**: no adaptation, no context
- **L1 Statistical**: matches surface statistics
- **L2 Reverse-Engineered**: knows the signal formulas, targets feature distributions
- **L3 Expert**: full signal inversion with gradient-guided forgery

### Gradient Forgers (`gradient_forger.py`)

- `GradientForger`: numerical gradient descent — 0% convergence (non-smooth objective)
- `EvolutionaryForger`: CMA-ES-inspired — 20% convergence, loss ratio 0.38

Optimization hardness without information-theoretic hardness: sequential computation finds solutions, parallel optimization does not (VDF analogy).

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
│   └── smoke_test.py
├── analysis/
│   ├── extended/              # 11 empirical + theoretical analyses (01-11)
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

Eleven analyses in `analysis/extended/` validate that process signals capture real cognitive phenomena:

| # | Analysis | Key Result |
|---|---|---|
| 01 | Cognitive spectrum classifier | Composition > transcription > password regime detectable |
| 02 | Synthesizer comparison | Grounds adversarial tier ranking in empirical forger proximity |
| 03 | Parkinson's validation | AUC 0.58 — signals measure real motor/cognitive processes |
| 04 | Emotion dynamics | Entropy and IKI modulate systematically across 5 affect states |
| 05 | Quality correlation | Spearman r with bootstrap CI; text-only detectors cannot match |
| 06 | Authorship fingerprinting | Per-writer signal vectors form stable pairwise distance clusters |
| 07 | Adversarial channel capacity | AUC degrades predictably; capacity bounds derived |
| 08 | Simulation calibration | KS-test and Wasserstein distance vs. real KLiCKe writers |
| 09 | Entropy floor | Human 5th-percentile IKI entropy above simulation 95th percentile |
| 10 | Channel capacity | AUC connected to Shannon bounds; monotonic degradation with adversarial knowledge |
| 11 | Regime detection | Rolling IKI entropy reveals within-document composition vs. recall switches |

Run all: `for f in analysis/extended/0*.py; do uv run python $f; done`

`analysis/hypotheses-tested/` archives the *spectrum hypothesis* (2026-04-07, FALSIFIED): the hypothesis predicted monotonic accuracy degradation from process-intrinsic to state-intrinsic tasks; the empirical pattern showed 3 non-monotonic reversals. Documented as required scientific practice.

---

## Real Keystroke Data

`~/.local/data/keystrokes.db` contains 637K events and 5K checkpoints from real writing sessions, directly validating the simulation (SYS-001, SYS-002). Analysis in `analysis/keystroke_recorder_analysis.py`.

---

## Information-Theoretic Bounds

`scripts/run_info_theoretic_analysis.py` derives Fano's inequality bounds and critical trace length requirements. Analysis 10 connects empirical AUC to Shannon upper bounds.

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

The current framework establishes the impossibility boundary but leaves several open problems:

- **Larger-scale evaluation**: the adversarial AUC matrix was computed at n=30-50; bootstrap confidence intervals at n=100+ are needed before the result is paper-ready.
- **Formal theorem**: the impossibility argument is stated informally. A rigorous reduction to a computational hardness assumption (e.g., VDF hardness) would strengthen the theoretical contribution.
- **Privileged-channel detection**: if the adversary cannot observe the raw keylog, what minimum channel bandwidth is required for reliable detection? The information-theoretic bound analysis is a starting point but not a full answer.
- **Multimodal process signals**: the current signals are keystroke-timing only. Eye-tracking, clipboard events, and pause structure could add independent channels that are harder to jointly forge.
- **Cross-language validation**: all current data is English. The cognitive calibration parameters (pause-burst ratio, IKI distributions) may not transfer to languages with different orthographic demands.
- **Dataset release**: HuggingFace and Zenodo releases with checksums, a model card, and a reproducibility bundle.
- **Real human baseline**: GPTZero and Originality.ai have not been tested on real human text passing through the pipeline; only synthetic traces. Running live participants through the detection stack is a required validity check.

---

## Citation

```bibtex
@inproceedings{wang2025scholawrite,
  title     = {ScholaWrite: A Writing Process Study Dataset for Scholarly Writing},
  author    = {Wang, Seulgi and Lee, Yoonna and Volkov, Ilya and Chau, Tuyen Luan and Kang, Dongyeop},
  booktitle = {Proceedings of the Joint Conference of the 59th Annual Meeting of the Association for Computational Linguistics},
  year      = {2025}
}
```

## License

Apache-2.0, inherited from ScholaWrite. See [docs/ATTRIBUTION_AND_LICENSE.md](docs/ATTRIBUTION_AND_LICENSE.md) for full terms including restrictions on reverse identification and PII.
