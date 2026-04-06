# ScholaWrite-Augmented

**Revision-level benchmarks for hybrid human-AI writing detection through causal process simulation.**

This repository is a high-fidelity fork of the original [ScholaWrite](https://github.com/minnesotanlp/scholawrite) project. While the original focused on analyzing human writing revisions, **ScholaWrite-Augmented** introduces an advanced framework for generating synthetic, process-grounded writing revisions that simulate the cognitive and metabolic constraints of human writers.

## 🚀 The Value of the "Augmented" Fork

Standard AI writing datasets consist of static "human vs. AI" snapshots. This fork shifts the paradigm to **Forensic Process Analysis** by simulating *how* text is produced over time.

### 🧠 1. Embodied Cognition Simulation
Unlike typical LLM wrappers, this engine models a stateful **Embodied Scholar**.
- **Metabolic Tracking:** Simulates irreversible glucose depletion and visual fatigue during writing sessions.
- **Cognitive Mapping:** Automatically maps the scholar's metabolic state to LLM generation parameters (temperature, top_p, presence penalty) to emulate realistic stylistic drift.
- **Syntactic Demand:** Calculates the cognitive load of a passage based on academic markers and deconstructs intentions token-by-token.

### 🧬 2. Forensic Causal Signatures
Our simulation produces genuine biometric markers that standard LLM outputs lack:
- **Repair Locality:** Generates "lexical starvation" and "syntactic collapse" events with human-like repair distances (1.0-3.5 tokens).
- **Resource Coupling:** Models the Pearson correlation between cognitive failures and subsequent syntactic simplification.
- **Plausibility Validation:** Every generation is statistically validated against human baselines before being committed to the dataset.

### 🏗️ 3. Robust Asynchronous Pipeline
Built for large-scale, cost-effective dataset construction:
- **Asynchronous Orchestration:** High-concurrency builder with `OpenRouter` integration for multi-model fallback.
- **Persistent LLM Cache:** Deterministic hashing and caching prevent redundant API calls.
- **Background I/O Worker:** Thread-safe, non-blocking disk writes ensure peak performance during heavy generation runs.
- **Adversarial Hardening:** Automated detection of LLM "prompt leakage" (e.g., "As an AI...") and iterative stylistic submergence.

---

## 🛠️ Installation

This project uses `uv` for lightning-fast dependency management.

```bash
cd augmented
uv pip install -e ".[dev]"
```

## 📖 Usage

Run all commands from the `augmented/` directory:

```bash
# 1. Run the test suite (includes architecture & simulation tests)
uv run python -m pytest tests/ -v

# 2. Build a high-fidelity augmented dataset
# This runs the full causal agentic loop across seed documents
uv run python scripts/build_augmented_dataset.py --config configs/production.json

# 3. Perform a quick system validation
uv run python scripts/smoke_test.py
```

## 📁 Project Structure

```text
augmented/
├── scholawrite/         # Core Python package
│   ├── agentic.py       # LLM orchestrator & AAA loop
│   ├── embodied.py      # Metabolic resource simulation
│   ├── causal_core.py   # Token-granular execution engine
│   ├── injection.py     # Disruption point detection
│   └── config.py        # Centralized simulation thresholds
├── configs/             # JSON-driven simulation & marker configs
├── docs/                # Deep-dive documentation
└── scripts/             # Dataset build & evaluation tools
```

## 📚 Documentation

Detailed specifications are available in the `docs/` folder:
- [Architecture Decisions](docs/ARCHITECTURE_DECISIONS.md) - Rationale for the process-driven design.
- [Threshold Calibration](docs/THRESHOLDS.md) - Mathematical justifications for human-baseline simulation.
- [Data Generation Protocol](docs/PROTOCOL.md) - The step-by-step pipeline for building forensics datasets.

## Keystroke Data Collection

To validate simulated writing traces against real human behavior (SYS-001, SYS-002), participants record keystroke timing data using the [Keystroke Recorder](https://github.com/writerslogic/keystroke-recorder):

```bash
brew tap writerslogic/tap
brew install --cask keystroke-recorder
```

See the [keystroke-recorder repository](https://github.com/writerslogic/keystroke-recorder) for source code, privacy details, and participant instructions.

## License

This project is licensed under the terms found in `docs/ATTRIBUTION_AND_LICENSE.md`.
