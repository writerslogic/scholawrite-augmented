# Data Card: ScholaWrite‑Augmented

## 1) Dataset Summary
ScholaWrite‑Augmented is a revision‑tracked scholarly writing dataset with **annotated external insertion events**, designed for process‑integrity research. It does **not** label origin (human vs AI) and adheres to `docs/TERMINOLOGY.md`.

## 1.1) Release
- Version: snapshot `d473c26d` of `Writerslogic/scholawrite-augmented` (gated), released 2026-01-29; not rebuilt since.
- Released files: `documents.jsonl`, `annotations.jsonl`, `anomalies.jsonl`, `injections.jsonl`, `stats.json`, `docs/`. The `checksums.txt` and `run_manifest.json` described in Sections 10-11, and the `failures.jsonl` described in `docs/PROTOCOL.md`, are produced by the build pipeline but were not included in the 2026-01-29 release.
- Reported counts (`stats.json`, seed 42): 5 documents, 126,246 revisions, 237 injections (naive 88, topical 101, contextual 48; cold 119, warm 79, assimilated 39).

## 2) Motivation
Existing corpora lack revision‑structured ground truth for **process‑inconsistent external insertion**. This dataset fills that gap and enables evaluation of process‑integrity signals without stigmatizing assistive tool use.

## 3) Source Data
- Seed dataset: ScholaWrite (revision‑tracked scholarly writing).
- Attribution and licensing: `docs/ATTRIBUTION_AND_LICENSE.md`.

## 4) Composition (Computed at Build Time)
All counts are produced in `data/augmented/stats.json`:
- Documents: `stats.documents`
- Revisions: `stats.revisions`
- Injections: `stats.injections`
- Trajectory distribution: `stats.trajectories`
- Label distribution: `stats.labels`

## 5) Data Structure
Primary artifacts:
- `data/augmented/documents.jsonl`: augmented revisions with metadata.
- `data/augmented/annotations.jsonl`: span‑level insertion annotations.
- `data/augmented/anomalies.jsonl`: process‑anomaly negative controls.

Each document record includes:
- `doc_id`, `revision_id`, `revision_index`
- normalized `text`
- timestamp fields (if available)
- provenance hash

Each annotation record includes:
- span offsets (char + sentence)
- injection level
- trajectory state
- ambiguity flag
- generator metadata

## 6) Labeling Policy
- Labels describe **process events**, not origin.
- No “AI‑generated” or “human‑written” labels are used.
- Ambiguity is explicit via `ambiguity_flag`.
- Label taxonomy is defined in `docs/LABEL_TAXONOMY.md`.

## 7) Intended Use
- Research on process integrity and revision‑trace analysis.
- Benchmarking methods that surface revision discontinuities.
- Studying ambiguity and erosion of insertion boundaries.

## 8) Non‑Use / Out‑of‑Scope
- Punitive or automated enforcement decisions.
- Sole determination of authorship or tool use.
- High‑stakes decisions without human review.
- Reverse identification of authors, papers, projects, or institutions.
- Any disclosure that could enable identification of source texts.

## 9) Ethical Considerations
- Synthetic text may encode generator biases.
- Misuse risk exists; see `docs/ETHICS.md`.
- The dataset is positioned to **separate assistance from misrepresentation**, not to stigmatize tool use.

## 9.1) Terms Compliance
This dataset inherits the ScholaWrite terms (Hugging Face). In particular:
- No reverse identification.
- No disclosure enabling identification.
- No PII introduced in derivatives.
- Use limited to explicitly permitted purposes.

## 10) Quality Assurance
- Deterministic pipeline with recorded RNG seeds.
- Annotation validation: offsets, non‑overlap, ambiguity constraints.
- Checksums for all artifacts in `data/augmented/checksums.txt`.

## 11) Reproducibility
- Rebuild steps in `docs/REPRODUCIBILITY.md`.
- Run metadata in `data/augmented/run_manifest.json`.

## 12) Citation
Include the ScholaWrite citation:

```bibtex
@inproceedings{le2026scholawrite,
  title={ScholaWrite: A Dataset of End-to-End Scholarly Writing},
  author={Khanh Chi Le and Linghe Wang and Minhwa Lee and Ross Volkov and Luan Tuyen Chau and Dongyeop Kang},
  booktitle={Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  year={2026},
  pages={34755--34788},
  publisher={Association for Computational Linguistics},
  doi={10.18653/v1/2026.acl-long.1606},
  url={https://aclanthology.org/2026.acl-long.1606/}
}

```

## 13) Terminology Contract
All interpretations must comply with `docs/TERMINOLOGY.md`.
