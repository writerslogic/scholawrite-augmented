# ScholaWrite-Augmented

**Revision-level benchmarks for hybrid human–AI writing and process-integrity research.**

**Created and developed by [David Condrey](https://github.com/writerslogic), Writerslogic, Inc.**

ScholaWrite-Augmented extends MinnesotaNLP's ScholaWrite with synthetic insertion tools, causal writing-process simulation, revision trajectory annotations, adversarial evaluation, and keystroke analysis. David Condrey authored the augmentation framework, its evaluation and analysis tools, and the accompanying manuscript, *ScholaWrite-Augmented: Revision-Level Benchmarks for Hybrid Human-AI Writing Detection through Causal Process Simulation*.

[Augmentation package](augmented/) · [Augmented dataset (gated)](https://huggingface.co/datasets/Writerslogic/scholawrite-augmented) · [Evaluation protocol](augmented/docs/EVALUATION.md) · [Attribution and data terms](augmented/docs/ATTRIBUTION_AND_LICENSE.md)

The accompanying manuscript, *ScholaWrite-Augmented: Revision-Level Benchmarks for Hybrid Human-AI Writing Detection through Causal Process Simulation* (David Condrey, 2026), is unpublished; no venue or DOI is claimed. It inspects this repository at commit `4f634d2`.

### Release history

- 2026-01-21: written derivative-release permission from Dongyeop Kang ([terms](augmented/docs/ATTRIBUTION_AND_LICENSE.md)).
- 2026-01-29: public release of this fork (first commit `cc3a6eb`) and of the gated Hugging Face dataset [`Writerslogic/scholawrite-augmented`](https://huggingface.co/datasets/Writerslogic/scholawrite-augmented) (data snapshot `d473c26d`). The data files have not been rebuilt since; the card was corrected on 2026-11-01 (UTC) to name this manuscript, its unpublished status, and the published ScholaWrite citation.
- 2026-04-03: `augmented/` merged into `minnesotanlp/scholawrite` (PR #1).
- 2026-04-05 to 2026-04-08: fork-only framework expansion (adversarial framework, external loaders, extended analyses, test suite); not part of the 2026-01-29 dataset snapshot.
- 2026-04-25: PR #5 (process attestation analysis pipeline) opened upstream; still open.
- 2026-10-31: manuscript finalized, nine months after the dataset release.

## Start with ScholaWrite-Augmented

The augmentation package requires Python 3.11 or later and uses `uv`:

```bash
cd augmented
uv sync --extra dev
uv run python scripts/pipeline/build_augmented_dataset.py --help
```

See the [package documentation](augmented/README.md) for the framework and the [evaluation protocol](augmented/docs/EVALUATION.md) for task definitions. Dataset counts and experimental results belong to specific recorded builds; source code availability does not imply unrestricted dataset redistribution.

## Original ScholaWrite

The original dataset, collection system, and writing-assistant experiments are the work of **Khanh Chi Le, Linghe Wang, Minhwa Lee, Ross Volkov, Luan Tuyen Chau, and Dongyeop Kang**, University of Minnesota. Their work provides the foundation for this extension.

[ScholaWrite paper (ACL 2026)](https://aclanthology.org/2026.acl-long.1606/) · [Original repository](https://github.com/minnesotanlp/scholawrite) · [Project page](https://minnesotanlp.github.io/scholawrite/) · [Seed dataset](https://huggingface.co/datasets/minnesotanlp/scholawrite)

### Original ScholaWrite abstract

Writing is a cognitively demanding activity that requires constant decision-making, heavy reliance on working memory, and frequent shifts between tasks of different goals. To build writing assistants that truly align with writers’ cognition, it is necessary to capture and analyze the complete thought process behind how writers transform ideas into final texts. We present **ScholaWrite**, the first dataset of end-to-end scholarly writing, tracing the multi-month journey from initial drafts to final manuscripts. The dataset traces nearly 62K LATEX-based edits from five computer science preprints over four months and is enriched with fine-grained annotations of cognitive writing intentions. We demonstrate the value of ScholaWrite through three complementary contributions: 
- (1) analysis of real-world writing behavior reveals that scholarly writing is highly non-linear and multiintentional, blending rapid drafting bursts with cognitively sustained writing sessions; 
- (2) evaluations of current large language models show that they struggle to provide meaningful support throughout the human writing process; and
- (3) models finetuned on SCHOLAWRITE demonstrate improved alignment with human writing workflows. SCHOLAWRITE underscores the value of capturing scientists’ cognitive writing process and provides actionable insights and resources for the development of future writing assistants.

## Repository Structure

| Directory | Description |
|---|---|
| `scholawrite_system/` | Data collection backend, admin page, annotation page, and Chrome extension |
| `scholawrite_finetune/` | Fine-tuning scripts for BERT, RoBERTa, and Llama-8B-Instruct |
| `gpt4o/` | GPT-4o inference for iterative writing and intention prediction |
| `meta_inference/` | Llama-8B-Instruct baseline inference |
| `experiments/` | Notebooks for LLM output generation and evaluation (session-level and intent-conditioned), research question analysis, and zero-shot vs. ScholaWrite fine-tuned model comparisons |
| `eval_tool/` | Web interface for human evaluation of model outputs |
| `analysis/` | Evaluation metrics, cosine similarity, lexical diversity, and figure generation |
| `augmented/` | ScholaWrite-Augmented: synthetic insertion tools, causal process simulation, revision-level annotations, and adversarial process evaluation |
| `seeds/` | Seed documents for iterative writing experiments |
| `outputs/` | Pre-computed inference outputs from all models |

## Original ScholaWrite Setup and Experiments

The remainder of this section is the upstream MinnesotaNLP documentation, retained unchanged; it is not required for the augmentation package.

### Prerequisites

- Docker & Docker Compose
- GPU with >= 16GB VRAM (for fine-tuning/inference)
- Python 3.8+

### Environment Setup

Create an `.env` file in the project root:

```bash
HUGGINGFACE_TOKEN="<Your Hugging Face access token>"
OPEN_AI_API="<Your OpenAI API key>"
```

Create a Docker container for fine-tuning and inference:

```bash
docker run --name scholawrite --gpus all -dt -v ./:/workspace --ipc=host pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel bash
docker exec -it scholawrite bash
pip install accelerate python-dotenv huggingface-hub datasets transformers trl unsloth diff_match_patch
```

---

### ScholaWrite System

The data collection system uses Flask + MongoDB with a Chrome extension for Overleaf.

#### 1. MongoDB

1. Install [MongoDB Community Edition](https://www.mongodb.com/docs/manual/administration/install-community/) and [MongoDB Compass](https://www.mongodb.com/try/download/compass).
2. Install [Database Tools](https://www.mongodb.com/docs/database-tools/installation/installation/#installing-the-database-tools).
3. Run MongoDB on the default port (27017). The `flask_db` database and `activity` collection are created automatically.

#### 2. Google OAuth

1. Create a [Google Cloud project](https://developers.google.com/workspace/guides/create-project) and an [OAuth client](https://developers.google.com/workspace/guides/create-credentials#oauth-client-id) for a **Desktop app**.
2. Download the credentials file, rename it to `sheet_credential.json`.
3. Update the volume paths in `scholawrite_system/docker-compose.yml` (lines 12-13):
   ```yaml
   volumes:
     - <path>/sheet_credential.json:/usr/local/src/scholawrite/flaskapp/sheet_credential.json
     - <path>/token.json:/usr/local/src/scholawrite/flaskapp/token.json
   ```

#### 3. Google Sheet

1. Add Overleaf project IDs to consecutive rows in a Google Sheet column.
2. In `scholawrite_system/flaskapp/App.py`, update:
   - Line 22: `SAMPLE_SPREADSHEET_ID` with your Sheet ID
   - Line 23: `SAMPLE_RANGE_NAME` with your cell range

#### 4. Ngrok

1. Get 3 static domains and auth tokens from [Ngrok](https://dashboard.ngrok.com/).
2. Create `ngrok_admin.yml`, `ngrok_annotation.yml`, and `ngrok_schola.yml` in `scholawrite_system/`:
   ```yaml
   version: 2
   authtoken: <Your AuthToken>
   ```
3. Paste your domains into `docker-compose.yml` on lines 36, 63, and 90.

#### 5. Launch

```bash
cd scholawrite_system
docker-compose up
```

#### Chrome Extension

1. Update the server URL in `scholawrite_system/extension/background.js` (line 3) and `popup.js` (line 5).
2. In Chrome, go to `chrome://extensions` → enable **Developer Mode** → **Load unpacked** → select the `extension/` folder.

> **Note:** Due to Overleaf UI updates, the Chrome extension can no longer record writer actions or perform AI paraphrase.

---

### Fine-Tuning

Inside the Docker container, navigate to `scholawrite_finetune/`:

**Llama-8B** (`llama8b_scholawrite_finetune/`):
```bash
# Iterative writing — set PURPOSE = "WRITING" in args.py
python3 train_writing.py

# Classification — set PURPOSE = "CLASS" in args.py
python3 train_classifier.py
```

**BERT / RoBERTa** (`bert_finetune/`):
```bash
python3 small_model_classifier.py
```

Fine-tuned models are saved to `results/` in the project root.

---

### Inference

#### Fine-tuned Llama-8B (`scholawrite_finetune/llama8b_scholawrite_finetune/`)

```bash
# Iterative writing — update model path in iterative_writing.py (lines 65, 80)
python3 iterative_writing.py

# Classification — update model path in classification.py (line 40)
python3 classification.py
```

#### Baseline Llama-8B (`meta_inference/llama8b_meta_instruction/`)

```bash
python3 iterative_writing.py
python3 classification.py
```

#### GPT-4o (`gpt4o/`)

```bash
python3 iterative_writing.py
python3 classification.py
```

Output structure: `<output_dir>/<seed>/generation/` and `<output_dir>/<seed>/intention/` (100 iterations each).

---

### Eval Tool

1. Set up Ngrok with 1 static domain. Create `eval_tool/ngrok.yml` with your auth token.
2. Update the domain in `eval_tool/run_eval_app.sh`.
3. Run:
   ```bash
   cd eval_tool
   docker-compose up -d
   docker exec -it scholawrite_eval bash
   ./run_eval_app.sh
   ```

---

## Citation

For the augmentation framework, cite David Condrey's manuscript:

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

## Code Contributors

**ScholaWrite-Augmented:** [David Condrey](https://github.com/writerslogic) — augmentation framework, causal simulation, adversarial evaluation, keystroke analyses, and manuscript.

**Original ScholaWrite:**

[Linghe Wang](https://github.com/Linghe-Wang), [Ross Volkov](https://github.com/rvolkov1), [Minhwa Lee](https://github.com/mimn97), [Khanh Chi Le](https://github.com/chile2706)

## License

The original ScholaWrite code carries the root [MIT License](LICENSE). The augmentation package declares Apache-2.0 in its [package metadata](augmented/pyproject.toml). The ScholaWrite seed dataset lists Apache-2.0 alongside additional access, redistribution, privacy, and permitted-use restrictions; derivatives must retain those restrictions. See [Attribution and License](augmented/docs/ATTRIBUTION_AND_LICENSE.md) for the recorded derivative-release permission and governing data terms.
