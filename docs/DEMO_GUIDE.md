# Phase 8 demo guide

This guide runs the existing local Medical-Rag-LLM stack as a short,
repeatable presentation. All records under `data/demo/` are fictional,
synthetic, and de-identified. They contain no real patient data and are **not
for clinical use**.

The Phase 8 runner reuses the production code paths: Phase 2 ingestion, Phase 3
indexing and Chroma, Phase 4 retrieval, Phase 5 `RAGService`, and Phase 6
evaluation metrics. It does not contain fixed answers or a mock success path.

## Prerequisites

- Python 3.12 virtual environment with `requirements-dev.txt` installed.
- The configured embedding model already present in the local model cache.
  Phase 8 forces Hugging Face/Transformers offline mode and will not download it.
- A compatible local vLLM server and model weights already installed on the
  target AMD ROCm host.
- Run every command from the repository root.

Copy `.env.example` to `.env` and review these existing settings:

```dotenv
EMBEDDING_MODEL_NAME=intfloat/multilingual-e5-small
EMBEDDING_MODEL_REVISION=614241f622f53c4eeff9890bdc4f31cfecc418b3
EMBEDDING_DEVICE=cpu
VECTOR_DB_DIR=vector_db
LLM_BASE_URL=http://127.0.0.1:8001/v1
LLM_MODEL_NAME=local-medical-model
LLM_TEMPERATURE=0.0
```

The demo forces the collection name to `medical_demo_v1` and pins its chunking
settings to 500/100/50 characters so evidence IDs remain stable. It does not
reset or seed the normal `medical_documents` collection.

## Install and start services

Install dependencies explicitly; the demo command itself never installs a
driver, package, or model:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

Follow [the ROCm runtime guide](rocm_local_llm.md) for the approved PyTorch,
ROCm, vLLM, and model installation. Start vLLM in terminal 1, replacing the
placeholder with an already-downloaded model path:

```bash
vllm serve <MODEL_PATH> \
  --served-model-name local-medical-model \
  --host 127.0.0.1 \
  --port 8001
```

The CLI calls the Phase 5 service directly, so FastAPI is optional. To include
the HTTP layer in preflight, start it separately in terminal 2:

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Health checks:

```bash
curl http://127.0.0.1:8001/v1/models
curl http://127.0.0.1:8000/health
```

## Clean-state demo commands

On a new demo collection, allow the first preflight to report an empty index:

```bash
python scripts/demo.py preflight --allow-unseeded
python scripts/demo.py seed --reset
python scripts/demo.py preflight
python scripts/demo.py run
python scripts/demo.py evaluate --output-dir reports/demo
```

Use `--require-api` on either preflight command when FastAPI is part of the
presentation. The single-command equivalent is:

```bash
python scripts/demo.py all --output-dir reports/demo
```

Useful focused commands:

```bash
python scripts/demo.py run --case DEMO-C001
python scripts/demo.py run --case DEMO-C001 --verbose
python scripts/demo.py reset --yes
python scripts/demo.py --help
```

`seed` is idempotent. `seed --reset` first verifies the exact dedicated demo
collection name, deletes only that collection, and then indexes all four fixed
records in filename order. A failure returns a non-zero exit code.

## Five-to-ten-minute presentation script

1. **Safety and architecture (one minute).** Say: “These are synthetic,
   de-identified records and the output is not clinical advice. The runner uses
   our normal ingestion, vector retrieval, local LLM, and evaluation layers.”
   Show `data/demo/README.md` and `python scripts/demo.py --help`.
2. **Environment check (one minute).** Run
   `python scripts/demo.py preflight --allow-unseeded`. Point out the fixed data
   count, dedicated collection, local vLLM check, and optional FastAPI status.
3. **Safe reproducibility (one minute).** Run
   `python scripts/demo.py seed --reset`. The audience should see
   `collection=medical_demo_v1 records=4 chunks=4`. Run `python scripts/demo.py
   seed` once more if time permits and note that the count remains four.
4. **Grounded answers (three minutes).** Run `python scripts/demo.py run`. For
   each case, point out the case ID, question, generated answer, and source
   document/chunk IDs. The answer text may vary; the evidence identifiers do not.
5. **Evaluation and handoff (two minutes).** Run `python scripts/demo.py
   evaluate --output-dir reports/demo`. Open the printed Markdown path, show the
   totals and citation/retrieval metrics, then mention that the adjacent JSON is
   machine-readable.

Example abbreviated output (illustrative synthetic output, not a patient result):

```text
[DEMO-C001] For patient DEMO-P001, what diabetes medicine ...?
ANSWER: The record documents metformin 500 mg twice daily and HbA1c 7.2% [1].
SOURCES:
  [1] DEMO-P001.txt document=672ff899b84c584a chunk=462bc507582ed594
STATUS: OK
```

## Evaluation artifacts

Each evaluation creates a timestamped pair under `reports/demo/`:

- `evaluation-<UTC-run-id>.json`: environment, totals, expected facts, answer,
  sources, Phase 6 metrics, status, and error for every case.
- `evaluation-<UTC-run-id>.md`: concise human-readable totals, metrics, per-case
  results, and limitations.

Runtime artifacts are ignored by Git. The writer refuses to overwrite an
existing artifact with the same run ID.

A case passes only when the existing Phase 6 metrics confirm an expected
retrieval hit, present/valid/relevant citations, and reference token F1 of at
least `0.30` when a reference answer exists. This threshold is only a
deterministic lexical-overlap sanity check for the fixed demo cases. It does not
measure semantic equivalence, medical correctness, or clinical safety. An
uncited or low-overlap answer fails even if its wording appears plausible.

## Recovery and troubleshooting

- **Demo collection is empty:** run `python scripts/demo.py seed --reset`.
- **Embedding model cannot load:** pre-cache the exact configured model and
  revision during setup. The demo intentionally stays offline.
- **vLLM unavailable:** confirm terminal 1 is running, then check
  `LLM_BASE_URL`, `LLM_MODEL_NAME`, and `/v1/models`.
- **FastAPI warning:** it is optional for the direct CLI. Start it or omit
  `--require-api`.
- **Embedding compatibility error:** confirm the pinned embedding settings,
  then rebuild only the demo collection with `seed --reset`.
- **Interrupted seed:** rerun `seed --reset`; the exact namespace guard prevents
  normal collections from being cleared.
- **Report already exists:** choose another output directory or wait for a new
  UTC run ID; existing reports are never silently overwritten.

To recover to an empty demo state without touching user data:

```bash
python scripts/demo.py reset --yes
```

## Stop services safely

In the terminals that own FastAPI and vLLM, press `Ctrl+C` and wait for each
process to exit. Do not kill unrelated Python processes or remove `vector_db/`.

## Known limitations

- LLM wording and citation placement can vary even at temperature zero.
- Retrieval/evaluation labels cover only the fixed synthetic demo cases.
- The output is not medical advice and must not be used for diagnosis or care.
- GPU, ROCm, model compatibility, and memory use must be verified on the target
  AMD Radeon RX 7900 XTX host; offline unit tests do not prove GPU readiness.
