# ROCm local LLM runtime guide

The inference runtime is a separate process from Medical-Rag-LLM:

```text
FastAPI RAG application
  -> loopback HTTP
  -> vLLM OpenAI-compatible server
  -> ROCm
  -> AMD GPU
```

vLLM is the primary example, not an application dependency. GPU support depends on the
specific AMD GPU, gfx architecture, ROCm stack, PyTorch build, and vLLM release. Select a
runtime or container image documented as compatible with the actual machine; this project
does not declare one unverified image or version universally compatible.

## 1. Inspect the machine safely

```bash
python scripts/check_runtime.py
rocminfo
rocm-smi --showproductname
amd-smi static --asic
docker --version
python -c "import torch; print(torch.__version__); print(torch.version.hip); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

PyTorch exposes ROCm devices through its `torch.cuda` compatibility API. A true result there
does not mean this project targets NVIDIA CUDA; confirm that `torch.version.hip` is present.
Missing utilities are not by themselves proof that ROCm is unavailable.

## 2. Start a compatible local server

Use an already approved chat/instruction model path or model ID. Do not place credentials in
`LLM_BASE_URL` and do not silently download large weights on shared infrastructure.

```bash
vllm serve <MODEL_PATH_OR_MODEL_ID> \
  --host 127.0.0.1 \
  --port 8001
```

The served model name must match the value configured below. `/v1/chat/completions` requires
an instruction/chat model with a usable chat template; consult the selected vLLM/model
documentation if the server reports that no template is available.

## 3. Configure the application

```dotenv
LLM_BASE_URL=http://127.0.0.1:8001/v1
LLM_MODEL_NAME=<SERVED_MODEL_NAME>
LLM_TEMPERATURE=0.0
LLM_MAX_TOKENS=512
LLM_TIMEOUT=60.0
LLM_ALLOW_PRIVATE_NETWORK=false
```

Loopback is required by default and no OpenAI cloud key is needed. With
`LLM_ALLOW_PRIVATE_NETWORK=false`, retrieved context stays on the application host. To use an
approved Lab server at an RFC1918 or IPv6 ULA address, set
`LLM_ALLOW_PRIVATE_NETWORK=true` explicitly. The application may then send retrieved context
from the application host across the private Lab network to that inference host. A private IP
does not automatically provide secure transport, authentication, or authorization; apply the
organization's network and medical-data controls. Public IPs and public hostnames remain
rejected.

## 4. Verify in order

```text
1. Check ROCm/GPU with scripts/check_runtime.py.
2. Launch the compatible local model server.
3. Run python scripts/smoke_test_llm.py.
4. Launch uvicorn app.main:app --host 127.0.0.1 --port 8000.
5. Ingest and index synthetic or de-identified documents.
6. Run python scripts/smoke_test_rag.py --query "<question about synthetic data>".
7. Run Phase 6 retrieval-only evaluation.
8. Run Phase 6 end-to-end RAG evaluation and save the JSON report.
```

```bash
python -m app.evaluation.cli --dataset data/evaluation/cases.jsonl --mode retrieval
python -m app.evaluation.cli --dataset data/evaluation/cases.jsonl --mode rag \
  --output data/evaluation/results.json
```

Unit tests need no GPU or running server. The LLM and RAG smoke tests intentionally fail with
an actionable message when the local runtime or indexed synthetic data is unavailable.
