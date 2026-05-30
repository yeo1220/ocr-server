#!/usr/bin/env bash
# DGX Spark 128GB: default Qwen2.5-14B-Instruct for Korean table cell refinement.
set -euo pipefail
cd "$(dirname "$0")"

# Load DGX defaults + .env
ROOT="$(cd .. && pwd)"
if [[ -f "${ROOT}/env.dgx-spark-128gb.sh" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/env.dgx-spark-128gb.sh"
fi
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

MODEL_ID="${REFINE_HF_MODEL}"
DIR_NAME="${REFINE_MODEL_DIR}"
DEST="models/${DIR_NAME}"

if [[ -f "${DEST}/config.json" ]]; then
  echo "Already present: ${DEST}"
  exit 0
fi

mkdir -p models
echo "Downloading ${MODEL_ID} -> ${DEST}"
echo "(DGX Spark 128GB 권장: 14B 한글 표 보정. 경량: REFINE_HF_MODEL=${REFINE_HF_MODEL})"

HF_TOKEN_ARGS=()
if [[ -n "${HF_TOKEN:-}" ]]; then
  HF_TOKEN_ARGS=(--token "${HF_TOKEN}")
fi

run_hf_download() {
  local hf_bin="$1"
  "${hf_bin}" download "${MODEL_ID}" --local-dir "${DEST}" "${HF_TOKEN_ARGS[@]}"
}

run_python_download() {
  local py_bin="$1"
  MODEL_ID="${MODEL_ID}" DEST="${DEST}" HF_TOKEN="${HF_TOKEN:-}" \
    "${py_bin}" - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=os.environ["MODEL_ID"],
    local_dir=os.environ["DEST"],
    token=os.environ.get("HF_TOKEN") or None,
)
PY
}

if [[ -x "${ROOT}/.venv/bin/hf" ]]; then
  run_hf_download "${ROOT}/.venv/bin/hf"
elif command -v hf >/dev/null 2>&1; then
  run_hf_download hf
elif [[ -x "${ROOT}/.venv/bin/python" ]] && "${ROOT}/.venv/bin/python" -c "import huggingface_hub" 2>/dev/null; then
  run_python_download "${ROOT}/.venv/bin/python"
elif python3 -c "import huggingface_hub" 2>/dev/null; then
  run_python_download python3
else
  echo "Install huggingface_hub in project venv: pip install -U huggingface_hub" >&2
  echo "  Then run: cd vllm && ./download-refine-model.sh" >&2
  exit 1
fi

echo ""
echo "Done. Start vLLM:"
echo "  cd $(pwd) && ./start.sh"
