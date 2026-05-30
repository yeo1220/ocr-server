#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

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

CHAT_DIR="${CHAT_MODEL_DIR:-gemma-4-26b-a4b-it}"
VL_DIR="${VL_MODEL_DIR:-VARCO-VISION-2.0-14B}"

missing=0
if [[ ! -f "models/${CHAT_DIR}/config.json" ]]; then
  echo "Chat 모델 없음: models/${CHAT_DIR}"
  echo "  ./download-chat-model.sh"
  missing=1
fi
if [[ ! -f "models/${VL_DIR}/config.json" ]]; then
  echo "VL OCR 모델 없음: models/${VL_DIR}"
  echo "  ./download-vl-model.sh"
  missing=1
fi
if [[ "${missing}" -eq 1 ]]; then
  exit 1
fi

export CHAT_GPU_MEMORY_UTIL="${CHAT_GPU_MEMORY_UTIL:-0.38}"
export VL_GPU_MEMORY_UTIL="${VL_GPU_MEMORY_UTIL:-0.55}"

docker compose up -d vllm-chat vllm-ocr-vl

echo ""
echo "Gemma chat:  curl -s http://127.0.0.1:8000/v1/models   # gemma-chat"
echo "VL OCR:      curl -s http://127.0.0.1:8003/v1/models   # qwen-vl-ocr"
echo "OCR API:     curl -s http://127.0.0.1:8001/health"
echo "ai-chat:     cd ../ai-chat && ./start.sh"

docker ps -a