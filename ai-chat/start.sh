#!/usr/bin/env bash
# AI Chat 스택: vLLM(백엔드) + nginx(웹/API 프록시)
set -euo pipefail

OCR_SERVER_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VLLM_DIR="${OCR_SERVER_DIR}/vllm"
AI_CHAT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "==> vLLM 시작 (${VLLM_DIR})"
cd "$VLLM_DIR"
docker compose up -d

echo "==> nginx AI Chat 프록시 시작"
cd "$AI_CHAT_DIR"
docker compose up -d

IP="$(hostname -I | awk '{print $1}')"
echo ""
echo "준비되면 접속:"
echo "  웹 채팅:  http://${IP}:8088/chat/"
echo "  OpenAI:  http://${IP}:8088/v1/"
echo ""
echo "vLLM 로딩 상태: docker logs -f vllm-qwen-ocr"
echo "모델 API 확인:  curl -s http://127.0.0.1:8000/v1/models"
