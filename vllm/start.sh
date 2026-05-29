#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
docker compose up -d
echo "vLLM 로딩: docker logs -f vllm-qwen-ocr"
echo "API 확인:   curl -s http://127.0.0.1:8000/v1/models"
