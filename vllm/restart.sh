#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
docker compose down
docker compose up -d
echo "vLLM 로딩: docker logs -f vllm-qwen-ocr"
