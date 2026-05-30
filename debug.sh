#!/usr/bin/env bash
# 원격 debugpy 디버깅용 시작 스크립트
# Cursor/VS Code에서 "Attach to OCR Server (debugpy)" 구성으로 연결한다.
#
# 사용법:
#   ./debug.sh              # 기본: 클라이언트 연결 대기 없이 즉시 시작
#   ./debug.sh --wait       # 클라이언트가 연결될 때까지 서버 시작을 차단
#
# debugpy 수신 포트: 5678 (0.0.0.0 — SSH 터널 또는 내부망 모두 접근 가능)
set -euo pipefail

cd "$(dirname "$0")"
[ -f ./env.sh ] && source ./env.sh

_cudnn_lib="${HOME}/local/cudnn/usr/lib/aarch64-linux-gnu"
if [ -d "$_cudnn_lib" ]; then
  case ":${LD_LIBRARY_PATH:-}:" in
    *":${_cudnn_lib}:"*) ;;
    *) export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}${_cudnn_lib}" ;;
  esac
fi

WAIT_FLAG=""
if [[ "${1:-}" == "--wait" ]]; then
  WAIT_FLAG="--wait-for-client"
  echo "[debug.sh] debugpy: 클라이언트 연결 대기 중 (port 5678)..."
else
  echo "[debug.sh] debugpy: port 5678 수신 중 (클라이언트 없이 즉시 시작)"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

exec .venv/bin/python -m debugpy \
  --listen 0.0.0.0:5678 \
  $WAIT_FLAG \
  -m uvicorn app:app \
  --host 0.0.0.0 \
  --port 8001 \
  --workers 1
