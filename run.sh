#!/usr/bin/env bash
# systemd / production entrypoint (no pkill)
set -euo pipefail

cd "$(dirname "$0")"
[ -f ./env.sh ] && source ./env.sh

# systemd가 프로세스를 관리하므로 pkill 사용 금지 (최소 PATH에서 pkill이 실패하거나 자기 자신을 죽임)
_cudnn_lib="${HOME}/local/cudnn/usr/lib/aarch64-linux-gnu"
if [ -d "$_cudnn_lib" ]; then
  case ":${LD_LIBRARY_PATH:-}:" in
    *":${_cudnn_lib}:"*) ;;
    *) export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}${_cudnn_lib}" ;;
  esac
fi
unset _cudnn_lib

if [ -x .venv/bin/uvicorn ]; then
  exec .venv/bin/uvicorn app:app --host 0.0.0.0 --port 8001 --workers 1
fi

if command -v uvicorn >/dev/null 2>&1; then
  exec uvicorn app:app --host 0.0.0.0 --port 8001 --workers 1
fi

echo "ERROR: uvicorn not found. Create .venv and install requirements first." >&2
exit 1
