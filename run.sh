#!/usr/bin/env bash
# systemd / production entrypoint (no pkill)
set -euo pipefail

cd "$(dirname "$0")"
[ -f ./env.sh ] && source ./env.sh

pkill -f "uvicorn app:app" 2>/dev/null || true
sleep 1

if [ -x .venv/bin/uvicorn ]; then
  exec .venv/bin/uvicorn app:app --host 0.0.0.0 --port 8001 --workers 1
fi

if command -v uvicorn >/dev/null 2>&1; then
  exec uvicorn app:app --host 0.0.0.0 --port 8001 --workers 1
fi

echo "ERROR: uvicorn not found. Create .venv and install requirements first." >&2
exit 1
