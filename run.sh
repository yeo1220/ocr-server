#!/usr/bin/env bash
# systemd / production entrypoint (no pkill)
cd "$(dirname "$0")"
source ./env.sh

pkill -f "uvicorn app:app" 2>/dev/null || true
sleep 1

exec .venv/bin/uvicorn app:app --host 0.0.0.0 --port 8001 --workers 1
