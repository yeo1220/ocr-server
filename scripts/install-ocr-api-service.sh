#!/usr/bin/env bash
# Install or refresh ocr-api systemd unit (requires sudo).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
UNIT_SRC="${ROOT}/ocr-api.service"
UNIT_DST="/etc/systemd/system/ocr-api.service"

if [ ! -f "$UNIT_SRC" ]; then
  echo "ERROR: missing $UNIT_SRC" >&2
  exit 1
fi

sudo cp "$UNIT_SRC" "$UNIT_DST"
sudo systemctl daemon-reload
sudo systemctl enable ocr-api.service
sudo systemctl restart ocr-api.service
systemctl status ocr-api.service --no-pager
echo
echo "Health:"
curl -sf "http://127.0.0.1:8001/health" && echo
