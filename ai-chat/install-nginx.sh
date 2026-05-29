#!/usr/bin/env bash
# nginx에 AI Chat 사이트 등록 (포트 8088)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
CONF_SRC="$ROOT/nginx/ai-chat.conf"
CONF_DST="/etc/nginx/sites-available/ai-chat"
ENABLED="/etc/nginx/sites-enabled/ai-chat"

if [[ "$(id -u)" -ne 0 ]]; then
  echo "sudo로 실행하세요: sudo $0"
  exit 1
fi

cp "$CONF_SRC" "$CONF_DST"
ln -sf "$CONF_DST" "$ENABLED"

nginx -t
systemctl reload nginx

echo "AI Chat nginx 설정 완료."
echo "  웹 UI:  http://$(hostname -I | awk '{print $1}'):8088/chat/"
echo "  API:    http://$(hostname -I | awk '{print $1}'):8088/v1/"
