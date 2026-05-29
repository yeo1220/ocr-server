#!/usr/bin/env bash
# systemd / production entrypoint (no pkill)
sudo cp /home/wslaw/ocr-server/ocr-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl restart ocr-api
sudo systemctl status ocr-api --no-pager