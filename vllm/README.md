# vLLM Qwen OCR (DGX Spark)

`vllm-qwen-ocr` 컨테이너 — **Qwen3-Next-80B-A3B-Thinking-FP8**, API 모델 ID `qwen-ocr`.

## 빠른 시작

```bash
# 최초 1회: HF 토큰 (기존 ~/vllm/.env 이 있으면 복사)
cp .env.example .env   # 또는 cp ~/vllm/.env .env

chmod +x start.sh stop.sh restart.sh
./start.sh
```

모델 가중치는 `models/Qwen3-Next-80B-A3B-Thinking-FP8/` 에 두세요.  
이전 경로(`~/vllm/models`)에서 이관 시 심볼릭 링크로 충분합니다:

```bash
ln -sfn /home/wslaw/vllm/models/Qwen3-Next-80B-A3B-Thinking-FP8 \
  models/Qwen3-Next-80B-A3B-Thinking-FP8
```

## AI Chat (nginx :8088)

웹 채팅·OpenAI 프록시는 상위 `ai-chat/` 스택과 함께 기동합니다:

```bash
/home/wslaw/ocr-server/ai-chat/start.sh
```

## 이관 (`~/vllm` → 여기)

```bash
# 1) 기존 컨테이너 중지
cd ~/vllm && docker compose down

# 2) 설정·모델 링크
cp ~/vllm/.env /home/wslaw/ocr-server/vllm/.env
ln -sfn ~/vllm/models/Qwen3-Next-80B-A3B-Thinking-FP8 \
  /home/wslaw/ocr-server/vllm/models/Qwen3-Next-80B-A3B-Thinking-FP8

# 3) 새 위치에서 기동
/home/wslaw/ocr-server/vllm/start.sh
/home/wslaw/ocr-server/ai-chat/start.sh
```

`~/vllm` 은 더 이상 필요 없으면 백업 후 제거해도 됩니다(모델을 실제로 옮기지 않고 symlink만 쓴 경우 `models/` 디렉터리는 유지).
