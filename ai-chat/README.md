# AI Chat (DGX Spark + nginx + OpenAI API)

## 선정 모델 (2026 — Gemma 4 chat)

| 항목 | 값 |
|------|-----|
| **모델** | google/gemma-4-26b-a4b-it |
| **컨테이너** | `vllm-gemma-chat` |
| **이유** | DGX Spark(128GB, GB10)에서 OCR(`vllm-qwen-vl-ocr`)과 동시 구동 가능한 경량 MoE, 한국어 대화·요약 균형 |
| **서빙** | `vllm/vllm-openai:gemma4-cu130` (CUDA 13 / GB10 sm_121) |
| **API 모델 ID** | `gemma-chat` |
| **포트** | vLLM `:8000` → nginx `:8088` |

OCR 스택과 동시 사용 시 GPU util은 `docker-compose.yaml`의 `CHAT_GPU_MEMORY_UTIL`(기본 0.38), `CHAT_MAX_MODEL_LEN`(기본 32768)로 조정합니다.

> 레거시 80B Thinking 스택(`vllm-thinking-chat`, 모델 ID `qwen-thinking-chat`)은 `--profile thinking` 으로만 기동됩니다.

## 아키텍처

```
브라우저 / 클라이언트
        │
        ▼  :8088
  ai-chat-nginx (Docker)
    ├─ /chat/  → 정적 웹 UI
    └─ /v1/    → vllm-gemma-chat:8000 (vLLM OpenAI API)
```

## 빠른 시작

vLLM 백엔드는 `ocr-server/vllm/` 에 있습니다.

```bash
chmod +x /home/wslaw/ocr-server/vllm/*.sh /home/wslaw/ocr-server/ai-chat/start.sh
/home/wslaw/ocr-server/ai-chat/start.sh
```

vLLM만 기동: `/home/wslaw/ocr-server/vllm/start.sh`

vLLM 첫 기동은 모델 로딩에 **약 8~15분** 걸릴 수 있습니다.

```bash
docker logs -f vllm-gemma-chat
curl -s http://127.0.0.1:8088/v1/models | jq '.data[].id'   # "gemma-chat"
```

## OpenAI 호환 API

```bash
curl http://localhost:8088/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer local" \
  -d '{
    "model": "gemma-chat",
    "messages": [{"role": "user", "content": "안녕하세요"}],
    "max_tokens": 512
  }'
```

Python:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8088/v1", api_key="local")
r = client.chat.completions.create(
    model="gemma-chat",
    messages=[{"role": "user", "content": "안녕"}],
)
print(r.choices[0].message.content)
```

> Django 운영 서버 연동(OCR + AI Chat): [../docs/DJANGO_INTEGRATION.md](../docs/DJANGO_INTEGRATION.md)

## 호스트 nginx 사용 (선택)

시스템 nginx에 등록하려면:

```bash
sudo /home/wslaw/ocr-server/ai-chat/install-nginx.sh
```

## systemd (선택)

```bash
sudo cp /home/wslaw/ocr-server/ai-chat/vllm-ai-chat.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now vllm-ai-chat
```
