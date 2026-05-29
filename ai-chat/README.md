# AI Chat (DGX Spark + nginx + OpenAI API)

## 선정 모델

| 항목 | 값 |
|------|-----|
| **모델** | [Qwen3-Next-80B-A3B-Thinking-FP8](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Thinking-FP8) |
| **이유** | DGX Spark(128GB 통합 메모리, GB10)에 맞는 MoE(80B 총량, 토큰당 ~3B 활성), FP8로 ~77GB, 한국어·추론·코딩 균형 |
| **서빙** | `hellohal2064/vllm-dgx-spark-gb10` (CUDA 13 / sm_121) |
| **API 모델 ID** | `qwen-ocr` (Hermes `config.yaml`과 동일) |

OCR 서버(~15GB)와 동시 사용 시 `GPU_MEMORY_UTIL=0.72`, `MAX_MODEL_LEN=32768`로 메모리를 나눕니다.

## 아키텍처

```
브라우저 / 클라이언트
        │
        ▼  :8088
  ai-chat-nginx (Docker)
    ├─ /chat/  → 정적 웹 UI
    └─ /v1/    → host:8000 (vLLM OpenAI API)
```

## 빠른 시작

```bash
chmod +x /home/wslaw/ocr-server/ai-chat/start.sh
/home/wslaw/ocr-server/ai-chat/start.sh
```

vLLM 첫 기동은 모델 로딩에 **약 8~15분** 걸릴 수 있습니다.

```bash
docker logs -f vllm-qwen-ocr
curl -s http://127.0.0.1:8000/v1/models
```

## OpenAI 호환 API

```bash
curl http://localhost:8088/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer local" \
  -d '{
    "model": "qwen-ocr",
    "messages": [{"role": "user", "content": "안녕하세요"}],
    "max_tokens": 512
  }'
```

Python:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8088/v1", api_key="local")
r = client.chat.completions.create(
    model="qwen-ocr",
    messages=[{"role": "user", "content": "안녕"}],
)
print(r.choices[0].message.content)
```

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
