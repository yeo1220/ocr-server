# vLLM on DGX Spark (128GB unified)

## 권장 모델 (2026 — VL OCR 전환)

| 용도 | HuggingFace | API ID | 포트 | GPU util |
|------|-------------|--------|------|----------|
| **OCR (PDF 페이지 이미지)** | **Qwen/Qwen2.5-VL-32B-Instruct-AWQ** | `qwen-vl-ocr` | 8003 | 0.55 |
| ai-chat | google/gemma-4-26b-a4b-it | `gemma-chat` | 8000 | 0.38 |
| (호스트) ocr-server | — | — | 8001 | — |

공개 **Qwen2.5-VL-32B** dense 가중치는 없습니다. 한글 행정 PDF·표·DocVQA 기준으로 **Qwen2.5-VL-32B**가 가장 적합합니다.  
OOM 시 `VL_HF_MODEL=Qwen/Qwen2.5-VL-32B-Instruct-AWQ` 로 전환하세요.

레거시(Paddle + 14B refine + 80B chat): `OCR_BACKEND=paddle`, `docker compose --profile refine --profile thinking up -d`

## 설치

```bash
cd /home/wslaw/ocr-server/vllm
cp .env.example .env   # HF_TOKEN

chmod +x download-chat-model.sh download-vl-model.sh start.sh
./download-chat-model.sh
./download-vl-model.sh

./start.sh
cd ../ai-chat && ./start.sh
source ../env.sh && ../restart.sh   # OCR API
```

## 확인

```bash
curl -s http://127.0.0.1:8003/v1/models | jq '.data[].id'   # "qwen-vl-ocr"
curl -s http://127.0.0.1:8000/v1/models | jq '.data[].id'   # "gemma-chat"
curl -s http://127.0.0.1:8001/health | jq '.ocr_backend, .vllm_vl_reachable'
```

## 파인튜닝

한글 재결서 9열 표는 VL **LoRA/전체 파인튜닝** 데이터(페이지 이미지 + JSON 라벨)가 있으면 정확도가 크게 오릅니다.  
이 저장소는 추론 파이프라인만 제공합니다.
