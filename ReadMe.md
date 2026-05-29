# OCR Server (DGX Spark)

NVIDIA DGX Spark(GB10, ARM64, CUDA 13) 환경에 최적화된 **스캔 PDF OCR API** 서버입니다.  
PaddleOCR PP-OCRv5(한국어) + 문서 보정(회전/왜곡) + GPU 추론을 FastAPI로 제공합니다.

## 주요 기능

- PDF / 이미지 OCR (한국어, PP-OCRv5)
- 스캔 PDF용 고해상도 렌더링 (기본 300 DPI, 병렬 처리)
- 스캔본 전처리 (노이즈 제거, adaptive threshold)
- 문서 방향 보정 및 왜곡 보정
- GB10 GPU 추론 (미지원 시 CPU 자동 fallback)
- 페이지별 텍스트, bounding box, confidence score 반환

## 프로젝트 구조

```
ocr-server/
├── app.py              # FastAPI 엔드포인트
├── config.py           # 환경변수 설정
├── ocr_engine.py       # PaddleOCR 초기화 및 추론
├── pdf_utils.py        # PDF → 이미지 변환
├── preprocess.py       # 스캔 이미지 전처리
├── start.sh            # 서버 실행 스크립트
├── log.sh              # systemd 로그 확인
├── ocr-api.service     # systemd 유닛 파일
├── scripts/
│   ├── build_paddle_gpu.sh   # ARM64 GPU PaddlePaddle 빌드
│   └── verify_gpu.py         # GPU / OCR 동작 검증
└── requirements.txt
```

## 사전 요구사항

| 항목 | 버전 / 비고 |
|------|-------------|
| OS | Ubuntu 24.04 (DGX OS) |
| CPU | ARM64 (aarch64) |
| GPU | NVIDIA GB10 (sm_121) |
| CUDA | 13.0 |
| Python | 3.12 |
| cuDNN | 9.x (`~/local/cudnn` 또는 시스템 설치) |

> 공식 pip wheel은 ARM64 GPU를 지원하지 않습니다. GPU 사용 시 [PaddlePaddle GPU 빌드](#paddlepaddle-gpu-빌드-dgx-spark)가 필요합니다.

## 설치

```bash
cd /home/wslaw/ocr-server

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

GPU PaddlePaddle이 이미 설치되어 있다면 다음으로 검증합니다.

```bash
export LD_LIBRARY_PATH="$HOME/local/cudnn/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH"
python scripts/verify_gpu.py
```

## PaddlePaddle GPU 빌드 (DGX Spark)

ARM64 + CUDA 13 + sm_121용 wheel을 소스에서 빌드합니다. (최초 1회, 약 30분~2시간)

```bash
# wheel 빌드만
./scripts/build_paddle_gpu.sh

# 빌드 후 venv에 설치
./scripts/build_paddle_gpu.sh --install
```

스크립트는 sudo 없이 다음을 자동 처리합니다.

- cuDNN, patchelf 로컬 설치 (`~/local/`)
- CMake/Ninja 빌드 (NCCL/TensorRT 비활성)
- wheel 생성: `~/Paddle/build/python/dist/paddlepaddle_gpu-*.whl`

이미 컴파일된 상태에서 wheel만 다시 만들려면:

```bash
export PATH="$HOME/local/patchelf/usr/bin:$HOME/wslaw/ocr-server/.venv/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/local/cudnn/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH"
cd ~/Paddle/build
.venv/bin/ninja python/build/.timestamp_wheel
pip install build/python/dist/paddlepaddle_gpu-*.whl
```

## 서버 실행

### 수동 실행

```bash
./run.sh
```

기본 포트: `8001`

### systemd 등록

```bash
sudo cp ocr-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable ocr-api
sudo systemctl start ocr-api
sudo systemctl status ocr-api
```

로그 확인:

```bash
./log.sh
# 또는
journalctl -u ocr-api -f
```

## API

### `GET /health`

서버 및 GPU 상태 확인

```bash
curl http://localhost:8001/health
```

응답 예시:

```json
{
  "status": "ok",
  "device": "gpu:0",
  "paddle_cuda": true,
  "ocr_dpi_default": 300,
  "ocr_workers": 8
}
```

### `POST /ocr`

PDF 또는 이미지 OCR

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `file` | file | (필수) | PDF 또는 이미지 파일 |
| `mode` | string | `scan` | `scan`: 전처리 적용, `raw`: 원본 그대로 |
| `dpi` | int | `300` | PDF 렌더 해상도 (PDF만 해당) |

```bash
# 스캔 PDF (권장)
curl -F "file=@document.pdf" \
     -F "mode=scan" \
     -F "dpi=300" \
     http://localhost:8001/ocr

# 원본 이미지 (전처리 없음)
curl -F "file=@scan.png" \
     -F "mode=raw" \
     http://localhost:8001/ocr
```

응답 예시:

```json
{
  "job_id": "...",
  "filename": "document.pdf",
  "mode": "scan",
  "dpi": 300,
  "device": "gpu:0",
  "preprocessed": true,
  "processing_time_sec": 0.74,
  "page_count": 1,
  "text": "인식된 전체 텍스트",
  "pages": [
    {
      "page": 1,
      "text": "페이지 텍스트",
      "avg_score": 0.9434,
      "blocks": [
        {
          "text": "텍스트",
          "score": 0.94,
          "box": [[95, 257], [315, 254], [315, 292], [96, 296]]
        }
      ]
    }
  ],
  "upload_path": "/tmp/ocr_uploads/{job_id}.pdf",
  "result_path": "/tmp/ocr_results/{job_id}.json",
  "page_image_paths": ["/tmp/ocr_pages/{job_id}_1.png"]
}
```

> Django 운영 서버 연동 상세 메뉴얼: [docs/DJANGO_INTEGRATION.md](docs/DJANGO_INTEGRATION.md)

## 환경변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `OCR_DEVICE` | `gpu:0` | 추론 디바이스 (`gpu:0`, `cpu`) |
| `OCR_DPI` | `300` | PDF 렌더 DPI |
| `OCR_WORKERS` | `8` | PDF 페이지 병렬 렌더 스레드 수 |
| `OCR_CPU_THREADS` | `20` | CPU fallback 시 Paddle 스레드 수 |
| `OCR_DET_LIMIT_SIDE_LEN` | `960` | 텍스트 검출 최대 변 길이 |
| `OCR_MAX_PAGE_PIXELS` | `25000000` | 페이지당 최대 픽셀 (메모리 보호) |
| `OCR_UPLOAD_DIR` | `/tmp/ocr_uploads` | 업로드 파일 보관 디렉터리 |
| `OCR_PAGE_DIR` | `/tmp/ocr_pages` | PDF 렌더/전처리 이미지 디렉터리 |
| `OCR_RESULT_DIR` | `/tmp/ocr_results` | OCR 결과 JSON 저장 디렉터리 |
| `CUDA_VISIBLE_DEVICES` | `0` | 사용할 GPU 번호 |
| `LD_LIBRARY_PATH` | (cuDNN 경로) | cuDNN 라이브러리 경로 |

## 처리 파이프라인

```
PDF/이미지 업로드
    ↓
PDF → PNG (300 DPI, 병렬 렌더)
    ↓
[mode=scan] 노이즈 제거 + 이진화
    ↓
PaddleOCR (방향 보정 + 왜곡 보정 + PP-OCRv5 한국어)
    ↓
JSON 응답 (텍스트, blocks, score) + /tmp/ocr_results/{job_id}.json 저장
```

## AI Chat (LLM + nginx)

DGX Spark에 맞춘 **Qwen3-Next-80B-A3B-Thinking-FP8** 모델을 vLLM으로 서빙하고, nginx로 웹 채팅·OpenAI 호환 API를 제공합니다.

| 항목 | URL |
|------|-----|
| 웹 채팅 | `http://<호스트>:8088/chat/` |
| OpenAI API | `http://<호스트>:8088/v1/` |
| 모델 ID | `qwen-ocr` |

상세: **[ai-chat/README.md](ai-chat/README.md)**

```bash
/home/wslaw/ocr-server/ai-chat/start.sh   # vllm + nginx
# vLLM만: /home/wslaw/ocr-server/vllm/start.sh
```

vLLM Docker 상세: **[vllm/README.md](vllm/README.md)**

## Django 연동

Django 운영 서버에서 OCR API를 호출하는 방법(설정, 클라이언트 코드, Celery 비동기, 트러블슈팅):

**[docs/DJANGO_INTEGRATION.md](docs/DJANGO_INTEGRATION.md)**

## 문제 해결

### `paddle_cuda: false` / CPU fallback

```bash
python -c "import paddle; print(paddle.is_compiled_with_cuda())"
python scripts/verify_gpu.py
```

- `False`이면 GPU wheel 재설치: `./scripts/build_paddle_gpu.sh --install`
- cuDNN 오류 시 `LD_LIBRARY_PATH`에 `~/local/cudnn/usr/lib/aarch64-linux-gnu` 포함 확인

### `libcudnn.so` not found

```bash
export LD_LIBRARY_PATH="$HOME/local/cudnn/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH"
```

`start.sh` 및 `ocr-api.service`에 이미 설정되어 있습니다.

### 첫 요청이 느림

서버 시작 시 모델 warm-up을 수행합니다. PP-OCRv5 및 문서 보정 모델이 `~/.paddlex/official_models/`에 캐시됩니다.

### systemd 반영 후에도 구버전 동작

```bash
sudo systemctl restart ocr-api
journalctl -u ocr-api -n 50
```

`Creating model` 로그와 `device=gpu:0` 확인.
