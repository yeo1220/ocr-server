# Django 운영 서버 ↔ DGX Spark (OCR + AI Chat) 연동 메뉴얼

Django 운영 서버에서 DGX Spark에 새로 세팅된 두 vLLM 서비스를 사용해
**OCR 분석**과 **AI Chat**을 처리하는 방법을 정리한 문서입니다.  
서버 자체 설치·운영은 [ReadMe.md](../ReadMe.md), vLLM 도커 구성은 [vllm/README.md](../vllm/README.md)를 참고하세요.

| 신규 vLLM 서비스 | 컨테이너 | 모델 ID | 포트 | 용도 | Django가 직접 호출? |
|------------------|----------|---------|------|------|---------------------|
| Vision OCR | `vllm-qwen-vl-ocr` | `qwen-vl-ocr` | `8003` | PDF/이미지 → 텍스트·표 | ❌ (OCR API `:8001` 경유) |
| AI Chat | `vllm-gemma-chat` | `gemma-chat` | `8000` | 대화/요약/질의응답 | ✅ (`:8088` nginx 경유) |

> 두 서비스 모두 **OpenAI 호환 API**(`/v1/chat/completions`)를 제공합니다.  
> OCR은 `vllm-qwen-vl-ocr`를 직접 부르지 않고, PDF→이미지 변환·표 구조화를 담당하는 **OCR API(FastAPI `:8001`)** 를 호출합니다.  
> Chat은 `vllm-gemma-chat` 앞단의 **ai-chat nginx(`:8088`)** 를 통해 호출합니다.

---

## 1. 시스템 구성

```
                              ┌──────────────────────────────────────────────┐
                              │            DGX Spark (GB10, 128GB)             │
                              │                                                │
  ┌─────────────────┐  /ocr   │  ┌────────────────┐     ┌────────────────────┐ │
  │                 │ ───────▶│  │ OCR API :8001  │────▶│ vllm-qwen-vl-ocr   │ │
  │  Django 운영     │  멀티파트 │  │ (FastAPI)      │ VL  │ :8003 qwen-vl-ocr  │ │
  │  서버            │ ◀─────── │  │ PDF→PNG, 표복원 │     └────────────────────┘ │
  │ (Gunicorn/      │  JSON    │  └────────────────┘                            │
  │  Celery)        │          │                                                │
  │                 │ /v1/chat │  ┌────────────────┐     ┌────────────────────┐ │
  │                 │ ───────▶│  │ ai-chat nginx  │────▶│ vllm-gemma-chat    │ │
  │                 │  OpenAI  │  │ :8088          │     │ :8000 gemma-chat   │ │
  │                 │ ◀─────── │  └────────────────┘     └────────────────────┘ │
  └─────────────────┘          └──────────────────────────────────────────────┘
```

| 구분 | OCR 분석 | AI Chat |
|------|----------|---------|
| Django 호출 대상 | OCR API `http://<dgx>:8001/ocr` | ai-chat `http://<dgx>:8088/v1/chat/completions` |
| 백엔드 | `vllm-qwen-vl-ocr` (`OCR_BACKEND=vllm_vl`) | `vllm-gemma-chat` |
| 프로토콜 | `multipart/form-data` 업로드 | OpenAI 호환 JSON (`requests`/`httpx`/`openai`) |
| 모델 ID | `qwen-vl-ocr` (서버 내부) | `gemma-chat` |
| 타임아웃 | **120~600초** (페이지·DPI에 비례) | **60~120초** (스트리밍 권장) |
| 권장 처리 | Celery 비동기 | 동기 또는 스트리밍 |

> Django 서버에는 PaddleOCR/모델을 설치할 필요 **없습니다**. 두 HTTP API만 호출하면 됩니다.

---

## 2. 사전 확인

### 2-1. OCR API 서버 상태 (`:8001`)

```bash
curl http://192.168.0.67:8001/health
```

정상 응답 예시 (`app.py` `/health` 기준):

```json
{
  "status": "ok",
  "ocr_backend": "vllm_vl",
  "device": "vllm_vl",
  "ocr_dpi_default": 400,
  "vllm_enabled": true,
  "vllm_chat_reachable": true,
  "vllm_chat_model": "gemma-chat",
  "vllm_vl_base_url": "http://127.0.0.1:8003/v1",
  "vllm_vl_model": "qwen-vl-ocr",
  "vllm_vl_reachable": true,
  "vllm_vl_max_image_side": 2048
}
```

| 필드 | 의미 |
|------|------|
| `status` | `"ok"` 이면 OCR API 가동 중 |
| `ocr_backend` | `"vllm_vl"` (신규 기본값, `vllm-qwen-vl-ocr` 사용) 또는 `"paddle"` (레거시) |
| `vllm_vl_reachable` | `true` — `vllm-qwen-vl-ocr`(`:8003`) 도달 가능 여부. **OCR 요청 전 필수 확인** |
| `vllm_vl_model` | `"qwen-vl-ocr"` (Vision OCR 모델 ID) |
| `vllm_chat_reachable` | `true` — `vllm-gemma-chat`(`:8088`/`:8000`) 도달 가능 여부 |
| `device` | VL: `"vllm_vl"`, Paddle: `"gpu:0"` / `"cpu"` |

### 2-2. AI Chat 서버 상태 (`:8088`)

```bash
# 모델 목록 → "gemma-chat" 확인
curl -s http://192.168.0.67:8088/v1/models | jq '.data[].id'

# 헬스
curl -s http://192.168.0.67:8088/health
```

### 2-3. 요청 테스트

```bash
# OCR
curl -X POST http://192.168.0.67:8001/ocr \
  -F "file=@sample.pdf" -F "mode=scan" -F "dpi=400"

# Chat (OpenAI 호환)
curl http://192.168.0.67:8088/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer local" \
  -d '{"model":"gemma-chat","messages":[{"role":"user","content":"안녕하세요"}],"max_tokens":256}'
```

### 2-4. 네트워크

- Django 서버 → DGX Spark **8001/TCP**(OCR), **8088/TCP**(Chat) 방화벽 허용
- 각 서비스는 `0.0.0.0` 바인딩 상태여야 함 (OCR: `start.sh`/`run.sh`, Chat: `ai-chat/docker-compose.yaml`)
- 내부망 IP 사용 권장 (예: `http://192.168.0.67:8001`, `http://192.168.0.67:8088`)

---

## 3. Django 설정

### 3-1. `settings.py`

```python
# settings.py

# --- OCR API (vllm-qwen-vl-ocr 백엔드) ---
OCR_API_BASE_URL = env("OCR_API_BASE_URL", default="http://192.168.0.67:8001")
OCR_API_TIMEOUT = env.int("OCR_API_TIMEOUT", default=300)       # 초
OCR_DEFAULT_MODE = env("OCR_DEFAULT_MODE", default="scan")      # scan | raw
OCR_DEFAULT_DPI = env.int("OCR_DEFAULT_DPI", default=400)
OCR_MAX_FILE_SIZE = env.int("OCR_MAX_FILE_SIZE", default=50 * 1024 * 1024)  # 50MB

# --- AI Chat (vllm-gemma-chat, OpenAI 호환) ---
CHAT_API_BASE_URL = env("CHAT_API_BASE_URL", default="http://192.168.0.67:8088/v1")
CHAT_API_KEY = env("CHAT_API_KEY", default="local")            # vLLM은 임의 키 허용
CHAT_MODEL = env("CHAT_MODEL", default="gemma-chat")
CHAT_API_TIMEOUT = env.int("CHAT_API_TIMEOUT", default=120)    # 초 (스트리밍 권장)
CHAT_MAX_TOKENS = env.int("CHAT_MAX_TOKENS", default=1024)
CHAT_TEMPERATURE = env.float("CHAT_TEMPERATURE", default=0.3)
```

`.env` 예시:

```env
OCR_API_BASE_URL=http://192.168.0.67:8001
OCR_API_TIMEOUT=300
OCR_DEFAULT_MODE=scan
OCR_DEFAULT_DPI=400

CHAT_API_BASE_URL=http://192.168.0.67:8088/v1
CHAT_API_KEY=local
CHAT_MODEL=gemma-chat
CHAT_API_TIMEOUT=120
```

> `CHAT_API_BASE_URL`은 nginx 경유(`:8088/v1`)를 권장합니다. nginx 없이 직접 호출하려면 `http://<dgx>:8000/v1` 로도 동일하게 동작합니다(같은 OpenAI API).

### 3-2. 의존 패키지

```bash
pip install requests        # OCR/Chat 공통 HTTP
pip install openai          # (선택) Chat OpenAI SDK 사용 시
pip install httpx           # (선택) 비동기/스트리밍
```

`requirements.txt`에 추가:

```
requests>=2.31.0
openai>=1.30.0
```

---

## 4. API 명세 (Django 연동 관점)

### 4-1. `GET /health`

헬스체크·모니터링용. Celery beat 또는 Django middleware에서 주기적으로 호출 가능.

### 4-2. `POST /ocr`

**Content-Type:** `multipart/form-data`

| 필드 | 필수 | 타입 | 기본값 | 설명 |
|------|------|------|--------|------|
| `file` | O | file | — | PDF 또는 이미지 (`.pdf`, `.png`, `.jpg`, `.jpeg`, `.tiff` 등) |
| `mode` | X | string | `scan` | `scan`: 스캔본 전처리 + 문서 보정, `raw`: 원본 그대로 (전처리는 `paddle` 백엔드에서만 동작) |
| `preprocess` | X | string | 서버 기본(`enhance`) | `enhance` \| `binary` \| `none` (스캔 전처리 방식, `paddle` 백엔드 전용. `vllm_vl`에서는 무시) |
| `dpi` | X | int | `400` | PDF 렌더 해상도 (PDF만 해당, 서버 `OCR_DPI` 기본값 `400`) |
| `format` | X | string | `text` | `table`: 고정 열 표 JSON (`table_cols` 필수) |
| `table_cols` | X* | int | — | `format=table`일 때 열 개수 (재결서 보상금내역: `9`) |
| `table_header_row` | X | int | `0` | 단일 헤더 행 인덱스 (레거시, `header_rows=1`일 때) |
| `table_header_rows` | X | int | `1` | 헤더로 볼 행 수 (재결서 2단 헤더: `2`) |
| `table_col_boundaries` | X | string | — | 열 경계 X좌표 JSON 배열 (예: `[0,75,150,...]`) |
| `refine` | X | string | 서버 기본(`none`) | `none` \| `vllm` (저신뢰 셀 LLM 보정. `vllm_vl` 백엔드에서는 `none`으로 강등) |
| `refine_threshold` | X | float | 서버 기본(`0.85`) | vLLM 보정 대상 신뢰도 상한 (예: `0.85`) |

\* `format=table`이면 `table_cols`(양의 정수) 필수. `table_header_rows`는 `>= 1`이어야 함.

**재결서 Celery(`decision_tasks`) 권장 요청 예시:**

```python
data = {
    "mode": "raw",
    "dpi": "400",
    "format": "table",
    "table_cols": "9",
    "table_header_row": "0",
    "table_header_rows": "2",
    "refine": "vllm",
    "refine_threshold": "0.85",
}
requests.post(f"{OCR_API_BASE_URL}/ocr", files={"file": (...)} , data=data, timeout=(5, 300))
```

**`format=table` 페이지 응답 (Django 파서 호환):**

| 필드 | 설명 |
|------|------|
| `pages[].raw_blocks` | 병합 전 OCR 블록 (좌표 기반 표 복원에 우선 사용) |
| `pages[].blocks` | 줄 단위 병합 블록 |
| `pages[].table.headers` | 헤더 행(2단이면 상·하 병합) |
| `pages[].table.data` | 데이터 행만 |
| `pages[].table.data_refined` | vLLM 보정 후 데이터 행 |
| `pages[].table.rows` | 헤더+데이터 전체 격자 (`all_rows`, Django `len==10` 분기용) |
| `pages[].table.rows_refined` | 헤더+보정 데이터 격자 |

Django는 `data_refined` → `rows_refined` → `data` → `rows` 순으로 표 행을 읽습니다.

**표 보정 동작 (백엔드별):**

- **신규 기본값 `vllm-qwen-vl-ocr` (`ocr_backend=vllm_vl`):** VL 모델이 페이지 이미지를 직접 읽어 표 JSON을 end-to-end로 생성합니다. 이 경우 `refine=vllm` 파라미터는 **무시**되며(`app.py`에서 `none`으로 강등), 별도 보정 LLM이 필요 없습니다. `data_refined`/`rows_refined`에는 `data`/`rows`와 동일한 값이 채워집니다.
- **레거시 `paddle` 백엔드:** PaddleOCR 1차 인식 후 저신뢰 셀을 별도 refine 스택(`qwen-refine`, `:8002`)으로 보정할 때만 `refine=vllm`이 적용됩니다.

> Django 입장에서는 **`OCR_API_BASE_URL`(`:8001`) 하나만** 호출하면 되고, 내부적으로 어떤 백엔드가 쓰이는지는 응답의 `ocr_backend`/`device` 필드로 확인할 수 있습니다.

**성공 응답 (HTTP 200, 기본 백엔드 `vllm_vl`, `format=text`):**

```json
{
  "job_id": "d4d03fae-b3bd-423c-acbb-591433e24286",
  "filename": "document.pdf",
  "mode": "scan",
  "format": "text",
  "dpi": 400,
  "ocr_backend": "vllm_vl",
  "device": "vllm_vl",
  "preprocessed": false,
  "preprocess_mode": null,
  "processing_time_sec": 12.721,
  "page_count": 3,
  "text": "1페이지 텍스트\n\n2페이지 텍스트\n\n3페이지 텍스트",
  "pages": [
    {
      "page": 1,
      "text": "1페이지 텍스트",
      "blocks": [
        { "text": "인식된 문장", "score": 1.0, "box": [] }
      ],
      "raw_blocks": [],
      "avg_score": 1.0,
      "ocr_backend": "vllm_vl",
      "vl_meta": { "model": "qwen-vl-ocr", "finish_reason": "stop" }
    }
  ],
  "upload_path": "/tmp/ocr_uploads/d4d03fae-b3bd-423c-acbb-591433e24286.pdf",
  "result_path": "/tmp/ocr_results/d4d03fae-b3bd-423c-acbb-591433e24286.json",
  "page_image_paths": [
    "/tmp/ocr_pages/d4d03fae-b3bd-423c-acbb-591433e24286_1.png"
  ]
}
```

> `format=table`이면 응답 최상위에 `table_cols`, `table_header_row`, `table_header_rows`, `refine`가 추가되고, 각 `pages[]`에 `table`(및 `text_refined`)이 포함됩니다.  
> 레거시 `paddle` 백엔드에서는 `device`가 `"gpu:0"`/`"cpu"`, `preprocessed`가 `true`(scan 모드 시), `pages[].blocks`에 실제 `box` 좌표·세부 `score`가 채워지고 `preprocess_mode`가 적용된 모드(`enhance` 등)로 설정됩니다.

**응답 필드 설명 (Django에서 활용):**

| 필드 | 설명 |
|------|------|
| `job_id` | OCR 작업 고유 ID (DB PK 또는 외부 참조키로 저장) |
| `format` | 요청한 출력 형식 (`text` \| `table`) |
| `ocr_backend` | 실제 사용된 백엔드 (`vllm_vl` \| `paddle`) |
| `device` | 처리 디바이스 (`vllm_vl` \| `gpu:0` \| `cpu`) |
| `preprocessed` | 스캔 전처리 적용 여부 (`paddle` + `mode=scan`에서만 `true`) |
| `preprocess_mode` | 적용된 전처리 모드 (`enhance` 등, 미적용 시 `null`) |
| `text` | 전체 페이지 텍스트 (`\n\n`로 페이지 구분) |
| `pages[].text` | 페이지별 텍스트 |
| `pages[].blocks` | 텍스트 블록 단위 (좌표 `box`, 신뢰도 `score` 포함. `vllm_vl`은 `box=[]`, `score=1.0`) |
| `pages[].raw_blocks` | 병합 전 OCR 블록 (`vllm_vl`은 빈 배열) |
| `pages[].avg_score` | 페이지 평균 인식 신뢰도 (0~1, `vllm_vl`은 `1.0`) |
| `pages[].ocr_backend` | 해당 페이지를 처리한 백엔드 |
| `pages[].vl_meta` | (VL 백엔드) 모델 메타 정보 |
| `processing_time_sec` | OCR 처리 소요 시간(초) |
| `upload_path` | OCR 서버에 보관된 원본 파일 경로 |
| `result_path` | OCR 서버에 저장된 결과 JSON 경로 |
| `page_image_paths` | 렌더/전처리 PNG 경로 목록 (OCR 서버 로컬) |

> `upload_path`, `result_path`, `page_image_paths`는 **OCR 서버(DGX Spark) 로컬 경로**입니다.  
> Django DB에는 `job_id`와 API 응답 JSON을 저장하고, 파일은 Django `MEDIA`에 별도 보관하는 것을 권장합니다.

**에러 응답:**

| HTTP | 원인 | Django 대응 |
|------|------|-------------|
| 422 | 잘못된 multipart/form 파라미터 (`format=table`인데 `table_cols` 누락, `table_header_rows < 1`, 잘못된 `preprocess`/`refine`/`table_col_boundaries` 등) | 요청 파라미터 검증 |
| 502 | (`format=table`) VL 모델이 해당 페이지의 표 JSON을 반환하지 못함 | 재시도 또는 입력 품질·`dpi` 확인 |
| 503 | Vision OCR(`vllm-qwen-vl-ocr`, `:8003`) 도달 불가 | vLLM 컨테이너 기동 확인(`/health`의 `vllm_vl_reachable`) |
| 500 | OCR 서버 내부 오류 (GPU/cuDNN 등) | 재시도 또는 관리자 알림 |
| 연결 실패 | OCR 서버 다운·방화벽 | `OCR_API_BASE_URL` / 방화벽 확인 |

---

## 5. Django 클라이언트 코드

### 5-1. OCR API 클라이언트 (`ocr_client.py`)

```python
# myapp/ocr_client.py

import logging
from typing import BinaryIO

import requests
from django.conf import settings

logger = logging.getLogger(__name__)


class OcrApiError(Exception):
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class OcrApiClient:
    def __init__(
        self,
        base_url: str | None = None,
        timeout: int | None = None,
    ):
        self.base_url = (base_url or settings.OCR_API_BASE_URL).rstrip("/")
        self.timeout = timeout or settings.OCR_API_TIMEOUT

    def health(self) -> dict:
        resp = requests.get(f"{self.base_url}/health", timeout=10)
        resp.raise_for_status()
        return resp.json()

    def ocr_file(
        self,
        file_obj: BinaryIO,
        filename: str,
        mode: str | None = None,
        dpi: int | None = None,
        *,
        fmt: str | None = None,
        table_cols: int | None = None,
        table_header_row: int | None = None,
        table_header_rows: int | None = None,
        table_col_boundaries: str | None = None,
        refine: str | None = None,
        refine_threshold: float | None = None,
    ) -> dict:
        """
        파일 객체를 OCR API에 전송하고 JSON 결과를 반환합니다.

        Args:
            file_obj: Django UploadedFile 또는 open(..., 'rb')
            filename: 원본 파일명 (확장자 포함)
            mode: 'scan' | 'raw'
            dpi: PDF 렌더 DPI (None이면 OCR 서버 기본값 400)
            fmt: 'text' | 'table' (재결서 표 추출 시 'table')
            table_cols: format='table'일 때 열 개수 (재결서 보상금내역: 9)
            table_header_row / table_header_rows: 헤더 행 인덱스 / 헤더 행 수(2단 헤더: 2)
            table_col_boundaries: 열 경계 X좌표 JSON 배열 문자열
            refine / refine_threshold: 저신뢰 셀 vLLM 보정 옵션(paddle 백엔드 전용)
        """
        mode = mode or settings.OCR_DEFAULT_MODE
        data: dict[str, str] = {"mode": mode}
        if dpi is not None:
            data["dpi"] = str(dpi)
        if fmt is not None:
            data["format"] = fmt
        if table_cols is not None:
            data["table_cols"] = str(table_cols)
        if table_header_row is not None:
            data["table_header_row"] = str(table_header_row)
        if table_header_rows is not None:
            data["table_header_rows"] = str(table_header_rows)
        if table_col_boundaries is not None:
            data["table_col_boundaries"] = table_col_boundaries
        if refine is not None:
            data["refine"] = refine
        if refine_threshold is not None:
            data["refine_threshold"] = str(refine_threshold)

        files = {"file": (filename, file_obj)}

        logger.info(
            "OCR request: filename=%s mode=%s dpi=%s format=%s",
            filename,
            mode,
            dpi,
            fmt,
        )
        resp = requests.post(
            f"{self.base_url}/ocr",
            files=files,
            data=data,
            timeout=self.timeout,
        )

        if resp.status_code != 200:
            logger.error(
                "OCR API error: status=%s body=%s",
                resp.status_code,
                resp.text[:500],
            )
            raise OcrApiError(
                f"OCR API failed: {resp.status_code} {resp.text[:200]}",
                status_code=resp.status_code,
            )

        return resp.json()
```

### 5-2. 동기 View 예시

```python
# myapp/views.py

import json

from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.conf import settings

from .models import OcrJob
from .ocr_client import OcrApiClient, OcrApiError


@require_POST
def upload_and_ocr(request):
    uploaded = request.FILES.get("file")
    if not uploaded:
        return JsonResponse({"error": "file is required"}, status=400)

    if uploaded.size > settings.OCR_MAX_FILE_SIZE:
        return JsonResponse({"error": "file too large"}, status=413)

    mode = request.POST.get("mode", settings.OCR_DEFAULT_MODE)
    dpi = request.POST.get("dpi")
    dpi = int(dpi) if dpi else None

    client = OcrApiClient()

    try:
        result = client.ocr_file(
            file_obj=uploaded,
            filename=uploaded.name,
            mode=mode,
            dpi=dpi,
        )
    except OcrApiError as e:
        return JsonResponse({"error": str(e)}, status=502)
    except requests.exceptions.Timeout:
        return JsonResponse({"error": "OCR timeout"}, status=504)
    except requests.exceptions.ConnectionError:
        return JsonResponse({"error": "OCR server unreachable"}, status=503)

    # DB 저장
    job = OcrJob.objects.create(
        job_id=result["job_id"],
        filename=result["filename"],
        mode=result["mode"],
        page_count=result["page_count"],
        text=result["text"],
        avg_score=_calc_avg_score(result),
        processing_time_sec=result["processing_time_sec"],
        device=result["device"],
        raw_response=result,
    )

    return JsonResponse(
        {
            "job_id": job.job_id,
            "text": job.text,
            "page_count": job.page_count,
            "processing_time_sec": job.processing_time_sec,
        }
    )


def _calc_avg_score(result: dict) -> float:
    scores = [p.get("avg_score", 0) for p in result.get("pages", [])]
    return sum(scores) / len(scores) if scores else 0.0
```

### 5-3. Model 예시

```python
# myapp/models.py

from django.db import models


class OcrJob(models.Model):
    job_id = models.UUIDField(unique=True, db_index=True)
    filename = models.CharField(max_length=512)
    mode = models.CharField(max_length=16, default="scan")
    page_count = models.PositiveIntegerField(default=0)
    text = models.TextField(blank=True)
    avg_score = models.FloatField(null=True, blank=True)
    processing_time_sec = models.FloatField(null=True, blank=True)
    device = models.CharField(max_length=32, blank=True)
    raw_response = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
```

---

## 6. Celery 비동기 처리 (운영 권장)

OCR은 페이지당 수 초~수십 초 걸릴 수 있으므로, **Django view에서 동기 호출하지 말고 Celery task로 위임**하는 것을 권장합니다.

### 6-1. Celery Task

```python
# myapp/tasks.py

import logging

from celery import shared_task
from django.core.files.storage import default_storage

from .models import OcrJob
from .ocr_client import OcrApiClient, OcrApiError

logger = logging.getLogger(__name__)


@shared_task(bind=True, max_retries=2, default_retry_delay=30)
def run_ocr_task(self, ocr_job_pk: int, storage_path: str, filename: str, mode: str = "scan", dpi: int | None = None):
    job = OcrJob.objects.get(pk=ocr_job_pk)
    job.status = "processing"
    job.save(update_fields=["status"])

    client = OcrApiClient()

    try:
        with default_storage.open(storage_path, "rb") as f:
            result = client.ocr_file(f, filename=filename, mode=mode, dpi=dpi)
    except OcrApiError as e:
        job.status = "failed"
        job.error_message = str(e)
        job.save(update_fields=["status", "error_message"])
        if e.status_code and e.status_code >= 500:
            raise self.retry(exc=e)
        return
    except Exception as e:
        job.status = "failed"
        job.error_message = str(e)
        job.save(update_fields=["status", "error_message"])
        raise self.retry(exc=e)

    job.job_id = result["job_id"]
    job.text = result["text"]
    job.page_count = result["page_count"]
    job.processing_time_sec = result["processing_time_sec"]
    job.device = result["device"]
    job.raw_response = result
    job.status = "completed"
    job.save()
```

### 6-2. View에서 Task 호출

```python
@require_POST
def upload_and_ocr_async(request):
    uploaded = request.FILES.get("file")
    if not uploaded:
        return JsonResponse({"error": "file is required"}, status=400)

    # Django MEDIA에 먼저 저장
    path = default_storage.save(f"ocr_uploads/{uploaded.name}", uploaded)

    job = OcrJob.objects.create(filename=uploaded.name, status="pending")

    run_ocr_task.delay(
        ocr_job_pk=job.pk,
        storage_path=path,
        filename=uploaded.name,
        mode=request.POST.get("mode", "scan"),
        dpi=int(request.POST["dpi"]) if request.POST.get("dpi") else None,
    )

    return JsonResponse({"id": job.pk, "status": "pending"}, status=202)
```

### 6-3. Celery 설정

```python
# settings.py
CELERY_TASK_TIME_LIMIT = 600        # hard limit 10분
CELERY_TASK_SOFT_TIME_LIMIT = 540   # soft limit 9분
```

---

## 7. `mode` / `dpi` 선택 가이드

| 문서 유형 | `mode` | `dpi` | 비고 |
|-----------|--------|-------|------|
| 스캔 PDF (권장) | `scan` | `300` | 노이즈 제거 + 이진화 + 방향/왜곡 보정 |
| 깨끗한 디지털 PDF | `raw` | `200` | 빠름, 전처리 생략 |
| 저화질 스캔 | `scan` | `300~400` | DPI↑ = 정확도↑, 시간↑ |
| 대용량 PDF (100p+) | `scan` | `200` | 타임아웃 주의, Celery 필수 |

---

## 8. 타임아웃·재시도 가이드

| 조건 | 권장 `OCR_API_TIMEOUT` |
|------|------------------------|
| GPU, 1~10페이지 | 120초 |
| GPU, 10~50페이지 | 300초 |
| CPU fallback | 600초 이상 |
| Celery task | `CELERY_TASK_TIME_LIMIT=600` |

```python
# requests 재시도 (선택)
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

session = requests.Session()
retry = Retry(total=2, backoff_factor=1, status_forcelist=[502, 503, 504])
session.mount("http://", HTTPAdapter(max_retries=retry))
```

---

## 9. Django Admin / API 폴링

비동기 처리 시 프론트엔드에서 상태 확인:

```python
# myapp/views.py

def ocr_status(request, pk):
    job = OcrJob.objects.get(pk=pk)
    return JsonResponse(
        {
            "id": job.pk,
            "status": job.status,
            "text": job.text if job.status == "completed" else "",
            "page_count": job.page_count,
            "processing_time_sec": job.processing_time_sec,
        }
    )
```

---

## 10. AI Chat 연동 (`vllm-gemma-chat`)

`vllm-gemma-chat`은 **OpenAI 호환 `/v1/chat/completions`** 를 제공합니다.  
Django에서는 `requests`/`openai`/`httpx` 중 무엇을 써도 되며, 별도 API 키 검증이 없으므로 임의의 `Bearer` 토큰(`local`)을 보냅니다.

### 10-1. Chat 클라이언트 (`chat_client.py`)

OCR API 서버의 `vllm_client.py`(`chat_json`)와 동일한 페이로드 구조(`model`, `messages`, `temperature`, `max_tokens`)를 사용합니다.

```python
# myapp/chat_client.py

import json
import logging
from typing import Iterator

import requests
from django.conf import settings

logger = logging.getLogger(__name__)


class ChatApiError(Exception):
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class ChatClient:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: int | None = None,
    ):
        self.base_url = (base_url or settings.CHAT_API_BASE_URL).rstrip("/")
        self.api_key = api_key or settings.CHAT_API_KEY
        self.model = model or settings.CHAT_MODEL
        self.timeout = timeout or settings.CHAT_API_TIMEOUT

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def chat(
        self,
        messages: list[dict],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """동기 호출 → 전체 응답 텍스트 반환."""
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens or settings.CHAT_MAX_TOKENS,
            "temperature": settings.CHAT_TEMPERATURE if temperature is None else temperature,
        }
        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=payload,
                timeout=self.timeout,
            )
        except requests.exceptions.RequestException as e:
            raise ChatApiError(f"chat request failed: {e}") from e

        if resp.status_code != 200:
            logger.error("Chat API error: %s %s", resp.status_code, resp.text[:500])
            raise ChatApiError(resp.text[:200], status_code=resp.status_code)

        body = resp.json()
        return body["choices"][0]["message"]["content"] or ""

    def chat_stream(
        self,
        messages: list[dict],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Iterator[str]:
        """스트리밍 호출 → 토큰 조각(delta)을 순차 yield (SSE)."""
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens or settings.CHAT_MAX_TOKENS,
            "temperature": settings.CHAT_TEMPERATURE if temperature is None else temperature,
            "stream": True,
        }
        with requests.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json=payload,
            timeout=self.timeout,
            stream=True,
        ) as resp:
            if resp.status_code != 200:
                raise ChatApiError(resp.text[:200], status_code=resp.status_code)
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                data = line[len("data: ") :]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                delta = chunk["choices"][0].get("delta", {}).get("content")
                if delta:
                    yield delta
```

> `openai` SDK를 선호하면:
>
> ```python
> from openai import OpenAI
> client = OpenAI(base_url=settings.CHAT_API_BASE_URL, api_key=settings.CHAT_API_KEY)
> r = client.chat.completions.create(model=settings.CHAT_MODEL, messages=[...])
> print(r.choices[0].message.content)
> ```

### 10-2. 동기 Chat View

```python
# myapp/views.py

import json

from django.http import JsonResponse
from django.views.decorators.http import require_POST

from .chat_client import ChatClient, ChatApiError


@require_POST
def chat(request):
    body = json.loads(request.body or "{}")
    messages = body.get("messages")
    if not messages:
        # 단일 질문 형태도 허용
        prompt = body.get("prompt", "").strip()
        if not prompt:
            return JsonResponse({"error": "messages or prompt required"}, status=400)
        messages = [{"role": "user", "content": prompt}]

    client = ChatClient()
    try:
        answer = client.chat(messages, max_tokens=body.get("max_tokens"))
    except ChatApiError as e:
        status = 504 if e.status_code is None else 502
        return JsonResponse({"error": str(e)}, status=status)

    return JsonResponse({"model": client.model, "answer": answer})
```

### 10-3. 스트리밍 Chat View (StreamingHttpResponse)

타이핑 효과(토큰 단위 출력)가 필요하면 SSE를 그대로 프론트로 흘려보냅니다.

```python
# myapp/views.py

from django.http import StreamingHttpResponse

from .chat_client import ChatClient, ChatApiError


@require_POST
def chat_stream(request):
    body = json.loads(request.body or "{}")
    messages = body.get("messages") or [
        {"role": "user", "content": body.get("prompt", "")}
    ]

    client = ChatClient()

    def event_stream():
        try:
            for delta in client.chat_stream(messages):
                yield f"data: {json.dumps({'delta': delta}, ensure_ascii=False)}\n\n"
        except ChatApiError as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        yield "data: [DONE]\n\n"

    resp = StreamingHttpResponse(event_stream(), content_type="text/event-stream")
    resp["Cache-Control"] = "no-cache"
    resp["X-Accel-Buffering"] = "no"   # nginx 버퍼링 방지
    return resp
```

> 스트리밍을 사용하려면 Django를 **ASGI(uvicorn/daphne)** 또는 `gunicorn --worker-class gthread`로 띄우고, 앞단 nginx에서 `proxy_buffering off;` 를 설정하세요.

### 10-4. OCR → Chat 결합 흐름 (요약/질의응답)

OCR로 추출한 텍스트를 그대로 Chat 모델에 넣어 **요약·핵심 추출·질의응답**을 수행하는 것이 가장 흔한 운영 패턴입니다.

```python
# myapp/services.py

from .chat_client import ChatClient

SUMMARY_SYSTEM = (
    "당신은 한국어 행정문서 분석 도우미입니다. "
    "주어진 OCR 텍스트만 근거로 정확하게 답하고, 추측하지 마세요."
)


def summarize_ocr_text(ocr_text: str, instruction: str = "핵심 내용을 5줄로 요약해줘") -> str:
    """OcrJob.text 를 받아 gemma-chat 으로 요약."""
    client = ChatClient()
    messages = [
        {"role": "system", "content": SUMMARY_SYSTEM},
        {"role": "user", "content": f"{instruction}\n\n--- OCR 텍스트 ---\n{ocr_text}"},
    ]
    return client.chat(messages, max_tokens=1024)


def ask_about_document(ocr_text: str, question: str) -> str:
    """OCR 문서 내용을 컨텍스트로 한 질의응답."""
    client = ChatClient()
    messages = [
        {"role": "system", "content": SUMMARY_SYSTEM},
        {"role": "user", "content": f"[문서]\n{ocr_text}\n\n[질문] {question}"},
    ]
    return client.chat(messages)
```

Celery에서 OCR 완료 후 자동 요약을 이어 붙이는 예시:

```python
# myapp/tasks.py (run_ocr_task 끝부분에 추가)

from .services import summarize_ocr_text

job.summary = summarize_ocr_text(job.text)
job.save(update_fields=["summary"])
```

> Gemma chat의 컨텍스트 한도는 `CHAT_MAX_MODEL_LEN`(기본 32768)입니다. 매우 긴 문서는 페이지 단위로 나눠 요약 후 합치는 map-reduce 방식을 권장합니다.

---

## 11. 운영 체크리스트

### Django 서버

- [ ] `OCR_API_BASE_URL`(`:8001`), `CHAT_API_BASE_URL`(`:8088/v1`) 환경변수 설정
- [ ] DGX Spark `:8001`, `:8088` 네트워크 연결 확인
- [ ] `OCR_API_TIMEOUT` 충분히 설정 (최소 120초), `CHAT_API_TIMEOUT` 60~120초
- [ ] 대용량 PDF는 Celery 비동기 처리
- [ ] OCR 결과 `raw_response` JSONField 저장
- [ ] Chat 스트리밍 사용 시 ASGI + nginx `proxy_buffering off`
- [ ] 서버 장애 시 503/502/504 사용자 메시지 처리

### DGX Spark 서버

- [ ] OCR API: `./start.sh` 또는 `systemctl start ocr-api`
- [ ] vLLM: `cd vllm && ./start.sh` → `vllm-qwen-vl-ocr`, `vllm-gemma-chat` 컨테이너 기동
- [ ] AI Chat nginx: `ai-chat/start.sh` → `:8088` 기동
- [ ] `curl :8001/health` → `vllm_vl_reachable: true`, `vllm_chat_reachable: true`
- [ ] `curl :8088/v1/models` → `gemma-chat` 노출 확인

### 모니터링

```bash
# Django cron / Celery beat에서 주기 실행
curl -sf http://192.168.0.67:8001/health || alert "OCR API down"
curl -sf http://192.168.0.67:8088/v1/models || alert "AI Chat down"
```

---

## 12. 트러블슈팅

### OCR — `502` / `OCR server unreachable`

1. OCR API 확인: `curl http://<dgx>:8001/health`
2. 방화벽: Django → DGX `8001/TCP`
3. `OCR_API_BASE_URL` 오타 확인

### OCR — `503 Vision OCR is not reachable`

`/health`에서 `vllm_vl_reachable: false`이면 `vllm-qwen-vl-ocr`(`:8003`)가 떠 있지 않은 것입니다.

```bash
docker ps | grep vllm-qwen-vl-ocr
docker logs -f vllm-qwen-vl-ocr      # 첫 로딩 8~15분
cd vllm && ./start.sh
```

### OCR — `504 timeout`

1. `OCR_API_TIMEOUT` 증가 (300 → 600)
2. Celery 비동기 전환
3. 대용량 PDF는 `dpi` 하향 또는 페이지 분할

### Chat — `502` / `gemma-chat` 응답 없음

1. `curl http://<dgx>:8088/v1/models` → `gemma-chat` 노출 확인
2. nginx·컨테이너 확인: `docker ps | grep -E 'ai-chat-nginx|vllm-gemma-chat'`
3. `docker logs -f vllm-gemma-chat` (첫 로딩 대기)
4. `CHAT_API_BASE_URL` 끝에 `/v1` 포함 여부 확인

### Chat — 스트리밍이 한 번에 몰려서 옴

- Django ASGI(uvicorn/daphne)로 기동
- 앞단 nginx: `proxy_buffering off;`, 응답 헤더 `X-Accel-Buffering: no`

### 응답 `text`가 비어 있음 (OCR)

- 스캔 품질 낮음 → `mode=scan`, `dpi=400` 시도
- 이미지만 있는 PDF여도 VL 백엔드는 글자를 읽으므로, 비어 있으면 `vllm-qwen-vl-ocr` 로그 확인

---

## 13. 전체 연동 흐름 요약

```
[OCR 분석]
1. 사용자 → Django View (파일 업로드)
2. Django → MEDIA 저장 → Celery task enqueue
3. Celery worker → POST :8001/ocr (multipart)
4. OCR API → vllm-qwen-vl-ocr(:8003)로 페이지 OCR → JSON 응답
5. Celery worker → OcrJob DB 업데이트 (text, pages, table)

[AI Chat / 요약]
6. (선택) Celery → summarize_ocr_text(job.text) → :8088/v1/chat/completions
7. 사용자 → Django chat View → gemma-chat 응답(동기/스트리밍) 표시
```

---

## 14. 참고

- 서버 설치·GPU 빌드·systemd: [ReadMe.md](../ReadMe.md)
- vLLM 도커(`vllm-qwen-vl-ocr`, `vllm-gemma-chat`): [vllm/README.md](../vllm/README.md)
- AI Chat 웹/nginx 구성: [ai-chat/README.md](../ai-chat/README.md)
- 포트: OCR API **8001**, AI Chat **8088**(nginx) / **8000**(vLLM 직접), Vision OCR **8003**(내부)
- 모델 ID: OCR `qwen-vl-ocr`, Chat `gemma-chat`
