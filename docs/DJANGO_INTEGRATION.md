# Django 운영 서버 → OCR API 연동 메뉴얼

Django 운영 서버에서 DGX Spark OCR API(`ocr-server`)를 호출하는 방법을 정리한 문서입니다.  
OCR 서버 자체 설치·운영은 [ReadMe.md](../ReadMe.md)를 참고하세요.

---

## 1. 시스템 구성

```
┌─────────────────────┐         HTTP (multipart)         ┌──────────────────────┐
│  Django 운영 서버     │  ─────────────────────────────▶  │  DGX Spark OCR API   │
│  (Gunicorn/uWSGI)   │         POST /ocr                │  FastAPI :8001       │
│                     │  ◀─────────────────────────────  │  PaddleOCR + GPU     │
└─────────────────────┘         JSON 응답                └──────────────────────┘
```

| 구분 | Django 서버 | OCR 서버 (DGX Spark) |
|------|-------------|----------------------|
| 역할 | 사용자 요청 수신, 파일 업로드, 결과 저장/표시 | PDF/이미지 OCR 수행 |
| 프로토콜 | HTTP 클라이언트 (`requests` / `httpx`) | HTTP API (FastAPI) |
| 기본 URL | 설정값 `OCR_API_BASE_URL` | `http://192.168.0.67:8001` (예시) |
| 타임아웃 | **120~600초** 권장 (페이지 수·DPI에 따라) | GPU: ~1초/페이지, CPU: ~30~60초/페이지 |

> Django 서버에 PaddleOCR을 설치할 필요 **없음**. OCR API만 호출하면 됩니다.

---

## 2. 사전 확인

### 2-1. OCR API 서버 상태

Django 서버(또는 배포 PC)에서 다음을 실행합니다.

```bash
# 연결 확인
curl http://192.168.0.67:8001/health
```

정상 응답 예시:

```json
{
  "status": "ok",
  "device": "gpu:0",
  "paddle_cuda": true,
  "ocr_dpi_default": 300,
  "ocr_workers": 8
}
```

| 필드 | 의미 |
|------|------|
| `status` | `"ok"` 이면 서버 가동 중 |
| `device` | `"gpu:0"` GPU 사용, `"cpu"` CPU fallback (느림) |
| `paddle_cuda` | GPU PaddlePaddle 빌드 여부 |

### 2-2. OCR 요청 테스트

```bash
curl -X POST http://192.168.0.67:8001/ocr \
  -F "file=@sample.pdf" \
  -F "mode=scan" \
  -F "dpi=300"
```

### 2-3. 네트워크

- Django 서버 → OCR 서버 **8001/TCP** 방화벽 허용
- OCR 서버는 `0.0.0.0:8001` 로 바인딩되어 있어야 함 (`start.sh` / `run.sh` 기본값)
- 내부망 IP 사용 권장 (예: `http://192.168.0.67:8001`)

---

## 3. Django 설정

### 3-1. `settings.py`

```python
# settings.py

OCR_API_BASE_URL = env("OCR_API_BASE_URL", default="http://192.168.0.67:8001")
OCR_API_TIMEOUT = env.int("OCR_API_TIMEOUT", default=300)       # 초
OCR_DEFAULT_MODE = env("OCR_DEFAULT_MODE", default="scan")    # scan | raw
OCR_DEFAULT_DPI = env.int("OCR_DEFAULT_DPI", default=300)
OCR_MAX_FILE_SIZE = env.int("OCR_MAX_FILE_SIZE", default=50 * 1024 * 1024)  # 50MB
```

`.env` 예시:

```env
OCR_API_BASE_URL=http://192.168.0.67:8001
OCR_API_TIMEOUT=300
OCR_DEFAULT_MODE=scan
OCR_DEFAULT_DPI=300
```

### 3-2. 의존 패키지

```bash
pip install requests
# 또는 비동기 사용 시
pip install httpx
```

`requirements.txt`에 추가:

```
requests>=2.31.0
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
| `mode` | X | string | `scan` | `scan`: 스캔본 전처리 + 문서 보정, `raw`: 원본 그대로 |
| `dpi` | X | int | `300` | PDF 렌더 해상도 (PDF만 해당) |
| `format` | X | string | `text` | `table`: 고정 열 표 JSON (`table_cols` 필수) |
| `table_cols` | X* | int | — | `format=table`일 때 열 개수 (재결서 보상금내역: `9`) |
| `table_header_row` | X | int | `0` | 단일 헤더 행 인덱스 (레거시, `header_rows=1`일 때) |
| `table_header_rows` | X | int | `1` | 헤더로 볼 행 수 (재결서 2단 헤더: `2`) |
| `table_col_boundaries` | X | string | — | 열 경계 X좌표 JSON 배열 (예: `[0,75,150,...]`) |
| `refine` | X | string | 서버 기본 | `none` \| `vllm` (저신뢰 셀 LLM 보정) |
| `refine_threshold` | X | float | 서버 기본 | vLLM 보정 대상 신뢰도 상한 (예: `0.85`) |

\* `format=table`이면 `table_cols` 필수.

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

**성공 응답 (HTTP 200):**

```json
{
  "job_id": "d4d03fae-b3bd-423c-acbb-591433e24286",
  "filename": "document.pdf",
  "mode": "scan",
  "dpi": 300,
  "device": "gpu:0",
  "preprocessed": true,
  "processing_time_sec": 2.721,
  "page_count": 3,
  "text": "1페이지 텍스트\n\n2페이지 텍스트\n\n3페이지 텍스트",
  "pages": [
    {
      "page": 1,
      "text": "1페이지 텍스트",
      "avg_score": 0.9434,
      "blocks": [
        {
          "text": "인식된 문장",
          "score": 0.94,
          "box": [[95, 257], [315, 254], [315, 292], [96, 296]]
        }
      ]
    }
  ],
  "upload_path": "/tmp/ocr_uploads/d4d03fae-b3bd-423c-acbb-591433e24286.pdf",
  "result_path": "/tmp/ocr_results/d4d03fae-b3bd-423c-acbb-591433e24286.json",
  "page_image_paths": [
    "/tmp/ocr_pages/d4d03fae-b3bd-423c-acbb-591433e24286_1.png",
    "/tmp/ocr_pages/d4d03fae-b3bd-423c-acbb-591433e24286_1_pre.png"
  ]
}
```

**응답 필드 설명 (Django에서 활용):**

| 필드 | 설명 |
|------|------|
| `job_id` | OCR 작업 고유 ID (DB PK 또는 외부 참조키로 저장) |
| `text` | 전체 페이지 텍스트 (`\n\n`로 페이지 구분) |
| `pages[].text` | 페이지별 텍스트 |
| `pages[].blocks` | 텍스트 블록 단위 (좌표 `box`, 신뢰도 `score` 포함) |
| `pages[].avg_score` | 페이지 평균 인식 신뢰도 (0~1) |
| `processing_time_sec` | OCR 처리 소요 시간(초) |
| `upload_path` | OCR 서버에 보관된 원본 파일 경로 |
| `result_path` | OCR 서버에 저장된 결과 JSON 경로 |
| `page_image_paths` | 렌더/전처리 PNG 경로 목록 (OCR 서버 로컬) |

> `upload_path`, `result_path`, `page_image_paths`는 **OCR 서버(DGX Spark) 로컬 경로**입니다.  
> Django DB에는 `job_id`와 API 응답 JSON을 저장하고, 파일은 Django `MEDIA`에 별도 보관하는 것을 권장합니다.

**에러 응답:**

| HTTP | 원인 | Django 대응 |
|------|------|-------------|
| 500 | OCR 서버 내부 오류 (GPU/cuDNN 등) | 재시도 또는 관리자 알림 |
| 422 | 잘못된 multipart/form 파라미터 | 요청 파라미터 검증 |
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
    ) -> dict:
        """
        파일 객체를 OCR API에 전송하고 JSON 결과를 반환합니다.

        Args:
            file_obj: Django UploadedFile 또는 open(..., 'rb')
            filename: 원본 파일명 (확장자 포함)
            mode: 'scan' | 'raw'
            dpi: PDF 렌더 DPI (None이면 OCR 서버 기본값 300)
        """
        mode = mode or settings.OCR_DEFAULT_MODE
        data = {"mode": mode}
        if dpi is not None:
            data["dpi"] = str(dpi)

        files = {"file": (filename, file_obj)}

        logger.info("OCR request: filename=%s mode=%s dpi=%s", filename, mode, dpi)
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

## 10. 운영 체크리스트

### Django 서버

- [ ] `OCR_API_BASE_URL` 환경변수 설정
- [ ] OCR 서버 `:8001` 네트워크 연결 확인
- [ ] `OCR_API_TIMEOUT` 충분히 설정 (최소 120초)
- [ ] 대용량 PDF는 Celery 비동기 처리
- [ ] OCR 결과 `raw_response` JSONField 저장
- [ ] OCR 서버 장애 시 503/502 사용자 메시지 처리

### OCR 서버 (DGX Spark)

- [ ] `./start.sh` 또는 `systemctl start ocr-api` 로 가동
- [ ] `curl http://<ocr-host>:8001/health` → `device: gpu:0` 확인
- [ ] systemd 사용 시 `run.sh` 기반 `ocr-api.service` 적용 (cuDNN 경로 포함)

### 모니터링

```bash
# Django cron / Celery beat에서 주기 실행
curl -sf http://192.168.0.67:8001/health || alert "OCR server down"
```

---

## 11. 트러블슈팅

### `502 Bad Gateway` / `OCR server unreachable`

1. OCR 서버 프로세스 확인: `curl http://<ocr-host>:8001/health`
2. 방화벽: Django → OCR `8001/TCP`
3. `OCR_API_BASE_URL` 오타 확인

### `504 OCR timeout`

1. `OCR_API_TIMEOUT` 증가 (300 → 600)
2. Celery 비동기 전환
3. OCR 서버 `/health`에서 `device`가 `cpu`이면 GPU 설정 점검 (CPU는 매우 느림)

### `500 Internal Server Error` (OCR 서버)

OCR 서버 로그 확인:

```bash
journalctl -u ocr-api -n 50
# 또는 OCR 서버에서
./log.sh
```

흔한 원인: cuDNN 미로드 → `./start.sh` 또는 `run.sh`로 재시작

### 응답 `text`가 비어 있음

- 스캔 품질 낮음 → `mode=scan`, `dpi=300` 시도
- 이미지만 있는 PDF → 정상 (OCR 대상 텍스트 없음)

---

## 12. 전체 연동 흐름 요약

```
1. 사용자 → Django View (파일 업로드)
2. Django → MEDIA 저장 (선택)
3. Django → Celery task enqueue (권장)
4. Celery worker → POST /ocr (multipart)
5. OCR 서버 → JSON 응답 + /tmp/ocr_results/{job_id}.json 저장
6. Celery worker → OcrJob DB 업데이트 (text, pages, job_id)
7. 프론트엔드 → /ocr/status/{id} 폴링 → 결과 표시
```

---

## 13. 참고

- OCR 서버 설치·GPU 빌드·systemd: [ReadMe.md](../ReadMe.md)
- API 기본 포트: **8001**
- 지원 언어: **한국어** (PP-OCRv5 `korean_PP-OCRv5_mobile_rec`)
