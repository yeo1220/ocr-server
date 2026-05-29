"""vLLM client: VL OCR, table refine, and chat endpoints."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

import httpx

from config import settings

logger = logging.getLogger(__name__)

_http_client: httpx.AsyncClient | None = None
_refine_base_url: str = settings.vllm_refine_base_url
_refine_model_id: str | None = None
_refine_profile: "VllmModelProfile | None" = None
_vl_base_url: str = settings.vllm_vl_base_url
_vl_model_id: str | None = None
_vl_profile: "VllmModelProfile | None" = None


@dataclass(frozen=True)
class VllmModelProfile:
    """Inference parameters for a served vLLM model."""

    model_id: str
    root: str
    is_thinking: bool
    is_small_instruct: bool
    max_output_tokens: int
    temperature: float
    use_json_mode: bool
    extra_body: dict[str, Any]
    base_url: str


def _is_small_instruct(meta: str) -> bool:
    if "qwen-refine" in meta:
        return True
    if re.search(r"qwen2[\._-]?5", meta) and re.search(
        r"(7b|8b|9b|14b|1\.5b)", meta
    ):
        return True
    if re.search(r"(7b|8b|9b|14b)", meta) and "instruct" in meta:
        return True
    return False


def _is_medium_instruct(meta: str) -> bool:
    return bool(re.search(r"qwen2[\._-]?5", meta) and re.search(r"14b", meta))


def _profile_from_model(
    model_id: str,
    root: str = "",
    *,
    base_url: str,
) -> VllmModelProfile:
    meta = f"{model_id} {root}".lower()
    is_thinking = "thinking" in meta
    is_small = _is_small_instruct(meta)
    is_medium = _is_medium_instruct(meta)
    is_large = any(x in meta for x in ("80b", "70b", "72b", "65b"))

    if is_medium:
        max_out = settings.vllm_refine_max_tokens_medium
    elif is_small:
        max_out = settings.vllm_refine_max_tokens_small
    elif is_thinking:
        max_out = settings.vllm_refine_max_tokens_thinking
    elif is_large:
        max_out = settings.vllm_refine_max_tokens_large
    else:
        max_out = settings.vllm_refine_max_tokens_default

    extra: dict[str, Any] = {}
    if is_thinking:
        extra["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": False},
        }

    return VllmModelProfile(
        model_id=model_id,
        root=root,
        is_thinking=is_thinking,
        is_small_instruct=is_small,
        max_output_tokens=max_out,
        temperature=0.0,
        use_json_mode=True,
        extra_body=extra,
        base_url=base_url.rstrip("/"),
    )


def strip_thinking_content(text: str) -> str:
    """Remove Qwen Thinking / reasoning wrappers before JSON parsing."""
    s = (text or "").strip()
    if not s:
        return s
    for pat in (
        r"<think[^>]*>[\s\S]*?</think[^>]*>",
        r"<reasoning>[\s\S]*?</reasoning>",
    ):
        s = re.sub(pat, "", s, flags=re.IGNORECASE)
    for marker in (
        r"<\|im_start\|>assistant\s*",
        r"<\|assistant\|>\s*",
        r"Final answer:\s*",
        r"최종\s*답변\s*[:：]\s*",
    ):
        m = re.search(marker, s, flags=re.IGNORECASE)
        if m:
            s = s[m.end() :].strip()
    j = re.search(r"[\{\[]", s)
    if j and j.start() > 0:
        s = s[j.start() :]
    return s.strip()


def extract_json_object(text: str) -> dict | None:
    """Parse a JSON object from model output (fences, thinking, trailing prose)."""
    s = strip_thinking_content(text)
    if not s:
        return None
    s = re.sub(r"^```[a-zA-Z]*\n?", "", s)
    s = re.sub(r"\n?```$", "", s.strip())
    try:
        parsed = json.loads(s)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    start, end = s.find("{"), s.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(s[start : end + 1])
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass
    return None


def get_model_profile() -> VllmModelProfile | None:
    return _refine_profile


def get_resolved_model_id() -> str:
    return _refine_model_id or settings.vllm_refine_model


def get_refine_base_url() -> str:
    return _refine_base_url


def _client(*, timeout: float | None = None) -> httpx.AsyncClient:
    global _http_client
    t = timeout if timeout is not None else settings.vllm_timeout
    if _http_client is None:
        _http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(t, connect=10.0),
            limits=httpx.Limits(max_keepalive_connections=8, max_connections=16),
        )
    return _http_client


def _vl_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=httpx.Timeout(settings.vllm_vl_timeout, connect=15.0),
        limits=httpx.Limits(max_keepalive_connections=4, max_connections=8),
    )


async def close_client() -> None:
    global _http_client
    if _http_client is not None:
        await _http_client.aclose()
        _http_client = None


async def _fetch_models(base_url: str) -> list[dict]:
    r = await _client().get(
        f"{base_url.rstrip('/')}/models",
        headers={"Authorization": f"Bearer {settings.vllm_api_key}"},
    )
    r.raise_for_status()
    return r.json().get("data") or []


async def _resolve_at_url(
    base_url: str,
    preferred_id: str,
) -> VllmModelProfile:
    data = await _fetch_models(base_url)
    if not data:
        return _profile_from_model(preferred_id, base_url=base_url)

    ids = [str(m.get("id") or "") for m in data if m.get("id")]
    root = ""
    if preferred_id in ids:
        chosen = preferred_id
        for m in data:
            if m.get("id") == preferred_id:
                root = str(m.get("root") or "")
                break
    else:
        chosen = ids[0]
        root = str((data[0] or {}).get("root") or "")

    return _profile_from_model(chosen, root, base_url=base_url)


async def resolve_refine_model() -> VllmModelProfile:
    """Resolve table-refine model (Qwen2.5 7B on :8002); fallback to chat 80B if needed."""
    global _refine_base_url, _refine_model_id, _refine_profile

    refine_url = settings.vllm_refine_base_url.rstrip("/")
    preferred = settings.vllm_refine_model.strip()

    try:
        prof = await _resolve_at_url(refine_url, preferred)
        _refine_base_url = prof.base_url
        _refine_model_id = prof.model_id
        _refine_profile = prof
        logger.info(
            "vLLM refine ready: id=%s small=%s url=%s max_tokens=%s",
            prof.model_id,
            prof.is_small_instruct,
            prof.base_url,
            prof.max_output_tokens,
        )
        return prof
    except Exception as e:
        logger.warning("vLLM refine endpoint unavailable (%s): %s", refine_url, e)

    if not settings.vllm_refine_fallback_to_chat:
        _refine_profile = _profile_from_model(preferred, base_url=refine_url)
        _refine_model_id = preferred
        _refine_base_url = refine_url
        return _refine_profile

    chat_url = settings.vllm_base_url.rstrip("/")
    logger.warning(
        "Falling back to chat vLLM for table refine: %s (model %s)",
        chat_url,
        settings.vllm_model,
    )
    prof = await _resolve_at_url(chat_url, settings.vllm_model.strip())
    _refine_base_url = prof.base_url
    _refine_model_id = prof.model_id
    _refine_profile = prof
    return prof


# Backward-compatible alias
resolve_served_model = resolve_refine_model


async def _reachable_at(base_url: str) -> bool:
    try:
        r = await _client().get(
            f"{base_url.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {settings.vllm_api_key}"},
            timeout=5.0,
        )
        return r.status_code == 200
    except Exception:
        return False


async def check_refine_reachable() -> bool:
    if not settings.vllm_enabled:
        return False
    if await _reachable_at(settings.vllm_refine_base_url):
        return True
    if settings.vllm_refine_fallback_to_chat:
        return await _reachable_at(settings.vllm_base_url)
    return False


async def check_chat_reachable() -> bool:
    return await _reachable_at(settings.vllm_base_url)


async def check_vllm_reachable() -> bool:
    if settings.ocr_backend == "vllm_vl":
        return await check_vl_reachable()
    return await check_refine_reachable()


async def resolve_vl_model() -> VllmModelProfile:
    """Resolve vision OCR model (Qwen2.5-VL on :8003)."""
    global _vl_base_url, _vl_model_id, _vl_profile

    vl_url = settings.vllm_vl_base_url.rstrip("/")
    preferred = settings.vllm_vl_model.strip()
    prof = await _resolve_at_url(vl_url, preferred)
    _vl_base_url = prof.base_url
    _vl_model_id = prof.model_id
    _vl_profile = VllmModelProfile(
        model_id=prof.model_id,
        root=prof.root,
        is_thinking=prof.is_thinking,
        is_small_instruct=prof.is_small_instruct,
        max_output_tokens=settings.vllm_vl_max_tokens,
        temperature=0.0,
        use_json_mode=True,
        extra_body={},
        base_url=prof.base_url,
    )
    logger.info(
        "vLLM VL OCR ready: id=%s url=%s max_tokens=%s",
        _vl_profile.model_id,
        _vl_profile.base_url,
        _vl_profile.max_output_tokens,
    )
    return _vl_profile


def get_vl_resolved_model_id() -> str:
    return _vl_model_id or settings.vllm_vl_model


def get_vl_base_url() -> str:
    return _vl_base_url


async def check_vl_reachable() -> bool:
    if not settings.vllm_enabled:
        return False
    return await _reachable_at(settings.vllm_vl_base_url)


async def chat_vision_json(
    messages: list[dict[str, Any]],
    *,
    max_tokens: int | None = None,
    profile: VllmModelProfile | None = None,
) -> tuple[str, dict[str, Any]]:
    """Vision chat/completions for page-image OCR."""
    prof = profile or _vl_profile
    if prof is None:
        prof = await resolve_vl_model()
    base = prof.base_url
    model_id = prof.model_id
    limit = max_tokens if max_tokens is not None else prof.max_output_tokens

    payload: dict[str, Any] = {
        "model": model_id,
        "messages": messages,
        "temperature": prof.temperature,
        "max_tokens": limit,
        "response_format": {"type": "json_object"},
    }
    payload.update(prof.extra_body)

    async with _vl_client() as client:
        r = await client.post(
            f"{base}/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.vllm_api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        r.raise_for_status()
        body = r.json()

    content = (
        body.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    )
    usage = body.get("usage") or {}
    meta = {
        "model": model_id,
        "base_url": base,
        "max_tokens": limit,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
    }
    return content or "", meta


def estimate_refine_max_tokens(
    num_rows: int,
    num_cols: int,
    num_fix_cells: int,
    *,
    profile: VllmModelProfile | None = None,
) -> int:
    prof = profile or _refine_profile
    cap = prof.max_output_tokens if prof else settings.vllm_refine_max_tokens_default
    est = 120 + num_rows * num_cols * 48 + num_fix_cells * 24
    return min(cap, max(256, est))


async def chat_json(
    messages: list[dict[str, str]],
    *,
    max_tokens: int | None = None,
    profile: VllmModelProfile | None = None,
) -> tuple[str, dict[str, Any]]:
    """Call refine vLLM /chat/completions; return (content, usage meta)."""
    prof = profile or _refine_profile or _profile_from_model(
        get_resolved_model_id(),
        base_url=_refine_base_url,
    )
    base = prof.base_url
    model_id = prof.model_id
    limit = max_tokens if max_tokens is not None else prof.max_output_tokens

    payload: dict[str, Any] = {
        "model": model_id,
        "messages": messages,
        "temperature": prof.temperature,
        "max_tokens": limit,
    }
    if prof.use_json_mode:
        payload["response_format"] = {"type": "json_object"}
    payload.update(prof.extra_body)

    r = await _client().post(
        f"{base}/chat/completions",
        headers={
            "Authorization": f"Bearer {settings.vllm_api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
    )
    r.raise_for_status()
    body = r.json()
    content = (
        body.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    )
    usage = body.get("usage") or {}
    meta = {
        "model": model_id,
        "base_url": base,
        "max_tokens": limit,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
    }
    return content or "", meta
