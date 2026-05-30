# DGX Spark (GB10, 128GB unified) — VL OCR + Gemma 4 chat.
# Source from env.sh:  source "$(dirname "$0")/env.dgx-spark-128gb.sh"

export OCR_BACKEND="${OCR_BACKEND:-vllm_vl}"
export OCR_REFINE_DEFAULT="${OCR_REFINE_DEFAULT:-none}"

# GPU memory split: Gemma 4 MoE ~28% (서브) + Qwen2.5-VL-32B-AWQ ~60% 메인 = 합계 88% (112.6GB / 128GB)
export CHAT_GPU_MEMORY_UTIL="${CHAT_GPU_MEMORY_UTIL:-0.28}"
export VL_GPU_MEMORY_UTIL="${VL_GPU_MEMORY_UTIL:-0.60}"
export CHAT_MAX_MODEL_LEN="${CHAT_MAX_MODEL_LEN:-8192}"
export VL_MAX_MODEL_LEN="${VL_MAX_MODEL_LEN:-16384}"

# ai-chat: Gemma 4 (MoE 26B, ~3.8B active)
export CHAT_HF_MODEL="${CHAT_HF_MODEL:-google/gemma-4-26b-a4b-it}"
export CHAT_MODEL_DIR="${CHAT_MODEL_DIR:-gemma-4-26b-a4b-it}"
export VLLM_MODEL="${VLLM_MODEL:-gemma-chat}"
export VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:8088/v1}"

# OCR: Qwen2.5-VL-32B-AWQ (한글 문서·표·PDF 페이지 이미지)
export VL_HF_MODEL="${VL_HF_MODEL:-NCSOFT/VARCO-VISION-2.0-14B}"
export VL_MODEL_DIR="${VL_MODEL_DIR:-VARCO-VISION-2.0-14B}"
export VLLM_VL_MODEL="${VLLM_VL_MODEL:-qwen-vl-ocr}"
export VLLM_VL_BASE_URL="${VLLM_VL_BASE_URL:-http://127.0.0.1:8003/v1}"
export VLLM_VL_MAX_TOKENS="${VLLM_VL_MAX_TOKENS:-8192}"
export VLLM_VL_TIMEOUT="${VLLM_VL_TIMEOUT:-180}"
export VLLM_VL_MAX_IMAGE_SIDE="${VLLM_VL_MAX_IMAGE_SIDE:-2048}"

# 경량 VL (OOM 시): VL_HF_MODEL=Qwen/Qwen2.5-VL-32B-Instruct-AWQ VL_MODEL_DIR=Qwen2.5-VL-32B-Instruct-AWQ VL_GPU_MEMORY_UTIL=0.35

# Paddle + 14B refine (OCR_BACKEND=paddle 일 때만)
export REFINE_HF_MODEL="${REFINE_HF_MODEL:-Qwen/Qwen2.5-14B-Instruct}"
export REFINE_MODEL_DIR="${REFINE_MODEL_DIR:-Qwen2.5-14B-Instruct}"
export VLLM_REFINE_MODEL="${VLLM_REFINE_MODEL:-qwen-refine}"
export VLLM_REFINE_BASE_URL="${VLLM_REFINE_BASE_URL:-http://127.0.0.1:8002/v1}"
export VLLM_REFINE_FALLBACK_TO_CHAT="${VLLM_REFINE_FALLBACK_TO_CHAT:-false}"
