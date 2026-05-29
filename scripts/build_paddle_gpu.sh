#!/usr/bin/env bash
set -euo pipefail

# Build PaddlePaddle GPU wheel for DGX Spark (ARM64 + CUDA 13 + sm_121)
# Usage: ./scripts/build_paddle_gpu.sh [--install]

INSTALL=0
if [[ "${1:-}" == "--install" ]]; then
    INSTALL=1
fi

VENV="${VENV:-/home/wslaw/ocr-server/.venv}"
export PATH="$VENV/bin:$PATH"
PADDLE_DIR="${PADDLE_DIR:-$HOME/Paddle}"
PADDLE_TAG="${PADDLE_TAG:-v3.2.2}"
JOBS="${JOBS:-$(nproc)}"

echo "==> Checking build dependencies..."
for cmd in git cmake python3 nvcc; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "ERROR: $cmd not found"
        exit 1
    fi
done

CUDNN_ROOT="${CUDNN_ROOT:-$HOME/local/cudnn/usr}"

_ensure_local_cudnn() {
    if ldconfig -p 2>/dev/null | grep -q libcudnn; then
        return 0
    fi
    if [[ -f "$CUDNN_ROOT/include/aarch64-linux-gnu/cudnn.h" ]]; then
        return 0
    fi

    echo "==> Setting up local cuDNN (no sudo required)..."
    mkdir -p "$HOME/local/cudnn"
    cd /tmp
    apt-get download libcudnn9-cuda-13 libcudnn9-dev-cuda-13 libcudnn9-headers-cuda-13 2>/dev/null || true
    for deb in libcudnn9-cuda-13_*.deb libcudnn9-dev-cuda-13_*.deb libcudnn9-headers-cuda-13_*.deb; do
        [[ -f "$deb" ]] && dpkg-deb -x "$deb" "$HOME/local/cudnn"
    done

    if [[ ! -f "$CUDNN_ROOT/include/aarch64-linux-gnu/cudnn.h" ]]; then
        echo "ERROR: cuDNN setup failed. Install manually:"
        echo "  sudo apt-get install -y cudnn9-cuda-13 libcudnn9-dev-cuda-13"
        exit 1
    fi
}

_ensure_local_cudnn
export LD_LIBRARY_PATH="$CUDNN_ROOT/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH:-}"

if [[ -x "$HOME/local/patchelf/usr/bin/patchelf" ]]; then
    export PATH="$HOME/local/patchelf/usr/bin:$PATH"
elif ! command -v patchelf >/dev/null 2>&1; then
    echo "==> Setting up local patchelf..."
    mkdir -p "$HOME/local/patchelf"
    cd /tmp
    apt-get download patchelf 2>/dev/null || true
    for deb in patchelf_*.deb; do
        [[ -f "$deb" ]] && dpkg-deb -x "$deb" "$HOME/local/patchelf"
    done
    export PATH="$HOME/local/patchelf/usr/bin:$PATH"
fi

NINJA="$VENV/bin/ninja"
if [[ ! -x "$NINJA" ]]; then
    echo "==> Installing ninja via pip..."
    "$VENV/bin/pip" install -q ninja
fi
export PATH="$VENV/bin:$PATH"

if ! command -v ccache >/dev/null 2>&1; then
    echo "==> ccache not found (optional); skipping"
fi

echo "==> Cloning PaddlePaddle ${PADDLE_TAG}..."
if [[ ! -d "$PADDLE_DIR/.git" ]]; then
    git clone --depth 1 --branch "$PADDLE_TAG" https://github.com/PaddlePaddle/Paddle.git "$PADDLE_DIR"
else
    cd "$PADDLE_DIR"
    git fetch --depth 1 origin "refs/tags/${PADDLE_TAG}" 2>/dev/null || true
    git checkout "$PADDLE_TAG" 2>/dev/null || git checkout develop
    cd -
fi

echo "==> Installing Python build deps..."
"$VENV/bin/pip" install -q --upgrade pip setuptools wheel
"$VENV/bin/pip" install -q numpy protobuf opt_einsum

echo "==> Configuring CMake..."
BUILD_DIR="$PADDLE_DIR/build"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

NVTX_INC="/usr/local/cuda/include/nvtx3"
if [[ ! -f "$NVTX_INC/nvToolsExt.h" ]]; then
    NVTX_INC="$HOME/local/nvtx/usr/local/cuda-13.0/targets/sbsa-linux/include/nvtx3"
fi

CMAKE_ARGS=(
    -GNinja
    -DCMAKE_BUILD_TYPE=Release
    -DWITH_GPU=ON
    -DWITH_ARM=ON
    -DCUDA_ARCH_NAME=Manual
    -DCUDA_ARCH_BIN="12.1"
    -DCUDNN_ROOT="$CUDNN_ROOT"
    -DCMAKE_CXX_FLAGS="-I${NVTX_INC}"
    -DCMAKE_CUDA_FLAGS="-U__ARM_NEON -DEIGEN_DONT_VECTORIZE=1 -I${NVTX_INC}"
    -DWITH_AVX=OFF
    -DWITH_MKL=OFF
    -DWITH_MKLDNN=OFF
    -DWITH_TENSORRT=OFF
    -DWITH_NCCL=OFF
    -DWITH_TESTING=OFF
    -DPYTHON_EXECUTABLE="$VENV/bin/python"
)

if command -v ccache >/dev/null 2>&1; then
    CMAKE_ARGS+=(-DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache)
fi

cmake .. "${CMAKE_ARGS[@]}" 2>&1 | tee cmake_output.log

echo "==> Building PaddlePaddle (this may take 1-2 hours)..."
"$VENV/bin/ninja" -j"$JOBS" 2>&1 | tee build_output.log

echo "==> Building wheel..."
cd "$PADDLE_DIR"
"$VENV/bin/python" setup.py bdist_wheel 2>&1 | tee wheel_output.log

WHEEL=$(ls -1 "$PADDLE_DIR"/build/python/dist/paddlepaddle_gpu-*.whl "$PADDLE_DIR"/dist/paddlepaddle_gpu-*.whl 2>/dev/null | tail -1)
if [[ -z "$WHEEL" ]]; then
    echo "ERROR: wheel not found in $PADDLE_DIR/dist/"
    exit 1
fi

echo "==> Built: $WHEEL"

if [[ "$INSTALL" -eq 1 ]]; then
    echo "==> Installing GPU wheel into $VENV..."
    "$VENV/bin/pip" uninstall -y paddlepaddle paddlepaddle-gpu 2>/dev/null || true
    "$VENV/bin/pip" install "$WHEEL"
    echo "==> Verifying installation..."
    "$VENV/bin/python" -c "import paddle; print('version:', paddle.__version__); print('cuda:', paddle.is_compiled_with_cuda())"
fi

echo "==> Done. Run with --install to install into venv, or:"
echo "    $VENV/bin/pip install $WHEEL"
