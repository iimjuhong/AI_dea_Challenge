#!/bin/bash
# Jetson Orin Nano 환경 설정 스크립트
# 시스템 라이브러리만 사용 - pip install 없음

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_DIR/venv"

echo "=== Jetson 환경 설정 스크립트 ==="
echo "프로젝트 디렉토리: $PROJECT_DIR"

# 1. 기존 venv 삭제 및 재생성
if [ -d "$VENV_DIR" ]; then
    echo "[1/2] 기존 venv 삭제..."
    rm -rf "$VENV_DIR"
fi

echo "[1/2] system-site-packages 포함 venv 생성..."
python3 -m venv --system-site-packages "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# 2. 시스템 라이브러리 검증
echo "[2/2] 시스템 라이브러리 검증..."
echo ""

check_module() {
    local module=$1
    local import_cmd=$2
    local version_cmd=$3

    if python3 -c "$import_cmd" 2>/dev/null; then
        local version
        version=$(python3 -c "$version_cmd" 2>/dev/null || echo "버전 확인 불가")
        echo "  [OK] $module: $version"
    else
        echo "  [MISSING] $module - 시스템에 설치되어 있지 않습니다"
    fi
}

check_module "OpenCV" "import cv2" "import cv2; print(cv2.__version__)"
check_module "NumPy" "import numpy" "import numpy; print(numpy.__version__)"
check_module "TensorRT" "import tensorrt" "import tensorrt; print(tensorrt.__version__)"
check_module "PyCUDA" "import pycuda" "import pycuda.driver as cuda; print(cuda.get_version())"
check_module "Flask" "import flask" "import flask; print(flask.__version__)"
check_module "PyYAML" "import yaml" "import yaml; print(yaml.__version__)"

echo ""
echo "=== 설정 완료 ==="
echo "활성화: source $VENV_DIR/bin/activate"
