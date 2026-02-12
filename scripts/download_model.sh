#!/bin/bash
# YOLOv8n ONNX 모델 다운로드 스크립트
set -e

MODEL_DIR="$(cd "$(dirname "$0")/.." && pwd)/models"
MODEL_FILE="$MODEL_DIR/yolov8n.onnx"
URL="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.onnx"
MIN_SIZE=1000000  # 최소 1MB — 정상 파일은 ~12MB

mkdir -p "$MODEL_DIR"

# 파일이 존재하더라도 크기가 너무 작으면 깨진 것으로 판단
if [ -f "$MODEL_FILE" ]; then
    FILE_SIZE=$(stat -c%s "$MODEL_FILE" 2>/dev/null || echo 0)
    if [ "$FILE_SIZE" -gt "$MIN_SIZE" ]; then
        echo "[INFO] 모델이 이미 존재합니다: $MODEL_FILE ($(du -h "$MODEL_FILE" | cut -f1))"
        exit 0
    else
        echo "[WARN] 기존 파일이 손상되었습니다 (${FILE_SIZE} bytes). 재다운로드합니다."
        rm -f "$MODEL_FILE"
    fi
fi

echo "[INFO] YOLOv8n ONNX 모델 다운로드 중..."
echo "[INFO] URL: $URL"

if command -v wget &> /dev/null; then
    wget -O "$MODEL_FILE" "$URL"
elif command -v curl &> /dev/null; then
    curl -L -o "$MODEL_FILE" "$URL"
else
    echo "[ERROR] wget 또는 curl이 필요합니다."
    exit 1
fi

# 다운로드 결과 검증
if [ ! -f "$MODEL_FILE" ]; then
    echo "[ERROR] 다운로드 실패: 파일이 생성되지 않았습니다."
    exit 1
fi

FILE_SIZE=$(stat -c%s "$MODEL_FILE" 2>/dev/null || echo 0)
if [ "$FILE_SIZE" -lt "$MIN_SIZE" ]; then
    echo "[ERROR] 다운로드 실패: 파일 크기가 비정상입니다 (${FILE_SIZE} bytes)"
    rm -f "$MODEL_FILE"
    exit 1
fi

echo "[INFO] 다운로드 완료: $MODEL_FILE ($(du -h "$MODEL_FILE" | cut -f1))"
