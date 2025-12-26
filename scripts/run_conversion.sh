#!/bin/bash
# RoCoChallenge 데이터셋을 LeRobot 형식으로 변환하는 배치 스크립트
# 192개 파일을 50개씩 4배치로 처리

set -e  # 에러 발생 시 중단

INPUT_REPO="https://huggingface.co/datasets/rocochallenge2025/rocochallenge2025"
OUTPUT_REPO="https://huggingface.co/datasets/yjsm1203/roco_3"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=========================================="
echo "RoCoChallenge -> LeRobot 변환 시작"
echo "입력: $INPUT_REPO"
echo "출력: $OUTPUT_REPO"
echo "=========================================="

# 배치 1: 0-49 (50개) - 새 데이터셋 생성
echo ""
echo "[배치 1/4] 에피소드 0-49 처리 중..."
python "$SCRIPT_DIR/to_lerobot_dataset.py" \
    --input "$INPUT_REPO" \
    --output "$OUTPUT_REPO" \
    --start-episode 0 \
    --end-episode 50 \
    --clean-cache

echo "[배치 1/4] 완료!"

# 배치 2: 50-99 (50개) - 기존에 추가
echo ""
echo "[배치 2/4] 에피소드 50-99 처리 중..."
python "$SCRIPT_DIR/to_lerobot_dataset.py" \
    --input "$INPUT_REPO" \
    --output "$OUTPUT_REPO" \
    --start-episode 50 \
    --end-episode 100 \
    --resume \
    --clean-cache

echo "[배치 2/4] 완료!"

# 배치 3: 100-149 (50개) - 기존에 추가
echo ""
echo "[배치 3/4] 에피소드 100-149 처리 중..."
python "$SCRIPT_DIR/to_lerobot_dataset.py" \
    --input "$INPUT_REPO" \
    --output "$OUTPUT_REPO" \
    --start-episode 100 \
    --end-episode 150 \
    --resume \
    --clean-cache

echo "[배치 3/4] 완료!"

# 배치 4: 150-끝 (나머지) - 기존에 추가
echo ""
echo "[배치 4/4] 에피소드 150-끝 처리 중..."
python "$SCRIPT_DIR/to_lerobot_dataset.py" \
    --input "$INPUT_REPO" \
    --output "$OUTPUT_REPO" \
    --start-episode 150 \
    --resume \
    --clean-cache

echo "[배치 4/4] 완료!"

echo ""
echo "=========================================="
echo "전체 변환 완료!"
echo "결과: $OUTPUT_REPO"
echo "=========================================="
