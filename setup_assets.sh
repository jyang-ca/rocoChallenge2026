#!/bin/bash
# setup_assets.sh - USD 파일 자동 다운로드 및 설치

set -e

echo "=========================================="
echo "GearboxAssembly Assets Setup"
echo "=========================================="

# 1. Git LFS 초기화
echo "[1/4] Initializing Git LFS..."
git lfs install

# 2. Git LFS pull 시도
echo "[2/4] Attempting to pull LFS files..."
if git lfs pull 2>&1 | grep -q "exceeded its LFS budget"; then
    echo "Git LFS bandwidth limit reached. Using alternative download method..."
    
    # 3. Google Drive에서 직접 다운로드
    echo "[3/4] Downloading assets from Google Drive..."
    
    # gdown 설치 확인
    if ! command -v gdown &> /dev/null; then
        echo "Installing gdown..."
        pip install -q gdown
    fi
    
    # 임시 디렉토리 생성
    TEMP_DIR=$(mktemp -d)
    ASSETS_ZIP="$TEMP_DIR/gearbox_assets.zip"
    
    # Google Drive에서 다운로드
    gdown 'https://drive.google.com/uc?id=1L7u89xxiHGkd72CzvZln3P5uPPqMu7b7' -O "$ASSETS_ZIP"
    
    # 압축 해제
    echo "[4/4] Extracting assets..."
    unzip -q -o "$ASSETS_ZIP" -d "$TEMP_DIR"
    
    # 파일 복사
    ASSETS_DIR="$HOME/gearboxAssembly/source/Galaxea_Lab_External/assets"
    cp -r "$TEMP_DIR/assets"/* "$ASSETS_DIR/"
    
    # 정리
    rm -rf "$TEMP_DIR" "$ASSETS_ZIP"
    
    echo "✅ Assets downloaded and installed successfully!"
else
    echo "✅ Git LFS pull successful!"
fi

# 4. 파일 확인
echo ""
echo "Verifying installation..."
if [ -f "$HOME/gearboxAssembly/source/Galaxea_Lab_External/assets/Props/table/OakTableLarge.usd" ]; then
    echo "✅ OakTableLarge.usd found"
    echo "✅ Total USD files: $(find "$HOME/gearboxAssembly/source/Galaxea_Lab_External/assets" -name '*.usd' -type f | wc -l)"
else
    echo "❌ Warning: OakTableLarge.usd not found"
    exit 1
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="