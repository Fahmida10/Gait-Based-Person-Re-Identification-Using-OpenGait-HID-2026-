#!/bin/bash
# ============================================================
# Step 1: Create and activate the conda environment
# ============================================================
set -euo pipefail

ENV_NAME="hid2026"
PYTHON_VERSION="3.9"
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=============================="
echo " Creating conda environment: ${ENV_NAME}"
echo "=============================="
conda create -y -n "${ENV_NAME}" python="${PYTHON_VERSION}"

echo "=============================="
echo " Installing PyTorch (CUDA 12.1)"
echo "=============================="
conda run -n "${ENV_NAME}" pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

echo "=============================="
echo " Installing remaining requirements"
echo "=============================="
conda run -n "${ENV_NAME}" pip install \
    numpy==2.0.2 \
    opencv-python==4.11.0.86 \
    Pillow==11.2.1 \
    PyYAML==6.0.3 \
    tqdm==4.67.1 \
    scipy==1.13.1 \
    einops==0.8.2 \
    matplotlib==3.9.4 \
    yacs==0.1.8 \
    yapf==0.43.0 \
    tensorboard

echo "=============================="
echo " Downloading pretrained baseline checkpoint"
echo "=============================="
OPENGAIT_DIR="${WORKDIR}/OpenGait"
cd "${OPENGAIT_DIR}"
wget -q --show-progress \
    https://github.com/ShiqiYu/OpenGait/releases/download/v1.1/pretrained_hid_model.zip \
    -O pretrained_hid_model.zip
unzip -o pretrained_hid_model.zip -d output/
rm pretrained_hid_model.zip

echo ""
echo "=============================="
echo " Environment setup complete!"
echo " To activate: conda activate ${ENV_NAME}"
echo "=============================="
