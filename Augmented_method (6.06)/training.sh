#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENGAIT_DIR="${SCRIPT_DIR}/OpenGait"
CONFIG="${SCRIPT_DIR}/training.yaml"
PYTHON=$(conda run -n hid2026 which python)

# -----------------------------------------------------------
# Patch training.yaml: replace placeholder paths with actual
# paths relative to where this script lives.
# The organiser must set CASIAB_ROOT, HID_ROOT, SUSTECH_ROOT
# to point to their local dataset pkl directories.
# -----------------------------------------------------------
CASIAB_ROOT="${CASIAB_ROOT:-/path/to/datasets/CASIAB-pkl}"
HID_ROOT="${HID_ROOT:-/path/to/datasets/HID2022-pkl}"
SUSTECH_ROOT="${SUSTECH_ROOT:-/path/to/datasets/SUSTech1K-pkl}"

sed -i \
    -e "s|work_dir:.*|work_dir: ${SCRIPT_DIR}/My_Methods|" \
    -e "s|opengait_root:.*|opengait_root: ${OPENGAIT_DIR}|" \
    -e "s|python:.*|python: ${PYTHON}|" \
    -e "s|pretrained_ckpt:.*|pretrained_ckpt: ${OPENGAIT_DIR}/output/HID/Baseline/Baseline/checkpoints/Baseline-60000.pt|" \
    -e "s|dataset_root: /path/to/datasets/CASIAB-pkl|dataset_root: ${CASIAB_ROOT}|" \
    -e "s|dataset_root: /path/to/datasets/HID2022-pkl|dataset_root: ${HID_ROOT}|g" \
    -e "s|dataset_root: /path/to/datasets/SUSTech1K-pkl|dataset_root: ${SUSTECH_ROOT}|" \
    -e "s|dataset_partition: /path/to/project/OpenGait|dataset_partition: ${OPENGAIT_DIR}|g" \
    "${CONFIG}"

cd "${OPENGAIT_DIR}"

echo "Starting full 4-stage training pipeline..."
${PYTHON} "${SCRIPT_DIR}/My_Methods/run_augmented_method.py" \
    --config "${CONFIG}" \
    --resume
echo "Training complete."
