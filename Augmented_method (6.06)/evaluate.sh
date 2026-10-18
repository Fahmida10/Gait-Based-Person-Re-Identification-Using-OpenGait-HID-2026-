#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENGAIT_DIR="${SCRIPT_DIR}/OpenGait"
CONFIG="${SCRIPT_DIR}/My_Methods/generated_configs_aug/submission_Augmented_stage4_02000.yaml"
LOG="${OPENGAIT_DIR}/submission_aug_stage4_02000.log"
PYTHON=$(conda run -n hid2026 which python)

# Set HID2025_TEST_ROOT to your local HID2025 test pkl directory
HID2025_TEST_ROOT="${HID2025_TEST_ROOT:-/path/to/HID2025_test_pkl}"

# Patch submission yaml with actual paths
sed -i \
    -e "s|dataset_root:.*|dataset_root: ${HID2025_TEST_ROOT}|" \
    -e "s|dataset_partition: /path/to/project/OpenGait|dataset_partition: ${OPENGAIT_DIR}|g" \
    -e "s|restore_hint:.*Checkpoint.*|restore_hint: ${OPENGAIT_DIR}/Checkpoint/Augmented_mixed_stage4_HID_final-02000.pt|g" \
    "${CONFIG}"

cd "${OPENGAIT_DIR}"

echo "Starting evaluation..."
CUDA_VISIBLE_DEVICES=0 ${PYTHON} \
    -m torch.distributed.launch \
    --nproc_per_node=1 \
    opengait/main.py \
    --cfgs "${CONFIG}" \
    --phase test \
    2>&1 | tee "${LOG}"
echo "Evaluation complete. Submission CSV saved to: ${OPENGAIT_DIR}/HID_result/"
