# Augmented Method — Reproduction Guide

Rank-1 Accuracy: **59.09%** on HID2025 test set.

---

## Requirements

- Linux OS
- CUDA 12.1
- Anaconda or Miniconda
- The following datasets pre-processed into `.pkl` format:
  - **CASIA-B** (124 identities)
  - **HID2022** (500 identities)
  - **SUSTech1K** (1050 identities)
  - **HID2025 test set** (gallery + probe, for evaluation only)

---

## Step 1 — Set Up the Environment

```bash
bash setup_env.sh
conda activate hid2026
```

This will:
- Create the `hid2026` conda environment with Python 3.9
- Install all dependencies (PyTorch 2.5.1 + CUDA 12.1, etc.)
- Download and extract the pretrained HID2022 Baseline checkpoint automatically

---

## Step 2 — Configure Paths (REQUIRED)

Before running training, you **must** update the following paths in `training.yaml`:

| Field | Location in `training.yaml` | What to set |
|---|---|---|
| `work_dir` | `project.work_dir` | Absolute path to the `My_Methods/` folder inside this project |
| `opengait_root` | `project.opengait_root` | Absolute path to the `OpenGait/` folder inside this project |
| `python` | `project.python` | Absolute path to the Python binary in your `hid2026` conda env (run `conda run -n hid2026 which python` to find it) |
| `pretrained_ckpt` | `checkpoint.pretrained_ckpt` | Absolute path to `OpenGait/output/HID/Baseline/Baseline/checkpoints/Baseline-60000.pt` (extracted by `setup_env.sh`) |
| `dataset_root` (Stage 1 — CASIA-B) | `datasets[0].dataset_root` | Absolute path to your local CASIA-B `.pkl` directory |
| `dataset_root` (Stage 2 — HID) | `datasets[1].dataset_root` | Absolute path to your local HID2022 `.pkl` directory |
| `dataset_root` (Stage 3 — SUSTech1K) | `datasets[2].dataset_root` | Absolute path to your local SUSTech1K `.pkl` directory |
| `dataset_root` (Stage 4 — HID) | `datasets[3].dataset_root` | Absolute path to your local HID2022 `.pkl` directory (same as Stage 2) |
| `dataset_partition` (all stages) | `datasets[*].dataset_partition` | Replace `/path/to/project/OpenGait` with the absolute path to your `OpenGait/` folder |

**Alternatively**, you can let `training.sh` patch these automatically by passing the dataset roots as environment variables:

```bash
CASIAB_ROOT=/your/path/to/CASIAB-pkl \
HID_ROOT=/your/path/to/HID2022-pkl \
SUSTECH_ROOT=/your/path/to/SUSTech1K-pkl \
bash training.sh
```

The script will auto-patch `work_dir`, `opengait_root`, `python`, and `pretrained_ckpt` from the script's own location — **you only need to set the three dataset roots**.

---

## Step 3 — Train

```bash
bash training.sh
```

This runs 4 sequential fine-tuning stages:

| Stage | Dataset | Iterations | Identities |
|---|---|---|---|
| 1 | CASIA-B | 4,000 | 124 |
| 2 | HID2022 | 3,000 | 500 |
| 3 | SUSTech1K | 4,000 | 1,050 |
| 4 | HID2022 | 2,000 | 500 |

The `--resume` flag is set, so if training is interrupted it will skip already-completed stages and continue from the last saved checkpoint.

The final checkpoint will be saved to:
```
OpenGait/output/HID/Baseline/Augmented_mixed_stage4_HID_final/checkpoints/Augmented_mixed_stage4_HID_final-02000.pt
```

> **Note:** The final checkpoint used for submission is already included at `OpenGait/Checkpoint/Augmented_mixed_stage4_HID_final-02000.pt`. You can skip training and go directly to Step 4 to reproduce the submission.

---

## Step 4 — Evaluate (Generate Submission CSV)

Before running evaluation, update the following path in
`My_Methods/generated_configs_aug/submission_Augmented_stage4_02000.yaml`:

| Field | What to set |
|---|---|
| `data_cfg.dataset_root` | Absolute path to your local HID2025 test `.pkl` directory (gallery + probe) |
| `data_cfg.dataset_partition` | Replace `/path/to/project/OpenGait` with your absolute path to `OpenGait/` |
| `evaluator_cfg.restore_hint` | Replace `/path/to/project/OpenGait` with your absolute path to `OpenGait/` |
| `trainer_cfg.restore_hint` | Same as above |

**Alternatively**, pass the test dataset path as an environment variable and let `evaluate.sh` patch automatically:

```bash
HID2025_TEST_ROOT=/your/path/to/HID2025_test_pkl bash evaluate.sh
```

The submission CSV will be saved to:
```
OpenGait/HID_result/<timestamp>.csv
```

---

## Summary — Three Commands

```bash
# 1. Create and activate conda environment
bash setup_env.sh && conda activate hid2026

# 2. Train (set your dataset paths)
CASIAB_ROOT=/your/CASIAB-pkl \
HID_ROOT=/your/HID2022-pkl \
SUSTECH_ROOT=/your/SUSTech1K-pkl \
bash training.sh

# 3. Evaluate
HID2025_TEST_ROOT=/your/HID2025_test_pkl bash evaluate.sh
```
