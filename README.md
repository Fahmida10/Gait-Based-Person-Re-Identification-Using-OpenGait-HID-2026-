# Gait-Based Person Re-Identification Using OpenGait – HID 2026

## Competition Submission

This repository contains my submission for the **Human Identification at a Distance (HID) 2026 competition**. The project focuses on gait-based person re-identification using deep learning, building upon the existing [OpenGait framework](https://github.com/ShiqiYu/OpenGait).

The work uses an existing OpenGait baseline and adapts it through a customised multi-stage fine-tuning strategy across multiple gait datasets. The objective was to improve the model's ability to learn discriminative gait representations and generalise across different data sources.

## Project Overview

Gait recognition is a biometric identification approach that identifies individuals through their walking patterns. It is particularly useful in scenarios where facial information may be unavailable or difficult to obtain.

For this competition, I developed a training pipeline based on the OpenGait baseline, with sequential fine-tuning across multiple datasets and additional data augmentation.

**Key aspects of the submission:**

- OpenGait Baseline architecture as the starting point.
- Pretrained model initialisation.
- Sequential multi-stage fine-tuning across CASIA-B, HID2022 and SUSTech1K.
- Data augmentation to improve training diversity.
- Triplet loss and cross-entropy loss for representation learning and classification.
- Evaluation on the HID competition test data.

## Methodology

Rather than introducing a new network architecture, this submission focuses on adapting and fine-tuning an existing gait recognition model.

The original OpenGait Baseline was used as the foundation. Its pretrained weights were used to initialise the training process, followed by sequential fine-tuning on multiple datasets.

### Multi-Stage Fine-Tuning Strategy

The training pipeline consists of four sequential stages:

| Stage | Dataset | Training Iterations |
|---|---|---:|
| 1 | CASIA-B | 4,000 |
| 2 | HID2022 | 3,000 |
| 3 | SUSTech1K | 4,000 |
| 4 | HID2022 | 2,000 |

The checkpoint obtained from each stage was transferred to the next stage, allowing the model to continue learning from the previously trained weights.

### Training and Optimisation

The training pipeline incorporates:

- **Data augmentation:** Random horizontal flipping, random rotation, random erasing and random perspective transformation.
- **Loss functions:** Triplet loss and cross-entropy loss.
- **Optimiser:** Stochastic Gradient Descent (SGD).
- **Learning-rate scheduler:** MultiStepLR.

These techniques were used as part of the fine-tuning process to improve feature learning and model generalisation.

## Results

The repository reports the following result:

| Evaluation Metric | Reported Performance |
|---|---:|
| Rank-1 Accuracy | 59.09% |

The reported result is from evaluation on the HID2025 test set, as documented in the project implementation.

## Repository Structure

```text
Gait-Based-Person-Re-Identification-Using-OpenGait-HID-2026/
│
└── Augmented_method (6.06)/
    ├── README.md
    ├── My_Methods/
    │   └── run_augmented_method.py
    ├── training.yaml
    ├── training.sh
    └── OpenGait/
```

The `Augmented_method (6.06)` directory contains the implementation and detailed reproduction instructions for the competition submission.

## Reproduction

For the complete instructions on setting up the environment, preparing the datasets, configuring the training pipeline and reproducing the experiments, see:

**[Competition Implementation and Reproduction Guide](./Augmented_method%20%286.06%29/README.md)**

## Framework Acknowledgement

This project builds upon [OpenGait](https://github.com/ShiqiYu/OpenGait), an open-source gait recognition framework.

The original OpenGait authors and contributors are acknowledged for providing the baseline framework used in this submission.

## Competition

**Human Identification at a Distance (HID) 2026**

This repository documents the implementation and training pipeline developed for the competition submission.
