# Comparative Evaluation of U-Net Variants for Lung Segmentation in Chest X-Ray Images

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-ff6f00.svg)](https://tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An empirical research study and reproducible benchmark comparing four canonical semantic segmentation architectures for automated pulmonary delineation on clinical chest radiographs (CXRs).

---

## 📌 Executive Summary & Key Results

| Architecture | Parameters | Input Dim | Mean Test Dice | Mean Test IoU | Precision | Recall | Pixel F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline U-Net** | 13.40 M | $512 \times 512$ | 0.9758 | 0.9532 | 0.9800 | **0.9748** | 0.9774 |
| **ResUNet** | **4.72 M** | $512 \times 512$ | 0.9679 | 0.9387 | 0.9749 | 0.9652 | 0.9700 |
| **2D Attention U-Net** | 8.74 M | $256 \times 256$ | **0.9795** | **0.9599** | **0.9883** | 0.9712 | **0.9797** |
| **U-Net++ (MC Dropout)**| 9.04 M | $256 \times 256$ | 0.8701 | 0.7714 | 0.9748 | 0.7912 | 0.8735 |

- **Top Accuracy**: **2D Attention U-Net** delivered the highest overlap and precision, eliminating background false-positives via spatial attention gating.
- **Top Efficiency**: **ResUNet** reduced parameter count by **64.8%** relative to baseline U-Net while maintaining 0.9679 Dice.
- **Audited Codebase**: Identified and fixed an upstream feature-routing defect in the 5th attention gate of Attention U-Net.

---

## 📂 Repository Structure

```
Lung-Segmentation-Research-Organized/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── environment.yml
├── .gitignore
├── docs/
│   ├── project_overview.md
│   ├── project_structure.md
│   ├── dataset_documentation.md
│   ├── experimental_protocol.md
│   ├── reproducibility_notes.md
│   ├── model_comparison.md
│   ├── source_attributions.md
│   ├── github_readiness_report.md
│   ├── integrity_verification_report.md
│   └── file_manifest.csv
├── research_paper/
│   ├── IEEE_research_paper_draft.md
│   ├── references.md
│   └── missing_information_checklist.md
├── models/
│   ├── unet/
│   │   ├── notebook/
│   │   ├── checkpoints/
│   │   └── results/
│   ├── resunet/
│   │   ├── notebook/
│   │   ├── checkpoints/
│   │   ├── results/
│   │   └── splits/
│   ├── attention_unet_2d/
│   │   ├── notebook/
│   │   ├── source_code/
│   │   └── results/
│   └── unet_plus_plus/
│       ├── notebook/
│       ├── checkpoints/
│       └── experiments/
├── datasets/
│   └── README.md
└── results/
    ├── metrics/
    ├── figures/
    └── tables/
```

---

## 🚀 Quick Start

### 1. Clone & Setup Environment
```bash
git clone https://github.com/YourUsername/Lung-Segmentation-Research.git
cd Lung-Segmentation-Research
pip install -r requirements.txt
```

### 2. Download Dataset
See instructions in `datasets/README.md` to acquire the Montgomery County CXR Set.

### 3. Open Notebooks
Each model has a dedicated notebook preserved under `models/<model_name>/notebook/`.

---

## 📄 IEEE Research Paper
The full academic manuscript is available in [`research_paper/IEEE_research_paper_draft.md`](research_paper/IEEE_research_paper_draft.md).
