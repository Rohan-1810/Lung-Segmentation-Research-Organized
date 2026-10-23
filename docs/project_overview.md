# Project Overview: Automated Lung Segmentation Research

## 1. Executive Summary
This repository contains the complete experimental artifacts, trained models, source code, and IEEE-formatted research paper draft for the major project:
**"Comparative Evaluation of U-Net Variants for Lung Segmentation in Chest X-Ray Images"**.

The research evaluates four deep learning semantic segmentation architectures:
1. **Baseline U-Net**
2. **ResUNet** (Deep Residual U-Net)
3. **2D Attention U-Net** (with corrected skip attention gating)
4. **U-Net++** (Nested Dense Skip Pathways with Monte Carlo Dropout)

All models are systematically evaluated on the **Montgomery County Chest X-ray Dataset** (138 frontal CXRs; 80 normal, 58 tuberculosis cases).

## 2. Key Research Findings
- **Highest Segmentation Overlap**: **2D Attention U-Net** achieved the top score across all overlap and precision metrics (Mean Test Dice: **0.9795**, IoU: **0.9599**, Precision: **0.9883**).
- **Most Parameter-Efficient**: **ResUNet** achieved **0.9679 Dice** with only **4.72M parameters** (64.8% fewer parameters than baseline U-Net's 13.40M), demonstrating exceptional feasibility for edge and point-of-care devices.
- **Strong Baseline Performance**: Classic **U-Net** achieved **0.9758 Dice** and **0.9532 IoU**, demonstrating that balanced compound loss functions enable standard U-Net to perform on par with advanced variants.
- **Under-segmentation in U-Net++**: Dense nested skip paths paired with heavy dropout ($p = 0.5$) resulted in conservative boundary predictions (Recall: **0.7912**), which was alleviated via test-time Monte Carlo dropout ($T = 20$).
- **Upstream Bug Discovery**: Code auditing revealed that the reference implementation of 2D Attention U-Net inadvertently bypassed the 5th attention gate during skip concatenation. Correcting this routing restored intended attention mechanics.
