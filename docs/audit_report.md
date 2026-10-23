# Comprehensive Implementation & Experimental Audit Report

## 1. Audit Scope
This audit report documents the systematic inspection of all four major project notebooks, configuration files, checkpoint binaries, and evaluation logs stored in the original workspace.

## 2. Notebook-by-Notebook Audit Findings

### A. U_Net.ipynb
- **Platform**: Google Colab / PyTorch 2.x
- **Cell Count**: 47 cells
- **Architecture**: Classic symmetrical encoder-decoder with double convolution units `(Conv2d -> BatchNorm2d -> ReLU) * 2`.
  - Contracting channels: `64 -> 128 -> 256 -> 512 -> 1024`
  - Expansive channels: `1024 -> 512 -> 256 -> 128 -> 64 -> 1`
  - Total Parameters: **13,395,329** (~13.40M)
- **Dataset & Split**:
  - Montgomery County CXR (138 frontal images, 80 normal, 58 TB).
  - Train: 110, Val: 13, Test: 15 (`torch.manual_seed(42)`).
  - Both lungs combined via logical OR. Resolution: $512 \times 512 \times 3$.
- **Training Config**: 20 epochs, batch size 1, Adam optimizer ($	ext{lr} = 10^{-4}$).
- **Loss Function**: `BCEWithLogitsLoss + SoftDiceLoss + InvSoftDiceLoss`.
- **Test Results Verified**:
  - Mean Test Dice: **0.9758** (97.58%)
  - Mean Test IoU: **0.9532** (95.32%)
  - Pixel Precision: **0.9800**, Recall: **0.9748**, F1: **0.9774**, Accuracy: **0.9883**
  - Best Test Case: `MCUCXR_0051_0.png` (Dice: 0.9856, IoU: 0.9717)
  - Worst Test Case: `MCUCXR_0399_1.png` (Dice: 0.9180, IoU: 0.8484)

### B. ResUnet.ipynb
- **Platform**: Google Colab / PyTorch 2.x
- **Cell Count**: 36 cells
- **Source Reference**: Adapted from `mhmodir/Chest-X-Ray-Image_Segmentation_ResUNet` (`main/Chest_Xray_resunet.ipynb`).
- **Architecture**: Residual blocks with pre-activation batch normalization and $1 \times 1$ conv identity projections.
  - Channels: `[16, 32, 64, 128, 256]`
  - Total Parameters: **4,715,761** (~4.72M) — 64.8% fewer parameters than U-Net.
- **Dataset & Split**: Identical 110 train, 13 val, 15 test cohort as U-Net.
- **Training Config**: 20 epochs, batch size 1, Adam ($	ext{lr} = 10^{-4}$), compound BCE + SoftDice + InvDice loss.
- **Test Results Verified**:
  - Mean Test Dice: **0.9679** (96.79%)
  - Mean Test IoU: **0.9387** (93.87%)
  - Pixel Precision: **0.9749**, Recall: **0.9652**, F1: **0.9700**, Accuracy: **0.9845**
  - Best Test Case: `MCUCXR_0051_0.png` (Dice: 0.9870, IoU: 0.9743)
  - Worst Test Case: `MCUCXR_0399_1.png` (Dice: 0.8944, IoU: 0.8090)

### C. 2D_Attention_U_Net.ipynb
- **Platform**: Google Colab / PyTorch 2.x
- **Cell Count**: 66 cells
- **Source Reference**: `aimalrehman92/Attention-U-Net-for-Lung-Segmentation-on-ChestXray-Scans`.
- **Architecture**: 2D Attention U-Net with additive spatial attention gates.
  - Channels: `[16, 32, 64, 128, 256, 512]`
  - Total Parameters: **8,744,945** (~8.74M).
- **Critical Discovery & Fix**:
  - In `source_code/attention_unet.py` line 132, the original code had:
    `x = torch.cat((x4, x), dim=1)`
    which bypassed the attention gate `Att5`!
  - The author corrected this to:
    `x = torch.cat((x5, x), dim=1)`
    and verified that attention features were properly concatenated.
- **Training Config**: 20 epochs, batch size 2, Adam ($	ext{lr} = 2 \times 10^{-4}$), CrossEntropyLoss (2-class).
- **Test Results Verified**:
  - Mean Test Dice: **0.9795** (97.95%) — **Highest overall score**
  - Mean Test IoU: **0.9599** (95.99%)
  - Pixel Precision: **0.9883**, Recall: **0.9712**, F1: **0.9797**, Accuracy: **0.9895**
  - Best Test Case: `MCUCXR_0051_0.png` (Dice: 0.9871, IoU: 0.9745)
  - Worst Test Case: `MCUCXR_0282_1.png` (Dice: 0.9644, IoU: 0.9313)

### D. Unet++.ipynb
- **Platform**: Google Colab / TensorFlow & Keras 3
- **Cell Count**: 206 cells
- **Architecture**: Nested dense skip pathways.
  - Channels: `[32, 64, 128, 256, 512]`
  - Total Parameters: **9,041,601** (~9.04M).
  - Deep supervision: Disabled (`deep_supervision = False`).
  - Regularization: Dropout rate $0.50$ after every standard unit.
- **Findings & Monte Carlo Dropout Protocol**:
  - Due to heavy dropout, deterministic evaluation yielded conservative boundaries (Recall: 0.7912).
  - In Experiment 04, test-time Monte Carlo (MC) dropout was applied ($T = 20$ stochastic forward passes).
  - Test Results Verified:
    - Mean Image Dice: **0.8701** (87.01%), Pooled Dice: **0.8735** (87.35%)
    - Mean Image IoU: **0.7714** (77.14%), Pooled IoU: **0.7754** (77.54%)
    - Precision: **0.9748**, Recall: **0.7912**, F1: **0.8735**
    - Total test inference time: 12.13 seconds.
