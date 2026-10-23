# Comparative Technical Profile of Evaluated Models

| Dimension | Baseline U-Net | ResUNet | 2D Attention U-Net (Corrected) | U-Net++ |
| :--- | :--- | :--- | :--- | :--- |
| **Original Notebook** | `U_Net.ipynb` | `ResUnet.ipynb` | `2D_Attention_U_Net.ipynb` | `Unet++.ipynb` |
| **Framework** | PyTorch 2.x | PyTorch 2.x | PyTorch 2.x | TensorFlow / Keras 3 |
| **Primary Reference** | Ronneberger et al. [7] | Zhang et al. [8] / `mhmodir` CXR repo | Oktay et al. [9] / `aimalrehman92` repo | Zhou et al. [10] / Keras nested |
| **Total Parameters** | 13,395,329 (~13.40M) | **4,715,761** (~4.72M) | 8,744,945 (~8.74M) | 9,041,601 (~9.04M) |
| **Input Shape** | $512 \times 512 \times 3$ | $512 \times 512 \times 3$ | $256 \times 256 \times 3$ | $256 \times 256 \times 1$ |
| **Number of Classes** | 1 (Sigmoid output) | 1 (Sigmoid output) | 2 (Softmax/Argmax output) | 1 (Sigmoid output) |
| **Skip Connection Type** | Hard Identity Concat | Residual Shortcut Additions | Additive Spatial Attention Gates | Nested Dense Convolutional Blocks |
| **Loss Function** | BCE + SoftDice + InvDice | BCE + SoftDice + InvDice | CrossEntropyLoss (2-class) | BCE + Dice Loss |
| **Optimizer** | Adam (lr = 1e-4) | Adam (lr = 1e-4) | Adam (lr = 2e-4) | Adam (lr = 2e-4) |
| **Batch Size** | 1 | 1 | 2 | 2 |
| **Epochs** | 20 (Best: 17) | 20 (Best: 20) | 20 (Best: 20) | 20 (Best checkpoint used) |
| **Test Set Size** | 15 images | 15 images | 15 images | 15 images |
| **Mean Test Dice** | **0.9758** | **0.9679** | **0.9795** | **0.8701** (Pooled: 0.8735) |
| **Mean Test IoU** | **0.9532** | **0.9387** | **0.9599** | **0.7714** (Pooled: 0.7754) |
| **Pixel Precision** | 0.9800 | 0.9749 | **0.9883** | 0.9748 |
| **Pixel Recall** | **0.9748** | 0.9652 | 0.9712 | 0.7912 |
| **Pixel F1-Score** | 0.9774 | 0.9700 | **0.9797** | 0.8735 |
| **Pixel Accuracy** | 0.9883 | 0.9845 | **0.9895** | 0.9412 |
| **Best Case** | `MCUCXR_0051_0` (0.9856) | `MCUCXR_0051_0` (0.9870) | `MCUCXR_0051_0` (0.9871) | - |
| **Worst Case** | `MCUCXR_0399_1` (0.9180) | `MCUCXR_0399_1` (0.8944) | `MCUCXR_0282_1` (0.9644) | - |
| **Special Observation**| Standard topology is robust | 64.8% fewer parameters than U-Net | Att5 skip bug identified & resolved | Severe dropout (0.5) causes under-seg |
