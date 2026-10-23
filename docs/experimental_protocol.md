# Experimental Protocols and Training Configurations

## 1. Overview of Experimental Controls
All experiments were conducted on Google Colab GPU runtimes (NVIDIA Tesla T4 / A100 environments).

## 2. Model-Specific Configurations

### A. Baseline U-Net
- **Framework**: PyTorch
- **Input**: $512 \times 512 \times 3$, normalized to $[0, 1]$
- **Batch Size**: 1
- **Epochs**: 20 (Best checkpoint saved at epoch 17, validation loss = 0.6252)
- **Optimizer**: Adam ($	ext{lr} = 10^{-4}$)
- **Loss Function**: Compound loss:
  $$\mathcal{L} = \text{BCEWithLogitsLoss} + \text{SoftDiceLoss} + \text{InvSoftDiceLoss}$$
- **Threshold**: 0.5 (Sigmoid output)

### B. ResUNet
- **Framework**: PyTorch
- **Input**: $512 \times 512 \times 3$, normalized to $[0, 1]$
- **Batch Size**: 1
- **Epochs**: 20 (Checkpoint saved at epoch 20)
- **Optimizer**: Adam ($	ext{lr} = 10^{-4}$)
- **Filters**: [16, 32, 64, 128, 256]
- **Loss Function**: Compound loss:
  $$\mathcal{L} = \text{BCEWithLogitsLoss} + \text{SoftDiceLoss} + \text{InvSoftDiceLoss}$$
- **Threshold**: 0.5 (Sigmoid output)

### C. 2D Attention U-Net (Corrected)
- **Framework**: PyTorch
- **Input**: $256 \times 256 \times 3$
- **Batch Size**: 2
- **Epochs**: 20 (Best validation epoch: 20)
- **Optimizer**: Adam ($	ext{lr} = 2 \times 10^{-4}$)
- **Loss Function**: CrossEntropyLoss (2-class: background and lung)
- **Augmentation**: Synchronized random horizontal flips ($p=0.5$), axial rotations ($\pm 5^\circ$)
- **Correction Applied**: Concatenation at `Up5` uses attention-filtered features $x_5$ instead of unfiltered encoder features $x_4$.

### D. U-Net++
- **Framework**: TensorFlow / Keras 3
- **Input**: $256 \times 256 \times 1$
- **Batch Size**: 2
- **Epochs**: 20
- **Optimizer**: Adam ($	ext{lr} = 2 \times 10^{-4}$)
- **Loss Function**: Binary Cross-Entropy + Dice Loss
- **Dropout Rate**: 0.50 after each convolutional unit
- **Inference Protocol**: Monte Carlo (MC) dropout with $T = 20$ forward stochastic passes, threshold $\tau = 0.5$.
