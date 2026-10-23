# Reproducibility Notes and Environment Specifications

## 1. Determinism and Random Seeds
- Random seeds across NumPy (`numpy.random.default_rng(42)`) and PyTorch (`torch.manual_seed(42)`) ensure strict reproducibility of dataset partitions.
- Exact split indices are preserved in:
  - `models/resunet/splits/split_indices_seed42.json`
  - `models/attention_unet_2d/results/split_manifest.json`

## 2. Preserved Original Notebooks
The four primary Jupyter notebooks are preserved byte-for-byte in the organized tree:
1. `models/unet/notebook/U_Net.ipynb`
2. `models/resunet/notebook/ResUnet.ipynb`
3. `models/attention_unet_2d/notebook/2D_Attention_U_Net.ipynb`
4. `models/unet_plus_plus/notebook/Unet++.ipynb`

## 3. Running Locally or in Google Colab
The notebooks originally ran in Google Colab with paths mapped to Google Drive (`/content/drive/MyDrive/...`).
When opening notebooks in Colab:
- Place the repository in Google Drive under `MyDrive/Major_Project` or mount Drive directly.
- Or, execute locally in a virtual environment with JupyterLab using `requirements.txt`.
