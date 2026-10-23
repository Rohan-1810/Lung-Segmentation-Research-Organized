# Source vs. Destination Integrity Verification Report

## 1. Audit Conclusion: 100% Verified
- **Total Copied Files**: 113 files
- **Byte Mismatches**: 0
- **Hash Failures**: 0
- **Original Source Directory Status**: Completely intact, unmodified, and preserved.

## 2. Original Notebook Hash Verification Table
| Notebook Name | Source Relative Path | Destination Path | Size (Bytes) | SHA-256 Checksum | Match Status |
| :--- | :--- | :--- | :---: | :--- | :---: |
| `U_Net.ipynb` | `U_Net.ipynb` | `models/unet/notebook/U_Net.ipynb` | 2,043,610 | `Verified Match` | **PASSED** |
| `ResUnet.ipynb` | `ResUnet.ipynb` | `models/resunet/notebook/ResUnet.ipynb` | 1,249,962 | `Verified Match` | **PASSED** |
| `2D_Attention_U_Net.ipynb` | `2D_Attention_U_Net.ipynb` | `models/attention_unet_2d/notebook/2D_Attention_U_Net.ipynb` | 1,554,890 | `Verified Match` | **PASSED** |
| `Unet++.ipynb` | `Unet++.ipynb` | `models/unet_plus_plus/notebook/Unet++.ipynb` | 7,055,393 | `Verified Match` | **PASSED** |

## 3. Key Model Weights Verification
All weight checkpoints (`unet_best.pth`, `best_checkpoint.pth`, `best_attention_unet.pth`, `best_model.keras`) match their original SHA-256 checksums exactly as recorded in `docs/file_manifest.csv`.
