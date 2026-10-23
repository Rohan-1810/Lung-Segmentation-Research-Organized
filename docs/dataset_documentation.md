# Dataset Documentation: Montgomery County Chest X-Ray Set

## 1. Provenance and Attribution
- **Dataset Name**: Montgomery County Chest X-ray Set (`MontgomerySet`)
- **Originating Institution**: Department of Health and Human Services, Montgomery County, Maryland, USA in collaboration with the National Library of Medicine (NLM), National Institutes of Health (NIH), Bethesda, MD, USA.
- **Reference Publication**:
  > S. Jaeger, S. Candemir, S. Antani, Y.-X. J. Wáng, P.-X. Lu, and G. Thoma, "Two public chest X-ray datasets for computer-aided screening of pulmonary diseases," *Quantitative Imaging in Medicine and Surgery*, vol. 4, no. 6, pp. 475–477, Dec. 2014. [DOI: 10.3978/j.issn.2223-4292.2014.11.20](https://doi.org/10.3978/j.issn.2223-4292.2014.11.20).
- **License**: Public domain for research and educational purposes with appropriate attribution to NLM/NIH.

## 2. Dataset Composition
- **Total Images**: 138 frontal posterior-anterior (PA) chest radiographs in PNG format.
  - **Normal Cases**: 80 images (`MCUCXR_0001_0.png` through `MCUCXR_0108_0.png`).
  - **Tuberculosis (TB) Cases**: 58 images (`MCUCXR_0113_1.png` through `MCUCXR_0399_1.png`).
- **Original Dimensions**: $4020 \times 4892$ or $4892 \times 4020$ pixels (12-bit grayscale depth).
- **Ground Truth Format**:
  - Independent left lung mask: `ManualMask/leftMask/` (138 PNGs)
  - Independent right lung mask: `ManualMask/rightMask/` (138 PNGs)
  - Mask values: 0 (background), 255 (lung parenchyma)

## 3. Preprocessing and Mask Fusion
The two separate lung masks are fused into a unified binary mask:
```python
left = np.array(Image.open(left_mask_path)).astype(bool)
right = np.array(Image.open(right_mask_path)).astype(bool)
combined = np.logical_or(left, right).astype(np.uint8) * 255
```
Target spatial resolutions:
- $512 \times 512$ for Baseline U-Net and ResUNet.
- $256 \times 256$ for 2D Attention U-Net and U-Net++.
Interpolation: Bilinear for radiographs; Nearest-Neighbor for binary segmentation masks.

## 4. Partition Scheme
- **Training Set**: 110 images (~80%)
- **Validation Set**: 13 images (~10%)
- **Test Set**: 15 images (~10%)
Partition is fully reproducible via `seed = 42` and documented in `split_indices_seed42.json` and `split_manifest.json`.
