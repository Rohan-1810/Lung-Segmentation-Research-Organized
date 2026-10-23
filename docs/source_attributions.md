# Source Attributions and Third-Party Dependencies

This project builds upon, benchmarks, and credits the following authoritative implementations and repositories:

1. **U-Net Reference Implementation**:
   - Original Authors: Olaf Ronneberger, Philipp Fischer, Thomas Brox (2015).
   - Paper: *U-Net: Convolutional Networks for Biomedical Image Segmentation* (MICCAI 2015).

2. **ResUNet Reference Implementation**:
   - Repository: [mhmodir/Chest-X-Ray-Image_Segmentation_ResUNet](https://github.com/mhmodir/Chest-X-Ray-Image_Segmentation_ResUNet)
   - Notebook: `main/Chest_Xray_resunet.ipynb`
   - License: MIT License

3. **2D Attention U-Net Implementation**:
   - Reference Repository: [aimalrehman92/Attention-U-Net-for-Lung-Segmentation-on-ChestXray-Scans](https://github.com/aimalrehman92/Attention-U-Net-for-Lung-Segmentation-on-ChestXray-Scans)
   - Authors: O. Oktay et al. (MIDL 2018)
   - Note: Audited and corrected for `Att5` skip concatenation bug in `models/attention_unet_2d/source_code/`.

4. **U-Net++ Implementation**:
   - Authors: Zongwei Zhou, Md Mahfuzur Rahman Siddiquee, Nima Tajbakhsh, Jianming Liang (2018).
   - Paper: *UNet++: A Nested U-Net Architecture for Medical Image Segmentation* (DLMIA 2018).
   - Implementation adapted for Keras 3 compatibility.

5. **Dataset Provider**:
   - Montgomery County Chest X-ray Set, National Library of Medicine (NLM), National Institutes of Health (NIH).
