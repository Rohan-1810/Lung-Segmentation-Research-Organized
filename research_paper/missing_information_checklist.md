# Missing Information & Research Gap Checklist

This checklist systematically documents every item that requires user clarification, institutional approval, or optional experimental extension prior to formal conference submission.

## 1. Author and Institutional Information
- [ ] **Author Names**: Full names and order of authors.
- [ ] **Affiliations**: Department, University/Institution, City, State/Province, Country.
- [ ] **Corresponding Author Email**: Official institutional email address.
- [ ] **Funding / Grant Acknowledgements**: Grant numbers or sponsoring agencies (if applicable).
- [ ] **Ethics / IRB Statement**: Since MontgomerySet is a de-identified public dataset, an exemption declaration is standard (confirm institutional requirements).

## 2. Experimental Harmonization Decisions
- [ ] **Resolution Standardization**: U-Net and ResUNet were evaluated at $512 \times 512$, whereas Attention U-Net and U-Net++ were evaluated at $256 \times 256$.
  *Status*: Documented clearly in Table II, Table III, and Section IV-E. If identical-resolution benchmarking is desired for camera-ready submission, note that models would require retraining under identical dimensions.
- [ ] **Framework Discrepancy**: U-Net, ResUNet, and Attention U-Net are implemented in PyTorch, whereas U-Net++ is implemented in TensorFlow/Keras.
  *Status*: Fully documented in methodology and Table II.
- [ ] **Test Set Overlap**: U-Net and ResUNet share the exact same 15 test radiographs (`torch.manual_seed(42)`). Attention U-Net and U-Net++ share an identical 15-case test cohort drawn from `split_manifest.json` (`numpy.default_rng(42)`). Overlap between the two groups includes `MCUCXR_0051_0.png`.
  *Status*: Fully disclosed in Section III-C and Table I.

## 3. GitHub & Codebase Decisions
- [ ] **License Selection**: Current draft includes a proposed MIT License (flagged for student/advisor confirmation).
- [ ] **Model Weights Hosting**: Total trained weights across the project exceed 1.5 GB. Git LFS or external release assets (e.g., GitHub Releases / Zenodo / Google Drive) are recommended for public deployment.
- [ ] **Dataset Redistribution**: The Montgomery County CXR dataset is public domain for research under NLM/NIH, but due to file size (589 MB), dataset files should remain excluded from Git (`.gitignore`) and downloaded via automated scripts (`datasets/README.md`).
