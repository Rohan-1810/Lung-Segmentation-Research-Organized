# Directory Structure and Provenance Map

This document outlines the organized repository architecture, mapping every subfolder to its scientific role and original source location.

```
Lung-Segmentation-Research-Organized/
│
├── README.md                           # GitHub landing page, key findings, and quick start guide
├── LICENSE                             # MIT Open Source License
├── CITATION.cff                        # Machine-readable academic citation metadata
├── requirements.txt                    # Exact Python pip dependencies
├── environment.yml                     # Conda virtual environment definition
├── .gitignore                          # Excludes large weights, local datasets, and caches
│
├── docs/                               # Comprehensive project documentation
│   ├── project_overview.md             # Clinical context, research aims, and summary of findings
│   ├── project_structure.md            # Detailed structural and provenance guide (this file)
│   ├── dataset_documentation.md        # Montgomery County CXR details, license, and split sizes
│   ├── experimental_protocol.md        # Model hyperparameters, compound losses, and optimizer settings
│   ├── reproducibility_notes.md        # Random seeds, deterministic configurations, and execution tips
│   ├── model_comparison.md             # Multi-dimensional side-by-side comparison table
│   ├── source_attributions.md          # Upstream repository credits and author citations
│   ├── github_readiness_report.md      # Sanitization, secret scanning, and release checklist
│   ├── integrity_verification_report.md# SHA-256 byte-for-byte copy verification
│   ├── audit_report.md                 # Full initial codebase and notebook audit report
│   └── file_manifest.csv               # Complete CSV inventory of all 113 preserved files
│
├── research_paper/                     # Complete academic research paper draft
│   ├── IEEE_research_paper_draft.md    # Formal IEEE conference format paper draft
│   ├── references.md                   # Verified academic references with DOIs and links
│   └── missing_information_checklist.md# Checklist for author details and camera-ready prep
│
├── models/                             # Preserved model artifacts, categorized by architecture
│   ├── unet/
│   │   ├── notebook/                   # Exact original byte-for-byte copy: U_Net.ipynb
│   │   ├── checkpoints/                # unet_best.pth (153.4 MB), unet_latest.pth
│   │   └── results/                    # final_test_results.json, predictions, loss curves
│   ├── resunet/
│   │   ├── notebook/                   # Exact original byte-for-byte copy: ResUnet.ipynb
│   │   ├── checkpoints/                # best_checkpoint.pth (54.2 MB), epoch checkpoints
│   │   ├── results/                    # test_summary.json, test_results_per_image.csv
│   │   ├── config/                     # resunet_experiment_config.json
│   │   ├── history/                    # training_validation_loss.png, training_history.json
│   │   ├── predictions/                # Per-image prediction masks
│   │   ├── splits/                     # split_indices_seed42.json
│   │   └── visualizations/             # test_predictions_overview.png
│   ├── attention_unet_2d/
│   │   ├── notebook/                   # Exact original byte-for-byte copy: 2D_Attention_U_Net.ipynb
│   │   ├── source_code/                # attention_unet.py, attention_unet_corrected.py, utilities.py
│   │   └── results/                    # additional_metrics.csv, test_metrics.json, training_config.json
│   └── unet_plus_plus/
│       ├── notebook/                   # Exact original byte-for-byte copy: Unet++.ipynb
│       ├── checkpoints/                # best_model.keras (103.8 MB), epoch checkpoints
│       ├── experiments/                # MC dropout experiments, experiment_02, experiment_03
│       └── logs/                       # training_history_20261022_090555.json
│
├── datasets/                           # Dataset documentation and automated setup scripts
│   ├── README.md                       # One-line download instructions via Hugging Face Hub
│   └── .gitkeep                        # Preserves directory in version control
│
└── results/                            # Centralized experimental visual artifacts
    └── figures/                        # High-resolution loss curves and prediction grids
```
