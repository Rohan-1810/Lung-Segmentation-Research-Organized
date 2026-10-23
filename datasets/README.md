# Dataset Setup Instructions: Montgomery County Chest X-Ray Set

## 1. Automatic Download via Hugging Face Hub
You can automatically download the complete MontgomerySet using `huggingface_hub`:

```bash
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
from pathlib import Path
target_dir = Path('./datasets/MontgomerySet')
print('Downloading MontgomerySet to', target_dir)
snapshot_download(repo_id='Famatsu123/montgomery-shenzhen-tuberculosis-cxr',
                  allow_patterns='MontgomerySet/**',
                  local_dir='./datasets',
                  repo_type='dataset')
print('Download complete!')
"
```

## 2. Expected Directory Structure
```
datasets/
└── MontgomerySet/
    ├── CXR_png/
    │   ├── MCUCXR_0001_0.png
    │   └── ... (138 radiographs)
    └── ManualMask/
        ├── leftMask/
        │   └── ... (138 PNG masks)
        └── rightMask/
            └── ... (138 PNG masks)
```
