# MROOD-TRAIN

Unified training and feature-extraction pipeline for medical anomaly detection models.

## Supported Models

| Model | Family |
|-------|--------|
| `draem` | Reconstruction |
| `dinomaly` | Reconstruction |
| `deepsvdd` | One-Class |
| `cutpaste` | One-Class |
| `stfpm` | Knowledge Distillation |
| `rd4ad` | Knowledge Distillation |
| `cfa` | Memory Bank |
| `patchcore` | Memory Bank |
| `fastflow` | Normalizing Flow |
| `cflow` | Normalizing Flow |

## Dataset Structure

All models expect the dataset root to follow this layout:

```
<data_root>/
├── train/
│   └── good/               # Normal training images (PNG)
│       ├── img_001.png
│       ├── img_002.png
│       └── ...
└── valid/
    ├── good/
    │   └── img/            # Normal validation images
    │       ├── img_001.png
    │       └── ...
    └── Ungood/
        ├── img/            # Abnormal validation images
        │   ├── img_001.png
        │   └── ...
        └── label/          # Binary segmentation masks (same filename as img/)
            ├── img_001.png
            └── ...
```

**Notes:**
- Images should be PNG format (NIfTI inputs are auto-converted via `data/dataset_conversion.py`).
- `label/` masks are optional for models that don't use pixel-level supervision (draem, dinomaly, fastflow, cflow, deepsvdd, cutpaste).
- `train/good/` contains only normal (healthy) images — no anomalous samples used during training.

## Usage

### Training

```bash
python train.py --config config/<model>.yaml \
                --data_root /path/to/dataset \
                --name my_experiment
```

Override YAML hyperparameters via CLI flags:

```bash
python train.py --config config/rd4ad.yaml \
                --data_root /data/RESC \
                --name resc_rd4ad \
                --epochs 200 \
                --batch_size 8 \
                --accelerator gpu
```

### Feature Extraction

```bash
python extract.py --config config/<model>.yaml \
                  --checkpoint results/<model>/my_experiment/checkpoints/last.ckpt \
                  --data_root /path/to/dataset \
                  --output_dir /path/to/outputs
```

## Directory Structure

```
MROOD-TRAIN/
├── train.py                  # Unified training entry point
├── extract.py                # Unified feature extraction entry point
├── config/
│   ├── cfa.yaml
│   ├── cflow.yaml
│   ├── custom_cutpaste.yaml  # BMAD-format config used by pytorch-cutpaste/run_training.py
│   ├── custom_DeepSVDD.yaml  # BMAD-format config used by Deep-SVDD/main.py
│   ├── cutpaste.yaml
│   ├── deepsvdd.yaml
│   ├── dinomaly.yaml
│   ├── draem.yaml
│   ├── fastflow.yaml
│   ├── patchcore.yaml
│   ├── rd4ad.yaml
│   └── stfpm.yaml
├── models/
│   ├── flow_models.py        # FastFlow, CFlow registry + builder functions
│   ├── kd_models.py          # RD4AD, STFPM registry
│   ├── memory_models.py      # CFA, PatchCore registry
│   ├── recon_models.py       # DRAEM, Dinomaly registry
│   └── radimagenet_utils.py  # RadImageNet weight loading
├── data/
│   └── dataset_conversion.py # NIfTI → PNG conversion
├── Deep-SVDD/                # BMAD Deep-SVDD implementation (copied from BMAD/Deep-SVDD/)
└── pytorch-cutpaste/         # BMAD CutPaste implementation (copied from BMAD/pytorch-cutpaste/)
```
