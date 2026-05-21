# Multi-Label Image Classifier Web

A training and inference system for multi-label object classification. The project has two connected workflows:

1. **Training/export**: train an EfficientNet classifier on VOC-style multi-label image annotations, evaluate it, tune thresholds, and export a reusable model artifact.
2. **Web inference**: users upload an image through a React frontend and receive predicted labels with confidence scores from a FastAPI backend.

The default class set is the full 20-class Pascal VOC label set:

```text
aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow,
diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa, train, tvmonitor
```

## Project layout

```text
multilabel-image-classifier-web/
├── backend/        # FastAPI inference API
├── frontend/       # React/Vite upload + prediction UI
├── src/            # Reusable ML package
├── scripts/        # train/evaluate/predict/export/inspect/dataset prep entrypoints
├── configs/        # YAML training configs
├── notebooks/      # Colab launcher notebook
├── checkpoints/    # exported model artifacts; ignored except .gitkeep
├── outputs/        # training reports/plots; ignored except .gitkeep
├── sample_images/  # optional local samples; ignored except .gitkeep
├── raw/            # local raw datasets; ignored
├── data/           # local processed datasets; ignored
├── tests/          # package tests
└── docs/           # architecture and usage docs
```

## Runtime split

Use your laptop for lightweight tasks and the web app. Use Colab GPU for real training.

```text
Laptop:
  prepare/inspect dataset
  run FastAPI backend
  run React frontend
  run small smoke tests

Colab GPU:
  train EfficientNet
  evaluate
  tune thresholds
  export model artifact

Laptop again:
  copy checkpoints/exported/model.pt and metadata.json back into checkpoints/exported/
  run the inference web app
```

CPU training on a laptop is possible but slow. For real VOC20 training, use Colab GPU.

## Install

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\activate
pip install -e ".[dev,train]"
```

macOS/Linux:

```bash
source .venv/bin/activate
pip install -e ".[dev,train]"
```

For CUDA-specific PyTorch builds, use the official PyTorch install selector when needed. The package does not force a single CUDA build because local and Colab environments differ.

## Prepare VOC2012 dataset

This project expects VOC-style processed data with image folders and multi-label CSV files. If you downloaded:

```text
raw/VOCtrainval_11-May-2012.tar
```

run:

```powershell
python scripts/prepare_voc2012.py
```

Expected output:

```text
data/voc2012_processed/
├── train/
│   └── images/
├── val/
│   └── images/
├── train_annotations.csv
└── val_annotations.csv
```

The generated CSV files include all 20 Pascal VOC classes as binary columns.

If you want to save disk space on Windows, try hardlinks:

```powershell
python scripts/prepare_voc2012.py --file-mode hardlink
```

Use relative paths in `configs/voc20.yaml`:

```yaml
dataset:
  train_images: data/voc2012_processed/train/images
  val_images: data/voc2012_processed/val/images
  train_annotations: data/voc2012_processed/train_annotations.csv
  val_annotations: data/voc2012_processed/val_annotations.csv
  image_name_column: image_name
  class_mode: all
```

Inspect the prepared dataset:

```powershell
python scripts/inspect_dataset.py --config configs/voc20.yaml
```

You can also inspect the legacy 8-class config:

```powershell
python scripts/inspect_dataset.py --config configs/voc8_legacy.yaml
```

## Train

After preparing the dataset and checking `configs/voc20.yaml`, run:

```powershell
python scripts/train.py --config configs/voc20.yaml
```

Training writes checkpoints and reports under:

```text
checkpoints/
outputs/
```

The training pipeline supports:

- full 20-class VOC training by default
- EfficientNet model factory
- BCE-with-logits multi-label loss
- class imbalance support through positive weights where configured
- macro/micro/per-class metrics
- per-class threshold tuning
- exported inference metadata

## Evaluate

```powershell
python scripts/evaluate.py --config configs/voc20.yaml
```

Evaluation reports per-class metrics and aggregate metrics. Multi-label accuracy can be misleading on imbalanced datasets, so macro F1, micro F1, precision, recall, and per-class F1 are more important.

## Export model artifact

The inference web app should load an exported artifact, not just a bare checkpoint.

Expected artifact:

```text
checkpoints/exported/model.pt
checkpoints/exported/metadata.json
```

Manual export:

```powershell
python scripts/export_model.py --config configs/voc20.yaml --checkpoint checkpoints/best_model.pt --output-dir checkpoints/exported
```

The metadata file stores:

```text
class order
image size
normalization mean/std
architecture name
per-class thresholds
validation metrics
```

This prevents class-order and preprocessing mismatches during inference.

## Run backend

```powershell
copy .env.example .env
python -m uvicorn app.main:app --app-dir backend --reload
```

On macOS/Linux:

```bash
cp .env.example .env
python -m uvicorn app.main:app --app-dir backend --reload
```

Backend endpoints:

```text
GET  /health
GET  /api/model
POST /api/predict
```

If no exported model exists, `/api/model` and `/api/predict` return a clear model-not-available response instead of fake predictions.

## Run frontend

```powershell
cd frontend
npm install
npm run typecheck
npm run build
npm run dev
```

Open:

```text
http://127.0.0.1:5173
```

The frontend supports:

- image upload
- local image preview
- predicted labels above threshold
- all 20 class confidence bars
- model metadata display
- clear errors for missing model artifacts or invalid images

## Colab training

Use:

```text
notebooks/colab_training_launcher.ipynb
```

The notebook should be a launcher only. The training logic lives in `src/` and `scripts/`.

Recommended Colab flow:

```text
1. Mount Google Drive.
2. Clone this repo.
3. Install package dependencies.
4. Point config paths to the dataset in Drive.
5. Run scripts/train.py.
6. Run evaluation/export.
7. Save checkpoints/exported/model.pt and metadata.json to Drive.
8. Copy the exported artifact back to your laptop project.
```

Colab cannot directly read your laptop's `C:\...` paths. Upload the processed dataset or raw tar to Google Drive, or prepare the dataset inside Colab from the tar.

## Dataset CSV format

Required columns:

```csv
image_name,aeroplane,bicycle,bird,boat,bottle,bus,car,cat,chair,cow,diningtable,dog,horse,motorbike,person,pottedplant,sheep,sofa,train,tvmonitor
```

Each class column must contain `0` or `1`.

Example:

```csv
image_name,aeroplane,bicycle,bird,boat,bottle,bus,car,cat,chair,cow,diningtable,dog,horse,motorbike,person,pottedplant,sheep,sofa,train,tvmonitor
2007_000033.jpg,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0
```

## Git hygiene

Do not commit raw datasets, processed images, checkpoints, exported models, local `.env` files, caches, virtual environments, or frontend build outputs. The `.gitignore` is configured for this.

Important ignored paths:

```text
raw/
data/
checkpoints/*
outputs/*
node_modules/
frontend/dist/
.venv/
.env
.pytest_cache/
```

The `.gitkeep` files preserve empty folders without committing heavy artifacts.

## Quality checks

Python:

```powershell
python -m compileall backend src scripts tests
pytest
ruff check .
```

Frontend:

```powershell
cd frontend
npm run typecheck
npm run build
npm audit
```

## Design notes

- The project includes all 20 Pascal VOC classes by default.
- `configs/voc8_legacy.yaml` preserves the original 8-class experiment.
- The notebook is no longer the main training implementation.
- `torch_xla` should remain optional and only be imported when TPU mode is explicitly requested.
- The web app uses model metadata for class order, preprocessing, and thresholds.
- The inference API is synchronous because EfficientNet prediction is fast enough for normal uploads.

## License

MIT License. See `LICENSE`.
