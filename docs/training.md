# Training

Prepare VOC2012 from the downloaded tar:

```bash
python scripts/prepare_voc2012.py
python scripts/inspect_dataset.py --config configs/voc20.yaml
python scripts/train.py --config configs/voc20.yaml
```

For real training, prefer Colab GPU. The notebook in `notebooks/` mounts Drive, updates dataset paths, verifies imports, and launches the training script.
