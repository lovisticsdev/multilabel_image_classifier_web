# Notebooks

`colab_training_launcher.ipynb` is a launcher for Colab GPU training. It mounts Google Drive, clones the repo, installs dependencies, verifies the package import, rewrites `configs/voc20.yaml` with Drive dataset/output paths, inspects the dataset, trains the model, and evaluates the exported artifact.

The notebook does not contain the model, dataset, or training logic. Those live in `src/` and `scripts/`.
