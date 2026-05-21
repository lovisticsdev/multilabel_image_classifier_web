from __future__ import annotations

from pathlib import Path


def save_checkpoint(model, path: str | Path, metadata: dict | None = None) -> None:
    import torch

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "metadata": metadata or {}}, output_path)


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict:
    import torch

    return torch.load(Path(path), map_location=map_location)
