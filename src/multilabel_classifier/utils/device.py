from __future__ import annotations


def select_device(device_name: str = "auto"):
    import torch

    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "tpu":
        try:
            import torch_xla.core.xla_model as xm
        except ImportError as exc:
            raise RuntimeError("TPU mode requires torch_xla to be installed.") from exc
        return xm.xla_device()
    return torch.device(device_name)
