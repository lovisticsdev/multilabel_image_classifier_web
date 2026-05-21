from __future__ import annotations


def test_data_module_imports():
    from multilabel_classifier.data.datamodule import create_dataloaders

    assert callable(create_dataloaders)
