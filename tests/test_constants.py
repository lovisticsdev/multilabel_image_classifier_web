from __future__ import annotations

from multilabel_classifier.constants import VOC20_CLASSES


def test_voc20_class_count():
    assert len(VOC20_CLASSES) == 20
    assert "person" in VOC20_CLASSES
    assert "tvmonitor" in VOC20_CLASSES
