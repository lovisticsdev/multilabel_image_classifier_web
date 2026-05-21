# Dataset format

Expected processed layout:

```text
data/voc2012_processed/
├── train/images/
├── val/images/
├── train_annotations.csv
└── val_annotations.csv
```

CSV columns:

```text
image_name,aeroplane,bicycle,bird,boat,bottle,bus,car,cat,chair,cow,diningtable,dog,horse,motorbike,person,pottedplant,sheep,sofa,train,tvmonitor
```

Each class column must contain binary `0` or `1` values.
