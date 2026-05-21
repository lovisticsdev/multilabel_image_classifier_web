# Architecture

The project has two workflows:

1. Training/export: load VOC-style multi-label data, train an EfficientNet classifier, tune thresholds, and export a model artifact.
2. Web inference: a FastAPI backend loads the exported artifact and a React frontend lets users upload images for predictions.

The backend does synchronous inference because EfficientNet prediction for one image is fast enough for a request/response flow.
