# Inference API

Start the backend:

```bash
python -m uvicorn app.main:app --app-dir backend --reload
```

Endpoints:

```text
GET  /health
GET  /api/model
POST /api/predict
```

`POST /api/predict` accepts a multipart image upload under the `file` field and returns predicted labels plus all class probabilities.
