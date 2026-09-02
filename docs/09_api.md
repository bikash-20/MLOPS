# REST API (FastAPI)

The Wine Quality model is served via FastAPI at `src/api/`.

## Run locally

```bash
make api
# open http://localhost:8000/docs  (Swagger UI)
```

Or programmatically:

```bash
uvicorn src.api.main:app --reload --port 8000
```

## Endpoints

| Method | Path          | Description                                  |
|--------|---------------|----------------------------------------------|
| GET    | `/`           | Service info                                 |
| GET    | `/health`     | Liveness probe + model-loaded flag           |
| GET    | `/model/info` | Version, metrics, feature names, class names |
| POST   | `/predict`    | Single-sample prediction                     |
| GET    | `/docs`       | Swagger UI (auto-generated)                  |
| GET    | `/redoc`      | ReDoc UI (auto-generated)                    |

## Example: predict

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "fixed_acidity": 7.0,
    "volatile_acidity": 0.27,
    "citric_acid": 0.36,
    "residual_sugar": 20.7,
    "chlorides": 0.045,
    "free_sulfur_dioxide": 45.0,
    "total_sulfur_dioxide": 170.0,
    "density": 1.001,
    "ph": 3.0,
    "sulphates": 0.45,
    "alcohol": 8.8
  }'
```

Response:

```json
{
  "label": "not_good",
  "confidence": 0.93,
  "probabilities": {
    "not_good": 0.93,
    "good": 0.07
  },
  "model_version": "v1"
}
```

## Validation

`POST /predict` validates each feature against realistic ranges defined
in `src/api/schemas.py`. Out-of-range values get a `422 Unprocessable
Entity`:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"fixed_acidity": 99.0, ...}'  # > 20.0
# 422 Unprocessable Entity
```

## Architecture

```
src/api/
├── main.py           # FastAPI app, routes, lifespan
├── model_loader.py   # cached registry singleton
└── schemas.py        # Pydantic request/response models
```

The `ModelRegistry` lazily loads `model.pth` + `scaler.joblib` +
`feature_names.json` + `metrics.json` from `models/wine_quality/v1/`
on first request (or during startup via the FastAPI `lifespan` hook).

## Test

```bash
make test-api
# or
pytest tests/api/ -v
```

The `fake_model_dir` fixture in `tests/api/test_api.py` builds a tiny
synthetic model in a temp directory, so tests don't depend on a real
trained model existing.

## Deploy

See [docs/10_deployment.md](10_deployment.md) for Docker, cloud, and
production considerations.
