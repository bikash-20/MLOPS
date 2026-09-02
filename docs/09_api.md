# REST API (FastAPI)

The Wine Quality model and the MNIST CNN are both served via FastAPI at
`src/api/`. One process, two registries, one set of routes.

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

| Method | Path                  | Description                                  |
|--------|-----------------------|----------------------------------------------|
| GET    | `/`                   | Service info                                 |
| GET    | `/health`             | Liveness probe + model-loaded flag           |
| GET    | `/model/info`         | Wine model metadata (version, metrics, features) |
| GET    | `/model/info/mnist`   | MNIST model metadata (version, metrics, classes) |
| POST   | `/predict`            | Wine single-sample prediction (JSON)         |
| POST   | `/predict/mnist`      | MNIST digit prediction (28x28 PNG upload)    |
| GET    | `/docs`               | Swagger UI (auto-generated)                  |
| GET    | `/redoc`              | ReDoc UI (auto-generated)                    |

## Example: wine prediction

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

## Example: MNIST digit prediction

Upload a 28x28 PNG (any color mode — the API converts to grayscale and
resizes to 28x28 automatically):

```bash
curl -X POST http://localhost:8000/predict/mnist \
  -F "file=@my_digit.png"
```

Response:

```json
{
  "label": "7",
  "confidence": 0.97,
  "probabilities": {
    "0": 0.001, "1": 0.002, ..., "7": 0.97, ...
  },
  "model_version": "v1"
}
```

If the file is not a valid image, you'll get `400 Bad Request`:

```bash
curl -X POST http://localhost:8000/predict/mnist \
  -F "file=@README.md"
# {"detail": "Could not decode image: ..."}
```

## Validation

`POST /predict` validates each wine feature against realistic ranges
defined in `src/api/schemas.py`. Out-of-range values get a
`422 Unprocessable Entity`.

`POST /predict/mnist` accepts any image format that PIL can decode
(PNG, JPEG, BMP, GIF, TIFF, WEBP); bad files get `400 Bad Request`.

## Architecture

```
src/api/
├── main.py             # FastAPI app, routes, lifespan
├── model_loader.py     # ModelRegistry (wine) + MnistRegistry (CNN)
└── schemas.py          # Pydantic request/response models
```

Both registries are cached singletons:

- **`ModelRegistry`** lazily loads `model.pth` + `scaler.joblib` +
  `feature_names.json` + `metrics.json` from `models/wine_quality/v1/`.
- **`MnistRegistry`** lazily loads `model.pth` + `model_arch.json` +
  `class_names.json` + `metrics.json` from `models/mnist/v1/`.

Each is lazy-loaded on first use (or during startup via the FastAPI
`lifespan` hook) and then reused process-wide.

## Test

```bash
make test-api
# or
pytest tests/api/ -v
```

The `fake_model_dir` and `fake_mnist_dir` fixtures in
`tests/api/test_api.py` build tiny synthetic models in temp directories,
so tests don't depend on real trained models existing.

## Deploy

See [docs/10_deployment.md](10_deployment.md) for Docker, cloud, and
production considerations (including the Modal one-click deploy).

