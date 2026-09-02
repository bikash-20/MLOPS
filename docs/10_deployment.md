# Deployment

This project ships four deployment surfaces: a local uvicorn process,
a container image, a docker-compose orchestration, and a one-click
serverless deploy to [Modal](https://modal.com).

## Local

```bash
make api  # uvicorn with reload on :8000
```

## Container (Docker)

Multi-stage Dockerfile — installs deps in a builder stage, copies only
the runtime artefacts into a slim final image, runs as non-root user.

```bash
make docker-build   # tag: neural-network:latest
make docker-run     # bind :8000
```

Image layers:
1. `python:3.11-slim` builder — pip install of `requirements.txt`
2. `python:3.11-slim` runtime — copy venv, src/, configs/, models/
3. Non-root `app` user, healthcheck, exposed port 8000

## docker-compose

Brings up the API and an MLflow server together, sharing the local
`./mlruns/` directory as a volume:

```bash
make docker-up      # api :8000 + mlflow :5000
make docker-down
```

```yaml
services:
  api:    # built from ./Dockerfile, serves the API
  mlflow: # ghcr.io/mlflow/mlflow:v2.16.0, server with SQLite backend
```

## CI (GitHub Actions)

`.github/workflows/ci.yml` runs on every push and PR:

1. Set up Python 3.11, cache pip
2. Install requirements
3. Run `ruff check src tests`
4. Run `pytest --cov=src`
5. Upload coverage to Codecov
6. Build the Docker image and smoke-test the API health endpoint

## Cloud (sketch)

The API is a stateless FastAPI app that loads its model from a
read-only filesystem path. Any of these work:

### AWS
- Push image to ECR
- Run on ECS Fargate or Lambda container
- Mount model artifacts from S3 on startup

### GCP
- Push to Artifact Registry
- Deploy to Cloud Run (auto-scales, scales to zero)
- Or GKE for full Kubernetes control

### Azure
- Push to ACR
- Deploy to Azure Container Apps or AKS

For all three: set `MODEL_DIR` env var if the registry path differs from
the baked-in `models/wine_quality/v1/`.

## Serverless (Modal)

The fastest way to get a public HTTPS URL for the API — no infra, no
Docker, no cloud account. Modal runs the same FastAPI app in a
container on their serverless platform (generous free tier).

### One-time setup

```bash
pip install modal
modal setup   # opens a browser to link your account
```

### Deploy

```bash
make deploy              # production deploy (one URL)
make deploy-serve        # live-reload dev (URL stays the same, code reloads)
```

`deploy/modal_app.py` builds a Debian-slim image with all the runtime
deps, bundles the `src/` tree and the `models/` registry, then serves
the FastAPI app via `@modal.asgi_app()`.

After `modal deploy` finishes, Modal prints a URL like

```
https://<your-workspace>--neural-network-api-fastapi-app.modal.run
```

### Test the live URL

```bash
URL=https://<your-workspace>--neural-network-api-fastapi-app.modal.run

curl $URL/health
# {"status":"ok","model_loaded":true}

curl -X POST $URL/predict -H "Content-Type: application/json" \
  -d '{"fixed_acidity":7.0,"volatile_acidity":0.27,"citric_acid":0.36,
       "residual_sugar":20.7,"chlorides":0.045,"free_sulfur_dioxide":45.0,
       "total_sulfur_dioxide":170.0,"density":1.001,"ph":3.0,
       "sulphates":0.45,"alcohol":8.8}'

curl -X POST $URL/predict/mnist -F "file=@tests/fixtures/sample_digit.png"
```

### Notes

- Cold-start is ~30-60 s on the first request after the image deploys
  (PyTorch import is heavy). Subsequent requests hit the warm
  instance in milliseconds.
- The image is cached — only rebuilds when `deploy/modal_app.py`
  changes. To force a rebuild after `requirements.txt` updates, bump
  the image by editing the file (e.g. add a comment).
- The free tier includes enough compute for demo traffic; check
  [modal.com/pricing](https://modal.com/pricing) for limits.

## Production checklist

Before going to production:

- [ ] Pin a specific model version, not just `v1`
- [ ] Set up health-check + liveness probes
- [ ] Add structured JSON logging
- [ ] Enable TLS termination at the load balancer
- [ ] Configure CORS if a browser will call the API directly
- [ ] Add rate limiting (e.g. via a gateway or `slowapi`)
- [ ] Set up request monitoring (Prometheus metrics, OpenTelemetry)
- [ ] Configure data-drift detection (Evidently AI / Whylogs)
- [ ] Schedule periodic model retraining (Prefect / Airflow)
- [ ] Define on-call runbook for `5xx` spikes

## References

- [FastAPI deployment guide](https://fastapi.tiangolo.com/deployment/)
- [Made With ML — MLOps](https://madewithml.com/)
- [Designing Machine Learning Systems (Chip Huyen)](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/)
