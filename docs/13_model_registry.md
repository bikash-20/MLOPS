# 13 — Model Registry & Deployment

How trained models move from disk into the served API, plus the
versioning scheme that lets the API load any past checkpoint.

## 1. The on-disk layout

Every training run writes a versioned directory under
``models/<project>/<version>/``:

```
models/
├── wine_quality/
│   ├── v1/                  ← first training run
│   │   ├── model.pth
│   │   ├── scaler.joblib
│   │   ├── feature_names.json
│   │   ├── metrics.json     ← includes a "registry" sub-object
│   │   └── config.yaml
│   └── v2/                  ← second run, promoted because it beat v1
│       ├── model.pth
│       └── ...
├── mnist/
│   ├── v1/{model.pth, model_arch.json, class_names.json, metrics.json, config.yaml}
│   └── v2/{...}
└── cifar/
    ├── v1/{best.pt, last.pt, model_arch.json, class_names.json, metrics.json, config.yaml}
    └── v2/{...}
```

Versions are plain ``vN`` integers. Each training run resolves the
**next** free version at startup:

- No previous versions → ``v1``.
- N previous versions → the next is ``v{max+1}``.

You can list everything on disk with:

```bash
make models
# wine_quality: ['v1', 'v2', 'v3']
# mnist:        ['v1']
# cifar:        ['v1', 'v2']
```

## 2. Auto-increment + promotion gate

Trainers no longer hardcode ``v1``. They call
`src.utils.versioning.resolve_promotion()` with this run's metric
(usually test accuracy) and a threshold (``train.min_acc_delta``):

```
candidate_accuracy >= previous_accuracy + min_acc_delta  ⇒ promoted
```

The gate is **optional** but the default behavior keeps the on-disk
registry clean:

- **First run:** always promoted, becomes ``v1``.
- **Subsequent runs:** if the new accuracy doesn't beat the previous by
  ``min_acc_delta``, the run is **not promoted**. The trainer still
  writes a new ``vN+1`` slot (so the experiment is preserved) but logs
  clearly that ``vN`` remains the production candidate.
- **Set `min_acc_delta=0.0`** to accept any improvement (or tie).
- **Set `min_acc_delta=0.05`** to require meaningful gains only.

Configure per trainer in the YAML / via CLI:

```bash
# CIFAR: require at least +1% accuracy over previous best to promote
python -m src.training.train_cifar train.min_acc_delta=0.01

# Wine: block promotion unless accuracy strictly improves
python -m src.training.train_wine train.min_acc_delta=0.001
```

If a previous version's `metrics.json` is missing or unreadable, the
gate fails closed by default (it logs a warning and refuses to
promote). Set `require_metrics=False` in `resolve_promotion()` if you
want unconditional promotion.

## 3. Pinning the API to a specific version

By default each FastAPI registry loads the **latest** version on disk.
Override per-deploy with the `MODEL_VERSION` env var:

```bash
# default — serves v2 (or whichever is highest)
make api

# pin to v1 (e.g. you're rolling back)
MODEL_VERSION=v1 make api-v2    # alias for clarity

# bare integer also accepted
MODEL_VERSION=1 make api

# In Docker / docker-compose, the same env var is read
MODEL_VERSION=v2 docker compose up
```

Resolution rules:

| `MODEL_VERSION` value | Resolves to |
|---|---|
| unset / empty / `"latest"` | highest version on disk |
| `"v2"` | exactly v2 (errors if not found) |
| `"2"` | exactly v2 (errors if not found) |
| `"v99"` | ValueError at startup (no silent fallback) |

Each model family is pinned independently — the same API can serve
`wine_quality@v1` and `mnist@v2` and `cifar@v1` simultaneously if you
ever need to roll one back without the others.

## 4. Inspecting the registry at runtime

`GET /models` returns the full registry state:

```bash
curl http://localhost:8000/models
```

```json
[
  {
    "project": "wine_quality",
    "active_version": "v2",
    "versions": [
      {
        "version": "v1",
        "available": true,
        "metrics": {"accuracy": 0.83, "precision_macro": 0.84, ...},
        "promoted": true,
        "reason": "first version for this project"
      },
      {
        "version": "v2",
        "available": true,
        "metrics": {"accuracy": 0.87, ...},
        "promoted": true,
        "reason": "accuracy 0.8700 >= previous 0.8300 (delta=+0.0400, threshold=+0.0000)"
      }
    ]
  },
  ...
]
```

This endpoint exists to answer the operator questions:

- "What version is the API currently serving?" → `active_version`.
- "What versions exist on disk?" → `versions[].version`.
- "Was the latest run actually deployed?" → `promoted` flag on the
  newest entry (true ⇒ it's the active one; false ⇒ an older
  `active_version` is still in production).

## 5. CI → Modal deploy

`.github/workflows/ci.yml` has four jobs:

1. **lint-and-test** — runs `ruff check` and the full pytest suite
   (with coverage uploaded to Codecov). Required to pass.
2. **docker-build** — multi-stage container build + `/health` smoke
   test inside the container. The image is **not** pushed to a registry
   (add that if you want to).
3. **deploy-modal** *(new)* — gated on `push` to `main`. Runs
   `modal deploy deploy/modal_app.py`, then smoke-tests the live URL.
   Requires two GitHub repo secrets:
   - `MODAL_TOKEN_ID`
   - `MODAL_TOKEN_SECRET`
4. *(manual)* — `workflow_dispatch` lets you re-run the deploy job from
   the GitHub UI without needing a fresh commit.

### One-time setup: Modal CI secrets

```bash
# Local
modal token new            # creates ~/.modal.toml with token_id + token_secret
cat ~/.modal.toml          # shows the values to paste into GitHub
```

In GitHub → Settings → Secrets and variables → Actions:

- New secret `MODAL_TOKEN_ID` → paste `token_id` value.
- New secret `MODAL_TOKEN_SECRET` → paste `token_secret` value.

After that, every merge to `main` will ship the latest code (and the
newest `models/vN/` slot) to Modal automatically.

### How the deployed Modal image picks the right version

`deploy/modal_app.py` does `add_local_dir(models, /root/models)` so the
whole registry tree ships inside the container. The API resolves the
version from `MODEL_VERSION` at startup. If you want the deployed
Modal service to serve a specific version, set the env var in the
Modal image:

```python
@app.function(
    image=image,
    env={
        "MODEL_VERSION": "v2",
    },
    ...
)
```

## 6. Rollback procedure

If `v3` regresses in production:

```bash
# 1. Roll Modal back to v2 via the env var (one-touch).
modal deploy deploy/modal_app.py   # with MODEL_VERSION=v2 in deploy/modal_app.py
# OR set MODEL_VERSION=v2 in the GitHub environment and re-run the CI deploy job.

# 2. Investigate. The full v3 weights are still on disk under
#    models/<project>/v3/ — nothing is deleted by a failed run.

# 3. Train a v4 with better config / data / hyperparams.
make train-wine      # or train-mnist / train-cifar
```

## 7. Tests

```bash
make test-unit          # 21 versioning tests (10 new in test_versioning.py)
make test-api           # 19 API tests, incl. /models listing endpoint
```

## 8. Operations cheatsheet

| Task | Command |
|---|---|
| List on-disk versions | `make models` |
| Pin API to a specific version (local) | `MODEL_VERSION=v2 make api` |
| Pin API to a specific version (Docker Compose) | `MODEL_VERSION=v2 docker compose up` |
| Inspect live registry | `curl http://localhost:8000/models` |
| Train (auto-increments, checks promotion gate) | `make train-cifar` |
| Require meaningful promotion | `train.min_acc_delta=0.01 make train-cifar` |
| Deploy to Modal | GitHub: merge to main, or Actions → run workflow |
