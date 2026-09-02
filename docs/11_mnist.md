# MNIST digit classifier

The third project in this repo: a small convolutional neural network
trained from scratch on MNIST, served through the same FastAPI app as
the wine model, and exposed publicly through Modal.

## Dataset

- **MNIST** — 60,000 training + 10,000 test images of handwritten
  digits (0-9), 28x28 grayscale.
- Downloaded automatically by `torchvision.datasets.MNIST` into
  `data/raw/MNIST/` on first run.
- Normalised to `[0, 1]` floats before training.

## Model

`SimpleCNN` in `src/models/mnist_cnn.py` — a 3-layer CNN with ~225K
parameters:

```
Input (1, 28, 28)
  ↓ Conv2d(1→32, 3, pad=1) → ReLU → MaxPool(2)         # → (32, 14, 14)
  ↓ Conv2d(32→64, 3, pad=1) → ReLU → MaxPool(2)        # → (64, 7, 7)
  ↓ Flatten
  ↓ Linear(64*7*7=3136 → 128) → ReLU → Dropout(0.25)
  ↓ Linear(128 → 10)
```

**Why this small?** It trains in seconds, fits comfortably on CPU, and
still hits ≥98% test accuracy. The point is to demonstrate the full
computer-vision pipeline (data → CNN → serving → deployment), not to
chase SOTA.

## Training

```bash
make train-mnist
# or
python -m src.training.train_mnist
```

Overridable from the CLI:

```bash
python -m src.training.train_mnist train.epochs=10 model.dropout=0.5 train.batch_size=64
```

Config lives at `configs/mnist.yaml`; dataclasses are in
`src/configs/schemas.py` (`MnistDataConfig`, `MnistModelConfig`,
`MnistTrainConfig`).

Training artefacts land in `models/mnist/v1/`:

```
models/mnist/v1/
├── model.pth          # state_dict
├── model_arch.json    # architecture (so API can rebuild)
├── config.yaml        # frozen Hydra config
├── metrics.json       # accuracy, f1_macro, f1_weighted
├── class_names.json   # ["0", "1", ..., "9"]
└── ...
```

MLflow tracking is automatic (run name = `mnist-cnn-dr25-lr0.001` by
default).

## Results

5 epochs on the full MNIST training set, default config:

| Metric       | Value  |
|--------------|--------|
| Test accuracy| ~0.99  |
| F1 (macro)   | ~0.99  |
| Train time   | ~90 s on CPU |

(The actual training loss/accuracy curves land in
`plots/mnist_training_history.png` and as an MLflow artefact.)

## API

`POST /predict/mnist` accepts a multipart upload of a single image and
returns:

```json
{
  "label": "7",
  "confidence": 0.97,
  "probabilities": {"0": 0.001, ..., "7": 0.97, ...},
  "model_version": "v1"
}
```

The endpoint:

1. Decodes the bytes via PIL (PNG, JPEG, BMP, GIF, TIFF, WEBP all work).
2. Converts to grayscale (`"L"` mode).
3. Resizes to 28x28 with bilinear interpolation.
4. Normalises to `[0, 1]` and reshapes to `[1, 1, 28, 28]`.
5. Runs the CNN → softmax → top class.

Errors:

- `400 Bad Request` — file cannot be decoded, or is empty.
- `503 Service Unavailable` — the registry failed to load (i.e. the
  model wasn't trained yet).
- `500 Internal Server Error` — unexpected inference failure.

## Deploy

`make deploy` (after `modal setup`) builds the image and pushes the
unified API — including this endpoint — to Modal. See
[docs/10_deployment.md](10_deployment.md) for the full deployment
guide.

## Tests

- Unit (`tests/unit/test_mnist_cnn.py`): forward shape, parameter
  count, dropout inactive in `eval()`, custom architecture.
- Integration (`tests/integration/test_train_mnist.py`): 1-step smoke
  that asserts loss decreases on synthetic data (offline, CI-friendly).
- API (`tests/api/test_api.py::test_mnist_*`): end-to-end through the
  FastAPI app using a synthetic `SimpleCNN` + PIL-generated PNGs.
