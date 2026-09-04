# 12 — CIFAR-10: ResNet From Scratch

Fourth project in this repo: image classification on CIFAR-10 with a
CIFAR-style ResNet-18 trained from scratch.

## Dataset

- **Name:** CIFAR-10 (Canadian Institute For Advanced Research, 10 classes).
- **Size:** 60,000 RGB images, 32x32 px each (50k train + 10k test).
- **Classes:** `airplane, automobile, bird, cat, deer, dog, frog, horse,
  ship, truck`. Each class has exactly 6,000 images in the standard split.
- **Auto-download** via `torchvision.datasets.CIFAR10` on first run; lives
  under `data/raw/cifar-10-batches-py/`.

Splitting (in `src/data/cifar_dataset.py`):

| Split | Source | Count |
|---|---|---|
| Train | 90% of the 50k train split, stratified | 45,000 |
| Val   | 10% of the 50k train split, stratified |  5,000 |
| Test  | the official 10k test split              | 10,000 |

The validation set is used for early stopping and learning-rate
monitoring. The test set is only touched once, at the end of training.

## Model

`src/models/cifar_resnet.py` defines a CIFAR-style ResNet — the variant
from He et al. (2015) "Deep Residual Learning for Image Recognition",
adapted for 32x32 inputs:

- **Stem:** 3x3 conv (no 7x7, no max-pool) at full resolution
- **Stage 1:** 2 residual blocks at 64 channels, 32x32
- **Stage 2:** 2 residual blocks at 128 channels, 16x16
- **Stage 3:** 2 residual blocks at 256 channels, 8x8
- **Stage 4:** 2 residual blocks at 256 channels, 4x4
- **Head:** Global avg pool -> dropout(0.2) -> linear(num_classes)

Total parameters: ~5.2M (default config — `base_channels=64`,
`num_blocks_per_stage=2`). The ImageNet-style ResNet-18 has ~11M params
because its stage widths grow to 512; our CIFAR variant uses 64-128-256-256
because 32×32 inputs don't need that much capacity. Reaches ~92-94% test
accuracy in 20 epochs with OneCycle LR.

### BasicBlock

```text
x ─┬─→ [Conv 3x3] → [BN] → [ReLU] → [Conv 3x3] → [BN] ─┐
   │                                                   + → [ReLU] → out
   └──→ [Identity or (Conv 1x1 + BN) if shape changes] ─┘
```

Residual connections let gradients propagate through ~20 conv layers
without vanishing — the original ResNet breakthrough.

## Training Recipe

This is where CIFAR differs from Iris/Wine/MNIST — it brings modern
training practices:

| Component | Choice | Rationale |
|---|---|---|
| Optimizer | SGD + momentum(0.9) + Nesterov + weight decay(5e-4) | Standard CIFAR recipe; beats Adam on this dataset |
| LR scheduler | OneCycleLR (`max_lr=0.1`, `pct_start=0.1`) | Modern best practice for image tasks; linear warmup + cosine decay |
| Epochs | 20 (configurable) | Reaches ~92-94% |
| Batch size | 128 | Standard for CIFAR on a single GPU |
| Early stopping | patience=5 on val accuracy | Avoids wasted compute if a config overfits |
| Normalisation | mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616) | Canonical CIFAR constants |

## How to Train

```bash
# Full training (downloads CIFAR on first run, ~10-30 min on GPU)
make train-cifar

# Smoke test (2 epochs, tiny model)
make train-cifar-fast

# Override hyperparameters
python -m src.training.train_cifar train.epochs=30 train.learning_rate=0.05
python -m src.training.train_cifar model.dropout=0.3 model.base_channels=16
```

Outputs:
- `models/cifar/v1/best.pt`, `models/cifar/v1/last.pt` — model weights
- `models/cifar/v1/model_arch.json` — arch config + normalisation stats
- `models/cifar/v1/config.yaml` — full Hydra config used
- `models/cifar/v1/metrics.json` — final test accuracy, per-class F1, best
  val accuracy, etc.
- `models/cifar/v1/class_names.json` — class index -> name mapping
- `plots/cifar_training_history.png` — loss + accuracy curves
- `plots/cifar_confusion_matrix.png` — row-normalised confusion matrix

All of the above are also logged as MLflow artifacts under the
`cifar10-resnet` experiment.

## How to Serve

The trained model is automatically exposed via the unified FastAPI service:

```bash
# Local
make api
# open http://localhost:8000/docs

# Inspect CIFAR model metadata
curl http://localhost:8000/model/info/cifar

# Predict on an image file
curl -X POST http://localhost:8000/predict/cifar \
     -F "file=@some_32x32_image.png"

# Serverless (Modal)
make deploy
# URL printed by Modal; same /predict/cifar endpoint works
```

The response includes the top-5 predicted classes (not just argmax),
useful when confidence is low:

```json
{
  "label": "airplane",
  "confidence": 0.74,
  "probabilities": { "airplane": 0.74, "ship": 0.12, ... },
  "top5": [
    { "label": "airplane", "probability": 0.74 },
    { "label": "ship",    "probability": 0.12 },
    ...
  ],
  "model_version": "v1"
}
```

## Tests

```bash
make test-unit         # 14 tests covering CifarResNet + BasicBlock
make test-integration  # 2 tests covering pipeline + OneCycle scheduler
make test-api          # 6 tests covering /predict/cifar contract
```

The unit tests verify forward shape, gradient flow, parameter count,
stage channel doubling, and the BasicBlock shape contract for every
(stride, channels) combination used in the network.

## Expected Results

With the default config (`base_channels=64`, 20 epochs, OneCycle), on a
single GPU:

- **best val accuracy:** ~93-95%
- **final test accuracy:** ~92-94%
- **training time:** ~10-15 minutes on a modern GPU, much longer on CPU

On CPU expect ~1-2 hours for the full 20 epochs; use
`make train-cifar-fast` for a quick sanity run.

## Lessons / Why ResNet for CIFAR-10

- The 7x7-stem / max-pool variant from `torchvision.models.resnet18`
  throws away too much spatial information at 32x32. The CIFAR variant
  keeps the full 32x32 until stage 2, doubling channels and halving
  spatial dims instead.
- BatchNorm + skip connections let us go deep without vanishing
  gradients — a 2-layer Iris MLP could solve 4-feature classification
  in 1,500 epochs; a 70-layer CIFAR ResNet converges in 20.
- OneCycle LR consistently beats cosine/step decay at a similar cost —
  the linear warmup avoids the early instability of large SGD steps,
  and the cosine decay gives finer convergence than a step schedule.

## Next Steps

- Stronger augmentation (RandAugment, Mixup, CutMix) typically adds
  another +1-2% accuracy.
- A wider backbone (base_channels=128) trades training time for ~+1%.
- AMP (`torch.cuda.amp`) roughly halves GPU training time.
- Reusable callbacks: top-k checkpointing, gradient clipping, label
  smoothing.
