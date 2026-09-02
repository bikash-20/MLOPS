"""Modal deployment for the neural-network API.

Deploys the FastAPI service (wine + MNIST) to Modal's serverless platform.

Usage:
    pip install modal
    modal setup                     # one-time: link your Modal account
    modal deploy deploy/modal_app.py
    modal serve  deploy/modal_app.py  # for live-reload dev

After deployment, Modal prints a public URL like
    https://<your-workspace>--neural-network-api-fastapi-app.modal.run

Test it:
    curl <URL>/health
    curl -X POST <URL>/predict -H "Content-Type: application/json" -d '{...}'
    curl -X POST <URL>/predict/mnist -F "file=@some_digit.png"
"""

from __future__ import annotations

import sys
from pathlib import Path

import modal

# --- Modal app + image ---------------------------------------------------

APP_NAME = "neural-network-api"

app = modal.App(APP_NAME)

# Pin Python + PyTorch + the project's runtime dependencies. Modal rebuilds
# the container image only when this file changes, so pinning keeps deploys
# deterministic and reproducible.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi>=0.110,<1.0",
        "uvicorn[standard]>=0.27,<1.0",
        "pydantic>=2.5,<3.0",
        "pydantic-settings>=2.1,<3.0",
        "python-multipart>=0.0.9,<1.0",
        "torch>=2.1,<3.0",
        "torchvision>=0.16,<1.0",
        "joblib>=1.3,<2.0",
        "numpy>=1.24,<3.0",
        "scikit-learn>=1.3,<2.0",
        "Pillow>=10.0,<12.0",
        "hydra-core>=1.3,<2.0",
        "mlflow>=2.10,<3.0",
        "python-json-logger>=2.0,<3.0",
        "omegaconf>=2.3,<3.0",
    )
    # Project source → /root/src so ``import src.api.main`` resolves.
    .add_local_dir(
        local_path=Path(__file__).resolve().parent.parent / "src",
        remote_path="/root/src",
    )
    # Trained model registry → /root/models (matches MODELS_DIR resolution).
    .add_local_dir(
        local_path=Path(__file__).resolve().parent.parent / "models",
        remote_path="/root/models",
    )
)


# --- ASGI app wrapped by Modal -------------------------------------------


@app.function(
    image=image,
    # 1 vCPU + 2 GB RAM is plenty for a FastAPI worker + a small CNN.
    cpu=1.0,
    memory=2048,
    # Allow longer cold-start tolerance for image pull + torch import.
    timeout=300,
    # Keep one warm instance to absorb bursty traffic.
    scaledown_window=120,
)
@modal.asgi_app()
def fastapi_app():
    """Return the FastAPI ASGI app so Modal can serve it over HTTPS."""
    # ``PROJECT_ROOT`` inside the container is /root (because the src tree
    # lives at /root/src). Make it importable so ``import src.api.main``
    # resolves from the wrapper function.
    if "/root" not in sys.path:
        sys.path.insert(0, "/root")
    from src.api.main import app as _app

    return _app
