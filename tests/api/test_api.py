"""FastAPI endpoint tests using TestClient.

These tests use tiny synthetic WineNet + SimpleCNN models registered at
temp directories so they never touch the real model registry on disk.
"""

import json
from io import BytesIO
from pathlib import Path

import joblib
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image
from sklearn.preprocessing import StandardScaler

from src.models.wine_nn import WineNet


@pytest.fixture
def fake_model_dir(tmp_path, monkeypatch) -> Path:
    """Build a tiny trained model + scaler + metadata, and patch the registry to use it."""
    from src.api import model_loader

    # Train a tiny model that predicts based on first feature only.
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 11)).astype(np.float32)
    y = (X[:, 10] > 0).astype(np.int64)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    torch.manual_seed(0)
    model = WineNet(input_size=11, hidden_sizes=[8, 4], output_size=2, dropout=0.0)
    optim = torch.optim.Adam(model.parameters(), lr=0.01)
    crit = torch.nn.CrossEntropyLoss()
    Xt = torch.from_numpy(Xs).float()
    yt = torch.from_numpy(y)
    for _ in range(20):
        optim.zero_grad()
        loss = crit(model(Xt), yt)
        loss.backward()
        optim.step()

    model_dir = tmp_path / "wine_quality" / "v1"
    model_dir.mkdir(parents=True)
    torch.save(model.state_dict(), model_dir / "model.pth")
    joblib.dump(scaler, model_dir / "scaler.joblib")
    (model_dir / "feature_names.json").write_text(json.dumps([
        "fixed acidity", "volatile acidity", "citric acid",
        "residual sugar", "chlorides", "free sulfur dioxide",
        "total sulfur dioxide", "density", "pH", "sulphates", "alcohol",
    ]))
    (model_dir / "metrics.json").write_text(json.dumps({"accuracy": 0.95, "f1_macro": 0.94}))

    # Patch the singleton path resolution.
    monkeypatch.setattr(model_loader, "_singleton", None)

    def fake_get_registry():
        reg = model_loader.ModelRegistry(
            project="wine_quality",
            version="v1",
            model_dir=str(model_dir),
        )
        reg.load({"input_size": 11, "hidden_sizes": [8, 4], "output_size": 2, "dropout": 0.0})
        return reg

    monkeypatch.setattr(model_loader, "get_registry", fake_get_registry)

    # Patch the FastAPI app's reference too.
    from src.api import main as api_main
    monkeypatch.setattr(api_main, "get_registry", fake_get_registry)
    return model_dir


@pytest.mark.api
def test_root_endpoint(fake_model_dir):
    from src.api.main import app

    client = TestClient(app)
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert body["service"] == "neural-network-classifier"
    assert "/predict" in body["endpoints"]
    assert "/predict/mnist" in body["endpoints"]


@pytest.mark.api
def test_health_endpoint(fake_model_dir):
    from src.api.main import app

    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


@pytest.mark.api
def test_model_info_endpoint(fake_model_dir):
    from src.api.main import app

    client = TestClient(app)
    r = client.get("/model/info")
    assert r.status_code == 200
    body = r.json()
    assert body["version"] == "v1"
    assert len(body["feature_names"]) == 11
    assert body["class_names"] == ["not_good", "good"]


@pytest.mark.api
def test_predict_valid_payload(fake_model_dir, valid_wine_payload):
    from src.api.main import app

    client = TestClient(app)
    r = client.post("/predict", json=valid_wine_payload)
    assert r.status_code == 200
    body = r.json()
    assert body["label"] in ("good", "not_good")
    assert 0 <= body["confidence"] <= 1
    assert set(body["probabilities"].keys()) == {"good", "not_good"}


@pytest.mark.api
def test_predict_missing_field_returns_422(fake_model_dir, valid_wine_payload):
    from src.api.main import app

    invalid = dict(valid_wine_payload)
    del invalid["alcohol"]
    client = TestClient(app)
    r = client.post("/predict", json=invalid)
    assert r.status_code == 422


@pytest.mark.api
def test_predict_out_of_range_returns_422(fake_model_dir, valid_wine_payload):
    from src.api.main import app

    invalid = dict(valid_wine_payload)
    invalid["ph"] = 99.0  # above Field(le=5)
    client = TestClient(app)
    r = client.post("/predict", json=invalid)
    assert r.status_code == 422


# --- MNIST endpoint tests --------------------------------------------------


@pytest.fixture
def fake_mnist_dir(tmp_path, monkeypatch) -> Path:
    """Build a tiny trained SimpleCNN + metadata, patch the MNIST registry."""
    from src.api import model_loader
    from src.models.mnist_cnn import SimpleCNN

    torch.manual_seed(0)
    model = SimpleCNN(
        in_channels=1, conv_channels=(8, 16), fc_hidden=32, num_classes=10, dropout=0.0,
    )

    model_dir = tmp_path / "mnist" / "v1"
    model_dir.mkdir(parents=True)
    torch.save(model.state_dict(), model_dir / "model.pth")
    (model_dir / "model_arch.json").write_text(json.dumps({
        "in_channels": 1,
        "conv_channels": [8, 16],
        "fc_hidden": 32,
        "num_classes": 10,
        "dropout": 0.0,
    }))
    (model_dir / "class_names.json").write_text(json.dumps([str(i) for i in range(10)]))
    (model_dir / "metrics.json").write_text(json.dumps({"accuracy": 0.99, "f1_macro": 0.99}))

    monkeypatch.setattr(model_loader, "_mnist_singleton", None)

    def fake_get_mnist_registry():
        reg = model_loader.MnistRegistry(
            project="mnist", version="v1", model_dir=str(model_dir),
        )
        reg.load({
            "in_channels": 1, "conv_channels": [8, 16],
            "fc_hidden": 32, "num_classes": 10, "dropout": 0.0,
        })
        return reg

    monkeypatch.setattr(model_loader, "get_mnist_registry", fake_get_mnist_registry)

    from src.api import main as api_main
    monkeypatch.setattr(api_main, "get_mnist_registry", fake_get_mnist_registry)

    # Build a 28x28 grayscale PNG in memory for the happy-path test.
    png_path = tmp_path / "sample_digit.png"
    Image.new("L", (28, 28), color=128).save(png_path)
    return model_dir


@pytest.mark.api
def test_mnist_model_info_endpoint(fake_mnist_dir):
    from src.api.main import app

    client = TestClient(app)
    r = client.get("/model/info/mnist")
    assert r.status_code == 200
    body = r.json()
    assert body["version"] == "v1"
    assert body["class_names"] == [str(i) for i in range(10)]
    assert body["metrics"]["accuracy"] == 0.99


@pytest.mark.api
def test_mnist_predict_valid_png(fake_mnist_dir):
    from src.api.main import app

    png_path = fake_mnist_dir.parent.parent / "sample_digit.png"
    with Image.open(png_path) as img:
        buf = BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

    client = TestClient(app)
    r = client.post(
        "/predict/mnist",
        files={"file": ("digit.png", buf.getvalue(), "image/png")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["label"] in {str(i) for i in range(10)}
    assert 0 <= body["confidence"] <= 1
    assert set(body["probabilities"].keys()) == {str(i) for i in range(10)}
    assert body["model_version"] == "v1"


@pytest.mark.api
def test_mnist_predict_non_image_returns_400(fake_mnist_dir):
    from src.api.main import app

    client = TestClient(app)
    r = client.post(
        "/predict/mnist",
        files={"file": ("foo.txt", b"not an image", "text/plain")},
    )
    assert r.status_code == 400
    assert "decode" in r.json()["detail"].lower() or "image" in r.json()["detail"].lower()


@pytest.mark.api
def test_mnist_predict_empty_upload_returns_400(fake_mnist_dir):
    from src.api.main import app

    client = TestClient(app)
    r = client.post(
        "/predict/mnist",
        files={"file": ("empty.png", b"", "image/png")},
    )
    assert r.status_code == 400
    assert "empty" in r.json()["detail"].lower()


@pytest.mark.api
def test_mnist_predict_accepts_rgb_png(fake_mnist_dir):
    """Non-grayscale PNGs should be accepted (registry converts to 'L')."""
    from src.api.main import app

    rgb = Image.new("RGB", (32, 32), color=(64, 64, 64))
    buf = BytesIO()
    rgb.save(buf, format="PNG")
    buf.seek(0)

    client = TestClient(app)
    r = client.post(
        "/predict/mnist",
        files={"file": ("rgb.png", buf.getvalue(), "image/png")},
    )
    assert r.status_code == 200, r.text
