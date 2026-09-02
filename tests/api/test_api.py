"""FastAPI endpoint tests using TestClient.

These tests use a tiny synthetic WineNet registered at a temp directory.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
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
    assert body["service"] == "wine-quality-classifier"
    assert "/predict" in body["endpoints"]


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
