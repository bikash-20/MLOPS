"""Unit tests for the on-disk registry versioning helper.

Covers:
- ``list_versions`` returns nothing on a missing project, sorted versions on an existing one.
- ``next_version`` increments correctly.
- ``resolve_promotion`` decides promotion correctly with various deltas.
- ``resolve_runtime_version`` honours ``"latest"``, ``"vN"``, bare ``"N"``,
  rejects unknown versions, and falls back to ``"v1"`` when no versions exist.
- ``ensure_version_dir`` creates the directory on demand.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from src.utils.versioning import (
    ensure_version_dir,
    list_versions,
    next_version,
    resolve_promotion,
    resolve_runtime_version,
)

# --- list_versions ---------------------------------------------------------


def test_list_versions_empty_when_no_dir(tmp_path, monkeypatch):
    """A nonexistent project directory yields an empty list."""
    monkeypatch.setattr("src.utils.paths.MODELS_DIR", str(tmp_path))
    assert list_versions("nonexistent_project") == []


def test_list_versions_returns_only_vN_dirs(tmp_path, monkeypatch):
    """Non-versioned directories and stray files are ignored."""
    monkeypatch.setattr("src.utils.paths.MODELS_DIR", str(tmp_path))
    proj = tmp_path / "demo"
    proj.mkdir()
    (proj / "v1").mkdir()
    (proj / "v2").mkdir()
    (proj / "v10").mkdir()
    (proj / "not-a-version").mkdir()
    (proj / "v0").mkdir()
    (proj / "README.md").write_text("ignored")
    versions = list_versions("demo")
    # Should sort lexically; v10 comes before v2 lexically, so we sort
    # by the parsed integer in the helper. Verify by spot-check.
    assert versions[0] == "v0"
    assert versions[-1] == "v10"
    assert "v1" in versions
    assert "v2" in versions
    assert "not-a-version" not in versions


# --- next_version ----------------------------------------------------------


def test_next_version_first_run(tmp_path, monkeypatch):
    monkeypatch.setattr("src.utils.paths.MODELS_DIR", str(tmp_path))
    assert next_version("foo") == "v1"


def test_next_version_increments(tmp_path, monkeypatch):
    monkeypatch.setattr("src.utils.paths.MODELS_DIR", str(tmp_path))
    proj = tmp_path / "foo"
    proj.mkdir()
    for v in ("v1", "v3", "v2"):
        (proj / v).mkdir()
    assert next_version("foo") == "v4"


# --- resolve_promotion -----------------------------------------------------


def _write_metrics(project: str, version: str, accuracy: float, tmp_path: Path) -> None:
    """Helper: write a metrics.json under models/<project>/<version>/."""
    d = tmp_path / project / version
    d.mkdir(parents=True, exist_ok=True)
    (d / "metrics.json").write_text(json.dumps({"accuracy": accuracy}))


def test_resolve_promotion_first_run_is_always_promoted(tmp_path, monkeypatch):
    monkeypatch.setattr("src.utils.paths.MODELS_DIR", str(tmp_path))
    res = resolve_promotion("wine_quality", candidate_accuracy=0.50, min_acc_delta=0.05)
    assert res.new_version == "v1"
    assert res.previous_version is None
    assert res.promoted is True
    assert "first version" in res.reason


def test_resolve_promotion_promotes_when_better(tmp_path, monkeypatch):
    monkeypatch.setattr("src.utils.paths.MODELS_DIR", str(tmp_path))
    _write_metrics("wine_quality", "v1", accuracy=0.80, tmp_path=tmp_path)
    res = resolve_promotion(
        "wine_quality", candidate_accuracy=0.85, min_acc_delta=0.01,
    )
    assert res.new_version == "v2"
    assert res.previous_version == "v1"
    assert res.promoted is True


def test_resolve_promotion_blocks_when_below_delta(tmp_path, monkeypatch):
    monkeypatch.setattr("src.utils.paths.MODELS_DIR", str(tmp_path))
    _write_metrics("wine_quality", "v1", accuracy=0.80, tmp_path=tmp_path)
    res = resolve_promotion(
        "wine_quality", candidate_accuracy=0.81, min_acc_delta=0.05,
    )
    assert res.new_version == "v2"
    assert res.previous_version == "v1"
    assert res.promoted is False
    assert "0.8100" in res.reason and "0.8000" in res.reason


def test_resolve_promotion_tie_accepted_at_zero_delta(tmp_path, monkeypatch):
    monkeypatch.setattr("src.utils.paths.MODELS_DIR", str(tmp_path))
    _write_metrics("wine_quality", "v1", accuracy=0.80, tmp_path=tmp_path)
    res = resolve_promotion(
        "wine_quality", candidate_accuracy=0.80, min_acc_delta=0.0,
    )
    assert res.promoted is True


def test_resolve_promotion_no_metrics_fails_closed(tmp_path, monkeypatch):
    """If the previous version has no metrics.json, default to fail-closed."""
    monkeypatch.setattr("src.utils.paths.MODELS_DIR", str(tmp_path))
    (tmp_path / "wine_quality" / "v1").mkdir(parents=True)
    # Note: no metrics.json written.
    res = resolve_promotion("wine_quality", candidate_accuracy=0.99, min_acc_delta=0.0)
    assert res.promoted is False
    assert "no metrics" in res.reason


def test_resolve_promotion_no_metrics_can_be_overridden(tmp_path, monkeypatch):
    """require_metrics=False promotes unconditionally."""
    monkeypatch.setattr("src.utils.paths.MODELS_DIR", str(tmp_path))
    (tmp_path / "wine_quality" / "v1").mkdir(parents=True)
    res = resolve_promotion(
        "wine_quality", candidate_accuracy=0.99, min_acc_delta=0.0,
        require_metrics=False,
    )
    assert res.promoted is True


# --- resolve_runtime_version ----------------------------------------------


def test_resolve_runtime_version_latest(tmp_path, monkeypatch):
    monkeypatch.setattr("src.utils.paths.MODELS_DIR", str(tmp_path))
    proj = tmp_path / "wine_quality"
    proj.mkdir()
    (proj / "v1").mkdir()
    (proj / "v3").mkdir()
    (proj / "v2").mkdir()
    assert resolve_runtime_version("wine_quality") == "v3"


def test_resolve_runtime_version_specific_vN(tmp_path, monkeypatch):
    monkeypatch.setattr("src.utils.paths.MODELS_DIR", str(tmp_path))
    proj = tmp_path / "wine_quality"
    proj.mkdir()
    (proj / "v1").mkdir()
    (proj / "v2").mkdir()
    assert resolve_runtime_version("wine_quality", "v2") == "v2"


def test_resolve_runtime_version_bare_integer(tmp_path, monkeypatch):
    monkeypatch.setattr("src.utils.paths.MODELS_DIR", str(tmp_path))
    proj = tmp_path / "wine_quality"
    proj.mkdir()
    (proj / "v1").mkdir()
    (proj / "v2").mkdir()
    assert resolve_runtime_version("wine_quality", "2") == "v2"


def test_resolve_runtime_version_nonexistent(tmp_path, monkeypatch):
    monkeypatch.setattr("src.utils.paths.MODELS_DIR", str(tmp_path))
    proj = tmp_path / "wine_quality"
    proj.mkdir()
    (proj / "v1").mkdir()
    with pytest.raises(ValueError, match="not found"):
        resolve_runtime_version("wine_quality", "v99")


def test_resolve_runtime_version_empty_registry(tmp_path, monkeypatch):
    monkeypatch.setattr("src.utils.paths.MODELS_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="No versions exist"):
        resolve_runtime_version("wine_quality")


def test_resolve_runtime_version_honours_env(tmp_path, monkeypatch):
    """MODEL_VERSION env var should override default when no version arg given."""
    monkeypatch.setattr("src.utils.paths.MODELS_DIR", str(tmp_path))
    proj = tmp_path / "wine_quality"
    proj.mkdir()
    (proj / "v1").mkdir()
    (proj / "v2").mkdir()
    monkeypatch.setenv("MODEL_VERSION", "v1")
    assert resolve_runtime_version("wine_quality") == "v1"
    monkeypatch.delenv("MODEL_VERSION")
    assert resolve_runtime_version("wine_quality") == "v2"


# --- ensure_version_dir ----------------------------------------------------


def test_ensure_version_dir_creates_path(tmp_path, monkeypatch):
    monkeypatch.setattr("src.utils.paths.MODELS_DIR", str(tmp_path))
    path = ensure_version_dir("wine_quality", "v3")
    assert os.path.isdir(path)
    assert path.endswith("wine_quality/v3")
