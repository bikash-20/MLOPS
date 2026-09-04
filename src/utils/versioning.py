"""Versioning utilities for the on-disk model registry.

The project's models live under ``models/<project>/v{N}/`` where ``N``
is a monotonically increasing integer. Every training run:

1. Resolves the next free version (1 if no versions exist, else
   ``max(existing) + 1``).
2. Optionally gates that promotion against a quality threshold
   (``MIN_ACC_DELTA``) when an existing version already exists -- if the
   new run does not improve accuracy by at least the delta, the run is
   either discarded, saved as ``vN+1`` but not "promoted", or rejected
   entirely (configurable).
3. Writes all artifacts to ``models/<project>/v{N+1}/`` (or the current
   ``N+1`` slot if promotion passed).

The API side resolves which version to serve from the ``MODEL_VERSION``
env var (defaulting to ``"latest"``). At runtime the helper returns the
resolved version string.

Why v1, v2, v3, ... and not timestamps or hashes:
- They sort lexically.
- They are reversible to the integer that produced them.
- They map cleanly onto the way MLflow also names runs.

This module is intentionally tiny and dependency-free so that the
training entrypoints and the API loader can call into it independently.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from src.utils.paths import ensure_dir, models_path

_VERSION_RE = re.compile(r"^v(\d+)$")


@dataclass(frozen=True)
class VersionResolution:
    """Result of resolving the next registry version for a project.

    Attributes:
        project: The project name (e.g. ``"wine_quality"``, ``"cifar"``).
        new_version: The version string that the trainer will write to
            (always ``vN`` where ``N = max(existing) + 1``).
        previous_version: The most recent existing version, or ``None``
            if this is the first training run.
        promoted: Whether the new run *passed* the promotion gate and is
            considered the new production candidate. Promotion requires
            either no previous version, or accuracy improvement of at
            least ``min_acc_delta`` over the previous best.
        reason: Human-readable explanation of the promotion decision.
    """

    project: str
    new_version: str
    previous_version: str | None
    promoted: bool
    reason: str


def list_versions(project: str) -> list[str]:
    """Return all ``v{N}`` subdirectories of ``models/<project>/``, sorted ascending by N.

    Returns an empty list if the project directory does not exist.
    Sorting by integer (not lexically) ensures ``v10`` comes after ``v2``.
    """
    project_dir = Path(models_path(project))
    if not project_dir.exists():
        return []
    pairs = [
        (int(m.group(1)), entry.name)
        for entry in project_dir.iterdir()
        if entry.is_dir() and (m := _VERSION_RE.match(entry.name))
    ]
    return [name for _, name in sorted(pairs)]


def _version_int(name: str) -> int:
    """Return the integer N from a ``vN`` string."""
    m = _VERSION_RE.match(name)
    if not m:
        raise ValueError(f"not a valid version string: {name!r}")
    return int(m.group(1))


def next_version(project: str) -> str:
    """Return ``"v1"`` if no versions exist, else ``"v{max+1}"``."""
    existing = list_versions(project)
    if not existing:
        return "v1"
    highest = max(_version_int(v) for v in existing)
    return f"v{highest + 1}"


def _read_metrics(project: str, version: str) -> dict | None:
    """Read ``metrics.json`` for a given project + version, or ``None``."""
    p = Path(models_path(f"{project}/{version}/metrics.json"))
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def resolve_promotion(
    project: str,
    candidate_accuracy: float,
    metric_key: str = "accuracy",
    min_acc_delta: float = 0.0,
    require_metrics: bool = True,
) -> VersionResolution:
    """Compute the next version + whether this run earns promotion.

    A run is "promoted" (i.e. considered the new production candidate)
    when:
    - There is no existing version (first ever run), OR
    - The candidate accuracy is at least ``previous + min_acc_delta``.

    Args:
        project: Project name (``"wine_quality"``, ``"mnist"``, ``"cifar"``).
        candidate_accuracy: This run's metric to compare against the
            previous best (typically test accuracy).
        metric_key: Key inside ``metrics.json`` to compare. Default
            ``"accuracy"``; CIFAR-10 uses the same key.
        min_acc_delta: Minimum improvement over the previous version's
            metric required for promotion. ``0.0`` means any improvement
            (or tie) counts; negatives would always fail.
        require_metrics: If True and a previous version exists but has
            no ``metrics.json``, fail closed (do not promote). If False,
            fall back to "always promote".

    Returns:
        ``VersionResolution`` with the new version string and promotion
        decision + reason.
    """
    existing = list_versions(project)
    if not existing:
        return VersionResolution(
            project=project,
            new_version="v1",
            previous_version=None,
            promoted=True,
            reason="first version for this project",
        )

    previous = max(existing, key=_version_int)
    prev_metrics = _read_metrics(project, previous)
    if prev_metrics is None:
        if require_metrics:
            return VersionResolution(
                project=project,
                new_version=next_version(project),
                previous_version=previous,
                promoted=False,
                reason=(
                    f"previous version {previous} has no metrics.json; "
                    "not promoting (require_metrics=True)"
                ),
            )
        return VersionResolution(
            project=project,
            new_version=next_version(project),
            previous_version=previous,
            promoted=True,
            reason=(
                f"previous version {previous} has no metrics.json; "
                "promoting unconditionally (require_metrics=False)"
            ),
        )

    prev_acc = float(prev_metrics.get(metric_key, float("-inf")))
    delta = candidate_accuracy - prev_acc
    promoted = delta >= min_acc_delta
    if promoted:
        reason = (
            f"accuracy {candidate_accuracy:.4f} >= previous {prev_acc:.4f} "
            f"(delta={delta:+.4f}, threshold={min_acc_delta:+.4f})"
        )
    else:
        reason = (
            f"accuracy {candidate_accuracy:.4f} < previous {prev_acc:.4f} "
            f"(delta={delta:+.4f}, threshold={min_acc_delta:+.4f}); "
            "not promoting"
        )

    return VersionResolution(
        project=project,
        new_version=next_version(project),
        previous_version=previous,
        promoted=promoted,
        reason=reason,
    )


def resolve_runtime_version(
    project: str,
    requested: str | None = None,
) -> str:
    """Resolve which ``v{N}`` to load at API startup.

    Args:
        project: Project name.
        requested: Caller-supplied version (e.g. ``"v2"``), the special
            token ``"latest"`` (or None) for the highest version, or an
            integer / integer-string like ``"2"``. Defaults to
            ``"latest"``.

    Returns:
        The resolved version string (e.g. ``"v2"``).

    Raises:
        FileNotFoundError: If no versions exist for the project and
            ``requested`` is None / ``"latest"``.
        ValueError: If ``requested`` is not a recognised form.
    """
    requested = (requested or os.getenv("MODEL_VERSION") or "latest").strip().lower()
    existing = list_versions(project)
    if not existing:
        raise FileNotFoundError(
            f"No versions exist for project {project!r}. "
            f"Train first: `make train-<project>`.",
        )

    # "latest"
    if requested in ("latest", ""):
        return max(existing, key=_version_int)

    # "vN"
    m = _VERSION_RE.match(requested)
    if m and requested in existing:
        return requested

    # bare integer "2"
    if requested.isdigit():
        candidate = f"v{int(requested)}"
        if candidate in existing:
            return candidate

    raise ValueError(
        f"Requested model version {requested!r} not found for "
        f"{project!r}. Available: {existing}",
    )


def ensure_version_dir(project: str, version: str) -> str:
    """Create ``models/<project>/<version>/`` if it does not exist and return the path."""
    return ensure_dir(models_path(f"{project}/{version}"))


__all__ = [
    "VersionResolution",
    "ensure_version_dir",
    "list_versions",
    "next_version",
    "resolve_promotion",
    "resolve_runtime_version",
]
