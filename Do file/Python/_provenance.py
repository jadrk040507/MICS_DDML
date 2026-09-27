"""Deterministic provenance fingerprints for expensive research checkpoints.

The fingerprint changes whenever an input dataset, the analysis script, the
locked environment, or an explicitly declared analysis setting changes. It
is intentionally independent of absolute machine-specific paths.
"""

from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path


@lru_cache(maxsize=None)
def _file_sha256(path_string, size, modified_nanoseconds):
    """Hash one file, caching by path plus filesystem identity metadata."""

    del size, modified_nanoseconds
    digest = sha256()
    with Path(path_string).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path):
    """Return portable identity metadata for one provenance input."""

    path = Path(path).resolve()
    if not path.exists():
        return {"status": "missing"}
    metadata = path.stat()
    return {
        "status": "present",
        "size": metadata.st_size,
        "sha256": _file_sha256(
            str(path),
            metadata.st_size,
            metadata.st_mtime_ns,
        ),
    }


def build_checkpoint_provenance(schema_version, files, settings):
    """Return a stable fingerprint and the complete human-auditable payload."""

    payload = {
        "checkpoint_schema_version": int(schema_version),
        "files": {
            label: _file_record(path)
            for label, path in sorted(files.items())
        },
        "settings": settings,
    }
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return sha256(serialized).hexdigest(), payload


def build_sensitivity_provenance(schema_version, model_fingerprint, files, settings=None):
    """Fingerprint sensitivity checkpoints separately from fitted models."""

    sensitivity_settings = {
        "model_checkpoint_fingerprint": str(model_fingerprint),
    }
    if settings:
        sensitivity_settings.update(settings)
    return build_checkpoint_provenance(
        schema_version,
        files,
        sensitivity_settings,
    )
