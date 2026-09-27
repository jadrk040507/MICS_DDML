"""Current checkpoint storage and deterministic provenance fingerprints."""

from functools import lru_cache
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import tempfile

import doubleml as dml
import joblib

from ddml import collect_convex_weights

def atomic_dump(value, path):
    """Publish a complete joblib file only after serialization succeeds."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as temporary:
            temp_path = Path(temporary.name)
        joblib.dump(value, temp_path, compress=3)
        os.replace(temp_path, path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def valid_sensitivity_rows(rows, *, dataset, outcome, specification,
                           method, group_name, group_columns):
    """Reject stale or incomplete benchmark rows before skipping a refit."""
    if not isinstance(rows, list) or len(rows) != (1 if method == "IRM" else 3):
        return False
    treatments = set()
    for row in rows:
        if not isinstance(row, dict):
            return False
        if any(row.get(key) != expected for key, expected in (
            ("dataset", dataset), ("outcome", outcome),
            ("specification", specification), ("method", method),
            ("benchmark_group", group_name),
            ("benchmark_columns", tuple(group_columns)),
        )):
            return False
        try:
            if not all(math.isfinite(float(row[key])) for key in (
                "rv", "rva", "cf_y", "cf_d", "rho",
            )):
                return False
            # NaN is a legitimate derived value for inadmissible finite-
            # sample rho/cf_y inputs; the raw benchmark and the other
            # independently calculated scenario remain useful.
            for key in ("r_equiv_empirical", "r_equiv_adversarial"):
                value = float(row[key])
                if not (math.isfinite(value) or math.isnan(value)):
                    return False
            treatments.add(row["treatment"])
        except (KeyError, TypeError, ValueError):
            return False
    return treatments == ({"Any Treatment"} if method == "IRM" else {"1", "2", "3"})



class OutcomeCheckpointBundle:
    """Expose saved models and lightweight table frames through one mapping.

    Parameters
    ----------
    paths : dict[str, pathlib.Path]
        Checkpoint paths for ``irm_cluster``, ``irm_no_cluster``,
        ``apos_cluster``, and ``apos_no_cluster``.
    table_frames : dict[str, pandas.DataFrame]
        Small in-memory frames used for sample statistics and PSU identifiers.

    Notes
    -----
    Accessing ``bundle[key]`` loads that model only for the current task. This
    avoids keeping several large DoubleML models in memory simultaneously.
    """

    def __init__(self, paths, table_frames):
        """Store checkpoint paths and already-small table data.

        Parameters
        ----------
        paths, table_frames : dict
            Model paths and in-memory descriptive frames documented above.
        """

        self.paths = paths
        self.table_frames = table_frames

    def __getitem__(self, key):
        """Return a table frame immediately or load the requested model.

        Parameters
        ----------
        key : str
            Model or table-frame name.

        Returns
        -------
        object
            DataFrame or fitted DoubleML model associated with ``key``.
        """

        if key in self.table_frames:
            return self.table_frames[key]

        saved_object = joblib.load(self.paths[key])
        if key == "apos_cluster":
            return saved_object["model"]
        return saved_object

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

class CheckpointStore:
    """Read and write only checkpoints matching the current fingerprints."""

    def __init__(
        self,
        checkpoint_dir,
        *,
        estimand,
        quick_sample,
        model_fingerprint,
        sensitivity_fingerprint,
        sample_fraction=0.05,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.estimand = str(estimand).upper()
        self.quick_sample = bool(quick_sample)
        self.model_fingerprint = str(model_fingerprint)
        self.sensitivity_fingerprint = str(sensitivity_fingerprint)
        self.sample_fraction = float(sample_fraction)

    def path(self, name):
        """Return the sole current path for a logical checkpoint name."""
        fingerprint = (
            self.sensitivity_fingerprint
            if name.startswith("sensitivity_")
            else self.model_fingerprint
        )
        sample_tag = (
            f"_sample{int(self.sample_fraction * 100):02d}"
            if self.quick_sample
            else ""
        )
        return self.checkpoint_dir / (
            f"{name}_{fingerprint[:12]}{sample_tag}.pkl"
        )

    def exists(self, name):
        """Return whether the current fingerprinted checkpoint exists."""
        return self.path(name).exists()

    def load(self, name):
        """Load the current checkpoint and propagate deserialization errors."""
        path = self.path(name)
        print(f"Loading checkpoint: {path.name}", flush=True)
        return joblib.load(path)

    def save(self, name, value, fitted_model=False):
        """Save a model or sensitivity checkpoint and return the input value."""
        if fitted_model:
            model = value["model"] if isinstance(value, dict) else value
            if not isinstance(model, dml.DoubleMLAPOS):
                model.convex_weights = collect_convex_weights(model)
                model._models = None

        path = self.path(name)
        if name.startswith("sensitivity_"):
            atomic_dump(value, path)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(value, path, compress=3)
        print(f"Saved checkpoint: {path.name}", flush=True)
        return value
