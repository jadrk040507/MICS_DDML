"""Atomic writes for resumable sensitivity benchmark checkpoints."""

import os
import math
from pathlib import Path
import tempfile

import joblib


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
