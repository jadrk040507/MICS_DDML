"""Reuse legacy fitted models only after checking their recorded inputs."""

from hashlib import sha256
import json
from pathlib import Path


def legacy_model_path(project, estimand, name, current_provenance, quick_sample):
    """Return an earlier fitted-model path, or None if provenance differs.

    Sensitivity rows never qualify: their definition changed and they must be
    refitted. This function is deliberately limited to completed full runs.
    """
    if quick_sample or name.startswith("sensitivity") or not (
        name.endswith(("_IRM_clustered", "_IRM_iid", "_APOS", "_APOS_iid"))
    ) or current_provenance is None:
        return None

    project = Path(project)
    old_root = project / "Output" / estimand
    manifest_path = old_root / "manifest.json"
    archived_script = (
        project / "Do file" / "Python" / "archive"
        / "replaced_2026_09_23" / f"{estimand.lower()}.py"
    )
    if not manifest_path.is_file() or not archived_script.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    old = manifest.get("checkpoint_provenance", {})
    if old.get("checkpoint_schema_version") != current_provenance.get("checkpoint_schema_version"):
        return None
    if old.get("settings") != current_provenance.get("settings"):
        return None
    old_files = old.get("files", {})
    new_files = current_provenance.get("files", {})
    if any(old_files.get(key) != new_files.get(key)
           for key in ("data_HH", "data_U5", "environment_lock", "shared_engine")):
        return None
    digest = sha256(archived_script.read_bytes()).hexdigest()
    if digest != old_files.get("analysis_script", {}).get("sha256"):
        return None
    fingerprint = manifest.get("checkpoint_fingerprint", "")
    if len(fingerprint) != 64:
        return None
    candidate = old_root / "checkpoints" / f"{name}_{fingerprint[:12]}.pkl"
    return candidate if candidate.is_file() else None
