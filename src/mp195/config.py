"""Settings and small helpers (owner: Tejaswita)."""

import hashlib
import yaml


def load_params(path="params.yaml"):
    """Read params.yaml into a dictionary."""
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def sha256_of_file(path):
    """SHA-256 fingerprint of a file's raw bytes."""
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()