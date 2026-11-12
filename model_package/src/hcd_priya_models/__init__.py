"""The trained models of hcd_priya, as files inside an installed package. Data only.

``hcd-priya`` depends on this package, so once both are published ``pip install hcd-priya`` brings the models.
Users do not call it:
``hcd_analysis.predictor.load_published`` asks it where a model's files are and checks them against the digests that
``hcd-priya`` pins before it loads anything. This package itself trusts nothing and loads nothing; it imports only
the standard library.

- ``__version__``: the version of the package, which is the version of the model bundle it holds.
- ``manifest()``: what the package holds: ``{"format": "hcd_priya.models/1", "version": ..., "synthetic": bool,
  "models": [{"name": ..., "format": ..., "path": ..., "anchor_sha256": ...}]}``.
- ``model_dir(name)``: the directory of one model's files.

A package built with a synthetic bundle (``"synthetic": true``; its version ends in ``+synthetic``) holds untrained
networks. It exists so that the tests of hcd-priya never need a real model, and it is never published.
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

from ._version import __version__

FORMAT = "hcd_priya.models/1"
__all__ = ["FORMAT", "__version__", "data_dir", "manifest", "model_dir"]


def data_dir() -> Path:
    """The directory that holds ``models.json`` and one sub-directory per model."""
    path = Path(str(resources.files(__name__).joinpath("data")))
    if not path.is_dir():
        raise FileNotFoundError(
            f"hcd_priya_models {__version__} holds no model files at {path}: the package was built without its "
            "data, or it is not installed as plain files"
        )
    return path


def manifest() -> dict:
    """What this package holds (the parsed ``models.json``); see the module docstring for the keys."""
    with open(data_dir() / "models.json", encoding="utf-8") as fh:
        doc = json.load(fh)
    if (
        not isinstance(doc, dict)
        or doc.get("format") != FORMAT
        or not isinstance(doc.get("models"), list)
    ):
        raise ValueError(
            f"hcd_priya_models {__version__}: models.json is not in the format {FORMAT}"
        )
    return doc


def model_dir(name: str) -> Path:
    """The directory of the model called ``name``. ``KeyError`` if the package does not hold it."""
    for model in manifest()["models"]:
        if isinstance(model, dict) and model.get("name") == name:
            sub = str(model.get("path"))
            path = data_dir() / sub
            if Path(sub).name != sub or sub in ("", ".", "..") or not path.is_dir():
                raise ValueError(
                    f"hcd_priya_models {__version__}: the model {name!r} has no directory {path}"
                )
            return path
    raise KeyError(name)
