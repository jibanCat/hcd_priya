"""The model versions that this version of hcd_priya accepts, with the digest each must have.

The trained models come in the companion package ``hcd-priya-models``, which ``hcd-priya`` requires, so installing
``hcd-priya`` installs them. ``hcd_analysis.predictor.load_published`` loads a model from that package only if the
package's version is listed here, and then only against the digest listed here: the trust is in this file, which
ships with the code, never in the model package. (The one exception is a synthetic test build, which holds untrained
networks and loads only with an explicit switch; see ``load_published``.)

``MODELS``: one entry per accepted model and version::

    {"name": "lfmf", "version": "<version>", "format": "hcd_priya.release/1",
     "anchor_sha256": "<64 hex digits>"}

- ``name``: what the user passes to ``load_published``.
- ``version``: the version of ``hcd-priya-models`` that holds it, which is the version of the model bundle.
- ``format``: ``hcd_priya.release/1``, an LF+MF release directory.
- ``anchor_sha256``: the sha256 of the release's ``release.json``. Loading verifies every file of the release
  against it.

The versioning rule: the version of ``hcd-priya-models`` is the version of the bundle it holds; a version of
``hcd-priya`` lists here the exact versions and digests it accepts, and its dependency on ``hcd-priya-models``
(``pyproject.toml``) is the range that contains them. ``model_package/prepare.py`` prints the entry for a release.

No model is published yet: ``MODELS`` is empty.
"""

MODELS = []
