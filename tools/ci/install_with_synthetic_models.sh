#!/usr/bin/env bash
# Install an hcd-priya wheel into an environment, together with a model package built with a SYNTHETIC bundle.
#
#   tools/ci/install_with_synthetic_models.sh PYTHON WHEEL[extra] MODELS_DIR [pip install options ...]
#
# hcd-priya requires hcd-priya-models. CI and development environments never need the real models: this script
# builds the model package with a synthetic bundle (untrained networks, version 0+synthetic, which a package index
# refuses) and installs both wheels. The steps, and why they are in this order:
#   1. every requirement of the wheel except the model package (the synthetic bundle is written with hcd_analysis
#      itself and the test suite's builder, so the wheel's requirements come first; pass the extra "test");
#   2. the model wheel, built into MODELS_DIR, unless one is already there (a second environment reuses it);
#   3. the two wheels, both given as files: the model package is taken from the file, never looked up on an index.
# Run it from the repository root. PYTHON is the interpreter of a fresh virtual environment.
set -euo pipefail
py="$1"; target="$2"; models="$3"; shift 3
wheel="$target"; extra=""
if [[ "$target" =~ ^(.*)\[([A-Za-z0-9_.-]+)\]$ ]]; then wheel="${BASH_REMATCH[1]}"; extra="${BASH_REMATCH[2]}"; fi
mkdir -p "$models"
req="$(mktemp)"
trap 'rm -f "$req"' EXIT
"$py" tools/ci/wheel_requires.py "$wheel" ${extra:+--extra "$extra"} --without hcd-priya-models > "$req"
"$py" -m pip install "$@" -r "$req" build
shopt -s nullglob
found=("$models"/hcd_priya_models-*.whl)
if [ "${#found[@]}" -eq 0 ]; then
  version_file=model_package/src/hcd_priya_models/_version.py
  if [ -e "$version_file" ] && ! grep -q '+synthetic' "$version_file"; then
    echo "model_package/ holds a prepared model that is not a synthetic build: not touching it." >&2
    echo "Build elsewhere, or run 'python model_package/prepare.py --clean' yourself first." >&2
    exit 1
  fi
  "$py" model_package/prepare.py --clean
  JAX_PLATFORMS=cpu "$py" model_package/prepare.py --synthetic
  "$py" -m build --wheel --outdir "$models" model_package
  "$py" model_package/prepare.py --clean
  found=("$models"/hcd_priya_models-*.whl)
else
  echo "reusing the model wheel ${found[0]}"
fi
if [ "${#found[@]}" -ne 1 ]; then
  echo "expected exactly one hcd_priya_models wheel in $models, found ${#found[@]}" >&2
  exit 1
fi
"$py" -m pip install "$@" "$target" "${found[0]}"
