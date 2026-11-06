"""Comparing computed floating-point test results across machines, and the opt-in exact-bitwise replay.

A sha256 of computed float64 arrays is reproducible only on the same machine and software stack. numpy dispatches its
elementary functions by the CPU's instruction set (AVX-512, AVX2, ...) and otherwise calls the system libm; OpenBLAS
chooses its kernels by CPU model; XLA compiles for the host. Any of these can move the last bits of a result, and
GitHub-hosted runners come with different CPU models. So:

- **By default, and always on a GitHub runner,** a test compares its computed float arrays with stored reference
  arrays (synthetic values: they come from the synthetic fixtures, never from an analysis), field by field, with an
  explicit relative and absolute tolerance for each field (``field_mismatches``). Each test file states and justifies
  its own per-field tolerances. There is no default tolerance.
- **``HCD_EXACT_REPLAY=1``** asks, in addition, for the bit-level golden digests: a replay on the same machine with
  the pinned stack. It requires ``HCD_PINNED_STACK=1`` and numpy, jax and jaxlib at their pinned versions, and it is
  refused on a GitHub runner (``GITHUB_ACTIONS=true``), which is not a same-machine replay (``exact_replay``).
- **Discrete invariants** (source and stored-file hashes, lineage, cache identity, split indices, schema, integer
  arrays) are not handled here: they stay exact everywhere.

``HCD_PINNED_STACK=1`` keeps its meaning: the run declares itself the supported pinned stack, and a numpy that is not
the pinned one fails the run (``check_declared_pinned_stack``)."""
import os
import re

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONSTRAINTS_FILE = os.path.join(REPO, "tools", "ci", "constraints-devenv.txt")
PINNED_STACK_ENV = "HCD_PINNED_STACK"
EXACT_REPLAY_ENV = "HCD_EXACT_REPLAY"
EXACT_REPLAY_PACKAGES = ("numpy", "jax", "jaxlib")
# The pins of Session B's tools/ci/constraints-devenv.txt, used only where that file is absent (the wheel tier runs a
# copy of tests/ alone); test_wc_map.py checks that they equal the file's pins.
PINNED_RECORDED = {"numpy": ("2.4.6",), "jax": ("0.10.1",), "jaxlib": ("0.10.1",)}

_STRICT_REQUIREMENT = re.compile(r"(?P<name>[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)\s*(?:\[[^\]]*\])?\s*"
                                 r"(?P<spec>[^;]*?)\s*(?:;(?P<marker>.*))?")
_EXACT_SPEC = re.compile(r"===?\s*(?P<version>[0-9][0-9A-Za-z.+!_-]*)")


def _requirement_strict(line):
    """``(canonical name, exact version or None)`` of one requirement line, without ``packaging``: a name (``-``,
    ``_`` and ``.`` runs equal, any case), optional extras, the specifiers and an optional environment marker. The
    version is returned only for one ``==`` or ``===`` specifier without a wildcard. Unparsable: ValueError."""
    m = _STRICT_REQUIREMENT.fullmatch(line)
    if m is None or (m.group("marker") is not None and not m.group("marker").strip()):
        raise ValueError(f"unparsable requirement line {line!r}")
    e = _EXACT_SPEC.fullmatch(m.group("spec").strip())
    return re.sub(r"[-_.]+", "-", m.group("name")).lower(), e.group("version") if e and "*" not in e.group(0) else None


def _requirement(line):
    """``(canonical name, exact version or None)`` of one requirement line, with the standard requirement grammar
    (``packaging.requirements``, a pytest dependency) when it is importable, else ``_requirement_strict``."""
    try:
        from packaging.requirements import InvalidRequirement, Requirement
        from packaging.utils import canonicalize_name
    except ImportError:
        return _requirement_strict(line)
    try:
        req = Requirement(line)
    except InvalidRequirement as e:
        raise ValueError(f"unparsable requirement line {line!r}: {e}") from None
    specs = list(req.specifier)
    exact = len(specs) == 1 and specs[0].operator in ("==", "===") and "*" not in specs[0].version
    return canonicalize_name(req.name), specs[0].version if exact else None


def package_pins(text, name, parse=None):
    """The exact pins of package ``name`` in a pip constraints text, in any valid spelling (spaces, case, an
    environment marker, an inline comment, per-requirement options such as ``--hash``); comment, blank and option
    (``-c``, ``-r``) lines are skipped. Fail closed: a line for ``name`` that is not one exact pin (a range, a wildcard,
    several specifiers) or any unparsable line raises ValueError. ``parse``: the line parser (default
    ``_requirement``)."""
    parse = parse or _requirement
    pins = []
    for raw in text.splitlines():
        line = re.sub(r"\s+--\S.*$", "", re.sub(r"(^|\s)#.*$", "", raw)).strip()
        if not line or line.startswith("-"):
            continue
        got, version = parse(line)
        if got == name:
            if version is None:
                raise ValueError(f"the {name} line {raw.strip()!r} is not an exact pin ({name}==X)")
            pins.append(version)
    return tuple(pins)


def numpy_pins(text, parse=None):
    """``package_pins(text, "numpy", parse)``."""
    return package_pins(text, "numpy", parse=parse)


def pinned_versions(name, path=CONSTRAINTS_FILE):
    """The version(s) of ``name`` in the supported, pinned environment (docs/compatibility.md; the PI's rule B-D3).

    - When the constraints file (Session B's ``tools/ci/constraints-devenv.txt``) exists, its exact pins of ``name``,
      read by ``package_pins``. Fail closed: a file that pins no version of ``name``, or whose line for it is not one
      exact pin, raises ValueError.
    - Only when the file is absent (a tests-only copy, or a tree without it): ``PINNED_RECORDED``."""
    if not os.path.exists(path):
        return PINNED_RECORDED[name]
    with open(path) as fh:
        pins = package_pins(fh.read(), name)
    if not pins:
        raise ValueError(f"{path} exists but pins no {name} version ({name}==X): the pinned stack cannot be told, so "
                         "the bit-level digests cannot be placed; fix the constraints file or this parser")
    return pins


def pinned_numpy_versions(path=CONSTRAINTS_FILE):
    """``pinned_versions("numpy", path)``."""
    return pinned_versions("numpy", path)


def pinned_stack_declared(environ=None):
    """Whether this run declares itself the supported pinned stack: ``HCD_PINNED_STACK`` exactly ``"1"`` (Session B's
    supported CI job and its pinned replays set it; the newer-dependency jobs set it empty). Unset, empty or any other
    value (``"true"``, ``" 1"``) is not a declaration."""
    return (os.environ if environ is None else environ).get(PINNED_STACK_ENV) == "1"


def running_versions():
    """The running numpy, jax and jaxlib versions (None for one that is not importable)."""
    out = {"numpy": np.__version__}
    for mod in ("jax", "jaxlib"):
        try:
            out[mod] = __import__(mod).__version__
        except ImportError:
            out[mod] = None
    return out


def check_declared_pinned_stack(versions=None, environ=None, pins=None):
    """A run that declares the pinned stack (``pinned_stack_declared``) MUST be on a pinned numpy: otherwise
    ValueError, so a supported job that drifted to another numpy fails instead of quietly passing. ``versions``: as
    ``running_versions()`` (default); ``pins``: ``{name: versions}`` (default ``pinned_versions``)."""
    version = (running_versions() if versions is None else versions)["numpy"]
    allowed = pinned_numpy_versions() if pins is None else tuple(pins["numpy"])
    if pinned_stack_declared(environ) and version not in allowed:
        raise ValueError(f"{PINNED_STACK_ENV}=1 declares the supported pinned stack, but numpy {version} is not one of "
                         f"the pinned versions {allowed}: fix the environment or the pin")


def exact_replay(environ=None, versions=None, pins=None):
    """Whether the tests also require the bit-level golden digests (the same-machine pinned replay).

    True only when ``HCD_EXACT_REPLAY`` is exactly ``"1"``. Then it is refused (ValueError, never a silent fallback)
    on a GitHub runner (``GITHUB_ACTIONS`` is ``"true"``), without ``HCD_PINNED_STACK=1``, or when numpy, jax or
    jaxlib is not at its pinned version. Unset, empty or any other value: False, the tolerance comparison alone (the
    default, and the only mode on a GitHub runner). ``check_declared_pinned_stack`` runs first, in every mode."""
    env = os.environ if environ is None else environ
    versions = running_versions() if versions is None else versions
    pins = {n: pinned_versions(n) for n in EXACT_REPLAY_PACKAGES} if pins is None else pins
    check_declared_pinned_stack(versions, env, pins)
    if env.get(EXACT_REPLAY_ENV) != "1":
        return False
    if env.get("GITHUB_ACTIONS") == "true":
        raise ValueError(f"{EXACT_REPLAY_ENV}=1 on a GitHub runner (GITHUB_ACTIONS=true): the exact digests are for a "
                         "same-machine replay with the pinned stack; a hosted runner's CPU is not known in advance")
    if not pinned_stack_declared(env):
        raise ValueError(f"{EXACT_REPLAY_ENV}=1 needs {PINNED_STACK_ENV}=1: the exact digests hold only on the pinned "
                         "stack")
    off = [f"{n} {versions.get(n)} (pinned {tuple(pins[n])})" for n in EXACT_REPLAY_PACKAGES
           if versions.get(n) not in tuple(pins[n])]
    if off:
        raise ValueError(f"{EXACT_REPLAY_ENV}=1, but not on the pinned stack: {'; '.join(off)}")
    return True


def field_mismatches(got, want, tols):
    """The names of the fields where ``got`` departs from the reference ``want`` (empty: they agree).

    ``got`` and ``want`` map field names to arrays; ``tols`` maps EVERY field of ``want`` to its own ``(rtol, atol)``
    (a missing or extra tolerance is a ValueError: there is no default). The key sets of ``got`` and ``want`` must be
    equal (else ``"keys"``). Per field: the same dtype and shape; a non-float field (integers, booleans) must be equal
    exactly, and its tolerance must be ``(0, 0)``; a float field must have the same non-finite entries in the same
    places, and every finite element must satisfy ``|got - want| <= atol + rtol * |want|``."""
    if set(tols) != set(want):
        raise ValueError(f"every reference field needs its own tolerance: missing {sorted(set(want) - set(tols))}, "
                         f"unknown {sorted(set(tols) - set(want))}")
    out = ["keys"] if set(got) != set(want) else []
    for name in sorted(want):
        if name not in got:
            continue
        g, w = np.asarray(got[name]), np.asarray(want[name])
        rtol, atol = tols[name]
        if not (rtol >= 0 and atol >= 0):
            raise ValueError(f"{name}: tolerances must be non-negative, got {(rtol, atol)}")
        if g.dtype != w.dtype or g.shape != w.shape:
            out.append(name)
            continue
        if w.dtype.kind != "f":
            if (rtol, atol) != (0, 0):
                raise ValueError(f"{name}: a {w.dtype} field is compared exactly; its tolerance must be (0, 0)")
            if not np.array_equal(g, w):
                out.append(name)
            continue
        fg, fw = np.isfinite(g), np.isfinite(w)
        if not np.array_equal(fg, fw) or not np.array_equal(g[~fw], w[~fw], equal_nan=True):
            out.append(name)
            continue
        if np.any(np.abs(g[fw] - w[fw]) > atol + rtol * np.abs(w[fw])):
            out.append(name)
    return out


def load_reference(path):
    """The reference arrays stored at ``path`` (an ``.npz`` written with ``np.savez``; no pickles)."""
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}
