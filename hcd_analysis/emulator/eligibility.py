"""Production-model eligibility: the one source of truth for the library loaders (``safe_h5``,
``hcd_analysis.predictor``) and the export and acceptance scripts (docs/release_format_h5.md).

A **production** model is trained on every simulation of its training cache. The **metadata rule**
(``production_recipe_reasons(meta)``, run by every release load) requires that the member's recipe is a JSON object,
declares ``all_sims`` true, carries no research marking (``recipe.research``), and that no key anywhere in the meta is
a reserved held-out or fold name (``RESERVED_FORMS``, rule ``reserved_keys/2``: matched by the normalised WHOLE key
name, at any depth of dicts and lists; a null value counts as absent). Choices derived from a cross-validation (an
epoch count taken from the folds) may be recorded in ``recipe.cv_selection``, a closed schema (``cv_selection/1``) whose
keys avoid every reserved name; the scan exempts nothing. The reserved list is a heuristic against accidental misuse,
not a parser of arbitrary metadata: a novel spelling outside it passes the loader.

Reasons are strings ``"[CODE] text"`` with a stable machine code (``reason_code``); comparisons use the codes, never
the wording. Nothing here resists deliberately forged metadata with a recomputed digest: that defence is the published
digest pinned after a recorded acceptance. This module imports neither the trainer nor the cache reader."""
from __future__ import annotations

import math
import re

# ------------------------------------------------------------------------------------------------ rule identifiers
RULE_RESERVED_KEYS = "reserved_keys/2"
RULE_CV_SELECTION = "cv_selection/1"

# the normalised forms (lower case, "_" and "-" removed) of the reserved held-out / fold key names (reserved_keys/2)
RESERVED_FORMS = frozenset({
    "holdoutsim", "holdoutsims", "heldoutsim", "heldoutsims", "heldout", "fold", "folds", "foldid", "foldssha256",
    "nheld", "nheldrows", "leftout", "leftoutsims", "withheld", "withheldsims", "excludesims", "excludedsims", "loo",
    "loso", "kfold", "cvfold"})

_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_CODE = re.compile(r"\[([A-Z][A-Z0-9_]*)(?::([^\]\s]*))?\] ")


# --------------------------------------------------------------------------------------------------- reason codes
def reason(code, text):
    """``"[CODE] text"``: a reason with its stable machine code (``CODE`` may carry ``:<detail>``)."""
    return f"[{code}] {text}"


def reason_code(r):
    """The full code of a reason string (``"RSV_KEY:recipe.fold"``), or None for an uncoded string."""
    m = _CODE.match(r) if isinstance(r, str) else None
    return None if m is None else (m.group(1) + (f":{m.group(2)}" if m.group(2) is not None else ""))


def code_class(r):
    """``"lineage"`` (LIN_*), ``"tie"`` (TIE_*), ``"research"`` (RESEARCH_BUNDLE), ``"production"`` (every other code:
    the member recipe rules), or None for an uncoded string."""
    c = reason_code(r)
    if c is None:
        return None
    base = c.split(":", 1)[0]
    if base.startswith("LIN_"):
        return "lineage"
    if base.startswith("TIE_"):
        return "tie"
    if base == "RESEARCH_BUNDLE":
        return "research"
    return "production"


def research_reason(purpose):
    """The ineligibility reason a research bundle's lineage block carries (``RESEARCH_BUNDLE``)."""
    return reason("RESEARCH_BUNDLE", f"research bundle: {purpose}")


# ----------------------------------------------------------------------------------------------- reserved keys
def normalise_key(k):
    return str(k).lower().replace("_", "").replace("-", "")


def find_reserved(obj, path=""):
    """Paths (dot-separated; list items by index) of every dict key whose normalised name is a reserved form and whose
    value is not None, at any depth of ``obj`` (dicts and lists). ``fold: 0``, ``[]`` and ``""`` count as present."""
    hits = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else str(k)
            if normalise_key(k) in RESERVED_FORMS and v is not None:
                hits.append(p)
            hits += find_reserved(v, p)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            hits += find_reserved(v, f"{path}.{i}" if path else str(i))
    return hits


def _hex64(x):
    return isinstance(x, str) and bool(_HEX64.match(x))


def _scalar(v):
    return v is None or isinstance(v, (bool, str)) or (isinstance(v, (int, float)) and math.isfinite(v))


# ------------------------------------------------------------------------------------------ cv_selection/1 schema
_CV_KEYS = {"schema", "source_cv_set_id", "source_folds_digest", "derived"}
_CV_OPTIONAL = {"source_record_sha256"}


def cv_selection_problems(cv, recipe):
    """What is wrong with a ``recipe.cv_selection`` record (``cv_selection/1``): a CLOSED schema
    ``{schema, source_cv_set_id, source_folds_digest, [source_record_sha256], derived: {knob: {value, rule}}}``. Each
    knob names an existing recipe key whose value equals the derived value (a JSON scalar). The record is NOT exempt
    from the reserved-key scan; its own keys avoid every reserved name. Empty list: well formed."""
    if not isinstance(cv, dict):
        return [f"recipe.cv_selection is {type(cv).__name__}, not a {RULE_CV_SELECTION} object"]
    out = []
    unknown = sorted(set(cv) - _CV_KEYS - _CV_OPTIONAL)
    missing = sorted(_CV_KEYS - set(cv))
    if unknown:
        out.append(f"recipe.cv_selection has keys outside {RULE_CV_SELECTION}: {unknown}")
    if missing:
        out.append(f"recipe.cv_selection lacks {missing}")
    if cv.get("schema") != RULE_CV_SELECTION:
        out.append(f"recipe.cv_selection schema {cv.get('schema')!r}, expected {RULE_CV_SELECTION}")
    if "source_cv_set_id" in cv and not (isinstance(cv["source_cv_set_id"], str) and cv["source_cv_set_id"]):
        out.append("recipe.cv_selection source_cv_set_id must be a non-empty string")
    if "source_folds_digest" in cv and not _hex64(cv["source_folds_digest"]):
        out.append("recipe.cv_selection source_folds_digest must be a sha256 (64 hex digits)")
    if "source_record_sha256" in cv and not _hex64(cv["source_record_sha256"]):
        out.append("recipe.cv_selection source_record_sha256 must be a sha256 (64 hex digits)")
    der = cv.get("derived")
    if "derived" in cv:
        if not isinstance(der, dict) or not der:
            out.append("recipe.cv_selection derived must be a non-empty object {knob: {value, rule}}")
        else:
            for knob, e in der.items():
                if normalise_key(knob) in RESERVED_FORMS:
                    out.append(f"recipe.cv_selection derives the reserved name {knob!r}: not a selection record")
                if not (isinstance(e, dict) and set(e) == {"value", "rule"} and isinstance(e["rule"], str)
                        and e["rule"].strip()):
                    out.append(f"recipe.cv_selection derived[{knob!r}] must be exactly {{value, rule}} with a rule")
                    continue
                if not _scalar(e["value"]):
                    out.append(f"recipe.cv_selection derived[{knob!r}] value must be a JSON scalar")
                if knob == "cv_selection" or not isinstance(recipe, dict) or knob not in recipe:
                    out.append(f"recipe.cv_selection derived[{knob!r}] names no recipe key")
                elif type(recipe[knob]) is not type(e["value"]) or recipe[knob] != e["value"]:
                    out.append(f"recipe.cv_selection derived[{knob!r}] value {e['value']!r} is not the recipe's "
                               f"{recipe[knob]!r}")
    return out


# -------------------------------------------------------------------------------------- the production metadata rule
def production_recipe_reasons(meta, i=None):
    """The coded reasons (empty: eligible by metadata) why the member with meta ``meta`` is not a production member:
    ``RECIPE_NOT_DICT``, ``ALL_SIMS_NOT_TRUE``, ``RECIPE_RESEARCH``, ``CV_SELECTION`` and one ``RSV_KEY:<path>`` per
    reserved key found anywhere in the meta. ``i`` names the member in the text."""
    who = "the member" if i is None else f"member {i}"
    if not isinstance(meta, dict):
        return [reason("RECIPE_NOT_DICT", f"{who}'s meta is not a JSON object")]
    why = []
    r = meta.get("recipe")
    if not isinstance(r, dict):
        why.append(reason("RECIPE_NOT_DICT", f"{who}'s recipe is {type(r).__name__}, not a JSON object: its training "
                                             "set is not declared"))
    else:
        if r.get("all_sims") is not True:
            why.append(reason("ALL_SIMS_NOT_TRUE", f"{who} does not declare all_sims true (recipe all_sims "
                                                   f"{r.get('all_sims')!r}): a production member is trained on every "
                                                   "simulation of its cache"))
        if r.get("research") is not None:
            why.append(reason("RECIPE_RESEARCH", f"{who} is marked as a research model (recipe.research)"))
        if r.get("cv_selection") is not None:
            why += [reason("CV_SELECTION", f"{who}: {w}") for w in cv_selection_problems(r["cv_selection"], r)]
    for p in find_reserved(meta):
        why.append(reason(f"RSV_KEY:{_code_path(p)}", f"{who} declares the held-out or fold key {p!r} "
                                                       f"({RULE_RESERVED_KEYS})"))
    return why


def _code_path(p):
    """A key path as it appears inside a code: characters other than letters, digits, ``.``, ``_`` and ``-``
    become ``_`` (the human text keeps the exact path)."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", p)
