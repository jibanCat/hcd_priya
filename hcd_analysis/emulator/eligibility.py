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

The **content rules** (they need the training cache; run at export and acceptance, never at load):
``production_content_check`` and ``research_content_check`` tie the cache first, then recompute the split from THIS
cache's rows with the trainer's own rule (no simulation count is assumed), refit the normaliser on the training rows
(``train._fit_norm``, every tensor compared bitwise: ``norm_fingerprint/1``) and the w_c map on the training
simulations (``wcmap_refit/1`` on the full cache, ``wcmap_refit_subset/1`` on a training subset; the whole payload and
the record's fit fields compared, never the writing run's code, time or settings digest). Each component is PASS,
MISMATCH or UNAVAILABLE (``classify``: a non-bitwise tensor within ``DRIFT_BOUND`` relative is UNAVAILABLE only with a
RECORDED training environment that differs from the current one; with none recorded the strict rule gives MISMATCH).
Production requires PASS everywhere; a research export refuses MISMATCH and records UNAVAILABLE. The trainer and the
cache reader are imported inside these functions only, never by a loader.

A **research** bundle is marked at the bundle level (``provenance.research``, schema ``research/1``:
``check_research_marking``). Every production loader refuses it whatever its split; it loads only through
``safe_h5.load_research`` / ``predictor.load_research_bundle``, and ``research_member_reasons`` decides which members it
admits (one declared split per bundle).

Reasons are strings ``"[CODE] text"`` with a stable machine code (``reason_code``); comparisons use the codes, never
the wording. Nothing here resists deliberately forged metadata with a recomputed digest: that defence is the published
digest pinned after a recorded acceptance."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re

import numpy as np

# ------------------------------------------------------------------------------------------------ rule identifiers
RULE_RESERVED_KEYS = "reserved_keys/2"
RULE_CV_SELECTION = "cv_selection/1"
RESEARCH_SCHEMA = "research/1"
RULE_PRODUCTION_SPLIT = "production_row_split/1"      # data.production_row_split: none or one simulation held out
RULE_MULTI_HOLDOUT = "multi_holdout_row_split/1"      # data.multi_holdout_row_split: a set of simulations held out
RULE_EXPLICIT_ROWS = "explicit_rows/1"                # reserved: needs trainers that record row-index digests
RULE_UNVERIFIABLE = "unverifiable"                    # a declared split whose content cannot be recomputed
SPLIT_RULES = (RULE_PRODUCTION_SPLIT, RULE_MULTI_HOLDOUT, RULE_UNVERIFIABLE)
RULE_NORM = "norm_fingerprint/1"
RULE_WCMAP = "wcmap_refit/1"
RULE_WCMAP_SUBSET = "wcmap_refit_subset/1"
RULE_CACHE_TIE = "cache_tie/1"

PASS, MISMATCH, UNAVAILABLE = "PASS", "MISMATCH", "UNAVAILABLE"      # the outcomes of a content-check component
DRIFT_BOUND = 1e-12           # max relative difference a recorded environment change may explain (v3 2.3)
RULES = {"reserved_keys": RULE_RESERVED_KEYS, "cv_selection": RULE_CV_SELECTION, "research": RESEARCH_SCHEMA,
         "split": [RULE_PRODUCTION_SPLIT, RULE_MULTI_HOLDOUT, RULE_UNVERIFIABLE], "normaliser": RULE_NORM,
         "wc_map": [RULE_WCMAP, RULE_WCMAP_SUBSET], "cache": RULE_CACHE_TIE}

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


class EligibilityError(ValueError):
    """A research marking, a selection record or a content-check input is malformed."""


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


# ---------------------------------------------------------------------------------------- the research marking
_RS_KEYS = {"schema", "purpose", "split_rule", "split", "content_checks", "wc_map"}
_RS_OPTIONAL = {"folds_digest", "cv_set_id", "fold_file_sha256", "restored_epoch"}
_SPLIT_KEYS = {RULE_PRODUCTION_SPLIT: ({"held_out_sims", "val_seed", "val_frac", "n_train", "n_val"}, set()),
               RULE_MULTI_HOLDOUT: ({"held_out_sims", "fold", "val_seed", "val_frac", "n_train", "n_val"}, set()),
               RULE_UNVERIFIABLE: ({"held_out_sims", "description"}, set())}
_WC_RS_KEYS = {"rule", "fitted_on_sims_sha256", "n_sims"}


def _int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def check_research_marking(rs):
    """Refuse (EligibilityError) a ``provenance.research`` that is not a well-formed ``research/1`` record; returns it.

        {"schema": "research/1", "purpose": str, "split_rule": one of SPLIT_RULES,
         "split": {held_out_sims: [sorted names], ...per rule}, "content_checks": [component records],
         "wc_map": {"rule", "fitted_on_sims_sha256", "n_sims"},
         optional: "folds_digest" (hex64 or null), "cv_set_id" (str or null), "fold_file_sha256" (hex64 or null),
         "restored_epoch" ({member: {...}} or null)}"""
    def bad(why):
        raise EligibilityError(f"provenance.research: {why}")
    if not isinstance(rs, dict):
        bad(f"is {type(rs).__name__}, not an object")
    if set(rs) - _RS_KEYS - _RS_OPTIONAL or _RS_KEYS - set(rs):
        bad(f"keys {sorted(rs)}, expected {sorted(_RS_KEYS)} plus optionally {sorted(_RS_OPTIONAL)}")
    if rs["schema"] != RESEARCH_SCHEMA:
        bad(f"schema {rs['schema']!r}, expected {RESEARCH_SCHEMA}")
    if not (isinstance(rs["purpose"], str) and rs["purpose"].strip()) or rs["purpose"] != rs["purpose"].strip():
        bad("purpose must be a non-empty string without surrounding space")
    if rs["split_rule"] == RULE_EXPLICIT_ROWS:
        bad(f"{RULE_EXPLICIT_ROWS} needs trainers that record the row-index digests; it is not supported yet")
    if rs["split_rule"] not in SPLIT_RULES:
        bad(f"split_rule {rs['split_rule']!r} is not one of {list(SPLIT_RULES)}")
    sp = rs["split"]
    need, opt = _SPLIT_KEYS[rs["split_rule"]]
    if not isinstance(sp, dict) or set(sp) - need - opt or need - set(sp):
        bad(f"split keys {sorted(sp) if isinstance(sp, dict) else sp!r}, expected {sorted(need)} for "
            f"{rs['split_rule']}")
    held = sp["held_out_sims"]
    if not (isinstance(held, list) and all(isinstance(s, str) and s for s in held) and held == sorted(set(held))):
        bad("split.held_out_sims must be a sorted list of distinct simulation names")
    if rs["split_rule"] == RULE_PRODUCTION_SPLIT and len(held) > 1:
        bad(f"{RULE_PRODUCTION_SPLIT} holds out at most one simulation")
    if rs["split_rule"] == RULE_MULTI_HOLDOUT and not held:
        bad(f"{RULE_MULTI_HOLDOUT} declares a non-empty held-out set")
    if rs["split_rule"] in (RULE_PRODUCTION_SPLIT, RULE_MULTI_HOLDOUT):
        if not (_int(sp["val_seed"]) and isinstance(sp["val_frac"], float) and 0 <= sp["val_frac"] < 1
                and _int(sp["n_train"]) and _int(sp["n_val"]) and sp["n_train"] > 0 and sp["n_val"] >= 0):
            bad("split val_seed, val_frac, n_train and n_val must be an int, a float in [0, 1) and counts")
        if rs["split_rule"] == RULE_MULTI_HOLDOUT and sp["fold"] is not None and not _int(sp["fold"]):
            bad("split.fold must be an int or null")
    elif not (isinstance(sp["description"], str) and sp["description"].strip()):
        bad("an unverifiable split carries a description")
    for k in ("folds_digest", "fold_file_sha256"):
        if rs.get(k) is not None and not _hex64(rs[k]):
            bad(f"{k} must be a sha256 (64 hex digits) or null")
    if rs.get("cv_set_id") is not None and not (isinstance(rs["cv_set_id"], str) and rs["cv_set_id"]):
        bad("cv_set_id must be a non-empty string or null")
    if not isinstance(rs["content_checks"], list) or not all(isinstance(c, dict) for c in rs["content_checks"]):
        bad("content_checks must be a list of component records")
    w = rs["wc_map"]
    if not (isinstance(w, dict) and set(w) == _WC_RS_KEYS and w["rule"] in (RULE_WCMAP, RULE_WCMAP_SUBSET)
            and (w["fitted_on_sims_sha256"] is None or _hex64(w["fitted_on_sims_sha256"])) and _int(w["n_sims"])
            and w["n_sims"] > 0):
        bad(f"wc_map must be {{rule, fitted_on_sims_sha256, n_sims}} with rule {RULE_WCMAP_SUBSET} or {RULE_WCMAP}")
    try:
        json.dumps(rs, allow_nan=False)
    except (TypeError, ValueError) as e:
        bad(f"is not plain finite JSON ({e})")
    return rs


def declared_exclusion(recipe):
    """``(kind, held_out_sims)`` the recipe declares: ``("multi", [...])`` for a ``holdout_sims`` list, ``("one",
    [name])`` for a ``holdout_sim`` name, ``("none", [])`` when it declares no exclusion, ``("other", [])`` when it
    carries a reserved key of no recognised rule (or a malformed one)."""
    hits = find_reserved(recipe)
    if not hits:
        return "none", []
    hs, h1 = recipe.get("holdout_sims"), recipe.get("holdout_sim")
    if hs is not None and h1 is None and isinstance(hs, list) and all(isinstance(s, str) for s in hs):
        return "multi", sorted(hs)
    if h1 is not None and hs is None and isinstance(h1, str):
        return "one", [h1]
    return "other", []


_MULTI_OK = {"holdout_sims", "fold", "folds_sha256"}       # the keys W-1-style fold members carry (multi_holdout)
_ONE_OK = {"holdout_sim"}


def research_member_reasons(metas, rs):
    """What keeps the members from this research bundle (``rs``: its checked ``provenance.research``); empty: admitted.

    (a) a member declaring an exclusion under the bundle's recognised split rule is admitted as it is (the existing
        W-1 fold members): a non-empty held-out set equal to ``split.held_out_sims``, its ``fold`` and
        ``folds_sha256`` equal to ``split.fold`` and ``folds_digest``, its ``val_seed`` / ``val_frac`` / ``n_train`` /
        ``n_val`` equal to the split's, no reserved key of another rule, and not ``all_sims`` true (contradictory);
    (b) a member declaring NO exclusion (a full-simulation research model) must carry ``recipe.research`` naming the
        bundle's purpose (otherwise it is indistinguishable from a production member), under a split that holds out
        nothing;
    under ``unverifiable`` (nothing is recomputed) a member that declares an exclusion needs a non-empty declared
    held-out set, and a member's ``holdout_sims`` / ``holdout_sim`` must equal it exactly, so the w_c map's training
    set (the cache minus the declared set) never includes a member's held-out simulation.
    A bundle mixing (a) and (b) members is refused, and a member whose ``recipe.research`` names another purpose."""
    rule, sp = rs["split_rule"], rs["split"]
    out, kinds = [], set()
    for i, m in enumerate(metas):
        r = m.get("recipe") if isinstance(m, dict) else None
        if not isinstance(r, dict):
            out.append(f"member {i}'s recipe is not a JSON object")
            continue
        rr = r.get("research")
        if rr is not None and not (isinstance(rr, dict) and rr.get("purpose") == rs["purpose"]):
            out.append(f"member {i}'s recipe.research names another purpose than the bundle's {rs['purpose']!r}")
        kind, held = declared_exclusion(r)
        kinds.add("exclusion" if kind != "none" else "full")
        if kind == "none":
            if rr is None:
                out.append(f"member {i} declares no exclusion and carries no recipe.research: it is "
                           "indistinguishable from a production member")
            if sp["held_out_sims"]:
                out.append(f"member {i} declares no exclusion, the bundle's split holds out {sp['held_out_sims']}")
        else:
            if r.get("all_sims") is True:
                out.append(f"member {i} declares an exclusion and all_sims true (contradictory)")
            if rule == RULE_UNVERIFIABLE:
                # nothing is recomputed under this rule, so the declared held-out set (from which the w_c map's
                # training set is taken) must be the members' own: never empty, and equal to a recognised exclusion
                if not sp["held_out_sims"]:
                    out.append(f"member {i} declares an exclusion ({sorted({p.split('.')[0] for p in find_reserved(r)})});"
                               " an unverifiable split must declare the held-out simulations")
                elif kind in ("multi", "one") and held != sp["held_out_sims"]:
                    out.append(f"member {i} holds out {held}, the bundle's split {sp['held_out_sims']}")
                continue
            keys = {p.split(".")[0] for p in find_reserved(r)}
            allowed = _MULTI_OK if rule == RULE_MULTI_HOLDOUT else _ONE_OK
            if kind not in (("multi",) if rule == RULE_MULTI_HOLDOUT else ("one",)) or keys - allowed:
                out.append(f"member {i}'s declared exclusion ({sorted(keys)}) is not one of the {rule} rule "
                           f"({sorted(allowed)}); declare the bundle {RULE_UNVERIFIABLE} instead")
                continue
            if not held:
                out.append(f"member {i} declares an empty held-out set: no exclusion")
            if held != sp["held_out_sims"]:
                out.append(f"member {i} holds out {held}, the bundle's split {sp['held_out_sims']}")
            if rule == RULE_MULTI_HOLDOUT:
                if r.get("fold") != sp["fold"]:
                    out.append(f"member {i}'s fold {r.get('fold')!r} is not the split's {sp['fold']!r}")
                if r.get("folds_sha256") != rs.get("folds_digest"):
                    out.append(f"member {i}'s folds_sha256 is not the bundle's folds_digest")
        if rule == RULE_UNVERIFIABLE:
            continue
        for k in ("val_seed", "val_frac", "n_train", "n_val"):
            if r.get(k) != sp[k] or type(r.get(k)) is not type(sp[k]):
                out.append(f"member {i}'s {k} {r.get(k)!r} is not the split's {sp[k]!r}")
    if len(kinds) > 1:
        out.append("the bundle mixes members that declare an exclusion with full-simulation members")
    return out


def split_summary(rs):
    """The research split as surfaced on research loads and predictions."""
    sp = rs["split"]
    return {"rule": rs["split_rule"], "held_out_sims": list(sp["held_out_sims"]), "fold": sp.get("fold"),
            "folds_digest": rs.get("folds_digest"), "n_train": sp.get("n_train"), "n_val": sp.get("n_val"),
            "cv_set_id": rs.get("cv_set_id")}


def sims_sha256(names):
    """sha256 of the canonical JSON of the sorted distinct simulation names (the identity of a simulation set)."""
    from .. import lineage as L
    return hashlib.sha256(L.canonical_json(sorted(set(map(str, names)))).encode("ascii")).hexdigest()


def component(name, rule, outcome, sub_reason, *, detail=None, diff=None):
    """One recorded content-check component outcome (finite JSON only): ``{component, rule, outcome, sub_reason,
    detail}`` plus the tensor-difference numbers when ``diff`` is given."""
    if outcome not in (PASS, MISMATCH, UNAVAILABLE):
        raise EligibilityError(f"outcome {outcome!r} is not PASS, MISMATCH or UNAVAILABLE")
    rec = {"component": name, "rule": rule, "outcome": outcome, "sub_reason": sub_reason, "detail": detail or {}}
    if diff is not None:
        rec.update(n_tensors=diff["n"], n_bitwise=diff["n_bitwise"], max_abs=diff["max_abs"], max_rel=diff["max_rel"],
                   worst=diff["worst"])
    json.dumps(rec, allow_nan=False)
    return rec


# ------------------------------------------------------------------------------------------------- content rules
def index_sha256(idx):
    """sha256 of row indices as little-endian int64 bytes."""
    return hashlib.sha256(np.ascontiguousarray(np.asarray(idx, dtype="<i8")).tobytes()).hexdigest()


def tensors_sha256(tensors):
    """sha256 over the sorted tensors: name, dtype, shape and bytes of each (the golden-digest rule)."""
    h = hashlib.sha256()
    for k in sorted(tensors):
        a = np.ascontiguousarray(tensors[k])
        h.update(k.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def environment():
    """The environment a content check ran in: Python, numpy, jax, jaxlib, scipy and h5py versions and the hcd_priya
    code identity (lineage.code_identity)."""
    import platform
    from .. import lineage as L
    out = {"python": platform.python_version(), "numpy": np.__version__}
    for mod in ("jax", "jaxlib", "scipy", "h5py"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:                                              # noqa: BLE001  (absent: recorded as None)
            out[mod] = None
    out["hcd_priya"] = L.code_identity()
    return out


def _finite(x):
    return float(x) if x is not None and math.isfinite(float(x)) else None


def tensor_differences(stored, recomputed):
    """``{n, n_bitwise, same_structure, max_abs, max_rel, worst}`` over two dicts of arrays. The relative difference
    of a pair is |a-b| / max(|a|, |b|, floor) with floor = 1e-300 + 1e-12 max(|a|, |b|) over the tensor (v3.1 E2)."""
    names = sorted(set(stored) | set(recomputed))
    out = {"n": len(names), "n_bitwise": 0, "same_structure": set(stored) == set(recomputed), "max_abs": 0.0,
           "max_rel": 0.0, "worst": None}
    for k in names:
        if k not in stored or k not in recomputed:
            continue
        a, b = np.asarray(stored[k]), np.asarray(recomputed[k])
        if a.shape != b.shape or a.dtype != b.dtype:
            out["same_structure"] = False
            continue
        if a.tobytes() == b.tobytes():
            out["n_bitwise"] += 1
            continue
        a64, b64 = a.astype(np.float64), b.astype(np.float64)
        if not (np.all(np.isfinite(a64)) and np.all(np.isfinite(b64))):
            out["same_structure"], out["worst"] = False, k
            continue
        d = np.abs(a64 - b64)
        floor = 1e-300 + 1e-12 * max(float(np.max(np.abs(a64))), float(np.max(np.abs(b64))))
        rel = float(np.max(d / np.maximum(np.maximum(np.abs(a64), np.abs(b64)), floor)))
        if rel > out["max_rel"] or out["worst"] is None:
            out["max_abs"], out["max_rel"], out["worst"] = float(d.max()), rel, k
    out["max_abs"], out["max_rel"] = _finite(out["max_abs"]), _finite(out["max_rel"])
    return out


_ENV_KEYS = ("python", "numpy", "jax", "jaxlib", "scipy", "h5py", "hcd_priya")


def recorded_environment(env):
    """Whether ``env`` is a recorded training environment: a dict with exactly the keys ``environment()`` records and
    no None value. Anything else (None, ``{}``, a partial or foreign record) counts as not recorded, so the strict
    rule applies."""
    return isinstance(env, dict) and set(env) == set(_ENV_KEYS) and all(v is not None for v in env.values())


def classify(diff, *, training_environment=None, current_environment=None):
    """``(outcome, sub_reason)`` of a tensor comparison (``tensor_differences``) under v3 2.3 and v3.1 E2:

    * at least one tensor, every tensor bitwise equal: PASS;
    * otherwise within ``DRIFT_BOUND`` relative, a RECORDED training environment (``recorded_environment``) that
      differs from the current one: UNAVAILABLE (``env-drift``);
    * within the bound with no recorded training environment: MISMATCH (``strict-env``);
    * anything else (no tensor at all, beyond the bound, another structure, or the same recorded environment):
      MISMATCH (``content``).
    No override exists."""
    if diff["n"] > 0 and diff["same_structure"] and diff["n_bitwise"] == diff["n"]:
        return PASS, "bitwise"
    within = diff["n"] > 0 and diff["same_structure"] and diff["max_rel"] is not None and \
        diff["max_rel"] <= DRIFT_BOUND
    recorded = recorded_environment(training_environment)
    if within and recorded and training_environment != current_environment:
        return UNAVAILABLE, "env-drift"
    if within and not recorded:
        return MISMATCH, "strict-env"
    return MISMATCH, "content"


class ContentCheck(dict):
    """The result of a content check: ``{"rules", "environment", "components": [...], "passed"}`` (a dict, recorded
    as is in provenance)."""

    @property
    def passed(self):
        return self["passed"]

    def outcomes(self):
        return {c["component"]: c["outcome"] for c in self["components"]}


def _result(components, env, kind):
    passed = all(c["outcome"] == PASS for c in components)
    return ContentCheck(kind=kind, rules=dict(RULES), environment=env, components=components, passed=passed)


# ------------------------------------------------------------------------------------- content: the components
def _check_inputs(metas, norms, wcm):
    if not metas or len(norms) != len(metas):
        raise EligibilityError(f"the content checks take one normaliser per member and at least one member "
                               f"({len(metas)} metas, {len(norms)} normalisers)")
    if wcm is None:
        raise EligibilityError("the content checks need the bundle's w_c map")


def _load(cache, cache_path):
    if cache is not None:
        return cache
    from .data import load_cache
    return load_cache(cache_path)


def _recipe_split(metas):
    r0 = metas[0].get("recipe") or {}
    return {k: r0.get(k) for k in ("val_seed", "val_frac", "n_train", "n_val")}


def split_component(d, metas, rule, held_out_sims):
    """The split recomputed from the cache's own rows (no hard-coded simulation count): ``(component record, train
    rows, val rows, training simulations)``; the training rows are None when the split cannot be recomputed."""
    from . import data as D
    sim = np.asarray(d["sim_name"]).astype(str)
    all_sims = sorted(set(sim.tolist()))
    rs = _recipe_split(metas)
    detail = {"n_rows": int(sim.size), "n_sims_in_cache": len(all_sims), "held_out_sims": list(held_out_sims),
              "recorded": rs}
    if any({k: (m.get("recipe") or {}).get(k) for k in rs} != rs for m in metas):
        return component("split", rule, MISMATCH, "content", detail=dict(detail, why="members record different "
                                                                                     "splits")), None, None, None
    if rule == RULE_UNVERIFIABLE:
        train_sims = sorted(set(all_sims) - set(held_out_sims))
        return component("split", rule, UNAVAILABLE, "unverifiable", detail=dict(
            detail, why="an unverifiable split rule: the split is not recomputed")), None, None, train_sims
    missing = sorted(set(held_out_sims) - set(all_sims))
    if missing:
        return component("split", rule, MISMATCH, "content", detail=dict(detail, why=f"held-out simulations not in "
                                                                                    f"the cache: {missing}")), \
            None, None, None
    if not (_int(rs["val_seed"]) and isinstance(rs["val_frac"], float)):
        return component("split", rule, MISMATCH, "content", detail=dict(detail, why="the recipe does not state "
                                                                                    "val_seed and val_frac")), \
            None, None, None
    if rule == RULE_MULTI_HOLDOUT:
        tr, va, held = D.multi_holdout_row_split(sim, held_out_sims, val_seed=rs["val_seed"], val_frac=rs["val_frac"])
    elif rule == RULE_PRODUCTION_SPLIT and len(held_out_sims) <= 1:
        tr, va, held = D.production_row_split(sim, val_seed=rs["val_seed"], val_frac=rs["val_frac"],
                                              holdout_sim=held_out_sims[0] if held_out_sims else None)
    else:
        return component("split", rule, MISMATCH, "content", detail=dict(detail, why=f"rule {rule!r} with "
                                                                                    f"{len(held_out_sims)} held out")), \
            None, None, None
    train_sims = sorted(set(sim[tr].tolist()))
    want_sims = sorted(set(all_sims) - set(held_out_sims))
    detail.update(n_train=int(tr.size), n_val=int(va.size), n_held=int(held.size), train_idx_sha256=index_sha256(tr),
                  val_idx_sha256=index_sha256(va), train_sims_sha256=sims_sha256(train_sims),
                  n_train_sims=len(train_sims))
    why = []
    if not (_int(rs["n_train"]) and _int(rs["n_val"])):
        why.append("the recipe does not state n_train and n_val")
    elif (rs["n_train"], rs["n_val"]) != (int(tr.size), int(va.size)):
        why.append(f"recomputed sizes {tr.size}/{va.size}, the recipe states {rs['n_train']}/{rs['n_val']}")
    if train_sims != want_sims:
        why.append(f"the training rows cover {len(train_sims)} simulations, the declared training set "
                   f"{len(want_sims)}")
    if why:
        return component("split", rule, MISMATCH, "content", detail=dict(detail, why="; ".join(why))), None, None, None
    return component("split", rule, PASS, "recomputed", detail=detail), tr, va, train_sims


def normaliser_component(d, norms, train_idx, *, training_environment, current_environment):
    """Every member's stored normaliser against ``train._fit_norm`` on the recomputed training rows (``norm_fingerprint/
    1``: all tensors, bitwise first)."""
    from . import safe_io as S
    from .train import _fit_norm                                     # imported here only (pickle, optax)
    ref = S.norm_tensors(_fit_norm(d, train_idx))
    worst, per = None, []
    for i, n in enumerate(norms):
        diff = tensor_differences(S.norm_tensors(n), ref)
        oc = classify(diff, training_environment=training_environment, current_environment=current_environment)
        per.append({"member": i, "outcome": oc[0], "sub_reason": oc[1], "n_bitwise": diff["n_bitwise"], "n": diff["n"],
                    "max_rel": diff["max_rel"]})
        rank = {PASS: 0, UNAVAILABLE: 1, MISMATCH: 2}[oc[0]]
        if worst is None or rank > worst[0]:
            worst = (rank, oc, diff)
    _, (outcome, sub), diff = worst
    return component("normaliser", RULE_NORM, outcome, sub, diff=diff,
                     detail={"members": per, "recomputed_tensors_sha256": tensors_sha256(ref)})


_WC_TENSORS = (("delta_c", "coeffs"), ("delta_c", "resid_std"), ("delta_c", "resid_max"), ("delta_c", "mean"),
               ("xbar", "coeffs"))


def wc_map_component(d, wcm, train_sims, *, cache_identity, cache_name, subset, training_environment,
                     current_environment, declared=None):
    """The bundle's w_c map ``wcm`` (a parsed ``wc_map.WcMap``) against the map refitted on ``train_sims``
    (``wc_map.refit`` for the full cache, ``wc_map.refit_subset`` otherwise). Compared (v3.1 E1, dossier section 6):
    the WHOLE payload (the coefficient and residual arrays classified as tensors, every other field exactly), and the
    record's whole input entry (role, kind, sha256, size, finder ids, info: the cache's file name ``cache_name`` and
    lineage-record digest) but its eligibility flag, the settings, finder_ids, product (its payload digest follows from
    the payload), kind, kind_version and the optional validation entry (presence included). Excluded: code,
    created_utc, run_settings_sha256 and the eligibility fields.
    The simulation set comes from the recomputation; ``declared`` (a research bundle's ``provenance.research.wc_map``)
    must name the same set."""
    from .. import lineage as L
    from . import wc_map as W
    rule = RULE_WCMAP_SUBSET if subset else RULE_WCMAP
    try:
        doc = W.refit_subset(d, train_sims) if subset else W.refit(d)
    except W.WcMapError as e:
        return component("wc_map", rule, MISMATCH, "content", detail={"why": f"refit refused: {e}"})
    ref = W.make_record(doc, d["lineage"], cache_identity, code={"commit": None, "dirty": None}, cache_name=cache_name)
    body = {k: v for k, v in wcm.doc.items() if k != "lineage"}
    rec = wcm.record
    why = []
    tensors_s, tensors_r = {}, {}
    for sec in ("delta_c", "xbar"):
        for k in sorted(set(body.get(sec, {})) | set(doc[sec])):
            if (sec, k) in _WC_TENSORS:
                tensors_s[f"{sec}.{k}"] = np.asarray(body[sec].get(k), dtype=np.float64)
                tensors_r[f"{sec}.{k}"] = np.asarray(doc[sec][k], dtype=np.float64)
            elif L.canonical_json(body[sec].get(k)) != L.canonical_json(doc[sec].get(k)):
                why.append(f"payload {sec}.{k}")
    for k in ("format", "class_order"):
        if body.get(k) != doc[k]:
            why.append(f"payload {k}")
    i0, r0 = rec["inputs"][0], ref["inputs"][0]
    for k in sorted((set(i0) | set(r0)) - {"release_eligible"}):
        if L.canonical_json(i0.get(k)) != L.canonical_json(r0.get(k)):
            why.append(f"record inputs[0].{k}")
    for k in ("settings", "finder_ids", "kind", "kind_version", "validation"):
        if (k in rec) != (k in ref) or L.canonical_json(rec.get(k)) != L.canonical_json(ref.get(k)):
            why.append(f"record {k}")
    pr, pf = dict(rec.get("product") or {}), dict(ref["product"])
    pr.pop("payload_sha256", None)
    pf.pop("payload_sha256", None)
    if L.canonical_json(pr) != L.canonical_json(pf):
        why.append("record product")
    fitted = sims_sha256(train_sims)
    if declared is not None:
        if declared.get("fitted_on_sims_sha256") != fitted:
            why.append("provenance.research.wc_map.fitted_on_sims_sha256 is not the recomputed training set")
        if declared.get("n_sims") != doc["delta_c"]["n_sims"]:
            why.append("provenance.research.wc_map.n_sims is not the recomputed map's")
        if declared.get("rule") != rule:
            why.append(f"provenance.research.wc_map.rule is not {rule}")
    diff = tensor_differences(tensors_s, tensors_r)
    detail = {"fitted_on_sims_sha256": fitted, "n_sims": doc["delta_c"]["n_sims"],
              "refit_payload_sha256": W.payload_sha256(doc), "bundle_wc_map_sha256": wcm.sha256}
    if why:
        return component("wc_map", rule, MISMATCH, "content", diff=diff, detail=dict(detail, why="; ".join(why)))
    outcome, sub = classify(diff, training_environment=training_environment, current_environment=current_environment)
    return component("wc_map", rule, outcome, sub, diff=diff, detail=detail)


def _cache_component(d, metas, expected_cache_sha256, wcm):
    got = d.get("cache_sha256")
    why = []
    if got != expected_cache_sha256:
        why.append("the cache file does not hash to the expected digest")
    if any(m.get("cache_sha256") != got for m in metas):
        why.append("a member was trained on another cache")
    if wcm is not None and wcm.cache_sha256 != got:
        why.append("the w_c map was fitted to another cache")
    detail = {"cache_sha256": got}
    if why:
        return component("cache", RULE_CACHE_TIE, MISMATCH, "content", detail=dict(detail, why="; ".join(why)))
    return component("cache", RULE_CACHE_TIE, PASS, "tied", detail=detail)


def _not_run(name, rule, why):
    return component(name, rule, UNAVAILABLE, "not-run", detail={"why": why})


def production_content_check(metas, norms, wcm, *, cache_path=None, cache=None, expected_cache_sha256,
                             cache_size=None, training_environment=None):
    """The production content rules (v2 3.4, v3 2.3, v3.1 E1): the cache tie first; the production split
    (``production_row_split/1``, every simulation of THIS cache, n_train / n_val required); every member's normaliser
    (``norm_fingerprint/1``); the w_c map refitted on the full cache (``wcmap_refit/1``). ``cache`` (a dict from the
    strict ``data.load_cache`` of ``cache_path``) or ``cache_path`` alone (loaded here); ``cache_path`` is required
    either way. Returns a ``ContentCheck``; production needs ``passed``."""
    _check_inputs(metas, norms, wcm)
    env = environment()
    d = _load(cache, cache_path)
    ident = _identity(d, cache_path, cache_size)
    name = os.path.basename(str(cache_path))
    comps = [_cache_component(d, metas, expected_cache_sha256, wcm)]
    if comps[0]["outcome"] != PASS:
        return _result(comps + [_not_run(n, r, "the cache tie failed") for n, r in
                                (("split", RULE_PRODUCTION_SPLIT), ("normaliser", RULE_NORM), ("wc_map", RULE_WCMAP))],
                       env, "production")
    sc, tr, _va, train_sims = split_component(d, metas, RULE_PRODUCTION_SPLIT, [])
    comps.append(sc)
    if tr is None:
        comps.append(_not_run("normaliser", RULE_NORM, "the split was not recomputed"))
    else:
        comps.append(normaliser_component(d, norms, tr, training_environment=training_environment,
                                          current_environment=env))
    all_sims = sorted(set(np.asarray(d["sim_name"]).astype(str).tolist()))
    comps.append(wc_map_component(d, wcm, all_sims, cache_identity=ident, cache_name=name, subset=False,
                                  training_environment=training_environment, current_environment=env))
    return _result(comps, env, "production")


def research_content_check(metas, norms, wcm, rs, *, cache_path=None, cache=None, expected_cache_sha256,
                           cache_size=None, training_environment=None, check_declared_map=True):
    """The research content rules for the declared split ``rs`` (a ``research/1`` record, or the part of it the
    research export has built: ``split_rule``, ``split`` and, with ``check_declared_map``, ``wc_map``): the cache tie;
    the split under its rule; the normaliser; the w_c map refitted on the training simulations only
    (``wcmap_refit_subset/1``; the full refit when nothing is held out). Under ``unverifiable`` the split and the
    normaliser are recorded UNAVAILABLE and the map is still refitted on the cache minus the declared held-out
    simulations. MISMATCH refuses a research export; UNAVAILABLE is allowed and recorded."""
    _check_inputs(metas, norms, wcm)
    env = environment()
    d = _load(cache, cache_path)
    ident = _identity(d, cache_path, cache_size)
    name = os.path.basename(str(cache_path))
    rule, held = rs["split_rule"], list(rs["split"]["held_out_sims"])
    comps = [_cache_component(d, metas, expected_cache_sha256, wcm)]
    if comps[0]["outcome"] != PASS:
        return _result(comps + [_not_run(n, r, "the cache tie failed") for n, r in
                                (("split", rule), ("normaliser", RULE_NORM), ("wc_map", RULE_WCMAP_SUBSET))],
                       env, "research")
    sc, tr, _va, train_sims = split_component(d, metas, rule, held)
    comps.append(sc)
    if tr is None:
        why = "an unverifiable split rule" if rule == RULE_UNVERIFIABLE else "the split was not recomputed"
        comps.append(_not_run("normaliser", RULE_NORM, why))
    else:
        comps.append(normaliser_component(d, norms, tr, training_environment=training_environment,
                                          current_environment=env))
    if train_sims is None:
        comps.append(_not_run("wc_map", RULE_WCMAP_SUBSET, "the training simulations are not known"))
    else:
        comps.append(wc_map_component(d, wcm, train_sims, cache_identity=ident, cache_name=name, subset=bool(held),
                                      training_environment=training_environment, current_environment=env,
                                      declared=rs.get("wc_map") if check_declared_map else None))
    return _result(comps, env, "research")


def _identity(d, cache_path, cache_size):
    if not cache_path:
        raise EligibilityError("the content checks need the cache_path (its file name is part of the w_c map's record)")
    size = cache_size if cache_size is not None else os.stat(cache_path).st_size
    return {"sha256": d["cache_sha256"], "size": int(size)}


def research_outcome_ok(check):
    """A research export proceeds unless a component is MISMATCH (UNAVAILABLE is allowed and recorded)."""
    return all(c["outcome"] != MISMATCH for c in check["components"])


def outcomes_differ(a, b):
    """Whether two content checks give different outcomes per component (the read-back check must repeat the
    pre-write one)."""
    return a.outcomes() != b.outcomes()

