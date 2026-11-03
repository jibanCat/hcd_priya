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

A **research** bundle is marked at the bundle level (``provenance.research``, schema ``research/1``:
``check_research_marking``). Every production loader refuses it whatever its split; it loads only through
``safe_h5.load_research`` / ``predictor.load_research_bundle``, and ``research_member_reasons`` decides which members it
admits (one declared split per bundle).

Reasons are strings ``"[CODE] text"`` with a stable machine code (``reason_code``); comparisons use the codes, never
the wording. Nothing here resists deliberately forged metadata with a recomputed digest: that defence is the published
digest pinned after a recorded acceptance. This module imports neither the trainer nor the cache reader."""
from __future__ import annotations

import hashlib
import json
import math
import re

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
    under ``unverifiable`` a member with any reserved key is admitted as declared (flagged; nothing is recomputed).
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

