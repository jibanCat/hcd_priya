"""The public tutorial notebook, notebooks/tutorial/: its builder, its helper cells and its stored files.

The notebook is written by ``notebooks/tutorial/_build_tutorial.py``, which holds every cell as source. The helper
functions the notebook needs are notebook cells; these tests take those cells from the builder
(``helper_source(cell_id)``) and run them in a namespace of their own, so what is tested is the code a reader
runs. Every number in these tests is synthetic.
"""

from __future__ import annotations

import ast
import functools
import hashlib
import http.server
import importlib.util
import json
import pathlib
import re
import sys
import threading

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
TUTORIAL = ROOT / "notebooks" / "tutorial"
BUILDER = TUTORIAL / "_build_tutorial.py"
NOTEBOOK = TUTORIAL / "hcd_priya_tutorial.ipynb"
MANIFEST = TUTORIAL / "assets" / "ASSETS.json"

TAG_URL = "https://raw.githubusercontent.com/jibanCat/hcd_priya/v1.0.0/"
RELEASE_URL = "https://github.com/jibanCat/hcd_priya/releases/download/v1.0.0/"
COMMIT_URL = re.compile(r"https://raw\.githubusercontent\.com/jibanCat/hcd_priya/[0-9a-f]{40}/")
# strings of a private machine that no public text holds; joined here so that this file does not hold them either
PRIVATE_PATHS = tuple("/" + name + "/" for name in ("home", "nfs", "tmp")) + ("tur" + "bo",)


def _builder():
    spec = importlib.util.spec_from_file_location("_build_tutorial", BUILDER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _namespace(*cell_ids):
    """The helper cells named, run in order in one namespace, as the notebook runs them."""
    builder = _builder()
    ns = {"__name__": "tutorial_cells"}
    for cell_id in cell_ids:
        exec(compile(builder.helper_source(cell_id), f"<cell {cell_id}>", "exec"), ns)
    return ns


def _notebook_cells(path):
    doc = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    return [(c["cell_type"], c["id"], "".join(c["source"])) for c in doc["cells"]]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ------------------------------------------------------------------------------------------ the builder
def test_builder_output_equals_the_committed_cells(tmp_path):
    builder = _builder()
    out = tmp_path / "built.ipynb"
    builder.build(out)
    built = _notebook_cells(out)
    assert built == [(kind, cell_id, source) for kind, cell_id, source in builder.CELLS]
    assert built == _notebook_cells(
        NOTEBOOK
    ), "run notebooks/tutorial/_build_tutorial.py and execute the notebook"
    ids = [cell_id for _, cell_id, _ in built]
    assert len(set(ids)) == len(ids)
    assert all(re.fullmatch(r"[a-z0-9-]{1,64}", cell_id) for cell_id in ids)
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert (doc["nbformat"], doc["nbformat_minor"]) == (4, 5)
    assert all(
        c["outputs"] == [] and c["execution_count"] is None
        for c in doc["cells"]
        if c["cell_type"] == "code"
    )


def test_helper_source_names_only_code_cells():
    builder = _builder()
    assert (
        builder.helper_source("helpers-assets")
        == dict((i, s) for _, i, s in builder.CELLS)["helpers-assets"]
    )
    with pytest.raises(KeyError):
        builder.helper_source("no-such-cell")
    with pytest.raises(KeyError):
        builder.helper_source("title")  # a text cell


def test_every_code_cell_parses():
    for kind, cell_id, source in _builder().CELLS:
        if kind == "code":
            try:
                ast.parse(source)
            except SyntaxError as exc:  # pragma: no cover - the message is the point
                raise AssertionError(f"cell {cell_id} does not parse: {exc}") from None


def test_no_private_attribute_no_private_path_no_em_dash():
    texts = {f"cell {cell_id}": source for _, cell_id, source in _builder().CELLS}
    for path in (BUILDER, MANIFEST, TUTORIAL / "README.md"):
        texts[path.name] = path.read_text(encoding="utf-8")
    for name, text in texts.items():
        assert "—" not in text, f"{name} holds an em-dash"
        for private in PRIVATE_PATHS:
            assert private not in text, f"{name} holds {private!r}"
    for kind, cell_id, source in _builder().CELLS:
        if kind != "code":
            continue
        for node in ast.walk(ast.parse(source)):
            names = []
            if isinstance(node, ast.Attribute):
                names = [node.attr]
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name.split(".")[-1] for a in node.names]
            for n in names:
                private = n.startswith("_") and not (n.startswith("__") and n.endswith("__"))
                assert not private, f"cell {cell_id} uses the private name {n}"


# ------------------------------------------------------------------------------------------ the stored files
def test_every_asset_url_is_immutable():
    records = json.loads(MANIFEST.read_text(encoding="utf-8"))["files"]
    assert records
    paths = [r["path"] for r in records]
    assert len(set(paths)) == len(paths)
    for r in records:
        assert set(r) == {"path", "sha256", "bytes", "url", "what", "source"}, r["path"]
        assert (
            re.fullmatch(r"[0-9a-f]{64}", r["sha256"])
            and isinstance(r["bytes"], int)
            and r["bytes"] > 0
        )
        url = r["url"]
        assert "/main/" not in url and "/master/" not in url and "/HEAD/" not in url, url
        immutable = url.startswith(TAG_URL) or url.startswith(RELEASE_URL) or COMMIT_URL.match(url)
        assert (
            immutable
        ), f"{r['path']}: {url} is not the tag v1.0.0, the release download or a commit"
        if not url.startswith(RELEASE_URL):
            assert url.endswith("/" + r["path"]), f"{r['path']}: the URL names another path"
            local = (
                ROOT / r["path"]
            )  # a file of this repository: the manifest pins the bytes that are here
            assert local.is_file(), f"{r['path']} is not in the repository"
            data = local.read_bytes()
            assert (_sha256(data), len(data)) == (r["sha256"], r["bytes"]), r["path"]


def test_the_assets_cell_holds_the_manifest():
    records = json.loads(MANIFEST.read_text(encoding="utf-8"))["files"]
    ns = _namespace("helpers-assets")
    assert ns["ASSETS"] == {r["path"]: {"sha256": r["sha256"], "url": r["url"]} for r in records}


# ------------------------------------------------------------------------------------------ tutorial_file
class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


@pytest.fixture
def served(tmp_path):
    """A local web server on a folder of synthetic files: (folder, base URL)."""
    folder = tmp_path / "served"
    folder.mkdir()
    server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(_Quiet, directory=str(folder))
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield folder, f"http://127.0.0.1:{server.server_address[1]}/"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.fixture
def cells(tmp_path, monkeypatch):
    """The asset cell with an empty table, a clone folder and a cache folder of its own."""
    ns = _namespace("helpers-assets")
    clone, cache = tmp_path / "clone", tmp_path / "cache"
    clone.mkdir()
    monkeypatch.setenv("HCD_PRIYA_TUTORIAL_CACHE", str(cache))
    ns["ASSETS"] = {}
    ns["LOCAL_ROOTS"] = (clone,)
    return ns, clone, cache


def _files_under(folder):
    return (
        sorted(str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file())
        if folder.exists()
        else []
    )


def test_tutorial_file_uses_a_local_copy_only_with_the_pinned_digest(cells, served):
    ns, clone, cache = cells
    folder, base = served
    good = b"synthetic table, the pinned bytes\n" * 7
    (folder / "data").mkdir()
    (folder / "data" / "table.txt").write_bytes(good)
    ns["ASSETS"]["data/table.txt"] = {"sha256": _sha256(good), "url": base + "data/table.txt"}
    (clone / "data").mkdir()
    # a local copy with the pinned digest is used, and nothing is downloaded
    (clone / "data" / "table.txt").write_bytes(good)
    assert ns["tutorial_file"]("data/table.txt") == clone / "data" / "table.txt"
    assert _files_under(cache) == []
    # a changed local copy is not used: the pinned bytes are downloaded, and the same bytes are read
    (clone / "data" / "table.txt").write_bytes(b"a changed local copy\n")
    got = ns["tutorial_file"]("data/table.txt")
    assert got == cache / "data" / "table.txt" and got.read_bytes() == good
    assert _files_under(cache) == ["data/table.txt"]
    # the cached copy is used the next time, also with the server's file gone
    (folder / "data" / "table.txt").unlink()
    assert ns["tutorial_file"]("data/table.txt") == got
    # a file that the table does not name is refused by name
    with pytest.raises(RuntimeError, match="data/other.txt"):
        ns["tutorial_file"]("data/other.txt")


def test_tutorial_file_refuses_a_wrong_digest_and_leaves_no_partial_file(cells, served):
    ns, clone, cache = cells
    folder, base = served
    good = b"synthetic table, the pinned bytes\n" * 7
    truncated = good[: len(good) // 2]
    (folder / "table.txt").write_bytes(truncated)
    ns["ASSETS"]["table.txt"] = {"sha256": _sha256(good), "url": base + "table.txt"}
    with pytest.raises(RuntimeError) as err:
        ns["tutorial_file"]("table.txt")
    message = str(err.value)
    assert _sha256(good) in message and _sha256(truncated) in message and "table.txt" in message
    assert _files_under(cache) == [], "a refused download left a file in the cache"
    # a cached file that was changed afterwards is not used either: it is replaced by the pinned bytes
    (folder / "table.txt").write_bytes(good)
    assert ns["tutorial_file"]("table.txt").read_bytes() == good
    (cache / "table.txt").write_bytes(truncated)
    assert ns["tutorial_file"]("table.txt").read_bytes() == good
    assert _files_under(cache) == ["table.txt"]


def test_tutorial_file_names_the_file_the_url_and_the_cache_when_the_download_fails(cells, served):
    ns, clone, cache = cells
    folder, base = served
    ns["ASSETS"]["missing.txt"] = {"sha256": _sha256(b"never served"), "url": base + "missing.txt"}
    with pytest.raises(RuntimeError) as err:
        ns["tutorial_file"]("missing.txt")
    message = str(err.value)
    assert "missing.txt" in message and base + "missing.txt" in message and str(cache) in message
    assert _files_under(cache) == []
    # no server at all
    ns["ASSETS"]["unreachable.txt"] = {
        "sha256": _sha256(b"x"),
        "url": "http://127.0.0.1:9/unreachable.txt",
    }
    with pytest.raises(RuntimeError) as err:
        ns["tutorial_file"]("unreachable.txt")
    assert "http://127.0.0.1:9/unreachable.txt" in str(err.value) and str(cache) in str(err.value)
    assert _files_under(cache) == []


# ------------------------------------------------------------------------------------------ the incidence helper
def _figure_script(name):
    scripts = ROOT / "scripts" / "figures"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location(name, scripts / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(scripts))
    return module


def test_cumulative_incidence_equals_the_figure_script():
    np = pytest.importorskip("numpy")
    ns = _namespace("helpers-incidence")
    script = _figure_script("fig_dndx_vs_ho21")
    log_nhi = 17.1 + 0.2 * np.arange(30)  # thirty bin centres, as a prediction has
    rng = np.random.default_rng(7)
    cddf = 10.0 ** (-17.0 - 1.5 * (log_nhi - 17.0)) * rng.uniform(
        0.5, 2.0, size=(4, 30)
    )  # synthetic, four rows
    cddf[1, 25:] = np.nan  # bins without a value are left out
    cddf[2, :] = np.nan  # a row with no value at all
    for threshold in (17.5, 17.2, 19.0, 20.37):
        ours = ns["cumulative_incidence"](cddf, log_nhi, threshold)
        assert ours.shape == (4,)
        assert np.array_equal(
            ours, script.cumulative_dndx(cddf, log_nhi, threshold), equal_nan=True
        )
    assert np.isnan(ns["cumulative_incidence"](cddf, log_nhi)[2])
    assert np.array_equal(
        ns["cumulative_incidence"](cddf, log_nhi),
        script.cumulative_dndx(cddf, log_nhi),
        equal_nan=True,
    )
    # the threshold inside a bin counts that bin from the threshold to its upper edge
    one = np.zeros(30)
    one[2] = 1.0  # the bin centred on 17.5, from 17.4 to 17.6
    assert ns["cumulative_incidence"](one, log_nhi, 17.5) == pytest.approx(
        10.0**17.6 - 10.0**17.5, rel=1e-12
    )
    with pytest.raises(ValueError):
        ns["cumulative_incidence"](cddf, log_nhi**1.01)


def test_absorption_distance_equals_the_figure_scripts():
    np = pytest.importorskip("numpy")
    ns = _namespace("helpers-incidence")
    sys.path.insert(0, str(ROOT / "scripts" / "figures"))
    try:
        common = _figure_script("figures_common")
    finally:
        sys.path.remove(str(ROOT / "scripts" / "figures"))
    for z1, z2, omega_m in ((2.0, 2.5, 0.3), (3.0, 4.0, 0.279), (1.5, 5.4, 0.25)):
        assert ns["absorption_distance"](z1, z2, omega_m) == pytest.approx(
            common.absorption_distance(z1, z2, omega_m), rel=1e-6
        )
    assert np.isclose(ns["absorption_distance"](2.0, 2.0, 0.3), 0.0)


# ------------------------------------------------------------------------------------------ the committed notebook
def test_the_committed_notebook_is_executed_in_order_without_an_error_or_a_private_path():
    raw = NOTEBOOK.read_text(encoding="utf-8")
    for private in PRIVATE_PATHS:
        assert private not in raw, f"the committed notebook holds {private!r}"
    assert "—" not in raw
    code = [c for c in json.loads(raw)["cells"] if c["cell_type"] == "code"]
    assert [c["execution_count"] for c in code] == list(
        range(1, len(code) + 1)
    ), "not executed top to bottom"
    for c in code:
        for output in c["outputs"]:
            assert output["output_type"] != "error", f"cell {c['id']} ended in an error"
            assert (
                output.get("name") != "stderr"
            ), f"cell {c['id']} wrote to stderr: {''.join(output['text'])[:200]}"


# ------------------------------------------------------------------------------------------ the posterior helper
ZS = (2.6, 3.0)  # two snapshot redshifts that the synthetic model serves under LF+MF


@pytest.fixture(scope="module")
def stand_in(tmp_path_factory):
    """A synthetic model with untrained networks: a stand-in for the code paths, never for a number."""
    pytest.importorskip("numpy")
    import hcd_analysis.predictor as hp
    from tests.synthetic_release import make_release

    info = make_release(tmp_path_factory.mktemp("tutorial") / "release", incidence=True)
    return hp.load_release(str(info["root"]), info["sha"])


def _made_up_data(emu, theta, *, p1d=None):
    """A data set at ZS: bins between the model's modes, a covariance that differs from its transpose by rounding
    (as a summed or inverted matrix does), and a factor on the model vector that is not one."""
    import numpy as np
    from hcd_analysis import likelihood as L

    nine = {name: value for name, value in theta.items() if name not in L.MEAN_FLUX_NAMES}
    pred = emu.predict_many(nine, z=list(ZS), tau0_amp=1.0, dtau0=0.0)
    rows, blocks = [], []
    for i, m in enumerate((4, 6)):
        k = np.asarray(pred.k[i])
        rows.append(np.linspace(k[len(k) // 4], k[len(k) // 2], m + 2)[1:-1])
        j = np.arange(m)
        blocks.append((0.04 + 0.05 * i) * 0.3 ** np.abs(j[:, None] - j[None, :]))
    blocks[0][0, 1] *= 1.0 + 4e-14  # the transpose keeps the other value
    n = sum(r.size for r in rows)
    return L.P1DData.from_blocks(
        z=ZS,
        k=rows,
        p1d=[np.linspace(3.0, 2.0, r.size) for r in rows] if p1d is None else p1d,
        cov_blocks=blocks,
        model_factor=np.split(np.linspace(0.9, 1.15, n), np.cumsum([r.size for r in rows])[:-1]),
    )


def _points(ns):
    truth = dict(ns["TRUTH"])
    return (
        truth,
        dict(truth, ns=0.87, Ap=2.3e-9),
        dict(truth, tau0_amp=1.12, ns=0.96),
    )


def test_manual_model_vector_equals_the_public_likelihood(stand_in):
    import numpy as np
    from hcd_analysis import likelihood as L

    ns = _namespace("helpers-posterior")
    data = _made_up_data(stand_in, ns["TRUTH"])
    assert data.cov[0, 1] != data.cov[1, 0] and not np.allclose(data.model_factor, 1.0)
    base = L.GaussianP1DLikelihood(stand_in, data)
    compared = 0
    for r_lls in (0.0, 0.5, 1.0, 2.0):
        for rel in (
            {"LLS": r_lls, "subDLA": 1.0, "DLA": 1.0},
            {"LLS": r_lls, "subDLA": 0.4, "DLA": 1.7},
        ):
            public = L.GaussianP1DLikelihood(stand_in, data, hcd_relative_to_priya=rel)
            for theta in _points(ns):
                manual = ns["model_vector_free_hcd"](base, theta, rel)
                want = public.model_vector(theta)
                assert manual.shape == want.shape == (data.n,)
                assert np.allclose(manual, want, rtol=1e-12, atol=0), (rel, theta)
                compared += 1
    assert compared == 24
    # the amplitude matters: the vectors at two LLS amplitudes differ (the equality above is not that of two constants)
    a = ns["model_vector_free_hcd"](base, ns["TRUTH"], {"LLS": 0.0, "subDLA": 1.0, "DLA": 1.0})
    b = ns["model_vector_free_hcd"](base, ns["TRUTH"], {"LLS": 2.0, "subDLA": 1.0, "DLA": 1.0})
    assert not np.allclose(a, b, rtol=1e-6, atol=0)


def test_manual_loglike_equals_the_public_likelihood(stand_in):
    import numpy as np
    from hcd_analysis import likelihood as L

    ns = _namespace("helpers-posterior")
    data = _made_up_data(stand_in, ns["TRUTH"])
    base = L.GaussianP1DLikelihood(stand_in, data)
    values = []
    for r_lls in (0.0, 0.5, 1.0, 2.0):
        rel = {"LLS": r_lls, "subDLA": 1.0, "DLA": 1.0}
        public = L.GaussianP1DLikelihood(stand_in, data, hcd_relative_to_priya=rel)
        for theta in _points(ns):
            manual = ns["log_likelihood_free_hcd"](base, theta, rel)
            want = public.loglike(theta)
            assert np.isfinite(want) and np.isclose(manual, want, rtol=1e-10, atol=0), (rel, theta)
            values.append(want)
    assert len(set(np.round(values, 6))) > 6  # the twelve values are not one constant
    # a mock made by the public likelihood at a point has chi-square zero there under the manual assembly
    rel = {"LLS": 1.0, "subDLA": 1.0, "DLA": 1.0}
    mock = _made_up_data(stand_in, ns["TRUTH"], p1d=None)
    vector = L.GaussianP1DLikelihood(stand_in, mock).model_vector(ns["TRUTH"])
    mock = _made_up_data(stand_in, ns["TRUTH"], p1d=[vector[s] for s in mock.slices()])
    like = L.GaussianP1DLikelihood(stand_in, mock)
    assert ns["chi_square_free_hcd"](like, ns["TRUTH"], rel) == pytest.approx(0.0, abs=1e-18)
    # and not elsewhere: without the LLS sightlines, and at another mean flux
    for theta, other in ((ns["TRUTH"], dict(rel, LLS=0.0)), (dict(ns["TRUTH"], tau0_amp=1.1), rel)):
        elsewhere = ns["chi_square_free_hcd"](like, theta, other)
        assert elsewhere > 1e-6, (elsewhere, theta, other)
        assert elsewhere == pytest.approx(
            L.GaussianP1DLikelihood(stand_in, mock, hcd_relative_to_priya=other).chi2(theta),
            rel=1e-9,
        )


def test_log_posterior_is_minus_infinity_outside_the_prior_and_on_a_refusal(stand_in):
    import math

    import numpy as np
    from hcd_analysis import likelihood as L

    ns = _namespace("helpers-posterior")
    truth = ns["TRUTH"]
    assert ns["FREE"] == ("ns", "Ap", "tau0_amp", "r_LLS") and ns["REL_TRUTH"] == {
        "LLS": 1.0,
        "subDLA": 1.0,
        "DLA": 1.0,
    }
    like = L.GaussianP1DLikelihood(stand_in, _made_up_data(stand_in, truth))
    ns["LIKE"] = like
    ns["PRIOR"] = {
        "ns": (0.8, 1.05),
        "Ap": (1.2e-9, 2.6e-9),
        "tau0_amp": (0.7, 1.3),
        "r_LLS": (0.0, 60.0),
    }
    x0 = np.array([truth["ns"], truth["Ap"], truth["tau0_amp"], 1.0])
    inside = ns["log_posterior"](x0)
    assert math.isfinite(inside)
    assert inside == pytest.approx(
        ns["log_likelihood_free_hcd"](like, truth, ns["REL_TRUTH"]), rel=1e-13
    )
    assert inside == pytest.approx(
        like.loglike(truth), rel=1e-10
    )  # all amplitudes at 1: the public default
    assert ns["REFUSALS"] == {}
    for i, name in enumerate(ns["FREE"]):
        low, high = ns["PRIOR"][name]
        for value in (low - 0.01 * (high - low), high + 0.01 * (high - low), math.nan):
            x = x0.copy()
            x[i] = value
            assert ns["log_posterior"](x) == -math.inf, (name, value)
        for value in (low, high):  # the edges belong to the box
            x = x0.copy()
            x[i] = value
            if name != "r_LLS":
                assert math.isfinite(ns["log_posterior"](x)), (name, value)
    assert ns["REFUSALS"] == {}, "a point outside the prior is not a refusal of the model"
    # inside the prior, a point the predictor refuses: so many LLS sightlines that the fractions sum to more than one
    x = x0.copy()
    x[3] = 55.0
    assert ns["log_posterior"](x) == -math.inf
    assert (
        list(ns["REFUSALS"]) == ["PredictorInputError"]
        and ns["REFUSALS"]["PredictorInputError"][0] == 1
    )
    assert ns["log_posterior"](x) == -math.inf and ns["REFUSALS"]["PredictorInputError"][0] == 2
    assert math.isfinite(ns["log_posterior"](x0))  # and the sampler goes on


def test_the_flat_prior_is_the_trained_box_and_the_recorded_mean_flux_range():
    import types

    import hcd_analysis.predictor as hp

    ns = _namespace("helpers-posterior")
    fake = types.SimpleNamespace(
        provenance={"lineage": {"cache": {"settings": {"alpha_range": [0.61, 1.37]}}}}
    )
    prior = ns["flat_prior"](fake)
    assert list(prior) == list(ns["FREE"])
    limits = dict(zip(hp.PARAM_NAMES, (tuple(float(v) for v in row) for row in hp.PARAM_LIMITS)))
    assert prior["ns"] == limits["ns"] and prior["Ap"] == limits["Ap"]
    assert prior["tau0_amp"] == (0.61, 1.37) and prior["r_LLS"] == (0.0, 3.0)


# ------------------------------------------------------------------------------------------ the sampler and its files
def test_metropolis_recovers_a_two_dimensional_gaussian():
    import numpy as np

    ns = _namespace("helpers-sampler")
    mean = np.array([1.0, -2.0])
    cov = np.array([[1.0, 0.6], [0.6, 2.0]])
    inverse = np.linalg.inv(cov)

    def logp(x):
        r = x - mean
        return -0.5 * float(r @ inverse @ r)

    proposal = (2.38**2 / 2.0) * cov
    chain, logps, acceptance = ns["metropolis"](
        logp, mean + 1.0, proposal, 40000, np.random.default_rng(11)
    )
    assert chain.shape == (40000, 2) and logps.shape == (40000,) and 0.2 < acceptance < 0.5
    assert np.array_equal(
        logps, [logp(x) for x in chain]
    )  # the stored value is that of the stored state
    kept = chain[2000:]
    assert np.allclose(kept.mean(axis=0), mean, atol=0.12)
    assert np.allclose(np.cov(kept.T), cov, rtol=0.2, atol=0.1)
    moves = np.any(np.diff(chain, axis=0) != 0.0, axis=1)
    assert moves.mean() == pytest.approx(
        acceptance, abs=2e-4
    )  # the acceptance is the fraction of steps that moved
    # the same seed gives the same chain; another seed another one
    again, _, _ = ns["metropolis"](logp, mean + 1.0, proposal, 500, np.random.default_rng(11))
    other, _, _ = ns["metropolis"](logp, mean + 1.0, proposal, 500, np.random.default_rng(12))
    assert np.array_equal(again, chain[:500]) and not np.array_equal(other, chain[:500])

    # where the log posterior is minus infinity (or not a number) the chain never goes, and it does not stop
    def walled(x):
        if x[0] < 0.5:
            return -np.inf
        return np.nan if x[1] > 0.0 else logp(x)

    chain, logps, acceptance = ns["metropolis"](
        walled, np.array([1.0, -2.0]), proposal, 3000, np.random.default_rng(3)
    )
    assert (
        chain[:, 0].min() >= 0.5
        and chain[:, 1].max() <= 0.0
        and np.all(np.isfinite(logps))
        and acceptance > 0.05
    )
    with pytest.raises(ValueError):
        ns["metropolis"](walled, np.array([0.0, -2.0]), proposal, 10, np.random.default_rng(3))


def test_rhat_is_near_one_for_independent_draws_and_large_for_shifted_chains():
    import numpy as np

    ns = _namespace("helpers-diagnostics")
    rng = np.random.default_rng(5)
    draws = rng.standard_normal((4, 2000, 3)) * [1.0, 10.0, 0.01] + [0.0, 5.0, -1.0]
    out = ns["split_rhat_and_ess"](draws)
    assert set(out) == {"rhat", "ess_bulk", "ess_tail"}
    assert out["rhat"].shape == out["ess_bulk"].shape == out["ess_tail"].shape == (3,)
    assert np.all(out["rhat"] < 1.01) and np.all(out["rhat"] > 0.999)
    assert np.all(out["ess_bulk"] > 0.85 * 8000) and np.all(out["ess_bulk"] < 1.15 * 8000)
    # the tail: independent draws keep about all of them (the estimate is noisier than the bulk's)
    assert np.all(out["ess_tail"] > 0.75 * 8000) and np.all(out["ess_tail"] < 1.25 * 8000)
    # the statistics do not depend on the scale or the location of a parameter
    same = ns["split_rhat_and_ess"](draws[:, :, :1] * 8.0 - 3.0)
    for key in out:
        assert same[key][0] == pytest.approx(out[key][0], rel=1e-6), key
    # one chain somewhere else: its mean off by two standard deviations
    shifted = draws.copy()
    shifted[0, :, 0] += 2.0
    out = ns["split_rhat_and_ess"](shifted)
    assert out["rhat"][0] > 1.1 and np.all(out["rhat"][1:] < 1.01)
    assert out["ess_bulk"][0] < 100
    # one chain three times as wide, the same mean: the folded half of the statistic sees it
    wide = draws.copy()
    wide[1, :, 2] = -1.0 + 3.0 * (wide[1, :, 2] + 1.0)
    assert ns["split_rhat_and_ess"](wide)["rhat"][2] > 1.05
    # a chain that drifts: its two halves differ, which the split sees with one chain alone
    drift = rng.standard_normal((1, 2000, 1)) + np.linspace(0.0, 3.0, 2000)[None, :, None]
    assert ns["split_rhat_and_ess"](drift)["rhat"][0] > 1.1
    # correlated draws: an autoregressive chain with coefficient 0.9 keeps about (1 - 0.9) / (1 + 0.9) of its draws
    ar = np.empty((4, 20000))
    ar[:, 0] = rng.standard_normal(4)
    noise = rng.standard_normal((4, 20000)) * np.sqrt(1.0 - 0.9**2)
    for t in range(1, 20000):
        ar[:, t] = 0.9 * ar[:, t - 1] + noise[:, t]
    out = ns["split_rhat_and_ess"](ar[:, :, None])
    expected = 80000 * (1.0 - 0.9) / (1.0 + 0.9)
    assert 0.75 * expected < out["ess_bulk"][0] < 1.3 * expected and out["rhat"][0] < 1.01
    # the tail of strongly correlated draws: far below the number of draws, and it is its own number
    assert 100 < out["ess_tail"][0] < 0.25 * 80000 and out["ess_tail"][0] != out["ess_bulk"][0]
    # draws whose width changes slowly (wide for 250 steps, narrow for 250, and so on) and whose sign is random: the
    # levels are uncorrelated, so the bulk keeps its draws, but the tails are visited in bursts
    width = np.where((np.arange(8000) // 250) % 2 == 0, 3.0, 1.0)
    bursts = rng.standard_normal((4, 8000, 1)) * width[None, :, None]
    out_bursts = ns["split_rhat_and_ess"](bursts)
    assert (
        out_bursts["ess_bulk"][0] > 0.6 * 32000
        and out_bursts["ess_tail"][0] < 0.25 * out_bursts["ess_bulk"][0]
    )
    with pytest.raises(ValueError):
        ns["split_rhat_and_ess"](draws[0])  # two dimensions: the chains axis is missing


def test_chain_files_round_trip_and_weights_sum_to_the_steps(tmp_path):
    import numpy as np

    ns = _namespace("helpers-chain-files")
    rng = np.random.default_rng(9)
    n_chains, n_steps, d = 3, 400, 4
    chains = np.empty((n_chains, n_steps, d))
    for c in range(
        n_chains
    ):  # a walk that stays where it is on two steps in three, as a sampler does
        x = rng.standard_normal(d) * [1.0, 1e-9, 1.0, 1.0]
        for t in range(n_steps):
            if rng.uniform() < 1.0 / 3.0:
                x = x + rng.standard_normal(d) * [0.1, 1e-10, 0.1, 0.1]
            chains[c, t] = x
    logps = -0.5 * np.sum(chains**2, axis=2)
    names = ("a", "b", "c", "r_x")
    ranges = {"a": (-5.0, 5.0), "b": (-1e-8, 1e-8), "c": (-5.0, 5.0), "r_x": (0.0, 3.0)}
    prefix = tmp_path / "out" / "synthetic_chain"
    prefix.parent.mkdir()
    ns["write_chains"](prefix, chains, logps, names, ranges)
    written = sorted(p.name for p in prefix.parent.iterdir())
    assert written == [f"synthetic_chain.{i}.txt" for i in (1, 2, 3)] + [
        "synthetic_chain.paramnames",
        "synthetic_chain.ranges",
    ]
    assert (prefix.parent / "synthetic_chain.paramnames").read_text().split() == list(names)
    assert [
        line.split() for line in (prefix.parent / "synthetic_chain.ranges").read_text().splitlines()
    ] == [[name, repr(float(low)), repr(float(high))] for name, (low, high) in ranges.items()]
    first = (prefix.parent / "synthetic_chain.1.txt").read_text().splitlines()
    assert (
        first[0].split() == ["#", "weight", "minuslogpost", *names]
        and len(first[1].split()) == 2 + d
    )

    # the second column is minus the log posterior of the row's state
    assert float(first[1].split()[1]) == pytest.approx(-logps[0, 0], rel=1e-11)
    assert int(first[1].split()[0]) >= 1

    samples, weights, index = ns["read_chains"](prefix)
    assert samples.shape[1] == d and weights.shape == index.shape == (samples.shape[0],)
    assert sorted(set(index)) == [1, 2, 3]
    assert [int(weights[index == i].sum()) for i in (1, 2, 3)] == [
        n_steps
    ] * 3  # the weights sum to the steps
    assert samples.shape[0] < n_chains * n_steps  # repeats are stored once, as a weight
    back = ns["expand_chains"](samples, weights, index)
    assert back.shape == chains.shape and np.allclose(back, chains, rtol=1e-12, atol=0)
    # and what is read is what a second writing gives, exactly: the files are the record
    again = tmp_path / "again" / "synthetic_chain"
    again.parent.mkdir()
    ns["write_chains"](again, back, logps, names, ranges)
    assert all((again.parent / n).read_bytes() == (prefix.parent / n).read_bytes() for n in written)
    # existing files are never overwritten
    with pytest.raises(FileExistsError):
        ns["write_chains"](prefix, chains, logps, names, ranges)
    # chains of unequal length cannot be stacked
    with pytest.raises(ValueError):
        ns["expand_chains"](samples[:-1], weights[:-1], index[:-1])


def test_levels_enclose_68_and_95_per_cent_of_a_gaussian():
    import numpy as np

    ns = _namespace("helpers-contours")
    rng = np.random.default_rng(21)
    draws = rng.multivariate_normal([0.0, 0.0], [[1.0, 0.7], [0.7, 2.0]], size=400000)
    h, _, _ = np.histogram2d(draws[:, 0], draws[:, 1], bins=70)
    l68, l95 = ns["levels_68_95"](h)
    assert l68 > l95 > 0
    assert h[h >= l68].sum() / h.sum() == pytest.approx(0.68, abs=0.01)
    assert h[h >= l95].sum() / h.sum() == pytest.approx(0.95, abs=0.01)
    weighted = np.histogram2d(draws[:, 0], draws[:, 1], bins=70, weights=np.full(len(draws), 2.5))[
        0
    ]
    assert np.allclose(
        ns["levels_68_95"](weighted), (2.5 * l68, 2.5 * l95)
    )  # the levels scale with the histogram
