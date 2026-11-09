"""The distribution-content scan (tools/ci/check_dist_content.py): what it finds in a built wheel and sdist, what it
leaves alone, that it never prints what it found, its two modes (the release gate and CI's known list), and that the
workflow and the release checklist use it. Synthetic archives only. The private-looking strings below are built from
pieces, so a plain search of the repository does not mistake this test for a leak, and none is a real path, address
or credential."""
import importlib.util
import io
import os
import tarfile
import warnings
import zipfile

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("_ci_check_dist_content",
                                               os.path.join(REPO, "tools", "ci", "check_dist_content.py"))
CC = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CC)

KNOWN_FILE = os.path.join(REPO, "tools", "ci", "dist_content_known.txt")
AUTHOR = "author" + "@" + "uni-a.xx"
OTHER = "someone" + "@" + "uni-b.xx"
METADATA = f"Metadata-Version: 2.4\nName: hcd-priya\nAuthor-email: Some Author <{AUTHOR}>\n\nA README.\n"
CLEAN = {"hcd_analysis/__init__.py": '"""A package."""\n__version__ = "0.0.0"\n'}

# Synthetic lines that each rule must find, in both archives.
H, S = "/" + "home/", "/" + "nfs/"
BAD = {
    "private-path": [f'root = "{H}someone/data/x.h5"', f"#   {S}cluster/project/sims", "/" + "tmp/claude-1/x",
                     "python=~/" + ".conda/envs/env/bin/python", "C:" + "\\Users\\someone\\x",
                     "/" + "global/cfs/cdirs/proj/x", "/" + "pscratch/sd/u/x", "/" + "Volumes/disk/x",
                     'os.path.join("/' + 'home", "someone", "x")', 'Path("/' + 'nfs") / "x"'],
    "internal-pointer": ["in .claude" + "/worktrees/x", "C:\\x\\.claude" + "\\y"],
    "cluster-directive": ["#SBATCH --" + "account=someacct", "#SBATCH --" + "mail-user=x", "#SBATCH -" + "A acct",
                          "#SBATCH -" + "Aacct", "#PBS -" + "A acct", "#BSUB -" + "P proj",
                          "subprocess.run(['sbatch', '--" + "account=x', 'job.sh'])"],
    "credential": ["-----BEGIN " + "RSA PRIVATE KEY-----", "-----BEGIN PGP " + "PRIVATE KEY BLOCK-----",
                   "token = gh" + "p_" + "a" * 36, "AKIA" + "ABCDEFGHIJKLMNOP", "password" + ' = "hunter22hunter"',
                   'api_key: "' + "x" * 12 + '"', "client_" + "secret = 'abcdefgh1234'",
                   "GITHUB_" + "TOKEN: abcd1234efgh5678ijkl", "gl" + "pat-" + "x" * 20, "hf_" + "a" * 34,
                   "AIza" + "B" * 35, "sk-" + "ant-" + "c" * 24, "https://user:" + "pw123@host.xx/path"],
    "confidential-marker": ["# CONFIDENTIAL" + ": do not share", "Draft, do not " + "distribute.",
                            "Confidentiality " + "notice", "This draft is " + "embargoed.",
                            "Please do not " + "circulate."],
    "email": [f"contact {OTHER} for data", "SOMEONE" + "@" + "UNI-B.SE", "x" + "@" + "lab.mx"],
}
# Ordinary code and text that must not be flagged.
GOOD = ["x = P @ np.linalg.inv(C)", "@dataclasses.dataclass", "base = 'lyaemu/data/kodiaq_squad/'",
        "cache = '~/.cache/hcd_" + "priya'", "os.path.join(home, 'x')", "see PU-0123 and GATE_E_SPEC section 2",
        "the confidence interval", "relative/home/path", f"maintainer: {AUTHOR}", "a@b", "arr @ w.T",
        "for internal use only", "token = tokens[i]", "n_tokens = 128", "SCHEMA = 'hcd_priya.registry/1'",
        "url = 'https://host.xx/project/a'", "user" + "@" + "example.com", "git" + "@" + "github.com:org/repo.git",
        "the tokenizer's secret_sauce = compute()"]


def _archives(tmp_path, files, metadata=METADATA, extra_names=(), sub="dist", owner=("", ""), link=None,
              zip_comment=b"", dirs=()):
    """A wheel and an sdist under tmp_path/sub holding `files` ({member: text}) and the metadata."""
    d = tmp_path / sub
    d.mkdir()
    with zipfile.ZipFile(d / "hcd_priya-0.0.0-py3-none-any.whl", "w") as zf:
        zf.comment = zip_comment
        for name, text in {**files, **{n: "x\n" for n in extra_names}}.items():
            zf.writestr(name, text)
        zf.writestr("hcd_priya-0.0.0.dist-info/METADATA", metadata)
    with tarfile.open(d / "hcd_priya-0.0.0.tar.gz", "w:gz") as tf:
        for name in dirs:
            info = tarfile.TarInfo(f"hcd_priya-0.0.0/{name}")
            info.type = tarfile.DIRTYPE
            tf.addfile(info)
        for name, text in {**files, "PKG-INFO": metadata, **{n: "x\n" for n in extra_names}}.items():
            data = text.encode()
            info = tarfile.TarInfo(f"hcd_priya-0.0.0/{name}")
            info.size = len(data)
            info.uname, info.gname = owner
            tf.addfile(info, io.BytesIO(data))
        if link:
            info = tarfile.TarInfo("hcd_priya-0.0.0/hcd_analysis/linked.py")
            info.type = tarfile.SYMTYPE
            info.linkname = link
            tf.addfile(info)
    return str(d)


def _known(tmp_path, entries):
    p = tmp_path / "known.txt"
    p.write_text("# a comment\n" + "".join(f"{r}\t{m}\t{h}\n" for r, m, h in entries))
    return str(p)


def test_a_clean_distribution_passes_in_both_modes(tmp_path, capsys):
    d = _archives(tmp_path, CLEAN)
    assert CC.main([d, "--release"]) == 0
    assert CC.main([d, "--known", _known(tmp_path, [])]) == 0
    assert "0 findings" in capsys.readouterr().out


@pytest.mark.parametrize("rule, line", [(r, line) for r, lines in BAD.items() for line in lines])
def test_each_rule_finds_its_content_in_both_archives_and_fails_the_release_gate(tmp_path, capsys, rule, line):
    d = _archives(tmp_path, {**CLEAN, "hcd_analysis/mod.py": f"a = 1\n{line}\n"})
    assert CC.main([d, "--release"]) == 1
    out = capsys.readouterr().out
    for kind in ("wheel", "sdist"):
        assert f"{kind}: hcd_analysis/mod.py:2: {rule} [sha256 {CC.sha256(line)}] NEW" in out


@pytest.mark.parametrize("line", GOOD)
def test_ordinary_code_and_allowed_codes_are_not_findings(line):
    assert CC.scan_text(line, {AUTHOR}) == []


def test_the_matched_text_is_never_printed(tmp_path, capsys):
    secret = f"{H}someone/very_private_dir"
    d = _archives(tmp_path, {**CLEAN, "hcd_analysis/mod.py": f'X = "{secret}"\n'})
    assert CC.main([d, "--release"]) == 1
    out = capsys.readouterr().out
    assert "private-path" in out and "someone" not in out and "very_private_dir" not in out


def test_a_matching_member_name_is_shown_only_as_its_digest(tmp_path, capsys):
    name = "hcd_analysis/.claude" + "/x.py"
    d = _archives(tmp_path, {**CLEAN, name: "a = 1\n"})
    assert CC.main([d, "--release"]) == 1
    out = capsys.readouterr().out
    assert f"name-sha256:{CC.sha256(name)}:0: internal-pointer" in out and ".claude" not in out


# The private note names are matched by digest (check_dist_content.POINTER_DIGESTS), so no test spells them. These tests
# exercise the mechanism with a synthetic name in place of the real digests; the real names are checked outside the
# repository.
SYNTH, SYNTH_MARK = "synthpriv_notes", "SYNTHMARK"


@pytest.fixture
def synthetic_digests(monkeypatch):
    CC._word_is_private.cache_clear()
    monkeypatch.setattr(CC, "POINTER_DIGESTS", frozenset({CC.hashlib.sha256(SYNTH.encode()).hexdigest()}))
    monkeypatch.setattr(CC, "EXACT_POINTER_DIGESTS", frozenset({CC.hashlib.sha256(SYNTH_MARK.encode()).hexdigest()}))
    yield
    CC._word_is_private.cache_clear()


@pytest.mark.parametrize("line", [f"see x/{SYNTH}/plan.md", f"x\\{SYNTH}\\y", SYNTH.upper(), SYNTH.replace("_", "-"),
                                  f"my_{SYNTH}_copy", f"'{SYNTH}'", f"under {SYNTH_MARK}/a", f"{SYNTH_MARK}_DIR"])
def test_a_private_name_is_found_by_its_digest_in_any_spelling(synthetic_digests, line):
    assert CC.line_rules(line) == ["internal-pointer"]


@pytest.mark.parametrize("line", ["synthpriv", "notes", "synthprivnotes", SYNTH_MARK.lower(), f"a {SYNTH_MARK.lower()} box"])
def test_a_part_of_a_private_name_or_the_ordinary_word_is_not(synthetic_digests, line):
    assert CC.line_rules(line) == []


def test_a_private_name_in_an_archive_fails_the_release_gate_without_being_printed(synthetic_digests, tmp_path, capsys):
    d = _archives(tmp_path, {**CLEAN, "hcd_analysis/mod.py": f"# design: {SYNTH}/spec.md\n"})
    assert CC.main([d, "--release"]) == 1
    out = capsys.readouterr().out
    assert "hcd_analysis/mod.py:1: internal-pointer" in out and SYNTH not in out


# The digests the scanner holds: five note names compared in lower case, one marker compared as written.
POINTER_DIGESTS = {
    "ba717dd9d4e3e0f52e338e04534394ebc5c5f45c948814476932d4a3a36ca713",
    "b225390a8984c8de4206c746772ab5ef2abb0a02bcb043da10f6d41c274d1a02",
    "9678f4c7207f3da2d59556c00fa5ceb8a3b107d1f539ba303e3ba91320d7aed1",
    "d3a4646482ceba965bbe9bcfc9fe9fb10eb0c8df829fa20ed9baff5b247964f2",
    "4e2f58503a2a022a1d09530042b39c555c797387916626f85e4d251e31f27989",
}
EXACT_POINTER_DIGESTS = {"d8e283d959d759036209e868ca4b18fb0306bbe9f04c135267e01131ada566cc"}


def test_the_digest_sets_are_pinned():
    assert CC.POINTER_DIGESTS == POINTER_DIGESTS and CC.EXACT_POINTER_DIGESTS == EXACT_POINTER_DIGESTS


@pytest.mark.parametrize("path", ["tools/ci/check_dist_content.py", "tests/test_dist_content.py",
                                  "tools/ci/dist_content_known.txt", "docs/release_checklist.md", "CHANGELOG.md"])
def test_the_gates_own_files_never_spell_a_private_name(path):
    """Not even in pieces: every word, and every word joined with its neighbours, is checked."""
    text = open(os.path.join(REPO, path), encoding="utf-8").read()
    assert not CC.has_private_name(text)
    assert not CC.has_private_name(text.replace('" + "', "").replace("' + '", ""))


def test_every_member_is_read_not_only_python_files(tmp_path, capsys):
    """Package data and the metadata (which embeds the README) ship too."""
    line = BAD["private-path"][0]
    d = _archives(tmp_path, {**CLEAN, "hcd_analysis/data.json": '{"path": "%s"}\n' % line},
                  metadata=METADATA + line + "\n")
    assert CC.main([d, "--release"]) == 1
    out = capsys.readouterr().out
    for where in ("wheel: hcd_analysis/data.json:1:", "sdist: hcd_analysis/data.json:1:",
                  "wheel: hcd_priya-0.0.0.dist-info/METADATA:6:", "sdist: PKG-INFO:6:"):
        assert f"{where} private-path" in out


def test_directory_names_link_targets_and_zip_comments_are_read(tmp_path, capsys):
    pointer = ".claude" + "/worktrees"
    for sub, kw in (("dir", {"dirs": [f"{pointer}/"]}), ("link", {"link": f"{H}someone/x"}),
                    ("comment", {"zip_comment": f"built from {pointer}".encode()})):
        assert CC.main([_archives(tmp_path, CLEAN, sub=sub, **kw), "--release"]) == 1, sub
    out = capsys.readouterr().out
    assert "(zip comment):1: internal-pointer" in out and pointer not in out


def test_every_copy_of_a_duplicated_zip_member_is_read(tmp_path):
    d = tmp_path / "dup"
    d.mkdir()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                       # zipfile warns about the duplicate name
        with zipfile.ZipFile(d / "hcd_priya-0.0.0-py3-none-any.whl", "w") as zf:
            zf.writestr("hcd_analysis/mod.py", BAD["private-path"][0] + "\n")
            zf.writestr("hcd_analysis/mod.py", "a = 1\n")      # a reader that takes the last copy sees this only
            zf.writestr("hcd_priya-0.0.0.dist-info/METADATA", METADATA)
    with tarfile.open(d / "hcd_priya-0.0.0.tar.gz", "w:gz") as tf:
        info = tarfile.TarInfo("hcd_priya-0.0.0/PKG-INFO")
        info.size = len(METADATA)
        tf.addfile(info, io.BytesIO(METADATA.encode()))
    assert CC.main([str(d), "--release"]) == 1


@pytest.mark.parametrize("owner, fails", [(("", ""), False), (("root", "root"), False), (("runner", "docker"), False),
                                          (("someuser", "somegroup"), True), (("runner", "somegroup"), True)])
def test_a_tar_header_naming_an_account_is_a_finding_never_printed(tmp_path, capsys, owner, fails):
    assert CC.main([_archives(tmp_path, CLEAN, owner=owner), "--release"]) == int(fails)
    out = capsys.readouterr().out
    assert ("(tar headers):0: archive-owner" in out) is fails and "someuser" not in out and "somegroup" not in out


def test_only_the_top_level_metadata_header_declares_authors(tmp_path, capsys):
    body = {**CLEAN, "hcd_analysis/mod.py": f"# {AUTHOR}\n"}
    assert CC.main([_archives(tmp_path, body), "--release"]) == 0
    other = {**CLEAN, "hcd_analysis/mod.py": f"# {OTHER}\n"}
    # another author in the metadata header is allowed; in the README body or a nested PKG-INFO it is not, in either
    # archive (each archive is checked on its own)
    assert CC.main([_archives(tmp_path, other, sub="h", metadata=METADATA.replace(AUTHOR, OTHER)), "--release"]) == 0
    nested = {**other, "hcd_analysis/fixtures/PKG-INFO": f"Author-email: <{OTHER}>\n",
              "hcd_analysis/fixtures/METADATA": f"Author-email: <{OTHER}>\n"}
    for sub, files, meta in (("r", other, METADATA + f"Author-email: <{OTHER}>\n"), ("n", nested, METADATA)):
        capsys.readouterr()
        assert CC.main([_archives(tmp_path, files, sub=sub, metadata=meta), "--release"]) == 1
        out = capsys.readouterr().out
        assert "wheel: hcd_analysis/mod.py:1: email" in out and "sdist: hcd_analysis/mod.py:1: email" in out, sub


def test_a_known_finding_passes_ci_mode_but_never_the_release_gate(tmp_path):
    line = BAD["private-path"][0]
    d = _archives(tmp_path, {**CLEAN, "hcd_analysis/mod.py": line + "\n"})
    known = _known(tmp_path, [("private-path", "hcd_analysis/mod.py", CC.sha256(line))])
    assert CC.main([d, "--known", known]) == 0
    assert CC.main([d, "--release"]) == 1


def test_a_new_finding_fails_ci_mode_even_next_to_a_known_one(tmp_path):
    a, b = BAD["private-path"][0], BAD["internal-pointer"][0]
    known = _known(tmp_path, [("private-path", "hcd_analysis/mod.py", CC.sha256(a))])
    assert CC.main([_archives(tmp_path, {**CLEAN, "hcd_analysis/mod.py": f"{a}\n{b}\n"}), "--known", known]) == 1
    # the same text moved to another file, or copied once more in the same file, is a new finding
    assert CC.main([_archives(tmp_path, {**CLEAN, "hcd_analysis/other.py": f"{a}\n"}, sub="moved"),
                    "--known", known]) == 1
    assert CC.main([_archives(tmp_path, {**CLEAN, "hcd_analysis/mod.py": f"{a}\n    {a}\n"}, sub="copied"),
                    "--known", known]) == 1


def test_a_listed_entry_that_no_longer_occurs_fails_so_the_list_shrinks(tmp_path, capsys):
    known = _known(tmp_path, [("private-path", "hcd_analysis/gone.py", "0" * 64)])
    assert CC.main([_archives(tmp_path, CLEAN), "--known", known]) == 1
    assert "(remove the entry)" in capsys.readouterr().out


@pytest.mark.parametrize("text", ["private-path hcd_analysis/x.py " + "0" * 64, "nosuchrule\tx.py\t" + "0" * 64,
                                  "private-path\tx.py\tnot-a-digest"])
def test_a_malformed_known_list_is_a_usage_error(tmp_path, text):
    p = tmp_path / "known.txt"
    p.write_text(text + "\n")
    assert CC.main([_archives(tmp_path, CLEAN), "--known", str(p)]) == 2


def test_the_dist_directory_must_hold_one_readable_wheel_and_sdist(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    assert CC.main([str(d), "--release"]) == 2
    with pytest.raises(SystemExit):
        CC.main([str(d)])                         # a mode is required: no silent default
    t = _archives(tmp_path, {**CLEAN, "hcd_analysis/big.py": "a = 1\n" * 20000}, sub="truncated")
    sdist = os.path.join(t, "hcd_priya-0.0.0.tar.gz")
    data = open(sdist, "rb").read()
    open(sdist, "wb").write(data[:len(data) // 2])
    assert CC.main([t, "--release"]) == 2


def test_line_numbers_count_newlines_only(tmp_path, capsys):
    line = BAD["private-path"][0]
    d = _archives(tmp_path, {**CLEAN, "hcd_analysis/mod.py": f"a = '\f'\r\nb = ' '\n{line}\n"})
    assert CC.main([d, "--release"]) == 1
    assert "wheel: hcd_analysis/mod.py:3: private-path" in capsys.readouterr().out


# The entries the committed known list may hold: the eight lines found in the archives when the gate was added,
# all already public on the default branch, waiting for their cleanup. A cleanup removes entries from the list; adding
# one means editing this set, which a reviewer sees.
KNOWN_BASELINE = {
    ("private-path", "hcd_analysis/config.py", "78d159c49cc0aca54f38c0dfc3d2b410771fb02aea77ce3df5ec74f3a20d7bd0"),
    ("private-path", "hcd_analysis/config.py", "d40478a1d152021d1a4b257d7cfbf04227067403539d2f065883582f7f5b336e"),
    ("private-path", "hcd_analysis/emulator/data_likelihood.py",
     "0b37abbd40dd71c20c09f9696d7c09e9b6dfe7e47c7de450bb0e74b693024154"),
    ("private-path", "hcd_analysis/emulator/data_likelihood.py",
     "209355a921108651c8e806d9d398772ba20a052766aebb08c5dba05e090a2c1d"),
    ("private-path", "hcd_analysis/emulator/data_likelihood.py",
     "2331a8c9681cdcfb0c640627905f0882eceaa6423654839f90ea02082261950f"),
    ("private-path", "hcd_analysis/emulator/data_likelihood.py",
     "71a9384ac89dc0b6c4b353a58d834ef439d2def5e7cd28b115064c236e874eed"),
    ("private-path", "hcd_analysis/emulator/hcd_lit_dndx_corrected.json",
     "00d98171a065519826a7b259d37efd6357afbadd3624501629b63b7757349161"),
    ("private-path", "hcd_analysis/emulator/multifidelity.py",
     "7fec9684082fea2cf0c0597f9bc0c021ab44d0da82dce3402d63eec9dc8d58fa"),
}


def test_the_committed_known_list_only_shrinks_and_holds_no_private_text():
    known = CC.read_known(KNOWN_FILE)
    assert set(known) <= KNOWN_BASELINE and all(n == 1 for n in known.values())
    assert CC.scan_text(open(KNOWN_FILE, encoding="utf-8").read()) == []


def _steps():
    import yaml
    wf = yaml.safe_load(open(os.path.join(REPO, ".github", "workflows", "tests.yml"), encoding="utf-8"))
    return wf["jobs"]["fixture-tests"]["steps"]


def test_the_workflow_runs_the_scan_right_after_the_contents_check_and_nothing_can_skip_it():
    steps = _steps()
    runs = [s.get("run", "") for s in steps]
    i = runs.index("python tools/ci/check_dist_content.py dist --known tools/ci/dist_content_known.txt")
    assert runs[i - 1] == "python tools/ci/check_dist.py dist" and steps[i + 1].get("id") == "install"
    assert not {"if", "continue-on-error"} & set(steps[i]) and "continue-on-error" not in steps[i + 1]
    assert "if" not in steps[i + 1]        # the install, and with it both test tiers, waits for a clean scan


def test_the_release_checklist_makes_the_release_gate_blocking():
    text = " ".join(open(os.path.join(REPO, "docs", "release_checklist.md"), encoding="utf-8").read().split())
    assert "**Blocking:** `python tools/ci/check_dist_content.py dist --release` reports 0 findings." in text
