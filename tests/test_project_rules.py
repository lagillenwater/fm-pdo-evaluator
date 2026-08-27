"""One test per numbered project rule in ``docs/PROJECT_SPEC.md``.

The project rules are what every task must satisfy, whichever rung or dataset it is about. They were prose only until
this module existed: three of them (the shared fold split as a project-wide rule, the gene-panel
guard, the embargo gate) had no test at all, and the rest were covered only through a helper's
own unit tests, which say nothing about whether the rest of the repo uses that helper.

Two kinds of test live here, and the difference decides how much a pass is worth:

* **Behavioural** — call the real function with a known answer and require the answer back.
  A pass is evidence the rule holds for that function.
* **Repository / artifact scan** — read the code or the promoted results and look for a
  specific violating shape. A pass means that shape is absent, which is weaker than "the rule
  holds": a scan cannot see a violation written a way the pattern does not match.
  A regex over source text read as stronger evidence than it was is a failure this project has
  already had, so each scan below says in its docstring what it does not prove.

Where the project violates a rule today, the case is marked ``xfail(strict=True)`` with a
pointer to where the gap is recorded. Strict means the test fails the day the gap closes, which
is the prompt to delete the marker rather than leave a stale exemption behind.
"""

from __future__ import annotations

import csv
import inspect
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fmharness import statistics as fmstatistics
from fmharness.deltas import assert_common_genes, fold_assignment, restrict_common_support

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "docs" / "results"

# Artifacts and drivers that violate a rule today. Every entry names the task that owns the fix,
# so an exemption is tracked debt rather than a silent carve-out, and each is applied as a STRICT
# xfail: the day the owning task lands, the test becomes an unexpected pass and the entry has to
# come out. The rules themselves stay generic — which instance is in violation is current state,
# and lives in docs/PROJECT_STATE.md and the owning task's spec, not in the rule.
# Keys are "ruleNN:name" — an instance exempt under one rule is NOT exempt under another.
KNOWN_GAPS: dict[str, str] = {
    # Classified: the violation is understood and a task owns the fix.
    "rule06:rung1_check1_fidelity.csv": "no floor and no positive control; owned by docs/tasks/rung1-controls-and-capacity",
    "rule07:scripts/alpine/34a_rung1_plan.sbatch": "pins --k 10; owned by docs/tasks/rung1-controls-and-capacity",
    # Unclassified: surfaced when discovery replaced a hardcoded list. Each needs one of three
    # answers — fix it, exempt it with a stated reason, or record that the artifact/script is
    # retired — and none may be guessed. The register is docs/tasks/project-rule-enforcement/design.md.
    "rule03:rung3_declared_variants.csv": "report_variants.py re-reports rung3_check2_grid.csv, whose own producing family carries the guards; the guard ran upstream of its input. Becomes a real gap only if it ever rescores raw deltas",
    "rule03:de_permutation_null_both_checkpoints.csv": "unclassified: same disposition question as its rule06 entry — if it is not a method comparison, neither test applies",
    "rule06:de_permutation_null_both_checkpoints.csv": "unclassified: a permutation-null table may not be a method comparison at all; if so the discovery predicate is what needs narrowing",
    "rule07:scripts/alpine/02_merge_score.sbatch": "unclassified: pins --k 10; predates the ladder, may be retired",
    "rule07:scripts/alpine/05_stack_score.sbatch": "unclassified: pins --k 10; predates the ladder, may be retired",
    "rule07:scripts/alpine/07_stack_emb_score.sbatch": "unclassified: pins --k 10; predates the ladder, may be retired",
}


def _case(rule: str, name: str) -> object:
    """A parametrised case, marked xfail when KNOWN_GAPS records it as an open violation."""
    key = f"{rule}:{name}"
    if key in KNOWN_GAPS:
        return pytest.param(name, marks=pytest.mark.xfail(strict=True, reason=KNOWN_GAPS[key]))
    return name


# --------------------------------------------------------------------------------------
# Project rule 1 — one shared way of splitting samples into cross-validation folds
# --------------------------------------------------------------------------------------


@pytest.mark.step_split
def test_rule_01_fold_split_is_order_free_deterministic_and_degenerates_to_loo() -> None:
    """The split may not depend on the order the caller happens to list its samples in."""
    lines = [f"L{i}" for i in range(12)]

    assert fold_assignment(lines, 5) == fold_assignment(list(reversed(lines)), 5)
    assert fold_assignment(lines, 5) == fold_assignment(lines, 5)
    # Both ends degenerate to leave-one-out: one sample per fold, no sample left unassigned.
    for folds in (0, 1, len(lines), 99):
        assignment = fold_assignment(lines, folds)
        assert sorted(assignment) == sorted(lines)
        assert len(set(assignment.values())) == len(lines)


# ``fold_assignment`` itself, and one diagnostic whose stated job is to reproduce a historical
# split exactly. Anything else writing this by hand is the rule-1 violation.
FOLD_MAP_ALLOWED = {
    "src/fmharness/deltas.py",
    "scripts/diagnose_oracle_additive.py",
}
HAND_WRITTEN_FOLD_MAP = re.compile(r"\{\s*\w+\s*:\s*\w+\s*%\s*\w*folds?\b")


@pytest.mark.step_split
def test_rule_01_no_analysis_code_writes_its_own_fold_map() -> None:
    """Scan: nothing but the shared helper builds a ``{line: i % n_folds}`` map.

    Does not prove every fold split in the repo comes from ``fold_assignment`` — a split
    written some other way (a different variable name, ``np.array_split``, a groupby) would
    not match this pattern. It catches the one shape that has actually recurred here, twice.
    """
    offenders = []
    for path in [*(REPO / "src").rglob("*.py"), *(REPO / "scripts").rglob("*.py")]:
        rel = path.relative_to(REPO).as_posix()
        if rel in FOLD_MAP_ALLOWED:
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if HAND_WRITTEN_FOLD_MAP.search(line) and not line.lstrip().startswith("#"):
                offenders.append(f"{rel}:{lineno}")
    assert not offenders, (
        "these build their own fold map instead of calling fmharness.deltas.fold_assignment, "
        f"which stops agreeing the moment two of them sort differently: {offenders}"
    )


# --------------------------------------------------------------------------------------
# Project rule 2 — one correlation, one way of averaging it, wherever two numbers are compared
# --------------------------------------------------------------------------------------


@pytest.mark.step_score
def test_rule_02_transfer_penalty_is_a_difference_of_like_quantities() -> None:
    """The one promoted number built from two others must be a difference of the same statistic.

    ``transfer_penalty`` subtracts the in-platform score from the cross-platform score for the
    same method. If the two columns were ever scored differently, this arithmetic would still
    produce a number, so the check is that the published column really is that difference and
    not a mix of two differently-scored quantities.
    """
    rows = list(csv.DictReader((RESULTS / "rung2_transfer_penalty.csv").open()))
    assert rows, "rung2_transfer_penalty.csv is empty"
    for row in rows:
        expected = float(row["cross_platform"]) - float(row["in_platform"])
        assert abs(float(row["transfer_penalty"]) - expected) < 1e-9, row["source"]


@pytest.mark.step_score
def test_rule_02_ladder_summary_publishes_no_number_without_saying_what_it_is_of() -> None:
    """A rung may publish a fraction only alongside the denominator it is a fraction of.

    Rung 1 is the live case: its ceiling comparison is blocked because rung 0 aggregates by
    median and rung 1 by mean, so the summary carries a reason instead of a number. This test
    pins the shape — number implies stated denominator — not that specific status, so it keeps
    holding once rung 1 unblocks.
    """
    rows = list(csv.DictReader((REPO / "docs" / "figures" / "ladder_summary.csv").open()))
    assert rows, "ladder_summary.csv is empty"
    for row in rows:
        if row["value"].strip():
            assert row["reason"].strip(), f"rung {row['rung']} publishes a bare number"


# --------------------------------------------------------------------------------------
# Project rule 3 — compare only on the samples, drugs and genes every method can cover
# --------------------------------------------------------------------------------------


def _source(pairs: list[tuple[str, str]], n_genes: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    delta = pd.DataFrame(
        {f"g{g}": [float(i) for i in range(len(pairs))] for g in range(n_genes)}
    )
    key = pd.DataFrame(pairs, columns=pd.Index(["patient", "drug"]))
    return delta, key


@pytest.mark.step_restrict
def test_rule_03_every_method_is_scored_on_the_identical_support() -> None:
    """After restriction, every method carries exactly the same (patient, drug) pairs."""
    wide = _source([("p1", "d1"), ("p2", "d1"), ("p1", "d2"), ("p2", "d2")])
    narrow = _source([("p1", "d1"), ("p2", "d1"), ("p3", "d1")])
    design = pd.DataFrame(
        {"patient": ["p1", "p2", "p3"], "drug": ["d1", "d1", "d1"], "y": [0.1, 0.2, 0.3]}
    )

    out = restrict_common_support({"wide": wide, "narrow": narrow}, design)

    supports = {
        name: list(zip(key["patient"], key["drug"], strict=True)) for name, (_, key) in out.items()
    }
    assert len({tuple(s) for s in supports.values()}) == 1, supports
    assert supports["wide"] == [("p1", "d1"), ("p2", "d1")]


@pytest.mark.step_build
def test_rule_03_a_method_on_a_different_gene_panel_is_rejected() -> None:
    """The gene-axis twin of the support rule: unequal panels must fail loudly, not score."""
    matched = {"a": _source([("p1", "d1")], n_genes=3), "b": _source([("p1", "d1")], n_genes=3)}
    assert_common_genes(matched)  # equal panels: no exception

    mismatched = {"a": _source([("p1", "d1")], n_genes=3), "b": _source([("p1", "d1")], n_genes=2)}
    with pytest.raises(ValueError, match="gene panel"):
        assert_common_genes(mismatched)


# --------------------------------------------------------------------------------------
# Project rule 4 — every reported statistic has a known-answer test that calls the real function
# --------------------------------------------------------------------------------------


@pytest.mark.step_score
def test_rule_04_every_reported_statistic_has_a_known_answer_test() -> None:
    """Scan: each public function in ``fmharness.statistics`` is named by some test.

    Scoped to that module because it exists to hold the statistics that get reported. A name
    appearing in a test file is not proof the test plants a signal and requires it recovered —
    that part is the reviewer's job — but a name appearing nowhere is proof no test calls it.
    """
    reported = [
        name
        for name, obj in vars(fmstatistics).items()
        if not name.startswith("_")
        and inspect.isfunction(obj)
        and obj.__module__ == fmstatistics.__name__
    ]
    assert reported, "fmharness.statistics exposes no public function; the scan would be vacuous"

    corpus = "\n".join(p.read_text() for p in (REPO / "tests").glob("test_*.py"))
    untested = [name for name in reported if name not in corpus]
    assert not untested, f"reported statistics with no test naming them: {untested}"


# --------------------------------------------------------------------------------------
# Project rule 5 — an average's null is resampled to that average, not to single draws
# --------------------------------------------------------------------------------------


@pytest.mark.step_null
def test_rule_05_aggregate_null_recovers_a_planted_shift_and_stays_null_without_one() -> None:
    """Known answer both ways, through the shared helper.

    The fuller battery — including the demonstration that the per-item form fails to recover
    the same planted effect — is in ``test_statistics_recover_known_answers.py``. This keeps one
    end-to-end check attached to the rule itself so the rule is not left resting on a test
    whose name does not mention it.
    """
    rng = np.random.default_rng(0)
    null_draws = rng.normal(0.0, 1.0, 500)  # individual mismatched-pair statistics
    n_obs = 40  # the observed statistic is a mean over this many pairs

    # A mean of 0.6 over 40 items sits ~3.8 standard errors above a null centred on zero.
    p_signal, lo, hi = fmstatistics.bootstrap_aggregate_pvalue(0.6, null_draws, n_obs, seed=0)
    assert p_signal < 0.05, p_signal
    assert lo <= hi

    # The same pool with nothing planted must not come out significant.
    p_none, _, _ = fmstatistics.bootstrap_aggregate_pvalue(0.0, null_draws, n_obs, seed=0)
    assert p_none > 0.05, p_none


# --------------------------------------------------------------------------------------
# Project rule 6 — every comparison table needs a floor, and a positive control where buildable
# --------------------------------------------------------------------------------------

FLOOR_NAMES = {"prior", "shuffled"}
POSITIVE_NAMES = {"planted"}
MIN_METHODS_FOR_A_COMPARISON = 3


def _comparison_tables() -> list[object]:
    """Every promoted table that compares methods, discovered rather than listed.

    A comparison table is one carrying a ``source`` column with at least
    ``MIN_METHODS_FOR_A_COMPARISON`` distinct values; a single-arm scoring output is not a
    comparison and cannot be asked for a floor. Discovering them means a table added later is
    covered without anyone remembering to add it here, which is the failure mode of a list.
    """
    found: list[object] = []
    for path in sorted(RESULTS.glob("*.csv")):
        try:
            rows = list(csv.DictReader(path.open()))
        except (OSError, csv.Error):
            continue
        if not rows or "source" not in rows[0]:
            continue
        if len({r["source"] for r in rows}) >= MIN_METHODS_FOR_A_COMPARISON:
            found.append(_case("rule06", path.name))
    return found


@pytest.mark.step_build
@pytest.mark.step_null
@pytest.mark.parametrize("table", _comparison_tables())
def test_rule_06_comparison_tables_carry_a_floor_and_a_positive_control(table: str) -> None:
    """A table of real methods with nothing that must fail and nothing that must succeed
    cannot separate "no method predicts this" from "nothing could have been detected here."
    """
    rows = list(csv.DictReader((RESULTS / table).open()))
    sources = {row["source"] for row in rows}

    floors = (sources & FLOOR_NAMES) | {s for s in sources if s.endswith("_random")}
    positives = sources & POSITIVE_NAMES
    assert floors, f"{table} has no floor row: {sorted(sources)}"
    assert positives, f"{table} has no positive control row: {sorted(sources)}"


# --------------------------------------------------------------------------------------
# Project rule 7 — model flexibility is chosen the same way for every method in a table
# --------------------------------------------------------------------------------------

FIXED_CAPACITY = re.compile(r"--(k|n-components|n_components)[= ]\s*\d+")


def _drivers() -> list[object]:
    """Every submission script, discovered rather than listed.

    Scoped to ``scripts/alpine/*.sbatch`` because that is what submits the work behind a
    promoted number. A script no longer in use is still checked: an unused driver that pins
    capacity is a template the next one gets copied from.
    """
    scripts = sorted((REPO / "scripts" / "alpine").glob("*.sbatch"))
    return [_case("rule07", s.relative_to(REPO).as_posix()) for s in scripts]


@pytest.mark.step_fit
@pytest.mark.parametrize("driver", _drivers())
def test_rule_07_drivers_do_not_pin_capacity_to_a_literal(driver: str) -> None:
    """Scan: no submission script hands its methods a fixed component count.

    A pass does not prove the script cross-validates capacity, only that it does not pin it on
    the command line; recording the selected capacity in the output table is what closes that.
    """
    text = (REPO / driver).read_text()
    hits = [
        line.strip()
        for line in text.splitlines()
        if FIXED_CAPACITY.search(line) and not line.lstrip().startswith("#")
    ]
    assert not hits, f"{driver} pins capacity instead of cross-validating it: {hits}"


# --------------------------------------------------------------------------------------
# Project rule 8 — every promoted result carries its provenance record
# --------------------------------------------------------------------------------------

SIDECAR_KEYS = ("name", "result", "script", "git_sha", "job_id", "inputs")


@pytest.mark.step_promote
def test_rule_08_every_promoted_result_carries_a_complete_provenance_record() -> None:
    """Artifact scan: sidecar present, required keys present, commit and inputs recorded.

    ``job_id`` is required to be present but may be empty — some promoted tables are inventories
    produced locally rather than by a cluster job. The clean-tree half of the rule is not
    checkable after the fact: ``promote_result.py`` records HEAD at promotion time and does not
    record whether the tree was dirty; see docs/PROJECT_STATE.md's open gaps.
    """
    problems = []
    for result in sorted(RESULTS.glob("*.csv")):
        sidecar = result.with_suffix(".provenance.json")
        if not sidecar.exists():
            problems.append(f"{result.name}: no sidecar")
            continue
        record = json.loads(sidecar.read_text())
        missing = [k for k in SIDECAR_KEYS if k not in record]
        if missing:
            problems.append(f"{result.name}: sidecar missing {missing}")
        if not record.get("git_sha"):
            problems.append(f"{result.name}: sidecar records no commit")
        if not record.get("inputs"):
            problems.append(f"{result.name}: sidecar records no inputs")
    assert not problems, problems


# --------------------------------------------------------------------------------------
# Project rule 9 — embargo is checked on what cells contain, not on what columns are named
# --------------------------------------------------------------------------------------


@pytest.mark.step_load
@pytest.mark.step_release
@pytest.mark.xfail(
    strict=True,
    reason="check_release.py scans only columns whose NAME is in SAMPLE_COLUMNS, so an "
    "identifier inside an ordinary cell is invisible to it; recorded in "
    "docs/PROJECT_STATE.md's open gaps",
)
def test_rule_09_embargo_gate_sees_an_identifier_inside_a_cell_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The live case: a table describing another table's schema, holding real organoid ids.

    ``docs/results/rung4_table_granularity.csv`` has the columns ``table, rows, columns, ...`` —
    none of them a sample-identifier name — while one row's ``columns`` cell lists
    ``SARC0128_Tumor;SARC0129_Tumor;...`` as ordinary text.
    """
    import check_release

    registry = tmp_path / "public_cell_lines.txt"
    registry.write_text("A549\nMCF7\n")
    monkeypatch.setattr(check_release, "PUBLIC_LINE_REGISTRY", registry)

    table = tmp_path / "table_granularity.csv"
    table.write_text(
        "table,rows,columns\n"
        'normalized_gene_counts.parquet,1000,"SARC0128_Tumor;SARC0129_Tumor;SARC0120_Organoids"\n'
    )

    cols = check_release.sample_columns_in(table)
    offending, status = check_release.nonpublic_line_values(table, cols)
    assert status == "ok"
    assert offending, "the gate found no embargoed identifier in the cell values"


# --------------------------------------------------------------------------------------
# Project rule 10 — a reversal gets a written decision, and every task/decision is indexed
# --------------------------------------------------------------------------------------


@pytest.mark.step_document
def test_rule_10_every_task_and_decision_is_named_in_the_spec_tree() -> None:
    """Scan: nothing under docs/tasks/ or docs/decisions/ is missing from PROJECT_SPEC.md.

    Cannot detect the thing rule 10 is really about — work that reverses a decision and
    writes nothing down. Only a reader comparing git log against docs/decisions/ finds that.
    What it does catch is the next step of the same failure: a decision or task document that
    exists but that nobody can find, because the index does not mention it.
    """
    index = (REPO / "docs" / "PROJECT_SPEC.md").read_text()

    unindexed = [
        f"docs/tasks/{d.name}/"
        for d in sorted((REPO / "docs" / "tasks").iterdir())
        if d.is_dir() and d.name not in index
    ]
    unindexed += [
        f"docs/decisions/{f.name}"
        for f in sorted((REPO / "docs" / "decisions").glob("*.md"))
        if f.name not in index
    ]
    assert not unindexed, f"present in the repo, absent from the spec tree: {unindexed}"


# ======================================================================================
# Edge-case tests — one per rule, covering exactly what the primary test cannot see.
# Where the repo cannot pass one today, it is a strict xfail naming the owning task, so
# the gap is a failing-by-record test rather than a sentence in a document.
# ======================================================================================

ALT_SPLIT = re.compile(r"\b(KFold|StratifiedKFold|ShuffleSplit|train_test_split|array_split)\s*\(")

# Files under scripts/ allowed to construct their own split, each with the verified reason.
# src/ is out of scope for THIS scan: KFold inside a fit (capacity selection during model
# fitting) is legitimate; the rule governs the EVALUATION partition, which scripts own.
SPLIT_EXEMPT: dict[str, str] = {
    "scripts/per_patient_eval.py": (
        "unclassified: GroupKFold over patients — a grouping the shared helper does not "
        "provide; not named by any promoted sidecar, may be retired"
    ),
}


@pytest.mark.step_split
def test_rule_01_edge_no_alternative_split_mechanism_in_scripts() -> None:
    """Rule 1's scan only sees `{ln: i % n_folds}`; this closes the named alternatives.

    Still a text scan, so still incomplete — a bespoke loop evades both — but the helper
    families sklearn offers are now as visible as the modulo shape.
    """
    offenders = []
    for path in (REPO / "scripts").rglob("*.py"):
        rel = path.relative_to(REPO).as_posix()
        if rel in SPLIT_EXEMPT:
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if ALT_SPLIT.search(line) and not line.lstrip().startswith("#"):
                offenders.append(f"{rel}:{lineno}")
    assert not offenders, (
        f"scripts building an evaluation split outside fold_assignment: {offenders}; "
        "add to SPLIT_EXEMPT only with a verified reason"
    )


@pytest.mark.step_score
@pytest.mark.xfail(
    strict=True,
    reason="no code-readable declaration of each rung's correlation and aggregation exists; "
    "owned by docs/tasks/project-rule-enforcement change 4",
)
def test_rule_02_edge_metric_declaration_exists_and_covers_every_rung_table() -> None:
    """Rule 2's edge: nothing states each rung's (correlation, aggregation) in a form code
    can read, so a cross-rung ratio can be computed by accident. The closing mechanism is a
    single declaration; this test is its acceptance criterion.
    """
    from fmharness import metric_declaration  # does not exist yet

    declared = metric_declaration.DECLARED  # {table_name: (correlation, aggregation)}
    rung_tables = {p.name for p in RESULTS.glob("rung*.csv")}
    undeclared = rung_tables - set(declared)
    assert not undeclared, f"promoted rung tables with no declared metric: {sorted(undeclared)}"


def _pipeline_text(script_rel: str) -> str:
    """The text of a producing script's whole pipeline family, plus their fmharness imports.

    A promoted table's sidecar names the LAST stage (`rung1_gather.py`), while the guards
    legitimately live in an earlier stage (`rung1_plan.py`, `rung1_build_one.py`) or in the
    library functions any stage calls. So: every script sharing the sidecar script's first
    name token (`rung1_*`, `check2_*`), plus one level of fmharness imports from each.
    """
    path = REPO / script_rel
    if not path.exists():
        return ""
    family_token = path.stem.split("_")[0]
    texts = [p.read_text() for p in sorted(path.parent.glob(f"{family_token}_*.py"))]
    if not texts:
        texts = [path.read_text()]
    for text in list(texts):
        for mod in re.findall(r"^\s*(?:from|import)\s+(fmharness[\w.]*)", text, re.M):
            mod_path = REPO / "src" / (mod.replace(".", "/") + ".py")
            if mod_path.exists():
                texts.append(mod_path.read_text())
    return "\n".join(texts)


def _rule03_cases() -> list[object]:
    cases = []
    for path in sorted(RESULTS.glob("*.csv")):
        try:
            rows = list(csv.DictReader(path.open()))
        except (OSError, csv.Error):
            continue
        if not rows or "source" not in rows[0]:
            continue
        if len({r["source"] for r in rows}) >= MIN_METHODS_FOR_A_COMPARISON:
            cases.append(_case("rule03", path.name))
    return cases


@pytest.mark.step_build
@pytest.mark.step_restrict
@pytest.mark.parametrize("table", _rule03_cases())
def test_rule_03_edge_producing_code_restricts_support_and_panel(table: str) -> None:
    """Rule 3's edge: the guards hold only where a caller invokes them. Until the calls move
    inside the scoring entry point, this pins each promoted comparison table's ACTUAL producing
    script (from its own sidecar) to code that names both guard families.
    """
    sidecar = json.loads((RESULTS / table).with_suffix(".provenance.json").read_text())
    text = _pipeline_text(sidecar["script"])
    assert text, f"{table}: producing script {sidecar['script']} not found on this branch"
    assert "restrict_common_support" in text, (
        f"{table}: {sidecar['script']} (plus its fmharness imports) never restricts to common "
        "(patient, drug) support"
    )
    assert re.search(r"common_gene_panel|assert_common_genes", text), (
        f"{table}: {sidecar['script']} (plus its fmharness imports) never pins a common gene panel"
    )


@pytest.mark.step_score
def test_rule_04_edge_known_answer_tests_carry_the_marker() -> None:
    """Rule 4's edge: a test can name a statistic without planting anything. Every public
    statistic must therefore be named in a test file that declares the ``known_answer``
    marker — the marker is the author's signed claim that a signal is planted and recovered,
    which a reviewer verifies once per file instead of once per mention.
    """
    reported = [
        name
        for name, obj in vars(fmstatistics).items()
        if not name.startswith("_")
        and inspect.isfunction(obj)
        and obj.__module__ == fmstatistics.__name__
    ]
    marked_corpus = "\n".join(
        p.read_text()
        for p in (REPO / "tests").glob("test_*.py")
        if "known_answer" in p.read_text()
    )
    unmarked = [name for name in reported if name not in marked_corpus]
    assert not unmarked, (
        f"statistics named only in unmarked test files: {unmarked}; a known-answer test "
        "declares `pytestmark = pytest.mark.known_answer`"
    )


MANUAL_PVALUE = re.compile(r"\b(?:np\.)?(?:mean|sum)\(\s*\w*null\w*\s*>=")

# Every site computing a p-value without the shared helper, with the reason each is correct:
# in all of them the null array holds PERMUTATION REPLICATES OF THE SAME AGGREGATE the observed
# statistic is (interaction_rho over the same pairs, recomputed per shuffle), so observed and
# null are the same kind of quantity and the aggregate-vs-single-draw inflation cannot occur.
# Verified by reading each site, 2026-08-26. A new hit must use the helper or be added HERE,
# deliberately, with its reason.
P_VALUE_MANUAL_SITES: dict[str, str] = {
    "src/fmharness/evaluation.py": "permutation p with +1 correction; null holds per-permutation replicates of the same statistic",
    "scripts/biomarker_anchored.py": "within-drug label permutation; null replicates the observed aggregate",
    "scripts/baselines_sarcoma_organoids_2024.py": "within-drug label permutation; null replicates the observed aggregate",
    "scripts/transfer_gdsc_sarcoma_organoids_2024.py": "within-drug label permutation; null replicates the observed aggregate",
    "scripts/transfer_pharmaformer_lite.py": "within-drug label permutation; null replicates the observed aggregate",
    "scripts/benchmark_sarcoma_organoids_2024.py": "label permutation with CV re-run per draw; null replicates the observed aggregate",
}


@pytest.mark.step_null
def test_rule_05_edge_every_manual_pvalue_site_is_allowlisted() -> None:
    """Rule 5's edge: the wrong form is not a text pattern — the same expression is right or
    wrong depending on what the null holds. So every site NOT using the shared helper must be
    listed above with the verified reason it is correct; a new hit fails until someone reads it
    and either routes it through the helper or allowlists it with the reason.

    This retires the open "no systematic audit" item: the audit
    is now this list, and it is enforced instead of pending.
    """
    unlisted = []
    for path in [*(REPO / "src").rglob("*.py"), *(REPO / "scripts").rglob("*.py")]:
        rel = path.relative_to(REPO).as_posix()
        if rel in P_VALUE_MANUAL_SITES:
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if MANUAL_PVALUE.search(line) and not line.lstrip().startswith("#"):
                unlisted.append(f"{rel}:{lineno}")
    assert not unlisted, (
        f"p-value computed outside bootstrap_aggregate_pvalue and not allowlisted: {unlisted}"
    )


SCORE_COLUMNS = ("mean_rho", "interaction", "in_platform", "r")


def _rule06_recovery_cases() -> list[object]:
    cases = []
    for path in sorted(RESULTS.glob("*.csv")):
        try:
            rows = list(csv.DictReader(path.open()))
        except (OSError, csv.Error):
            continue
        if not rows or "source" not in rows[0]:
            continue
        sources = {r["source"] for r in rows}
        if "planted" in sources and (sources & FLOOR_NAMES):
            cases.append(_case("rule06edge", path.name))
    return cases


@pytest.mark.step_build
@pytest.mark.step_null
@pytest.mark.parametrize("table", _rule06_recovery_cases())
def test_rule_06_edge_positive_control_is_recovered_not_just_present(table: str) -> None:
    """Rule 6's edge: a planted row that scores at chance is indistinguishable from a broken
    harness — it happens when the control is substituted at scoring time instead of threaded
    into the fit target. So in every table carrying both, the planted row must actually beat
    every floor row on the table's primary score.
    """
    rows = list(csv.DictReader((RESULTS / table).open()))
    col = next((c for c in SCORE_COLUMNS if c in rows[0]), None)
    assert col, f"{table}: no recognised score column among {SCORE_COLUMNS}; extend the list"
    by_source: dict[str, float] = {}
    for r in rows:
        if r[col] not in ("", "nan"):
            by_source.setdefault(r["source"], float(r[col]))
    floors = [v for s, v in by_source.items() if s in FLOOR_NAMES or s.endswith("_random")]
    assert by_source["planted"] > max(floors), (
        f"{table}: planted ({by_source['planted']:.3f}) does not clear the best floor "
        f"({max(floors):.3f}) on '{col}' — the positive control is present but not recovered"
    )


@pytest.mark.step_fit
@pytest.mark.xfail(
    strict=True,
    reason="no promoted table records the selected capacity per method; owned by "
    "docs/tasks/project-rule-enforcement change 6",
)
def test_rule_07_edge_tables_record_selected_capacity() -> None:
    """Rule 7's edge: a driver can pin capacity in code where no command-line scan sees it.
    The closing mechanism is the artifact recording what was selected; this is its acceptance
    criterion: every comparison table carries a selected-capacity column.
    """
    tables = [RESULTS / str(c) for c in _rule03_cases() if isinstance(c, str)]
    assert tables
    missing = []
    for path in tables:
        rows = list(csv.DictReader(path.open()))
        if not any(c in rows[0] for c in ("selected_k", "capacity", "n_components_selected")):
            missing.append(path.name)
    assert not missing, f"tables with no selected-capacity column: {missing}"


@pytest.mark.step_promote
@pytest.mark.xfail(
    strict=True,
    reason="promote_result.py records neither tree cleanliness nor the producing commit; "
    "owned by docs/tasks/project-rule-enforcement change 1",
)
def test_rule_08_edge_sidecars_record_clean_tree_and_producing_commit() -> None:
    """Rule 8's edge: the clean-checkout requirement cannot be reconstructed after the fact.
    Acceptance criterion for the fix: every NEW sidecar records ``clean_tree`` and the commit
    that PRODUCED the result (distinct from HEAD at promotion time).
    """
    missing = []
    for sidecar in sorted(RESULTS.glob("*.provenance.json")):
        record = json.loads(sidecar.read_text())
        if "clean_tree" not in record or "produced_at_sha" not in record:
            missing.append(sidecar.name)
    assert not missing, f"sidecars without clean_tree/produced_at_sha: {missing}"
