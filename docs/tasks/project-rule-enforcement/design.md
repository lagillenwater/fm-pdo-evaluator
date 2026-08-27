# Project-rule enforcement: close the gaps the tests cannot see

**Status** OPEN.
**Steps** split, restrict, score, null, promote.
**Project rules relied on** 1, 2, 3, 4, 5, 7, 8.
**Scope** project-wide by construction — none of this belongs to a rung, and every rung inherits it.

## The problem

Each project rule in `docs/PROJECT_SPEC.md` carries a line saying what a passing test does **not** prove.
Those lines are honest, not decorative: a scan over source text is weaker evidence than a behavioural check, and this project has already been burned by reading one as the other (`audit_ladder.py`'s `controls_*` columns report `True` for a rung that has no controls).

The gaps fall into two kinds.
Some are unclosable and should stay stated — rule 10 cannot detect a reversal nobody wrote down.
The rest are closable by making the guarantee **structural** instead of scanned: if the only way to get a fold split is to call the helper, no scan for hand-written splits is needed.
This task closes those.

## The changes

Each is small and independent; they are grouped because they are one review and one PR's worth of work, and because two of them change output schemas and should land before any rung re-runs.

1. **Rule 8 — provenance that can be trusted.**
   `scripts/promote_result.py` records `HEAD` at promotion time, not the commit that produced the result, and never records whether the tree was clean.
   Record both: the producing commit (from the job's own sidecar or an explicit argument) and a `clean_tree` boolean from `git status --porcelain`.
   Grandfather existing sidecars; require it for new promotions.
2. **Rule 4 — known-answer tests that are actually known-answer.**
   Add a `known_answer` marker. Rule 4's test currently accepts any test file that merely names a reported statistic; make it require a test carrying that marker.
3. **Rule 5 — every p-value from the shared helper.**
   `fmharness.statistics.bootstrap_aggregate_pvalue` is the only correct route for an aggregate.
   The five `np.mean(null >= observed)` sites under `scripts/` are **correct** — their nulls are permutation replicates of the same aggregate, verified by reading each one — but that is a fact established by a human read and recorded nowhere.
   Add an explicit allowlist naming those five with that reason, and a test that any new occurrence must either use the helper or be added to the allowlist deliberately.
   This closes the open audit item honestly, rather than by grep.
4. **Rule 2 — one declaration of each rung's correlation and averaging.**
   Nothing in the repo states, in a form code can read, that rung 0 is Pearson-median and rung 1 is Pearson-mean.
   The mismatch that blocks rung 1's fraction-of-ceiling therefore lives only in prose, and any two rungs can be divided by accident.
   Add one declaration the scorers read, and a test that a cross-rung number is only computed when both sides agree.
   This is the rule whose gap a test alone cannot close, which is why the declaration has to exist first.
5. **Rules 1 and 3 — guarantees inside the scorer.**
   Move `restrict_common_support` and `assert_common_genes` into the scoring entry point, and have it take folds only from `fold_assignment`, so neither depends on each caller remembering.
   The scans then become a backstop rather than the enforcement.
6. **Rule 7 — capacity recorded in the output.**
   Every scoring table gains a column naming the selected capacity per method, so tuned-vs-pinned is answerable from the artifact.
   Coordinate with `rung1-controls-and-capacity`, which needs the same column: land this first, so that rung re-runs once rather than twice.

## The exemption register

`tests/test_project_rules.py`'s `KNOWN_GAPS` is the authoritative list: every current violation of a project rule, keyed by rule and instance, each applied as a strict `xfail` so that fixing one turns the test into an unexpected pass and forces the entry out.
Two kinds of entry live there.

**Classified** — the violation is understood and a task owns it.
Rung 1's missing controls and its pinned capacity are both owned by `rung1-controls-and-capacity`.
`rung3_declared_variants.csv` is exempt from the support/panel check with a stated reason: its producer re-reports a table whose own pipeline carries the guards upstream of its input.

**Unclassified — the first work item of this task.**
Four entries surfaced when the rule-6 and rule-7 tests began discovering their cases instead of running against a hardcoded list, and each needs one of three answers, none of which may be guessed: fix it, exempt it with a stated reason, or record that the artefact is retired.

| Case | Rule | Question to answer |
|---|---|---|
| `de_permutation_null_both_checkpoints.csv` | 6, 3 | Is a permutation-null table a method comparison at all? If not, the discovery predicate is what needs narrowing, and one decision clears both entries |
| `scripts/alpine/02_merge_score.sbatch` | 7 | Pins a component count; predates the ladder — retired, or still a live path? |
| `scripts/alpine/05_stack_score.sbatch` | 7 | Same question |
| `scripts/alpine/07_stack_emb_score.sbatch` | 7 | Same question |

An unclassified exemption is worse than a failing test, because it looks handled.

## Tests this task must pass

Task-specific:

- `tests/test_promote_result.py` — a promotion from a dirty tree records `clean_tree: false`; the producing commit is recorded even when `HEAD` has moved past it.
- `tests/test_project_rules.py::test_rule_04_*` — extended to require the `known_answer` marker, with a fixture proving an unmarked test no longer satisfies it.
- `tests/test_project_rules.py::test_rule_05_*` — a new occurrence of the wrong shape, outside the allowlist, fails the scan.
- `tests/test_evaluation.py` — the scorer restricts support and asserts a common panel *itself*, proven by calling it with unrestricted sources and requiring the restriction to happen.
- A cross-rung number computed from two rungs with different aggregation raises rather than returning a float.

Project rules, from `tests/test_project_rules.py`: the whole file, plus `-m "step_split or step_restrict or step_score or step_null or step_promote"`.
No `xfail` flips here — this task strengthens what the passing tests prove, which is exactly the kind of change that can pass the old tests while breaking the guarantee, so review it against the "does not prove" lines rather than against the green run.

## Promotion to a project rule

Changes 1-3 and 6 alter what the existing project-rule tests check, so their assertions belong in `tests/test_project_rules.py` from the start.
Change 4's declaration, once it exists, makes rule 2 mechanically enforceable for the first time; update rule 2's "does not prove" line in the same change, since it will no longer be true.

## Done when

Every "does not prove" line under rules 1-5, 7 and 8 is either closed or restated to match what the strengthened test now proves, and `docs/PROJECT_STATE.md` records which.
Rule 10's gap stays open and stated: detecting an unwritten reversal is a human read, and a CI heuristic for it would be noise.
