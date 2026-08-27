# Project state

**Reads against** [`docs/PROJECT_SPEC.md`](PROJECT_SPEC.md).
The spec says what each rung must establish and what a passing result means; this document says where each one stands and what stands in the way.
It carries no history: what a rung's result *is* belongs here, how it came to be is in git and in the task's own spec.

**As of** 2026-08-27.
This branch carries the project's plan — the spec tree, the rules, and this status — ahead of the implementation and results, which arrive one rung at a time.

A number not carried here with its provenance record is not evidence.
Promotion means a result in `docs/results/` with a `.provenance.json` beside it recording the commit, job and inputs that produced it — project rule 8.

---

## Ladder status

| Rung | What the spec requires | Status | Evidence |
|---|---|---|---|
| 0 — replicate ceiling | A reproducibility ceiling clearing its null, on the declared panel | **Measured, not in this repository yet** | Arrives with rung 0's implementation |
| 1 — held-out line | Prediction beating a floor and recovering a planted signal, as a fraction of rung 0 | **Measured, not in this repository yet** | — |
| 2 — cross-platform | Retention separable from a scrambled-line control | **Measured, not in this repository yet** | — |
| 3 — GDSC2 viability | Interaction above zero after correction, against the screen-agreement ceiling | **Measured, not in this repository yet** | — |
| 4 — organoid viability | The transfer number, on a frozen embargoed holdout | **Not measured** | — |

**Measured, not in this repository yet** means the number exists, obtained on the working branch, and lands here with the pull request that brings its code and provenance record.
Until a result and its `.provenance.json` are in `docs/results/`, nothing in this repository backs the number, which is the standard project rule 8 sets and this table holds itself to.

**Measured, not closed** means a promoted number exists but the rung does not yet satisfy everything its spec asks of it.
A rung closes when its result is promoted with provenance, this table records it, and the project-rule tests for the steps it touches pass — including any exemption it was meant to remove.
No rung is closed.

## Where each rung stands

### Rung 0 — replicate ceiling

Split-half median **0.109**, Spearman-Brown full-data **0.197**, both clearing their diff-drug and same-drug nulls at p=0.0005, on rung 1's declared 14,121-gene panel.

*To close*: rung 0 aggregates by median while rung 1 aggregates by mean, so rung 1 cannot yet be expressed as a fraction of this ceiling (project rule 2).
Pick the shared aggregation, declare it where the scorers read it, republish the headline.

### Rung 1 — held-out line, delta fidelity

`knn`/`pca`/`nmf` reach ~0.28-0.32; the two Stack checkpoints reach ~0.018/0.040.
The direction is matched by the differential-expression table's within-drug p-values.

*To close*: the table has no floor and no positive control, so it cannot yet separate "the baselines genuinely beat Stack" from "nothing here fits anything", and baseline capacity is pinned rather than cross-validated.
Owned by [`rung1-controls-and-capacity`](tasks/rung1-controls-and-capacity/design.md).

### Rung 2 — cross-platform

Every fitted baseline loses 0.17-0.53 of correlation moving from same-platform to cross-platform fitting, landing at the scrambled-line control (pca 0.035 against a control of 0.033).
Stack's own arm is null under both checkpoints, matching its rung-1 result — its failure is not a platform artefact.

*To close*: `cmapPy` is a hard dependency of four rung-2 files and is declared in no dependency file, so the rung cannot be reproduced from a clean environment.

### Rung 3 — GDSC2 viability

`base` embedding under L2 penalty is the only representation clearing Bonferroni across 24 declared variants (interaction 0.137, p=0.001), about 30% of the 0.457 screen-agreement ceiling.

*To close*: the `perdrug` and `global` columns are computed on non-residualised predictions while `interaction` residualises, so reading `perdrug` across rows reports fold-intercept structure rather than ranking signal; and the variant reporter cannot be run without a `PYTHONPATH` workaround.

### Rung 4 — organoid viability

Not measured. The scoring path is built and the holdout is frozen.

*To start*: the committed drug crosswalk still carries pre-rename source labels, so the organoid design resolves to zero drugs.
Rebuild it, verify the `source` column, then resubmit; after that, the two provenance gaps named in the rung's spec must close before the holdout is unfrozen.
Detail in [`tasks/rung4-organoid-viability/design.md`](tasks/rung4-organoid-viability/design.md).

## Open gaps

Each is owned by a task spec; none is a rung result.

| Gap | Rule | Owner |
|---|---|---|
| No promoted result carries a leakage profile — the registry path that filters pretraining contamination runs on nothing | — | [`modular-harness-core`](tasks/modular-harness-core/design.md) |
| The release gate matches column *names*, so an identifier inside an ordinary cell is invisible to it; one committed table carries organoid ids that way | 9 | [`embargo-gate-cell-values`](tasks/embargo-gate-cell-values/design.md) |
| The gate is not enforced anywhere — no pre-commit hook, no CI step, no test invokes it | 9 | [`embargo-gate-cell-values`](tasks/embargo-gate-cell-values/design.md) |
| Promotion records `HEAD` at promotion time rather than the commit that produced the result, and does not record whether the tree was clean | 8 | [`project-rule-enforcement`](tasks/project-rule-enforcement/design.md) |
| No machine-readable declaration of each rung's correlation and aggregation, so a cross-rung ratio can be computed by accident | 2 | [`project-rule-enforcement`](tasks/project-rule-enforcement/design.md) |
| Four rule exemptions are unclassified — they need a fix, a stated reason, or a record that the artefact is retired | 6, 7 | [`project-rule-enforcement`](tasks/project-rule-enforcement/design.md) |
| The ladder audit's control columns are a regex over script text and point at scripts one rung does not run | 6 | [`rung1-controls-and-capacity`](tasks/rung1-controls-and-capacity/design.md) |
| CI runs lint before tests and lint fails, so no test or coverage result has ever been reported for this branch | — | (unowned — blocks the first pull request) |

## Where things live

- **Results** `docs/results/*.csv` with a matching `.provenance.json`. No sidecar, no evidence.
- **Figures** `docs/figures/*.png`, generated by `scripts/plot_ladder_results.py`.
- **Rung and task specs** `docs/tasks/<slug>/design.md`, branching from the spec's ladder.
- **Rung protocol** `docs/transfer_ladder_protocol.md` — panels, aggregation and control lists per rung.
- **Decisions** `docs/decisions/YYYY-MM-DD-<slug>.md` for anything reversed or chosen against an alternative.
- **Rules and their tests** `docs/PROJECT_SPEC.md` and `tests/test_project_rules.py`.
