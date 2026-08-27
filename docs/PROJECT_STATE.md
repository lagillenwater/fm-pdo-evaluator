# Project state

**Reads against** [`docs/PROJECT_SPEC.md`](PROJECT_SPEC.md).
The spec says what each rung must establish and what a passing result means; this document says where each one stands.
It carries no history: a rung's result belongs here, how it came to be belongs in git and in that rung's spec.

**As of** 2026-08-27.

A number not carried here with its provenance record is not evidence.
Promotion means a result in `docs/results/` with a `.provenance.json` beside it recording the commit, job and inputs that produced it — project rule 8.
`docs/results/` does not exist yet, so nothing in this repository is evidence of anything.

---

## Ladder status

| Rung | What the spec requires | Status |
|---|---|---|
| 0 — replicate ceiling | A reproducibility ceiling clearing its null, on the declared panel | Next |
| 1 — held-out line | Prediction beating a floor and recovering a planted signal, as a fraction of rung 0 | Not started |
| 2 — cross-platform | Retention separable from a scrambled-line control | Not started |
| 3 — GDSC2 viability | Interaction above zero after correction, against the screen-agreement ceiling | Not started |
| 4 — organoid viability | The transfer number, on a frozen embargoed holdout | Not started |

A rung closes when its result is promoted with provenance, this table records it, and the project-rule tests for the steps it touches pass.
Rungs are built in order, because each one is read against the one below it: rung 0's ceiling is the denominator for rung 1, and a rung whose denominator is unmeasured reports a ratio to an imaginary 1.0.

## Repository conditions that affect every rung

Neither is a rung result; both change what a rung can claim when it lands.

| Condition | Consequence |
|---|---|
| CI runs `ruff check` and `pyright` before `pytest`, and lint currently fails | No test result or coverage figure has been reported for this repository. A rung landing under this ordering gets a red check that says nothing about whether its tests pass |
| Existing design documents live under `docs/superpowers/` with dated filenames | `docs/PROCESS.md` places task documents in `docs/tasks/<slug>/`. Each existing document moves when it is next touched, rather than in one sweep that would obscure the rung it belongs to |

## Where things live

- **Results** `docs/results/*.csv` with a matching `.provenance.json`. No sidecar, no evidence.
- **Figures** `docs/figures/*.png`, generated rather than hand-made.
- **Rung and task specs** `docs/tasks/<slug>/design.md`, one folder per task, arriving with the work it specifies.
- **Decisions** `docs/decisions/YYYY-MM-DD-<slug>.md` for anything reversed or chosen against an alternative.
- **Rules and their tests** `docs/PROJECT_SPEC.md` and `tests/test_project_rules.py`.
