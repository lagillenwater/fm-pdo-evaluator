# Embargo gate: check cell values, not column names

**Status** OPEN.
**Steps** load, release.
**Project rules relied on** 9 (embargo), 8 (re-promotion of any file this remediates).
**Scope** project-wide — the gate governs every file in the repo, not one rung.

## The problem

`scripts/check_release.py` decides what to scan from the *header*.
`sample_columns_in` returns only those columns whose name is in `SAMPLE_COLUMNS` (`patient`, `line`, `cell_line`, …), and `nonpublic_line_values` scans only those columns' values against the public cell-line registry.
A table whose columns are named anything else is never opened for content, however many patient identifiers its cells contain.

That is not hypothetical here.
`docs/results/rung4_table_granularity.csv` has the schema `table, rows, columns, dose_or_replicate_columns, …`.
None of those names trips the check, and one row's `columns` cell is a semicolon-joined string holding `SARC0128_Tumor;SARC0129_Tumor;SARC0120_Organoids`, because that row describes `normalized_gene_counts.parquet`'s own column list.
The file is committed today with those identifiers in it.

Project rule 9 already states the correct behaviour — checked value by value, on what the cells contain.
The gate does not implement it.

## The change

1. Scan **cell values across every text-valued column** of an in-scope table, not only columns whose name looks like a sample identifier.
   Keep the existing three-outcome status (`ok` / `no-registry` / `unreadable`): the caller's message has to distinguish "this identifier is not public" from "I could not check", and collapsing them has already produced a message that blamed a missing registry for a `NameError`.
2. Skip numeric and boolean columns rather than stringifying them.
   Parquet carries types; for CSV, a column whose every value parses as a number is not an identifier column.
   This is a performance guard, not a correctness one, and it must be stated as such in the code — a column of numeric-looking sample ids would slip through it.
3. Keep the gate free of pandas.
   It runs wherever a commit happens, including interpreters without the project environment, which is why the current implementation reads headers by hand and imports `pyarrow` only inside the parquet branch.
   Reaching for pandas here broke that once already, and a bare `except` turned the breakage into a wrong message instead of a crash.
4. Remediate `docs/results/rung4_table_granularity.csv`: either regenerate it without the verbatim column-list cell, or classify it as embargoed in `data/release_manifest.yaml`.
   Whichever is chosen, record which in `docs/decisions/` — reclassifying a file that has been public-by-default is a data-source decision under project rule 10.

## Tests this task must pass

Task-specific, to be written in `tests/test_check_release.py`:

- A table whose header carries no sample-identifier column, and whose cell holds an embargoed identifier, is **rejected**. This is the case the gate misses today.
- A table whose cells hold only publicly catalogued lines (`A549`, `MCF7`) **passes**.
- A numeric-only table passes without being scanned value-by-value, and the test asserts the skip happened rather than inferring it from the pass.
- A missing registry returns `no-registry` and the gate still refuses — fail-closed, not fail-open.
- The remediated `docs/results/rung4_table_granularity.csv` passes the gate.

Project rules, from `tests/test_project_rules.py`:

- `test_rule_09_embargo_gate_sees_an_identifier_inside_a_cell_value` — currently `xfail(strict=True)`. This task flips it, and **deletes the marker in the same change**.
- `-m "step_load or step_release"` for the steps touched, and the full file, since the scans are repo-wide.

## Promotion to a project rule

The five task-specific tests above are about one script, so they stay in this task's file.
If the value-scanning helper is later reused by anything other than `check_release.py`, the first of them becomes a project rule and moves.

## Done when

The gate rejects the case above, the committed file is remediated with a decision entry, rule 9's `xfail` is gone, and `docs/PROJECT_STATE.md` §3 records the gap as closed with the commit that closed it.
