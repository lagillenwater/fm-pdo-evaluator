"""Render the COMPLETE declared variant table for a result, so prose cannot drop rows.

data/model_matrix.yaml declares which variants belong in a check and check_matrix.py verifies a
result covers them. Both worked. What repeatedly failed is the step after: writing a summary by
hand and quietly omitting variants, which produced a table that read as complete and was not --
on 2026-08-26 a rung-3 summary showed three Stack variants of four and led with the one that
scored.

This renders every declared variant for a check, in one place, from the promoted artifact. A
variant absent from the result appears as an explicit MISSING row rather than being skipped, and
every penalty is shown because reporting only the best one is a selection that flatters whatever
has no signal.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts_check_matrix_shim import expected_for, load_matrix  # noqa: F401


def main() -> None:
    """Print every declared variant's rows, marking any that are absent."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--result", required=True, type=Path)
    ap.add_argument("--check", required=True)
    ap.add_argument("--matrix", type=Path, default=Path("data/model_matrix.yaml"))
    ap.add_argument("--metric", default="interaction")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    matrix = load_matrix(args.matrix)
    declared = expected_for(matrix, args.check)
    aliases: dict[str, str] = {}
    for v in matrix.get("variants", []):
        for a in v.get("aliases") or []:
            aliases[str(a)] = str(v["id"])

    df = pd.read_csv(args.result)
    df["source"] = [aliases.get(str(s), str(s)) for s in df["source"]]

    keep = [c for c in ("source", "method", args.metric, "p_label", "null_p95", "z_random",
                        "p_random", "same_as") if c in df.columns]
    rows = []
    for variant in declared:
        sub = df[df["source"] == variant]
        if sub.empty:
            rows.append({"source": variant, "method": "MISSING",
                         **{c: "" for c in keep if c not in ("source", "method")}})
        else:
            rows.extend(sub[keep].to_dict("records"))

    out = pd.DataFrame(rows)

    # MULTIPLE TESTING. This grid is one test per (variant, penalty) -- more than thirty -- so
    # at an uncorrected 0.05 roughly one and a half rows clear by chance. Reporting the rows
    # that clear without saying how many tests were run is how a grid manufactures a finding,
    # and picking the single best row out of the grid by hand is the same error with extra
    # steps. Bonferroni is reported alongside the raw p, never instead of it.
    real = out[out["method"] != "MISSING"].copy()
    if "p_label" in real.columns:
        pv = pd.to_numeric(real["p_label"], errors="coerce")
        n_tests = int(pv.notna().sum())
        alpha = 0.05 / max(n_tests, 1)
        out["clears_uncorrected"] = ""
        out["clears_bonferroni"] = ""
        for i, row in out.iterrows():
            v = pd.to_numeric(pd.Series([row.get("p_label")]), errors="coerce").iloc[0]
            if pd.notna(v):
                out.at[i, "clears_uncorrected"] = "yes" if v < 0.05 else ""
                out.at[i, "clears_bonferroni"] = "yes" if v < alpha else ""
        print(f"{n_tests} tests in this grid; Bonferroni alpha = {alpha:.5f}")
        print(f"  clearing uncorrected p<0.05 : {int((out['clears_uncorrected'] == 'yes').sum())}")
        print(f"  clearing Bonferroni         : {int((out['clears_bonferroni'] == 'yes').sum())}")
        print(f"  expected false positives at 0.05 across {n_tests} tests: {0.05 * n_tests:.1f}")

    n_missing = int((out["method"] == "MISSING").sum())
    print(f"{args.check}: {len(declared)} declared variants, {n_missing} missing from the result")
    print(out.to_string(index=False))
    if n_missing:
        print(f"\n{n_missing} declared variant(s) ABSENT -- the result does not cover the matrix.")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
