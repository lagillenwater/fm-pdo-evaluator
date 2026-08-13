"""Check 1 (delta-Pearson generation quality) through the harness-core registries.

Reproduces docs/tahoe_generation_results.md's Check-1 table via the same underlying
functions scripts/score_generation_eval.py uses (loo_baseline_source / build_generated_deltas
/ delta_fidelity, all shared from fmharness.deltas as of the plan that added this script).
The Stack row's SCORING reuses build_generated_deltas directly, unchanged -- it is already
correct and already vectorized; there is no reason to re-derive it. What is new is that the
Stack checkpoint's LeakageQueryable declaration (fmharness.models.stack_generator's
PregeneratedStackGenerator) drives filter_leakage (fmharness.leakage) before any source is
built, so a checkpoint's measured pretraining overlap with the eval cohort actually strips
contaminated rows -- the composition this whole harness-core effort exists for -- rather than
leakage-filtering staying a dormant, uncalled function. Baselines (additive/knn/pca/nmf) are
not the swappable dimension this proves -- they stay exactly as score_generation_eval.py
already builds them.

Run (once Alpine has produced the inputs -- see the implementation plan's Task 9 for the
exact `ralpine pull` commands and real-data invocation). Prefer --deltas-bundle over --context
unless you have specifically verified the two agree: a live --context rebuild is a snapshot of
whatever Tahoe context currently sits on Alpine, which is NOT guaranteed to match the bundle
docs/tahoe_generation_results.md's published table was actually computed from (confirmed to
diverge in production 2026-08-12 -- see --deltas-bundle's own --help text):
  uv run python scripts/check1_registry_driver.py \\
      --deltas-bundle tahoe_deltas \\
      --query-baseline tahoe_query.h5ad \\
      --generated-dir generated \\
      --pert-map context_by_drug/pert_to_cid.tsv \\
      --checkpoint-label cytokine-aligned
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from fmharness.deltas import (
    build_generated_deltas,
    build_tahoe_deltas,
    learned_gene_panel,
    load_pert_map,
    loo_baseline_source,
)
from fmharness.evaluation import score_delta_sources
from fmharness.leakage import (
    corpus_declared_partially,
    filter_leakage,
    ground_truth_source_declared_ambiguously,
    parse_corpus_set,
)
from fmharness.models.stack_generator import PregeneratedStackGenerator
from fmharness.schema import TaskSignal

_ROW = "_row"  # positional tag carried through filter_leakage so the delta stays row-aligned


def run_check1(
    real_delta: pd.DataFrame,
    real_key: pd.DataFrame,
    base: pd.DataFrame,
    *,
    query_baseline: Path,
    generated_dir: Path,
    pert_to_drug: dict[str, str],
    checkpoint_label: str,
    hallmark_path: Path,
    n_hvg: int = 2000,
    k: int = 10,
    pretraining_lines: set[str] | None = None,
    pretraining_drugs: set[str] | None = None,
    task_signal_in_pretrain: TaskSignal = "none",
) -> pd.DataFrame:
    """Check-1 table: one row per delta source, including the Stack generator.

    ``real_delta``/``real_key``/``base`` are the ground-truth triple from
    ``build_tahoe_deltas`` (or the parquet-bundle equivalent). ``query_baseline`` is the
    AnnData path fed to Stack generation as ``--test-adata`` -- what ``build_generated_deltas``
    needs to compute ``generated - baseline``; it is a different representation from ``base``
    (CPM-normalized query file vs. raw pseudobulk), matching score_generation_eval.py's own
    ``--context``/``--query-baseline`` split. ``pretraining_lines``/``pretraining_drugs``
    (both default ``None``) declare the Stack checkpoint's measured pretraining overlap with
    this eval cohort, if known -- BOTH must be given together for filtering to activate;
    giving only one is equivalent to giving neither, since ``filter_leakage`` itself requires
    a measured declaration on both axes before it will drop anything (``basis="unknown"``
    otherwise, unfiltered). When both are given, ``filter_leakage`` always drops the
    doubly-exposed (line AND drug) pairs from every source before scoring; when
    ``task_signal_in_pretrain="direct"`` (the checkpoint was trained on actual response
    labels, not just the raw line/drug identities), it additionally drops single-axis overlap
    (line OR drug) -- the tiered rule the rest of the harness already applies.
    """
    model = PregeneratedStackGenerator(
        generated_dir,
        pert_to_drug,
        checkpoint_label=checkpoint_label,
        pretraining_lines=pretraining_lines,
        pretraining_drugs=pretraining_drugs,
        task_signal_in_pretrain=task_signal_in_pretrain,
    )
    # filter_leakage reset_index()es the surviving design, so its index no longer points at the
    # original rows -- carry an explicit positional tag through it and select the delta by
    # POSITION, or the filtered key would be paired with the first N deltas (different lines).
    design = real_key.reset_index(drop=True).assign(**{_ROW: np.arange(len(real_key))})
    filtered_design, profile = filter_leakage(design, model)
    if profile.basis == "measured":
        print(
            f"Check-1 leakage filter: basis=measured, "
            f"doubly_exposed_frac={profile.doubly_exposed_frac:.3f}, "
            f"line_overlap_frac={profile.line_overlap_frac:.3f}, "
            f"drug_overlap_fraction={profile.drug_overlap_fraction:.3f}"
        )
    else:
        print(f"Check-1 leakage filter: basis={profile.basis} (no corpus declared -- unfiltered)")
    keep = filtered_design[_ROW].to_numpy()
    fd = real_delta.iloc[keep].reset_index(drop=True)
    fk = filtered_design.drop(columns=[_ROW]).reset_index(drop=True)

    learned_genes = learned_gene_panel(fd, hallmark_path, n_hvg=n_hvg)
    sources: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {
        "additive": loo_baseline_source("additive", fd, fk, base, k=k),
        "knn": loo_baseline_source("knn", fd, fk, base, k=k),
        "pca": loo_baseline_source("pca", fd, fk, base, k=k, genes=learned_genes),
        "nmf": loo_baseline_source("nmf", fd, fk, base, k=k, genes=learned_genes),
        "stack": build_generated_deltas(generated_dir, query_baseline, pert_to_drug),
    }

    # delta_fidelity (inside score_delta_sources) inner-joins pred_key/real_key on
    # (patient, drug) itself (evaluation.py's own pk.merge(rk, on=["patient","drug"],
    # how="inner")) -- the stack source's key (built from the full generated directory,
    # independent of the leakage filter above) is automatically restricted to fk's
    # already-filtered pairs by that join; no separate pre-filter is needed here, and
    # adding one would just duplicate delta_fidelity's own contract.
    return score_delta_sources(sources, fd, fk, n_hvg=n_hvg)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--context", default=None, help="Tahoe context AnnData (build_tahoe_context)")
    ap.add_argument(
        "--deltas-bundle",
        default=None,
        help="dir of real_delta.parquet/real_key.parquet/base.parquet (pseudobulk shortcut, "
        "matching score_generation_eval.py's own --deltas-bundle) -- provide this OR --context, "
        "not both. A live single-cell --context rebuild is NOT guaranteed to match a bundle "
        "built earlier from a different Tahoe context snapshot: caught in production 2026-08-12,"
        " where a --context rebuild (dated after the bundle) gave baselines ~2.4x the published"
        " table's values despite identical code -- --deltas-bundle tahoe_deltas is what actually"
        " reproduces docs/tahoe_generation_results.md.",
    )
    ap.add_argument(
        "--query-baseline", required=True, help="AnnData fed to stack-generation as --test-adata"
    )
    ap.add_argument("--generated-dir", required=True, help="dir of Stack-generated <pert>.h5ad")
    ap.add_argument("--pert-map", required=True, help="TSV 'pert_id<TAB>cid' (context split)")
    ap.add_argument("--checkpoint-label", required=True, help="e.g. cytokine- or drug-aligned")
    ap.add_argument("--n-hvg", type=int, default=2000)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument(
        "--hallmark-path", default="data/static/hallmark_signatures.gmt", help="Hallmark .gmt path"
    )
    ap.add_argument("--corpus-lines", default=None, help="comma-separated declared pretrain lines")
    ap.add_argument("--corpus-drugs", default=None, help="comma-separated declared pretrain drugs")
    args = ap.parse_args()

    if corpus_declared_partially(args.corpus_lines, args.corpus_drugs):
        ap.error(
            "--corpus-lines and --corpus-drugs must both be given together (or neither, to "
            "run unfiltered) -- giving only one silently disables leakage filtering"
        )
    if ground_truth_source_declared_ambiguously(args.context, args.deltas_bundle):
        ap.error("provide exactly one of --context (single-cell) or --deltas-bundle (pseudobulk)")

    repo = Path(__file__).resolve().parent.parent
    if args.deltas_bundle:
        bdir = Path(args.deltas_bundle)
        bdir = bdir if bdir.is_absolute() else repo / bdir
        real_delta = pd.read_parquet(bdir / "real_delta.parquet")
        real_key = pd.read_parquet(bdir / "real_key.parquet")
        base = pd.read_parquet(bdir / "base.parquet")
    else:
        real_delta, real_key, base = build_tahoe_deltas(ad.read_h5ad(args.context))
    pert_to_drug = load_pert_map(Path(args.pert_map))
    table = run_check1(
        real_delta,
        real_key,
        base,
        query_baseline=Path(args.query_baseline),
        generated_dir=Path(args.generated_dir),
        pert_to_drug=pert_to_drug,
        checkpoint_label=args.checkpoint_label,
        hallmark_path=repo / args.hallmark_path,
        n_hvg=args.n_hvg,
        k=args.k,
        pretraining_lines=parse_corpus_set(args.corpus_lines),
        pretraining_drugs=parse_corpus_set(args.corpus_drugs),
        task_signal_in_pretrain="adjacent" if args.corpus_lines else "none",
    )
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
