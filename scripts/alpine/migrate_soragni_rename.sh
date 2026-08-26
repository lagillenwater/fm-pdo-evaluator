#!/usr/bin/env bash
# Rename the soragni-named artifacts on Alpine to match the repo (commit a6c8976).
# Generated 2026-08-25 from the live listing; every target was checked against a path
# the renamed scripts actually reference. Run from the repo root ON ALPINE.
set -euo pipefail
# The artifacts live at the REPO ROOT, not beside this script. The first version cd'd to
# $(dirname "$0") -- scripts/alpine/ -- so the guards all missed and the run was a no-op that
# looked like a success. Resolve the root from git instead of from this file's location.
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
echo "working in: $PWD"
echo "before:"; ls -d *soragni* 2>/dev/null || echo "  (none)"

[ -e context_by_drug_soragni                    ] && mv -vn context_by_drug_soragni                    context_by_drug_sarcoma_organoids_2024           # referenced by the code
[ -e generated_soragni_drug_aligned             ] && mv -vn generated_soragni_drug_aligned             generated_sarcoma_organoids_2024_drug_aligned    # referenced by the code
[ -e generated_soragni_drug_aligned_agg         ] && mv -vn generated_soragni_drug_aligned_agg         generated_sarcoma_organoids_2024_drug_aligned_agg   # referenced by the code
[ -e generated_soragni_faithful                 ] && mv -vn generated_soragni_faithful                 generated_sarcoma_organoids_2024_faithful        # referenced by the code
[ -e generated_soragni_faithful_agg             ] && mv -vn generated_soragni_faithful_agg             generated_sarcoma_organoids_2024_faithful_agg    # referenced by the code
[ -e soragni_aggregation_summary.csv            ] && mv -vn soragni_aggregation_summary.csv            sarcoma_organoids_2024_aggregation_summary.csv   # referenced by the code
[ -e soragni_aggregation_summary_drug_aligned.csv ] && mv -vn soragni_aggregation_summary_drug_aligned.csv sarcoma_organoids_2024_aggregation_summary_drug_aligned.csv   # NOT referenced - check before trusting
[ -e soragni_pathb_results.csv                  ] && mv -vn soragni_pathb_results.csv                  sarcoma_organoids_2024_pathb_results.csv         # referenced by the code
[ -e soragni_pathb_results_drug_aligned.csv     ] && mv -vn soragni_pathb_results_drug_aligned.csv     sarcoma_organoids_2024_pathb_results_drug_aligned.csv   # NOT referenced - check before trusting
[ -e soragni_query_pool.h5ad                    ] && mv -vn soragni_query_pool.h5ad                    sarcoma_organoids_2024_query_pool.h5ad           # referenced by the code
[ -e stack_input_soragni.h5ad                   ] && mv -vn stack_input_soragni.h5ad                   stack_input_sarcoma_organoids_2024.h5ad          # referenced by the code
[ -e stack_soragni.csv                          ] && mv -vn stack_soragni.csv                          stack_sarcoma_organoids_2024.csv                 # referenced by the code
[ -e stack_soragni.npy                          ] && mv -vn stack_soragni.npy                          stack_sarcoma_organoids_2024.npy                 # referenced by the code

echo; echo "after:"; ls -d *sarcoma_organoids_2024* 2>/dev/null | head -20
echo; echo "any soragni left?"; ls -d *soragni* 2>/dev/null || echo "  none - migration complete"

# data/raw/soragni was missed by the pass above (repo-root artifacts only). The loader hardcodes
# repo_root/data/raw/sarcoma_organoids_2024/tables (src/fmharness/data/loaders/sarcoma_organoids_2024.py:141)
# with no override, so without this rename load_sarcoma_organoids_2024() fails closed with
# "raw manifest missing" -- caught running rung 4 (job 31679380, 2026-08-26). Same mv -vn
# safety: never overwrites, guarded, no rm.
echo; echo "data/raw:"; ls -d data/raw/soragni data/raw/sarcoma_organoids_2024 2>/dev/null || true
[ -e data/raw/soragni ] && mv -vn data/raw/soragni data/raw/sarcoma_organoids_2024   # referenced by the code (loader)
echo; echo "after:"; ls -d data/raw/sarcoma_organoids_2024 2>/dev/null || echo "  (still missing)"
