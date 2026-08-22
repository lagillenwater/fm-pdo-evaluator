# Path B (Soragni sarcoma organoids) — results

**Question.** Does a Stack-generated, patient-specific transcriptome predict real Soragni
organoid drug-response (viability AUC) better than simple non-FM baselines — and can the
harness's own controls be trusted before spending the much longer leave-one-line-out (LOO)
compute on Check 1/2's cell-line reruns?

**Design.** Per (patient, drug), a treated-minus-baseline delta -- `additive` (drug-mean,
patient-independent floor), `pca`/`nmf` (learned baselines, CV-tuned `n_components`), and
`stack` (Stack-generated delta, `--mode mdm` faithful generation). Every source scored on
the SAME (patient, drug) support (`restrict_common_support`) through the same readouts:
Hallmark (broken out per signature, not averaged), and CV-tuned penalized regression
(l1/l2). Two Stack checkpoints: cytokine-aligned (default) and drug-aligned (sci-Plex
fine-tuned). Two files: `soragni_pathb_results.csv` (cytokine-aligned) and
`soragni_pathb_results_drug_aligned.csv` (drug-aligned).

This pass ran the CHEAP 5-fold penalized-grid CV only (the `additive`/`pca`/`nmf` sources
built once, not leave-one-line-out) -- specifically to validate the new controls before
committing to the much longer Check 1/2 Tahoe/GDSC2 LOO reruns, which now also CV-tune
`n_components`/`k` per held-out line (multiplicatively more expensive).

---

## Controls (2026-08-22)

Every control below is scored via the same `fmharness.check2.penalized_preds` pipeline as
the real sources, so a control's behavior is directly comparable to a real row.

| control | purpose | cytokine-aligned | drug-aligned |
|---|---|---|---|
| **`planted`** (l1/l2) | positive: a KNOWN, controlled-effect-size interaction planted into the tumor-RNA baseline (`fmharness.controls.plant_interaction`) -- must recover, or the fold structure/model/metric itself is broken | interaction **0.536**/0.510, p_label **0.000**/0.000 | identical (same design/baseline, checkpoint-independent) |
| **`*_random`** (l1/l2) | negative: same-width i.i.d. Gaussian noise per source, fit AND applied on matching-shape noise -- must NOT show real signal | l2 near-null (−0.20 to 0.20); l1 identical across sources (LassoCV correctly zeros all coefficients on pure noise, converging to the same "predict the mean" answer regardless of which noise) -- one cell (`pca_random`/l2) at p=0.013, expected multiple-comparison noise across ~8 tested cells, not a flag | same pattern |
| **`target_gene`** | known-biology positive control: each of the 26 screened drugs' own molecular target gene's baseline expression (`fmharness.drug_targets`) | interaction 0.010, p_label 0.282 (n=237) -- null, consistent with this being a much weaker/messier hypothesis than a guaranteed-by-construction planted signal, especially given several of the 26 drugs are classical cytotoxics (docetaxel, gemcitabine, topotecan, vinorelbine) where target-expression dependence is a poor fit to begin with | same |
| oracle/ceiling (validation) | N/A for Path B -- no real treated-organoid RNA-seq exists to build a ceiling from (only the untreated tumor-RNA baseline); Check 1/2's Tahoe/GDSC2 pipelines have this via `score_check2`'s `oracle=` | -- | -- |

**Conclusion: the controls behave as a trustworthy harness should.** The positive control
recovers cleanly and significantly; the negative controls are null modulo ordinary
multiple-comparison noise. Two real bugs were caught and fixed during this validation pass
(fixed random-control seeding causing identical "independent" draws; a planted signal that
couldn't be recovered because it was planted in a PCA subspace but scored on the raw
high-dimensional gene space with far too few training patients per fold to fit it) --
exactly the kind of problem this staged check-before-LOO-rerun step exists to catch.

---

## Main comparison (both checkpoints, l1/l2 + Hallmark-per-signature)

Full detail in the two CSVs. Headline: still a null result for Stack, consistent with the
2026-08-20/21 corrected (post-support-fix) finding -- `stack`+hallmark interaction stays in
the 0.01-0.09 range for most signatures, not exceeding the `additive`/`pca`/`nmf` baselines
by a margin that clears `p_label < 0.05` (cytokine-aligned). The drug-aligned checkpoint
shows two nominally significant Hallmark cells (`stack`+`HALLMARK_P53_PATHWAY` interaction
0.889 p=0.005; `HALLMARK_APOPTOSIS` interaction 0.926 p=0.002, both n=25) -- flagged, not
yet trusted: n=25 is small, this table doesn't yet carry the random-GENE-SET control
(`fmharness.signatures.score_signatures`'s `rnd_p95`/`p_vs_random`, distinct from this
pass's random-FEATURE control) that would show whether this is Hallmark-specific biology or
a generic perturbation-magnitude artifact -- the same check the Tahoe pipeline's "Gate"
already does and Path B doesn't yet. Worth a follow-up, not a headline claim yet.

---

## Next

Controls validated -- proceeding to the expensive Check 1/2 Tahoe/GDSC2 LOO reruns (now with
CV-tuned `n_components`/`k` per held-out line for `additive`/`knn`/`pca`/`nmf`, and the same
new controls: `oracle`, `planted`, per-representation `random`).
