# Modular harness core — design

**Date:** 2026-08-07
**Status:** approved, pending implementation plan
**ORPHANED, 2026-08-26:** the library layer here is implemented and correct. Its acceptance
driver, `check1_registry_driver.py`, is not invoked by any current sbatch — every promoted
rung-1 result comes from a parallel `rung1_plan/build_one/gather.py` path that does not run
`filter_leakage`. See `docs/PROJECT_SPEC.md`'s spec index before relying on this spec's
leakage guarantee.

## Purpose

Give the harness a model-agnostic, modality-agnostic core so a new foundation model or a new
phenotype target can be added without editing the driver scripts. This is the direct
operationalization of the two-score evaluation: an in silico
score now, a prospective score later, and the gap between them is the number that matters. The
companion manuscript (`greenelab/fm-pm-eval-manuscript`, "Prospective Evaluation of Foundation
Model Performance in Precision Medicine") names the concrete requirements this spec answers:
held-out axes (cohort, drug, organoid/subtype, model system), multiple benchmarked models
(Stack, STATE, X-Cell, and more), an "evaluation adapter" that scores new phenotypes and
modalities, and positive/negative controls.

## Context: reconciling three existing plans

Three planning artifacts already exist and partly disagree; this spec is what's left standing.

| artifact | what it got right | why it's not enough alone |
|---|---|---|
| `docs/fm-pdo-evaluator-plan.md` (original 3-week MVP plan, May 2026) | Designed a full `ModelAdapter` protocol, registry, leakage scan, container/determinism discipline. Most of the ideas this spec reuses trace back here. | Most of it was never built (Week 2/3 items are unstarted). Too much infrastructure to build before validating the core abstraction against a real second model. |
| `docs/superpowers/specs/2026-08-06-arm2-harness-validation-design.md` (yesterday) | Concrete, grounded 4-registry design (Representation/Estimator/CV/Readout) that already matches what's actually built (`docs/models.md`). Phases 0-6 with real acceptance criteria. | Scoped to Arm 2 (Tahoe/cell-line) only — explicitly excludes the Soragni/Arm-1 cohort and any cross-substrate work, which today's session did anyway at direction of the user. Assumes one model (Stack) and one representation shape (embedding); has no place for a model that generates. |
| `docs/adapter_contract.md` / `docs/models.md` | `adapter_contract.md` designed a clean `embed`/`metadata`/`predict_native` contract with required behaviors (determinism, no row reordering, gene-panel-mismatch refusal, content-addressed caching). `docs/models.md` documents what's actually running: a simpler representation × head split, `predict_parts` returning `(base, residual)`. | `adapter_contract.md`'s contract was never implemented. `docs/models.md`'s bilinear and biomarker models are explicitly "standalone — they do not use this head interface," so today's actual model catalogue does not fully conform to any single existing contract. |

This spec keeps yesterday's four registries as-is, extends them with what today's Arm-1/Arm-2
cross-substrate work and the manuscripts require, and reuses `adapter_contract.md`'s
already-designed pieces (`ModelMetadata`, required behaviors) rather than re-inventing them.

## What this spec covers

**In scope:** the model-level protocols (`Encoder`, `Generator`), the `Modality` registry,
leakage filtering, and the additions to yesterday's four registries needed to support all of
the above. This is "sub-project 1" of a three-part decomposition agreed during design:

1. **This spec.** The modular core.
2. **Tranches** (Pre-Tranche → Tranche 1/2/3, prospective prediction-locking). Manuscript's
   evaluation protocol layer, built on top of (1). Separate spec.
3. **Controls infrastructure beyond leakage** (systematic positive/negative controls across
   every model × modality pairing). Leakage was pulled into this spec because it biases
   results at the most fundamental level; the rest of (3) — e.g. running `plant_interaction`
   as standard practice on every new pairing — remains a separate spec, though the mechanism
   already composes for free (see "Controls already compose," below).

**Initial validation target:** the full harness (all registries below) built and validated
against **Stack + the existing baselines** (expr PCA/NMF, kernel head, bilinear, biomarker,
drug-mean) from `docs/models.md` — this is where the operational knowledge already is. A
**second, architecturally distinct model — scFoundation** — is the acceptance bar for the core
being genuinely modular, not just modular-in-name: non-Arc-Institute, encoder-**decoder**
(native generation, not a bolt-on alignment step like Stack's), and it explicitly claims both
GEARS-style perturbation prediction and DeepCDR drug-response prediction, so it exercises
`Encoder`, `Generator`, `context_coverage()`, and `LeakageQueryable` all at once. This is a
stress test of the registries, not a full scFoundation benchmark — enough integration to prove
the seams are in the right place.

`docs/models.md`'s Table 1 catalogue (Baek et al. 2025, "Single-cell foundation models bringing
artificial intelligence into cell biology") is the backlog for models to test later, organized
by relevance:

| category | models | relevance |
|---|---|---|
| explicit perturbation prediction | Geneformer, scGPT, scFoundation, GeneCompass | Stress-test the `Generator` path |
| embedding-only, architecturally diverse | scBERT, tGPT (decoder-only), CellPLM, UCE, Nicheformer | `Encoder`-only stress tests, no generation claim to hold them to |
| drug-response-specific downstream | scFoundation (DeepCDR), GeneCompass (DeepCDR) | Directly relevant to the `Modality` adapter side |

## The registries

### Model protocols: `Encoder` and `Generator`

Split rather than a single protocol with optional methods, so a model's type signature says
exactly what it can do — no `None`-checking, and `isinstance(model, Generator)` is how the
harness detects capability (both are `@runtime_checkable`).

```python
@runtime_checkable
class Encoder(Protocol):
    """Produces a per-sample representation vector."""
    def embed(self, adata: AnnData) -> np.ndarray:
        """(n_obs, embedding_dim), row-aligned to adata.obs_names. Deterministic."""
    def metadata(self) -> ModelMetadata: ...
    def version(self) -> str: ...

@runtime_checkable
class Generator(Protocol):
    """Produces a predicted post-perturbation profile."""
    def generate(self, baseline: AnnData, perturbation: str) -> AnnData:
        """Predicted profile for each row of `baseline` under `perturbation`. Same
        obs/var contract as `baseline`; full profile or delta, declared in metadata().
        Raises PerturbationNotInContext if `perturbation` cannot be represented."""
    def context_coverage(self, perturbations: Iterable[str]) -> set[str]:
        """Subset of `perturbations` this model can actually represent. Computed once
        per model, before generation runs -- not discovered as failures partway
        through an expensive GPU job."""
    def metadata(self) -> ModelMetadata: ...
    def version(self) -> str: ...
```

A model implements one, the other, or both by having the right methods present — Stack-base
only satisfies `Encoder`; Stack-aligned and scFoundation satisfy both.

`context_coverage()` is required on `Generator` specifically because generation is meaningless
without it: Tahoe-100M's own drug panel overlapped only 6 of sci-Plex's 187 compounds in the
Phase-1 leakage check, and this session's Tahoe↔GDSC2 join landed on ~30 of GDSC2's 403
compounds. Most drugs of clinical interest will not be in any given model's context. Without
this method, a model asked to generate for an out-of-context drug can silently extrapolate to
something that looks like a real prediction but isn't — confident-looking noise, worse than a
null result. Reporting coverage also sharpens the manuscript's "held-out drug" axis: a drug the
model *could* represent but was excluded from training (a real generalization test — the CV
registry's leave-drug-out) is different from a drug never in context at all (a coverage gap,
not a generalization result), and conflating them makes a generalization claim vacuous.

### `ModelMetadata`

Reused from `docs/adapter_contract.md` rather than redesigned, plus one addition:

| field | required | purpose |
|---|---|---|
| `pretraining_corpus` | yes | Free-form name (`"tahoe_100m"`, `"none"` for baselines) |
| `pretraining_cutoff_date` | yes | ISO date the corpus was frozen |
| `task_signal_in_pretrain` | yes | `"none"` / `"adjacent"` / `"direct"` — did the corpus contain drug-response labels |
| `model_weights_hash` | yes for FM models | sha256 of the loaded checkpoint |
| `expected_input` | yes | `"raw_counts"` / `"cpm"` / `"log1p_cpm"` — checked before dispatch |

`expected_input` is new. Phase 1's sci-Plex work hit exactly this gap: Stack's NB likelihood
silently breaks on normalized input, and the mismatch wasn't caught until diagnosis, not before
the job ran (`docs/superpowers/specs/2026-08-06-arm2-harness-validation-design.md`, Phase 1
blocker 3). Declaring it lets the harness validate at the boundary instead of downstream.

**Required behaviors** (from `adapter_contract.md`, restated for the `Encoder`/`Generator`
split rather than the single `ModelAdapter` they were originally written for): deterministic
output for identical input given a fixed seed; must not reorder rows (`adata.obs_names[i]`
corresponds to output row `i`); must raise `GenePanelMismatch` rather than silently realigning
when the input gene panel doesn't match what the model expects; must declare its container
digest for GPU models; embedding/generation caches keyed by content hash of input + model
version.

### Representation registry

Unchanged from yesterday's spec: `name -> (drug: str) -> DataFrame[line x feature]`.
`Encoder.embed()` output wraps into this via the pattern already proven for `--stack-emb`:
`lambda _drug: embedding_df` for a drug-independent entry. `Generator.generate()` output feeds
this too, as a **drug-dependent** entry: `generate(baseline, drug) - baseline` becomes a
per-drug delta, wrapped exactly the way `additive`/`knn`/`pca`/`nmf` already are — this is what
the existing `stack (gen delta)` row already does ad hoc; `Generator` formalizes it.

`Generator` output has a second consumer besides the Representation registry: a
**generation-fidelity readout** (Check-1-style — delta-Pearson, specificity rank against real
measured deltas). This has no `Estimator` or CV step; it's a property of the model alone, not
the model-plus-adapter pipeline.

### `Modality` registry (new)

The piece missing from yesterday's four registries. Today, "which phenotype are we predicting"
is hardcoded per script — `per_patient_eval.py`, `benchmark_sarcoma_organoids_2024.py`, `label_ceiling.py`,
and `score_generation_eval.py` each build their own `(patient, drug, y)` frame with their own
sign convention. This is what the manuscript's "evaluation adapter" language names.

```python
@runtime_checkable
class Modality(Protocol):
    def load(self, repo: Path) -> pd.DataFrame:
        """design[patient, drug, y] on this modality's native scale."""
    def direction(self) -> Literal["lower_is_better", "higher_is_better"]: ...
    def recommended_cv(self) -> str:
        """A CV-registry key sized to this modality's n (organoid: loo; cell-line: 5fold)."""
    def task_type(self) -> Literal["regression", "classification"]: ...
    def name(self) -> str: ...
```

Instances: `Gdsc2Auc`, `CtrpAuc`, `PrismAuc`, `SoragniViability`, later `PatientOutcome`.

**Boundary: substrate is a Representation concern, not a Modality one.** "Soragni tumor RNA
through Stack" and "Soragni organoid RNA through Stack" are two different `Representation`
instances aimed at the same Modality (Soragni viability) — not two modalities. This is the
distinction that was implicit and easy to get wrong in today's tumor-vs-organoid decomposition
(within-pair fidelity 0.86 vs. between-patient signal preservation 0.45-0.53); making it
explicit here prevents re-deriving it by hand each time.

**Classification modalities wrap a regression modality plus a threshold**, rather than
duplicating data-loading logic:

```python
class ThresholdedModality:
    """Wraps a regression Modality, emits binary y at a threshold."""
    def __init__(self, base: Modality, threshold: float, responder_is: Literal["below", "above"]): ...
    def task_type(self) -> Literal["classification"]:
        return "classification"
    # load() calls base.load() then thresholds; direction/recommended_cv delegate to base
```

E.g. `SoragniResponder = ThresholdedModality(SoragniViability(), threshold=50, responder_is="below")`.
This revives `sensitivity.py`'s binary responder/non-responder task from the original MVP plan,
which was scoped once (with `top_k_hit_rate`, `brier_score`, `expected_calibration_error` as
its metrics) and dropped when the more recent Arm-2 work stayed AUC/interaction-focused.

**Held-out axes fall out of `Modality` + the existing `CV` registry, with no new abstraction.**
Cohort and drug are already CV-registry variants (leave-line-out, leave-drug-out). Subtype is a
third variant, `LeaveSubtypeOut` — already designed and built once in the original MVP plan
(May 2026, verified end-to-end: 17 LPO + 11 LSO + 3 stratified folds on Soragni), just needs
porting into the CV registry table below. **Model-system** (train on cell-line, test on
organoid) falls out of composing two `Modality` instances: fit an `Estimator` on one Modality's
`(representation, label)` pairs, `predict_parts` against a different Modality's representation,
score with the same `Readout` — exactly what today's GDSC2→Soragni transfer work already did
by hand. A "Tranche" is then a thin, named bundle over `(Modality, CV scheme, allowed
representations)`, not a new registry — deferred to the Tranches spec (sub-project 2).

**`PooledModality`** (average GDSC2+CTRP+PRISM for shared lines, discussed as a reliability
lever earlier in design) follows the same wrapper pattern as `ThresholdedModality`. Noted as a
supported pattern; not built in this spec.

### `Estimator` registry

Unchanged contract from yesterday: `fit(features, drugs, y, groups)` / `predict_parts(...)`,
reusing `ProbeBase` (`src/fmharness/probe/`). **Resolved gap:** `docs/models.md` states bilinear
and biomarker are "standalone — they do not use this head interface." Decision: write thin
`Estimator`-conforming wrappers around both rather than leaving them outside the registry
system, since bilinear in particular is the mechanism that matters for future drug-coverage
questions (raised and tabled earlier in design — see "Out of scope").

### `CV` registry

Yesterday's three (leave-line-out, leave-drug-out, repeat-k-fold(seed)) plus **`LeaveSubtypeOut`**,
ported from the original MVP plan rather than rebuilt.

### `Readout` registry

Yesterday's four (raw-AUC gap@k, percentile-within-drug gap@k, MOA hit-rate@k,
interaction/global/per-drug rho) plus **classification readouts** for `task_type="classification"`
modalities: `top_k_hit_rate`, `brier_score`, `expected_calibration_error` — named in the
original MVP plan, never built.

**Testing standard, paid for by today's bug.** Every `Readout` implementation must ship a test
asserting it returns a null/expected value for a zero-information predictor scored against a
*realistically unbalanced* panel shape, not just a synthetic balanced one. This is the exact
gap that let `interaction_rho`'s missingness artifact ship undetected: a predictor with zero
cell-line information scored +0.118 on the real 44×30 Tahoe panel and +0.166 on the real
62%-filled Soragni panel, both purely from panel structure, both invisible on a balanced
synthetic test. `tests/test_evaluation.py`'s `test_interaction_rho_ignores_drug_only_signal_on_an_unbalanced_panel`
is the template.

### Leakage: `LeakageQueryable` and `filter_leakage`

Distinguished from `context_coverage()`, which looks similar but answers a different question:
coverage asks *can this model represent this perturbation* (capability); leakage asks *did this
model already see the answer during pretraining* (validity). A model can be fully capable of
representing a drug and still have memorized its specific response label.

```python
@runtime_checkable
class LeakageQueryable(Protocol):
    """Optional: a model that can expose what its pretraining corpus actually covered."""
    def pretraining_lines(self) -> set[str] | None: ...
    def pretraining_drugs(self) -> set[str] | None: ...

@dataclass
class LeakageProfile:
    line_overlap_frac: float | None
    drug_overlap_frac: float | None
    doubly_exposed_frac: float | None
    task_signal_in_pretrain: Literal["none", "adjacent", "direct"]
    basis: Literal["measured", "declared", "unknown"]

def filter_leakage(
    design: pd.DataFrame,       # [patient, drug, y]
    model: Encoder | Generator,
) -> tuple[pd.DataFrame, LeakageProfile]:
    """
    - Always drop doubly-exposed rows (line AND drug both in pretraining) -- the
      sharpest risk, matches Phase 1's actual exclusion (6 of 1,568 pairs, 0.4%).
    - If task_signal_in_pretrain == "direct" (model trained on actual response
      labels), also drop single-axis overlap (line OR drug).
    - If task_signal_in_pretrain in ("none", "adjacent"), single-axis overlap is
      reported but not hard-excluded -- a blanket single-axis filter would make a
      model pretrained on a broad public atlas (which will show large nominal line
      overlap with almost any cancer cohort) untestable on nearly anything.
    - If the model doesn't implement LeakageQueryable: filtered_design == design,
      profile.basis == "unknown". Filtering requires measured overlap; it never
      guesses clean.
    """
```

Filtering happens on the *whole panel* for a given model, before any CV split — not
fold-specific. If a line was in a model's own pretraining, it's invalid as a test case for that
model's representation quality regardless of which CV fold it would otherwise land in; this is
a different, earlier-stage contamination than ordinary train/test CV leakage (which the CV
registry's grouped splitting already handles).

**`basis="unknown"` policy:** the run proceeds — refusing to evaluate any model that doesn't
expose exact training-set membership would make most published FMs untestable — but every
downstream row for that model carries a visible "leakage: unfiltered" marker in any report.
Never silently dropped, never silently assumed clean.

**Scope drawn for this spec:** the interface and the "every result carries a profile, filtering
applied by default" requirement. Actually indexing a given model's pretraining corpus (building
a queryable line/drug set for Stack, scFoundation, etc.) is real data-wrangling work that stays
a follow-on, the same way concrete `Modality` subclasses are named here but not fully
implemented.

### Controls already compose

`permute_within_drug` and `plant_interaction`/`plant_response` (`src/fmharness/controls.py`)
operate on the same `(patient, drug, y)` design frame `Modality.load()` produces, so they work
against the new registries with no modification — the same way CV variants and Modality-swap
composed for free. What's still missing is running them as *standard practice* on every new
model × modality pairing (Arm 2 has never had a positive control run against it at all) — that
systematic coverage is sub-project 3, not a design gap in the core.

## Confirmed, no new capability needed

- **Pathway-level representations** (the Perspective paper's "the molecular level may be the
  wrong target for invariance" argument): a Hallmark/ssGSEA-scored representation is just
  another `Representation` entry — this is already how Check 1's fixed readouts work.
- **"World models" / causal structure** (the paper's `P(Y | do(X), context))` framing): such a
  model still just implements `Generator`. The protocol is agnostic to whether the model
  reasons causally or pattern-matches — it only constrains the interface, not the modeling
  philosophy behind it.

## Out of scope

- **Generalizing to genuinely novel compounds** outside a model's `context_coverage()` — via
  chemical similarity or any other mechanism. Same boundary the approved Arm-2 spec already
  drew around leave-drug-out ("the stress test, not the primary claim"). `context_coverage()`
  stays purely diagnostic.
- **Prospective prediction-locking** (hash/timestamp predictions before ground truth reveal) —
  sub-project 2 (Tranches).
- **Result provenance/versioning** — which code and registry-entry version produced a given
  number. The original MVP plan's `EnvironmentSnapshot`/`PredictionRecord` idea would have made
  "was this computed before or after the `interaction_rho` fix" automatic instead of the
  archaeology it actually required this session. Real value, bigger lift — a later spec, not
  this one.
- **Full leakage-corpus indexing** for every candidate model — the interface is in scope, the
  data engineering to populate it per-model is not.
- Extending `data/static/manifest.json` sha256 tracking to ad hoc comparator-screen downloads
  (today's CTRP/PRISM pull via `coderdata.download`) — minor, ops-level.

## Acceptance criteria

1. All four original registries (Representation, Estimator, CV, Readout) plus `Modality`,
   `Encoder`/`Generator`, and `filter_leakage` are implemented per the protocols above.
2. Re-running the existing Arm-2 ladder (`docs/tahoe_generation_results.md`) through the
   registries reproduces the published numbers, or every difference is explained and justified
   in writing — carried forward from yesterday's Phase 2 acceptance criterion.
3. Bilinear and biomarker models conform to `Estimator` via wrapper adapters; no model in
   `docs/models.md`'s catalogue remains outside the registry system.
4. scFoundation integrates through `Encoder`, `Generator`, `context_coverage()`, and (if it
   exposes enough to implement) `LeakageQueryable` without any protocol change — the modularity
   stress test. A full scFoundation benchmark is not required to close this spec.
5. Every `Readout` in the registry has a test against a zero-information predictor on an
   unbalanced synthetic panel.
6. Every result the harness produces carries a `LeakageProfile`, with `basis="unknown"` visibly
   marked wherever it occurs.
