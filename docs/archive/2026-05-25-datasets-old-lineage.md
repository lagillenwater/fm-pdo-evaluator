# ARCHIVED — old-lineage dataset notes (2026-05-25)

Preserved verbatim from the pre-rebuild development lineage (its `docs/datasets.md`, which was
never committed anywhere — this copy was the only one). Superseded as a registry by
[`docs/DATA.md`](../DATA.md). The loaders, cohorts, and crosswalks described below belong to the
archived lineage and are not landed here; these notes get reconciled into the registry entries
of the rungs that read these datasets, when those rungs arrive.

---

# Datasets — Soragni and GDSC2 sarcoma

Decisions made at loader / crosswalk time, recorded here so the substrate-gap
interpretation in the headline report has a single auditable source of truth.

## Soragni 2024 — PDTO sarcoma biobank

- **Source:** Al Shihabi et al., *Cell Stem Cell* 2024. Synapse project
  `syn55180195` (synapse.org/PDTOSarcoma). Controlled-access Synapse DUA.
- **Expression:** pre-computed normalized gene counts at Synapse Table
  `syn64333318`. 39,342 genes × 38 valid sample columns (Tumor + Organoid
  pairs across 19 patients; plus two empty `col` / `col1` stragglers that
  the loader silently drops). Gene IDs are Ensembl (ENSG...) with
  Gene_Symbol + biotype as supplementary `var` columns. Normalization is
  median-of-ratios per the Soragni 2024 protocol; accepted upstream, no
  local re-quantification. Biotype mix: 18,516 protein-coding + 12,549
  lncRNA + various pseudogenes / RNA classes — full set retained,
  protein-coding filter is a downstream concern.
- **Drug screen:** Synapse Table `syn61892224`. 1,350 rows; 94 unique
  patients × 34 unique drugs (very sparse — Soragni screened each PDTO
  against a custom small panel of drugs relevant to its tumor type).
  Response metric is `Viability_Score` = % of vehicle control (range
  ~0.3–315, median ~98 → close to control). The screen is performed on
  the **organoid** specifically; the loader attaches each `DrugAssay` row
  to the Organoid `sample_id`, leaving Tumor samples baseline-only.
- **WES:** `sample_info` / `snv` / `sv` / `cnv` tables (Table1_a–d in the
  paper) — 15 patients, used for positive-control biomarker pairs.
- **Usable matched cohort (drug × RNA):** **17 patients** (drug screen
  intersected with RNA-seq sample availability — verified end-to-end
  2026-05-25 by the live loader; plan's "14 patients" was an underestimate).
  Per-patient drug coverage varies wildly (5–24 drugs per patient), so the
  panel is *sparse*; **26** unique drugs screened across the matched
  cohort (the other 8 of 34 panel drugs were screened only on non-matched
  patients).
- **`DrugAssay` row count:** 276 (one per matched-cohort `(organoid, drug)`).
- **Sample-ID conventions handled by `canonicalize_patient_id`:**
  - Drug screen: lowercase `sarcNNNN` and `sarcNNNN_<digits|letters>`
    (e.g. `sarc0001`, `sarc0024_B`, `sarc0028_biopsy`, `sarc0053_a`)
  - WES / sample_info: uppercase `SARCNNNN[_<digits>]`
  - RNA-seq columns: `SARCNNNN[_<digits>]_Tumor|_Organoids`
  - All canonicalize to `SARC<digits>[_<digits|letters>]` (upper-case)
  - `SARC0139_1` and `SARC0139_2` are treated as **separate patients**
    (two timepoints; the data does not mark them as the same physical
    person — being conservative)
- **Subtype distribution of the 17-patient cohort** (free-text from drug
  screen `Diagnosis`):

  | Subtype | Patients |
  |---|---:|
  | osteosarcoma | 4 |
  | leiomyosarcoma | 3 |
  | rhabdomyosarcoma | 2 |
  | 8 singleton subtypes (MPNST, chondrosarcoma, CIC-rearranged, epithelioid, well/dediff liposarcoma, spindle cell, synovial) | 1 each |

  Day-6 LSO splits will need coarse-graining to be well-powered — same
  story as the GDSC2 28-line cohort.

## GDSC2 sarcoma — cell-line drug-response substrate

- **Drug response source:** GDSC2 release 8.5 (Sanger Institute, 27 Oct 2023).
  Open access via `https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.5/`.
  295 compounds screened across ~970 cell lines (all tissues).
- **Expression source:** DepMap 26Q1 `OmicsExpressionRawReadCountHumanProteinCodingGenes.csv`
  (RSEM raw read counts, per protein-coding gene). The loader runs
  `pydeseq2` median-of-ratios on the sarcoma subset to match Soragni's
  normalization scheme.
- **Cell-line crosswalk:** DepMap `Model.csv` provides the ACH↔COSMIC
  mapping plus `OncotreeLineage` / `OncotreeSubtype` columns.

### Identifier-alignment chain (verified end-to-end 2026-05-25)

```
Model.csv (DepMap, 2,154 rows)
  filter OncotreeLineage ∈ {Soft Tissue, Bone}
        ↓
  200 sarcoma cell lines (ACH IDs)
        ↓ Model.csv has COSMICID for 56 of them
  56 ACH IDs with a COSMIC crosswalk
        ↓ inner-join GDSC2 dose-response on COSMIC_ID
  55 cell lines present in DepMap Model.csv + GDSC2 screen
        ↓ additional filter: must have RNA-seq in DepMap (IsDefaultEntryForModel="Yes")
  28 cell lines in BOTH DepMap RNA-seq AND GDSC2 screen
```

**Final usable sarcoma cohort: 28 cell lines × 295 drugs × 2 metrics = 14,746 DrugAssay rows.**
DepMap RNA-seq coverage of sarcoma is the rate-limiting filter — only 28 of
the 56 sarcoma cell lines with a COSMIC crosswalk also have RNA-seq in the
26Q1 release. Original plan estimate (~25-35 lines) was accurate.

### Subtype distribution of the 28-line cohort

| Subtype | Lines |
|---:|---:|
| Osteosarcoma | 10 |
| Ewing Sarcoma | 6 |
| Leiomyosarcoma | 2 |
| Embryonal Rhabdomyosarcoma | 2 |
| 8 singleton subtypes (Synovial, Chondrosarcoma variants, Rhabdo, Fibrosarcoma, etc.) | 1 each |

Implication for splits (Day 6): coarse-grained LeaveSubtypeOut (Osteo vs
Ewing vs everything-else) is the only well-powered LSO axis. Fine-grained
LSO would leave 8 of 12 subtypes with zero training data and collapse toward
leave-patient-out for the rare ones.

## Drug-panel crosswalk

Generated by `scripts/build/build_drug_xref.py` from the two raw drug lists
(`screened_compounds_rel_8.5.csv` for GDSC2; the `drug_screen` Synapse Table
for Soragni). Output: `data/static/drug_xref.parquet`, sha256-tracked in
`data/static/manifest.json`, loaded at runtime by `fmharness.data.drug_xref`.

Canonical drug ID is the **PubChem CID** (resolved via PubChem PUG REST).
InChIKey and DrugBank ID resolved as secondary identifiers (UniChem for
DrugBank).

### Panel overlap (built 2026-05-25)

Re-derive at any time with `fmharness.data.drug_xref.overlap_report(xref)`.

- GDSC2 raw drug entries: 621 (the screened-compounds file has multiple
  `DRUG_ID`s per drug name across screening campaigns)
- Soragni panel size: 34 unique drugs
- **GDSC2:** 486/621 entries resolved to a PubChem CID → 403 unique drugs
- **Soragni:** 34/34 resolved to a PubChem CID (100%)
- **Drugs present in BOTH Soragni and GDSC2: 21**
- Soragni-only drugs (resolved, no GDSC2 match): 13
- GDSC2-only entries with no resolvable PubChem CID: 135 (mostly
  research-coded compounds with no PubChem entry: LGK974, Wnt-C59, etc.)

### Which panel slice for which evaluation

| Evaluation context | Drug panel | Rationale |
|---|---|---|
| Within-Soragni matrix cells (per-model × per-split) | All **34** Soragni drugs | Soragni tests sarcoma-relevant biology under PDTO biology; use the full panel for statistical power |
| Within-GDSC2 matrix cells (per-model × per-split) | All **403** unique GDSC2 drugs (post-CID resolution) | GDSC2 tests broader drug-response under cell-line biology; the full panel exercises the model across the diversity it was trained on |
| **Cross-substrate rank correlation** (Soragni rankings ↔ GDSC2 rankings) | The **21 shared drugs** | Apples-to-apples comparison requires the same drug set on both sides; the shared subset is the only fair-comparison surface |
| Substrate-gap headline cells | The 21 shared drugs | Same reason |
| Leakage scan (Tahoe-100M drug-overlap fraction) | All resolved CIDs from both panels | The leakage question is "how much of my evaluation surface did the pretraining see" — count over the full surface |

The within-dataset cells are not directly comparable across the two datasets
(they measure different things on different drug surfaces); they're the
per-dataset rows of the 3×3×2 matrix. The cross-substrate comparison is the
headline finding and lives in the rank-correlation table.

### Shared drugs (the 21-drug substrate-gap candidate set)

| GDSC2 name | Soragni name | DrugBank |
|---|---|---|
| Topotecan | Topotecan | DB01030 |
| Gemcitabine | Gemcitabine | DB00441 |
| Gefitinib | Gefitinib | DB00317 |
| Docetaxel | Docetaxel | DB01248 |
| Lapatinib | Lapatinib | DB01259 |
| Sorafenib | Sorafenib | DB00398 |
| Dasatinib | Dasatinib | DB01254 |
| Rapamycin | Rapamycin | DB00877 |
| Vinorelbine | Vinorelbine | DB00361 |
| Palbociclib | Palbociclib | DB09073 |
| Panobinostat | Panobinostat | DB06603 |
| OSI-930 | OSI-930 | DB05913 |
| Cediranib | Cediranib | DB04849 |
| Pazopanib | Pazopanib | DB06589 |
| BI-2536 | BI 2536 | DB16107 |
| Crizotinib | Crizotinib | DB08865 |
| Linsitinib | Linsitinib | — |
| Trametinib | Trametinib | DB08911 |
| Olaparib | Olaparib | DB09074 |
| Cabozantinib | Cabozantinib | DB08875 |
| Ruxolitinib | Ruxolitinib | DB08877 |

Notable: Rapamycin (mTOR inhibitor) is in the shared set — gives the plan's
mTOR/sirolimus positive-control pairing a direct cross-substrate test.

### Positive-control biomarker–drug pairings (Day 12 finalization)

Per plan §12, the loader pre-resolves candidate pairs at load time so the
controls module can pick from them. Initial candidates (to be filtered to
those actually present in the panel + cohort):

- **Soragni** — NTRK/larotrectinib, EZH2/tazemetostat, mTOR/sirolimus
- **GDSC2 sarcoma** — MDM2-amplified dedifferentiated liposarcoma /
  nutlin-3a; SS18-SSX synovial sarcoma / EZH2 inhibitor

## Subtype taxonomy alignment (Soragni × GDSC2)

Soragni's clinical labels (free-text `Diagnosis` column) vs DepMap's
`OncotreeSubtype` use different ontologies. Crosswalk table will live at
`data/static/sarcoma_subtype_xref.csv` (Day 4-5). Coarse-grained categories
(leiomyosarcoma, liposarcoma, osteosarcoma, Ewing, rhabdomyosarcoma,
synovial, fibrosarcoma, chondrosarcoma, chordoma) are unambiguous;
finer-grained subtypes will need collapsing to a shared granularity.
