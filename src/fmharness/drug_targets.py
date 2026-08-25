"""Known-biology positive control: each drug's own molecular target gene, as a
hand-curated PubChem-CID -> HGNC-symbol map for the 26 drugs actually screened in the
Soragni sarcoma viability assay (``fmharness.data.loaders.load_tranche("sarcoma", ...)``,
``build_sample_design(..., drug_key="pubchem_cid")`` -- confirmed 2026-08-22 by running
that loader directly, not guessed from a name list).

Rationale (target-dependency hypothesis): a cancer's dependence on its drug's own
molecular target is a well-established, independently-motivated prior for sensitivity --
distinct from this harness's other two "does a real signal get detected" checks: the
measured_delta/real-delta ceiling (a data-derived VALIDATION, not a control -- real data, no planted
effect size) and ``fmharness.controls.plant_interaction`` (the flowchart's actual positive
control: a SIMULATED signal with a known, controlled effect size). This one is a hand-picked,
externally-justified predictor from real pharmacology, not real data or a simulation. Higher
target-gene expression is hypothesized to track GREATER dependence on that gene/pathway,
hence greater sensitivity to a drug that inhibits it (direction=+1 in
``score_target_gene_predictors``'s sense: higher expression -> lower AUC).

Caveats (read before trusting a result built on this):
  * One PRIMARY target per drug, not the full target/off-target profile -- multi-target
    TKIs (pazopanib, dovitinib, sorafenib, cabozantinib, lenvatinib) inhibit several
    kinases; the gene here is the most commonly cited primary target, not the only one.
  * The target-dependency hypothesis is well-supported for oncogene-addicted targeted
    agents (EGFR/ALK/CDK4/BCR-ABL inhibitors, etc.) but much weaker/mixed for classical
    cytotoxic chemotherapies (docetaxel, vinorelbine, topotecan, gemcitabine) -- their
    mechanism (microtubule/topoisomerase/nucleotide-analog poisoning) is not really an
    "addiction to an overexpressed target" story the same way. Included for completeness,
    flagged here so a null result on those specific drugs isn't over-interpreted as a
    harness bug.
  * PubChem CID identity was checked against a file that is NOT tracked here (data/ ships only
    data/static/), so this is unverified from the repo alone:
    ``data/raw/coderdata/sarcoma_drugs.tsv.gz`` synonym list where present (22/26); the
    remaining 4 (sirolimus, ceralasertib, topotecan -- degrasyn was in the synonym list)
    were confirmed via a live PubChem CID lookup, not recalled from memory alone.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from fmharness.evaluation import score_predictions

# CID (str, PubChem) -> primary molecular target gene (HGNC symbol).
DRUG_TARGET_GENES: dict[str, str] = {
    "10113978": "KDR",  # pazopanib -- VEGFR2 (multi-target anti-angiogenic TKI)
    "11222830": "USP9X",  # degrasyn / WP1130 -- deubiquitinase inhibitor
    "11442891": "AURKA",  # danusertib -- pan-Aurora kinase inhibitor
    "11556711": "PSMB5",  # carfilzomib -- 20S proteasome beta5 subunit
    "11626560": "ALK",  # crizotinib -- ALK/MET/ROS1 (ALK is the namesake target)
    "11640390": "IGF1R",  # linsitinib
    "11707110": "MAP2K1",  # trametinib -- MEK1
    "123631": "EGFR",  # gefitinib
    "135398510": "FGFR1",  # dovitinib -- multi-target TKI (FGFR/VEGFR/PDGFR/KIT/FLT3)
    "148124": "TUBB1",  # docetaxel -- taxane, microtubule stabilizer
    "208908": "ERBB2",  # lapatinib -- EGFR/HER2 dual inhibitor
    "216239": "BRAF",  # sorafenib -- multi-target TKI (RAF/VEGFR2/PDGFRB)
    "23725625": "PARP1",  # olaparib
    "25102847": "MET",  # cabozantinib -- multi-target TKI (MET/VEGFR2/RET/AXL)
    "25126798": "JAK2",  # ruxolitinib -- JAK1/2
    "3062316": "ABL1",  # dasatinib -- BCR-ABL/SRC-family
    "5284616": "MTOR",  # sirolimus (rapamycin)
    "5311497": "TUBB1",  # vinorelbine -- vinca alkaloid, microtubule destabilizer
    "5330286": "CDK4",  # palbociclib -- CDK4/6
    "54761306": "ATR",  # ceralasertib
    "60700": "TOP1",  # topotecan
    "60750": "RRM1",  # gemcitabine -- ribonucleotide reductase
    "6442177": "MTOR",  # everolimus (RAD001)
    "6918837": "HDAC1",  # panobinostat -- pan-HDAC inhibitor
    "9823820": "KDR",  # lenvatinib -- multi-target TKI (VEGFR2/3/FGFR1/PDGFRB/KIT/RET)
    "9865515": "HDAC1",  # mocetinostat -- class I HDAC inhibitor
}


def score_target_gene_predictors(
    design: pd.DataFrame,
    baseline: pd.DataFrame,
    targets: dict[str, str] | None = None,
    *,
    n_perm: int = 1000,
    seed: int = 0,
) -> dict[str, float]:
    """Known-biology positive control: each drug's own target gene's baseline expression
    as a sensitivity predictor (target-dependency hypothesis, see module docstring).

    ``design`` is (patient, drug, y=AUC); ``baseline`` is patient x gene expression
    (e.g. the same tumor-RNA baseline every other Path-B source is built from). Only
    (patient, drug) pairs whose drug has a known target IN ``baseline``'s columns are
    scored -- pools across every covered drug into ONE score via
    ``fmharness.evaluation.score_predictions``, the same within-drug interaction rho +
    label-permutation null every other source in this harness uses. Returns an empty-n
    dict (not a raise) if no drug's target gene is covered, since that's a legitimate
    "this cohort's panel doesn't include this control's genes" outcome, not an error.
    """
    tmap = DRUG_TARGET_GENES if targets is None else targets
    patient = design["patient"].astype(str).reset_index(drop=True)
    drug = design["drug"].astype(str).reset_index(drop=True)
    y = design["y"].reset_index(drop=True)
    gene_of = drug.map(lambda d: tmap.get(d))
    known_cols = list(baseline.columns)
    covered = drug.isin(list(tmap)) & gene_of.isin(known_cols) & patient.isin(list(baseline.index))
    if not bool(covered.any()):
        return {"global": float("nan"), "interaction": float("nan"), "n": 0.0}
    patient, drug, y, gene_of = (
        cast("pd.Series", patient[covered]).reset_index(drop=True),
        cast("pd.Series", drug[covered]).reset_index(drop=True),
        cast("pd.Series", y[covered]).reset_index(drop=True),
        cast("pd.Series", gene_of[covered]).reset_index(drop=True),
    )
    expr = np.array(
        [baseline.loc[p, g] for p, g in zip(patient, gene_of, strict=True)], dtype=np.float64
    )
    preds = pd.DataFrame(
        {
            "patient": patient.to_numpy(),
            "drug": drug.to_numpy(),
            "y_true": y.to_numpy(dtype=np.float64),
            "y_pred": -expr,  # higher target expression -> predicted more sensitive -> lower AUC
        }
    )
    return score_predictions(preds, n_perm=n_perm, seed=seed)
