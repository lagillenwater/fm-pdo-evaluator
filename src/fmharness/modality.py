"""Modality registry: swappable phenotype targets.

Substrate (which RNA source feeds a representation -- tumor vs. organoid vs.
cell-line RNA) is a Representation concern, not a Modality one: "Soragni tumor
RNA through Stack" and "Soragni organoid RNA through Stack" are two different
representations aimed at the *same* Modality (Soragni viability). Modality owns
only the label side: which dataset, which metric, which sign convention.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol, cast, runtime_checkable

import pandas as pd

from fmharness.data.loaders import load_tranche
from fmharness.evaluation import build_sample_design, cpm_bundle

Direction = Literal["lower_is_better", "higher_is_better"]
TaskType = Literal["regression", "classification"]


@runtime_checkable
class Modality(Protocol):
    """A swappable phenotype target: the label side of an evaluation.

    A Modality owns only which dataset supplies the response, which metric it is
    measured in, and which sign convention makes a value "good". It deliberately
    does NOT own the substrate -- which RNA source feeds the representation
    (tumor vs. organoid vs. cell-line RNA) is a Representation concern, so
    "Soragni tumor RNA through Stack" and "Soragni organoid RNA through Stack"
    are two representations aimed at this same Modality. Keeping that boundary
    is what lets a representation sweep and a target sweep vary independently.
    """

    def load(self, repo: Path) -> pd.DataFrame:
        """design[patient, drug, y] on this modality's native scale."""
        ...

    def load_with_features(self, repo: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
        """(features, design) from a single underlying load.

        ``load()`` alone only needs the design, but a driver that also needs
        the expression matrix (fitting an Estimator on it) would otherwise have
        to load and renormalize the same tranche a second time by hand. This
        returns both from one internal load; ``load()`` is a thin wrapper over
        this method's design half.
        """
        ...

    def direction(self) -> Direction: ...

    def recommended_cv(self) -> str:
        """A CV-registry key sized to this modality's n (organoid: loo; cell-line: 5fold)."""
        ...

    def task_type(self) -> TaskType: ...

    def name(self) -> str: ...


class Gdsc2Auc:
    """GDSC2 AUC. Lower AUC = more sensitive = better response.

    NOTE on ``cancer_type_filter``: it is NOT an arbitrary cancer-type slice. The
    native GDSC2 loader supports one restriction -- sarcoma lineages -- and
    ``load_tranche`` (``data/loaders/adapt.py``) treats any non-None value here
    purely as the on/off signal for it, ignoring the list's contents. So
    ``Gdsc2Auc(cancer_type_filter=["Lung"])`` returns SARCOMA lines, not lung
    lines. Pass ``None`` (the default) for the full pan-cancer panel, or any
    non-None value (e.g. ``["sarcoma"]``, for readability) for sarcoma only.
    """

    def __init__(self, cancer_type_filter: list[str] | None = None) -> None:
        """``cancer_type_filter``: non-None restricts to sarcoma lineages
        regardless of its contents -- see the class docstring and
        ``load_tranche``. None (default) keeps the full cell-line panel."""
        self.cancer_type_filter = cancer_type_filter

    def load_with_features(self, repo: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
        bundle = cpm_bundle(
            load_tranche("gdscv2", repo, cancer_type_filter=self.cancer_type_filter)
        )
        x_df, design = build_sample_design(bundle, "all", "auc", drug_key="pubchem_cid")
        return cast(pd.DataFrame, x_df), cast(pd.DataFrame, design[["patient", "drug", "y"]])

    def load(self, repo: Path) -> pd.DataFrame:
        return self.load_with_features(repo)[1]

    def direction(self) -> Direction:
        return "lower_is_better"

    def recommended_cv(self) -> str:
        return "5fold"

    def task_type(self) -> TaskType:
        return "regression"

    def name(self) -> str:
        return "gdsc2_auc"


class CtrpAuc:
    """CTRPv2 AUC via CoderData (``data/raw/coderdata``). Lower = more sensitive."""

    def load_with_features(self, repo: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
        bundle = cpm_bundle(load_tranche("ctrpv2", repo))
        x_df, design = build_sample_design(bundle, "all", "auc", drug_key="pubchem_cid")
        return cast(pd.DataFrame, x_df), cast(pd.DataFrame, design[["patient", "drug", "y"]])

    def load(self, repo: Path) -> pd.DataFrame:
        return self.load_with_features(repo)[1]

    def direction(self) -> Direction:
        return "lower_is_better"

    def recommended_cv(self) -> str:
        return "5fold"

    def task_type(self) -> TaskType:
        return "regression"

    def name(self) -> str:
        return "ctrp_auc"


class PrismAuc:
    """PRISM AUC via CoderData (``data/raw/coderdata``). Lower = more sensitive."""

    def load_with_features(self, repo: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
        bundle = cpm_bundle(load_tranche("prism", repo))
        x_df, design = build_sample_design(bundle, "all", "auc", drug_key="pubchem_cid")
        return cast(pd.DataFrame, x_df), cast(pd.DataFrame, design[["patient", "drug", "y"]])

    def load(self, repo: Path) -> pd.DataFrame:
        return self.load_with_features(repo)[1]

    def direction(self) -> Direction:
        return "lower_is_better"

    def recommended_cv(self) -> str:
        return "5fold"

    def task_type(self) -> TaskType:
        return "regression"

    def name(self) -> str:
        return "prism_auc"


class SoragniViability:
    """Soragni organoid Viability_Score (% of vehicle). Lower = more sensitive."""

    def __init__(self, rna_source: Literal["tumor", "organoid", "all"] = "tumor") -> None:
        self.rna_source = rna_source

    def load_with_features(self, repo: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
        bundle = cpm_bundle(load_tranche("sarcoma", repo))
        x_df, design = build_sample_design(bundle, self.rna_source, "viability")
        return cast(pd.DataFrame, x_df), cast(pd.DataFrame, design[["patient", "drug", "y"]])

    def load(self, repo: Path) -> pd.DataFrame:
        return self.load_with_features(repo)[1]

    def direction(self) -> Direction:
        return "lower_is_better"

    def recommended_cv(self) -> str:
        return "loo"

    def task_type(self) -> TaskType:
        return "regression"

    def name(self) -> str:
        return f"sarcoma_organoids_2024_viability_{self.rna_source}"


class ThresholdedModality:
    """Wraps a regression Modality, emits binary y at a threshold.

    Reuses the base Modality's exact data-loading path -- no duplicated
    normalization or join logic -- so a classification target is always
    derived from, and stays consistent with, its regression counterpart.
    """

    def __init__(
        self,
        base: Modality,
        threshold: float,
        responder_is: Literal["below", "above"],
    ) -> None:
        self.base = base
        self.threshold = threshold
        self.responder_is = responder_is

    def load_with_features(self, repo: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
        x_df, design = self.base.load_with_features(repo)
        design = design.copy()
        if self.responder_is == "below":
            design["y"] = (design["y"] < self.threshold).astype(float)
        else:
            design["y"] = (design["y"] > self.threshold).astype(float)
        return x_df, design

    def load(self, repo: Path) -> pd.DataFrame:
        return self.load_with_features(repo)[1]

    def direction(self) -> Direction:
        """Always ``higher_is_better``: y is now 1 == responder.

        The base's direction applies to its continuous scale (GDSC2 AUC is
        lower_is_better), but thresholding has already absorbed that sign into
        the label -- ``responder_is`` decides which tail becomes 1 -- so a higher
        predicted responder probability is unconditionally the better outcome.
        Delegating here would flip the sign of every metric on a
        lower_is_better base.
        """
        return "higher_is_better"

    def recommended_cv(self) -> str:
        return self.base.recommended_cv()

    def task_type(self) -> TaskType:
        return "classification"

    def name(self) -> str:
        """Includes ``responder_is``: the same base and threshold with opposite
        tails are semantically opposite targets and must not share a name."""
        return f"{self.base.name()}_responder_{self.responder_is}_{self.threshold:g}"
