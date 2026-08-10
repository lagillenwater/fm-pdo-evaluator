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
    def load(self, repo: Path) -> pd.DataFrame:
        """design[patient, drug, y] on this modality's native scale."""
        ...

    def direction(self) -> Direction: ...

    def recommended_cv(self) -> str:
        """A CV-registry key sized to this modality's n (organoid: loo; cell-line: 5fold)."""
        ...

    def task_type(self) -> TaskType: ...

    def name(self) -> str: ...


class Gdsc2Auc:
    """GDSC2 AUC. Lower AUC = more sensitive = better response."""

    def __init__(self, cancer_type_filter: list[str] | None = None) -> None:
        self.cancer_type_filter = cancer_type_filter

    def load(self, repo: Path) -> pd.DataFrame:
        bundle = cpm_bundle(
            load_tranche("gdscv2", repo, cancer_type_filter=self.cancer_type_filter)
        )
        _, design = build_sample_design(bundle, "all", "auc", drug_key="pubchem_cid")
        design = cast(pd.DataFrame, design.rename(columns={"y": "y"}))  # type: ignore[call-overload]
        return cast(pd.DataFrame, design[["patient", "drug", "y"]])

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

    def load(self, repo: Path) -> pd.DataFrame:
        bundle = cpm_bundle(load_tranche("ctrpv2", repo))
        _, design = build_sample_design(bundle, "all", "auc", drug_key="pubchem_cid")
        design = cast(pd.DataFrame, design.rename(columns={"y": "y"}))  # type: ignore[call-overload]
        return cast(pd.DataFrame, design[["patient", "drug", "y"]])

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

    def load(self, repo: Path) -> pd.DataFrame:
        bundle = cpm_bundle(load_tranche("prism", repo))
        _, design = build_sample_design(bundle, "all", "auc", drug_key="pubchem_cid")
        design = cast(pd.DataFrame, design.rename(columns={"y": "y"}))  # type: ignore[call-overload]
        return cast(pd.DataFrame, design[["patient", "drug", "y"]])

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

    def load(self, repo: Path) -> pd.DataFrame:
        bundle = cpm_bundle(load_tranche("sarcoma", repo))
        _, design = build_sample_design(bundle, self.rna_source, "viability")
        design = cast(pd.DataFrame, design.rename(columns={"y": "y"}))  # type: ignore[call-overload]
        return cast(pd.DataFrame, design[["patient", "drug", "y"]])

    def direction(self) -> Direction:
        return "lower_is_better"

    def recommended_cv(self) -> str:
        return "loo"

    def task_type(self) -> TaskType:
        return "regression"

    def name(self) -> str:
        return f"soragni_viability_{self.rna_source}"


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

    def load(self, repo: Path) -> pd.DataFrame:
        design = self.base.load(repo).copy()
        if self.responder_is == "below":
            design["y"] = (design["y"] < self.threshold).astype(float)
        else:
            design["y"] = (design["y"] > self.threshold).astype(float)
        return design

    def direction(self) -> Direction:
        return self.base.direction()

    def recommended_cv(self) -> str:
        return self.base.recommended_cv()

    def task_type(self) -> TaskType:
        return "classification"

    def name(self) -> str:
        return f"{self.base.name()}_responder_{self.threshold:g}"
