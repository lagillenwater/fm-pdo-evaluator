"""``Generator``-protocol wrapper over Stack's already-generated output.

Stack generation runs on Alpine GPU (``scripts/alpine/04_stack_generate.sbatch``), writing
``<pert_id>.h5ad`` files under an output directory -- this class does not run inference; it
resolves a requested perturbation to its pre-generated file using the same rule
``fmharness.deltas.build_generated_deltas``'s file-matching already implements (filename
stem, then the same stem with spaces sanitized to underscores, matching how stack-generation
writes output), so a driver using only the ``Generator`` protocol reaches the identical
generation output the existing bulk-scoring path already validates.

Optionally ``LeakageQueryable`` (``pretraining_lines``/``pretraining_drugs``): both default
to ``None`` (no declared corpus, ``filter_leakage`` reports ``basis="unknown"``) so a
checkpoint whose overlap with the eval cohort has not been measured is never silently
assumed clean; pass real sets once a measurement exists (see Task 9).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import anndata as ad

from fmharness.model_protocols import PerturbationNotInContext
from fmharness.schema import ModelMetadata, TaskSignal


class PregeneratedStackGenerator:
    def __init__(
        self,
        generated_dir: Path,
        pert_to_drug: dict[str, str],
        *,
        checkpoint_label: str,
        pretraining_lines: set[str] | None = None,
        pretraining_drugs: set[str] | None = None,
        task_signal_in_pretrain: TaskSignal = "none",
    ) -> None:
        self.generated_dir = Path(generated_dir)
        self.pert_to_drug = pert_to_drug
        self.drug_to_pert = {v: k for k, v in pert_to_drug.items()}
        self.checkpoint_label = checkpoint_label
        self._pretraining_lines = pretraining_lines
        self._pretraining_drugs = pretraining_drugs
        self._task_signal_in_pretrain: TaskSignal = task_signal_in_pretrain

    def version(self) -> str:
        return f"stack-generated@{self.checkpoint_label}"

    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            pretraining_corpus=f"pregenerated:{self.checkpoint_label}",
            pretraining_cutoff_date=date(1970, 1, 1),
            task_signal_in_pretrain=self._task_signal_in_pretrain,
            expected_input="raw_counts",
        )

    def context_coverage(self, perturbations: object) -> set[str]:
        return {p for p in perturbations if p in self.drug_to_pert}  # type: ignore[attr-defined]

    def generate(self, baseline: ad.AnnData, perturbation: str) -> ad.AnnData:
        pert_id = self.drug_to_pert.get(perturbation)
        if pert_id is None:
            raise PerturbationNotInContext(
                f"{perturbation!r} has no pre-generated file (checkpoint "
                f"{self.checkpoint_label!r}) -- not in the declared pert_to_drug map"
            )
        path = self._resolve_file(pert_id)
        if path is None:
            raise PerturbationNotInContext(
                f"{perturbation!r} (pert_id {pert_id!r}) is declared but no matching file "
                f"exists under {self.generated_dir}"
            )
        return ad.read_h5ad(path)

    def _resolve_file(self, pert_id: str) -> Path | None:
        direct = self.generated_dir / f"{pert_id}.h5ad"
        if direct.exists():
            return direct
        sanitized = self.generated_dir / f"{pert_id.replace(' ', '_')}.h5ad"
        if sanitized.exists():
            return sanitized
        return None

    def pretraining_lines(self) -> set[str] | None:
        return self._pretraining_lines

    def pretraining_drugs(self) -> set[str] | None:
        return self._pretraining_drugs
