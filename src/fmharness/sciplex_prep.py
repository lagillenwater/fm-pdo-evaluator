"""Input validation for sci-Plex 3 data feeding Stack's drug-alignment fine-tune.

``scripts/build_sciplex_finetune.py`` reformats a sci-Plex AnnData for
``stack-finetune``; these checks catch the two failure modes found (2026-08-06) but never
enforced -- a pre-subset gene panel and non-raw-count input -- plus a name-collision
diagnostic for a third (upstream perturbation-name truncation). Kept importable and testable
independently of the CLI script, which only orchestrates argument parsing and I/O.
"""

from __future__ import annotations


def check_gene_count(n_vars: int, *, min_genes: int = 5000) -> None:
    """Raise if the input looks like a pre-subset HVG panel, not a near-full transcriptome.

    Stack's generation panel is 15,012 genes; fine-tuning the generation head on a much
    smaller subset (e.g. the chemCPA sci-Plex release's 2,000-gene HVG subset) and then
    generating over the full panel is a train/test mismatch. ``min_genes=5000`` is a loose
    floor -- well above 2,000, well below a full transcriptome (~20,000+) -- so it catches
    the known-bad case without demanding an exact match to Stack's panel size.
    """
    if n_vars < min_genes:
        raise SystemExit(
            f"input has only {n_vars} genes (< {min_genes}) -- looks like a pre-subset HVG "
            "panel, not a near-full transcriptome. Stack's generation panel is 15,012 genes; "
            "fine-tuning on a much smaller subset would be a train/test mismatch. Use a "
            "full-gene source instead (see scripts/alpine/08_sciplex_prep.sbatch's SCIPLEX_URL)."
        )
