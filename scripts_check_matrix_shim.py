"""Re-export check_matrix's matrix loader so report_variants shares ONE declaration reader.

Two readers of the same YAML drift, and a reporting tool that disagrees with the gate about
what is declared is worse than no reporting tool.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "_check_matrix", Path(__file__).resolve().parent / "scripts" / "check_matrix.py"
)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)

expected_for = _mod.expected_for
load_matrix = _mod.load_matrix
