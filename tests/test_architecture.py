"""The module boundaries are a design constraint, so they get enforced.

  brain/  connectome + LIF        - no MuJoCo, no FlyGym, no body/bridge
  body/   physics                 - no torch, no brain/bridge
  env/    presets                 - no brain/body/bridge
  bridge/ the only module allowed to see both a brain and a body
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FORBIDDEN = {
    "brain": ("mujoco", "flygym", "body", "bridge"),
    "body": ("torch", "brain", "bridge"),
    "env": ("brain", "body", "bridge"),
}


def _imports(path: Path) -> set[str]:
    pattern = re.compile(r"^\s*(?:from|import)\s+([A-Za-z_][\w.]*)", re.M)
    return {m.split(".")[0] for m in pattern.findall(path.read_text())}


@pytest.mark.parametrize("package,banned", FORBIDDEN.items())
def test_package_does_not_import(package: str, banned: tuple[str, ...]):
    for path in (ROOT / package).rglob("*.py"):
        # env/loader.py imports mujoco INSIDE apply_physics on purpose, so the
        # loader stays testable without MuJoCo installed; only module-level
        # imports are checked here.
        leaked = _imports(path) & set(banned)
        assert not leaked, f"{path.relative_to(ROOT)} imports {sorted(leaked)}"
