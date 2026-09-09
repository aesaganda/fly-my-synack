"""Load an environment preset and apply it to BOTH the physics and the brain.

A preset is a YAML file in `environments/`. Nothing about a preset is hardcoded
here: adding `environments/vacuum.yaml` makes `--env vacuum` work with no code
change. See `environments/_README.md` for the schema and the unit warning.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

from env.q10 import scale_tau_ms

# The only unit system the presets may declare. FlyGym's NeuroMechFly model is
# in mm/g/s, so SI fluid values are wrong by up to 1e6. Rejecting anything else
# is cheaper than debugging a fly that sinks like a stone.
REQUIRED_UNITS = "mm_g_s"

PRESET_DIR = Path(os.environ.get("FLY_ENV_DIR", Path(__file__).resolve().parent.parent / "environments"))

_REQUIRED_PHYSICS = ("density", "viscosity", "wind", "gravity", "integrator")
_REQUIRED_NEURAL = ("temperature_c", "q10", "tau_clamp_ms")
_REQUIRED_SENSORY = ("mechanosensory_gain", "johnstons_organ_gain")

# MuJoCo integrator names accepted in YAML -> mjtIntegrator enum member name.
_INTEGRATORS = {
    "euler": "mjINT_EULER",
    "rk4": "mjINT_RK4",
    "implicit": "mjINT_IMPLICIT",
    "implicitfast": "mjINT_IMPLICITFAST",
}


@dataclass(frozen=True)
class EnvPreset:
    name: str
    description: str
    physics: dict
    neural: dict
    sensory: dict
    path: Path

    @property
    def temperature_c(self) -> float:
        return float(self.neural["temperature_c"])

    def scaled_taus(self, base_taus_ms: dict[str, float]) -> dict[str, float]:
        """Apply this preset's Q10 scaling to a dict of base time constants."""
        clamp = tuple(self.neural["tau_clamp_ms"])
        return {
            k: scale_tau_ms(v, self.temperature_c, float(self.neural["q10"]), clamp)
            for k, v in base_taus_ms.items()
        }

    def fluid_is_active(self) -> bool:
        """MuJoCo disables lift/drag at density 0 and viscous forces at viscosity 0.

        A `wind` value with both of these at zero is a silent no-op, which is
        exactly the kind of bug that looks like 'wind just doesn't do much'.
        """
        return float(self.physics["density"]) > 0.0 or float(self.physics["viscosity"]) > 0.0


def list_presets(preset_dir: Path | None = None) -> list[str]:
    d = Path(preset_dir or PRESET_DIR)
    return sorted(p.stem for p in d.glob("*.yaml") if not p.stem.startswith("_"))


def load_preset(name: str, preset_dir: Path | None = None) -> EnvPreset:
    d = Path(preset_dir or PRESET_DIR)
    path = d / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"no preset {name!r} in {d} (have: {', '.join(list_presets(d)) or 'none'})")

    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a YAML mapping")

    units = raw.get("units")
    if units != REQUIRED_UNITS:
        raise ValueError(
            f"{path}: units must be {REQUIRED_UNITS!r}, got {units!r}. "
            "FlyGym is in mm/g/s - SI fluid values are off by ~1e6. "
            "See environments/_README.md."
        )

    for section, required in (
        ("physics", _REQUIRED_PHYSICS),
        ("neural", _REQUIRED_NEURAL),
        ("sensory", _REQUIRED_SENSORY),
    ):
        block = raw.get(section)
        if not isinstance(block, dict):
            raise ValueError(f"{path}: missing '{section}:' block")
        missing = [k for k in required if k not in block]
        if missing:
            raise ValueError(f"{path}: {section} is missing {missing}")

    integrator = str(raw["physics"]["integrator"]).lower()
    if integrator not in _INTEGRATORS:
        raise ValueError(f"{path}: unknown integrator {integrator!r}, expected one of {sorted(_INTEGRATORS)}")

    for vec, n in (("wind", 3), ("gravity", 3)):
        if len(raw["physics"][vec]) != n:
            raise ValueError(f"{path}: physics.{vec} must have {n} components")
    if len(raw["neural"]["tau_clamp_ms"]) != 2:
        raise ValueError(f"{path}: neural.tau_clamp_ms must be [lo, hi]")

    preset = EnvPreset(
        name=raw.get("name", name),
        description=str(raw.get("description", "")).strip(),
        physics=raw["physics"],
        neural=raw["neural"],
        sensory=raw["sensory"],
        path=path,
    )

    wind = [float(w) for w in preset.physics["wind"]]
    if any(wind) and not preset.fluid_is_active():
        raise ValueError(
            f"{path}: wind is set but density and viscosity are both 0. MuJoCo "
            "scales wind forces by density/viscosity, so this preset would be a "
            "silent no-op."
        )
    return preset


def apply_physics(mj_model, preset: EnvPreset) -> None:
    """Write a preset's physics onto an already-compiled MuJoCo model.

    density/viscosity/wind/gravity/integrator all live in mjOption, which is
    plain mutable memory in the Python bindings - no recompile needed. That is
    what makes switching presets mid-run cheap.

    Must be called AFTER `world.add_fly()`, because add_fly overwrites <option>
    from the fly's own mujoco_globals.yaml.
    """
    import mujoco  # local import: keeps the loader testable without MuJoCo

    p = preset.physics
    mj_model.opt.density = float(p["density"])
    mj_model.opt.viscosity = float(p["viscosity"])
    mj_model.opt.wind[:] = [float(w) for w in p["wind"]]
    mj_model.opt.gravity[:] = [float(g) for g in p["gravity"]]
    mj_model.opt.integrator = getattr(mujoco.mjtIntegrator, _INTEGRATORS[str(p["integrator"]).lower()])
