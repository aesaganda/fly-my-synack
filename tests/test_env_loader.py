"""Preset loader: schema, the unit trap, and the silent-no-op wind trap."""

from __future__ import annotations

import textwrap

import pytest

from env.loader import EnvPreset, apply_physics, list_presets, load_preset
from env.q10 import q10_factor, scale_tau_ms

ALL = ["cold", "dry_land", "hot", "humid_air", "submerged_water", "temperate", "windy"]


def test_all_presets_present_and_loadable():
    assert list_presets() == ALL
    for name in ALL:
        assert load_preset(name).name == name


def test_water_density_is_in_model_units_not_si():
    """The single most likely silent error: pasting SI values into a mm/g/s model.

    Water is ~1e-3 g/mm^3, NOT ~1000 kg/m^3. A preset carrying 1000 would make
    drag wrong by a factor of a million.
    """
    water = load_preset("submerged_water")
    assert 5e-4 < float(water.physics["density"]) < 5e-3
    assert 5e-4 < float(water.physics["viscosity"]) < 5e-3

    air = load_preset("temperate")
    assert 1e-7 < float(air.physics["density"]) < 1e-5
    # Water must be ~1000x denser than air, as in reality.
    assert float(water.physics["density"]) / float(air.physics["density"]) > 500


def test_submerged_avoids_euler():
    """MuJoCo warns Euler handles body viscosity poorly; FlyGym defaults to Euler."""
    assert load_preset("submerged_water").physics["integrator"] == "implicitfast"


def test_windy_has_nonzero_density_or_wind_is_a_noop():
    windy = load_preset("windy")
    assert any(float(w) for w in windy.physics["wind"])
    assert windy.fluid_is_active(), "wind without density/viscosity is silently ignored by MuJoCo"


def _write(tmp_path, body: str, name: str = "custom"):
    p = tmp_path / f"{name}.yaml"
    p.write_text(textwrap.dedent(body))
    return p


BASE = """
    name: custom
    units: {units}
    physics:
      density: {density}
      viscosity: 0.0
      wind: {wind}
      gravity: [0.0, 0.0, -9810.0]
      integrator: Euler
    neural:
      temperature_c: 25.0
      q10: 2.3
      tau_clamp_ms: [2.0, 40.0]
    sensory:
      mechanosensory_gain: 1.0
      johnstons_organ_gain: 1.0
"""


def test_si_units_are_rejected(tmp_path):
    _write(tmp_path, BASE.format(units="SI", density=1000.0, wind="[0,0,0]"))
    with pytest.raises(ValueError, match="units must be"):
        load_preset("custom", tmp_path)


def test_wind_without_fluid_is_rejected(tmp_path):
    _write(tmp_path, BASE.format(units="mm_g_s", density=0.0, wind="[500.0, 0.0, 0.0]"))
    with pytest.raises(ValueError, match="silent no-op"):
        load_preset("custom", tmp_path)


def test_missing_section_is_rejected(tmp_path):
    (tmp_path / "broken.yaml").write_text("name: broken\nunits: mm_g_s\nphysics: {}\n")
    with pytest.raises(ValueError, match="physics is missing"):
        load_preset("broken", tmp_path)


def test_unknown_preset_lists_alternatives(tmp_path):
    with pytest.raises(FileNotFoundError, match="no preset"):
        load_preset("atlantis")


# ---------------------------------------------------------------- Q10 -----

def test_q10_direction_and_clamp():
    assert q10_factor(25.0, 2.3) == pytest.approx(1.0)
    assert q10_factor(35.0, 2.3) < 1.0   # hot -> faster
    assert q10_factor(15.0, 2.3) > 1.0   # cold -> slower

    hot = load_preset("hot").scaled_taus({"tau_m_ms": 20.0})
    cold = load_preset("cold").scaled_taus({"tau_m_ms": 20.0})
    base = load_preset("dry_land").scaled_taus({"tau_m_ms": 20.0})
    assert hot["tau_m_ms"] < base["tau_m_ms"] < cold["tau_m_ms"]

    # The clamp is a guard rail against blowing up the integrator.
    assert scale_tau_ms(20.0, -100.0, 2.3, (2.0, 40.0)) == 40.0
    assert scale_tau_ms(20.0, 500.0, 2.3, (2.0, 40.0)) == 2.0


def test_q10_rejects_nonsense():
    with pytest.raises(ValueError):
        q10_factor(25.0, 0.0)
    with pytest.raises(ValueError):
        scale_tau_ms(20.0, 25.0, 2.3, (40.0, 2.0))


# ----------------------------------------------------- MuJoCo plumbing -----

def test_apply_physics_writes_every_field_to_mjoption():
    """The preset must actually reach mjOption - not just parse cleanly."""
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><body><geom size='1'/><freejoint/></body></worldbody></mujoco>"
    )
    assert model.opt.density == 0.0  # MuJoCo default: fluid forces OFF

    apply_physics(model, load_preset("submerged_water"))
    assert model.opt.density == pytest.approx(9.98e-4)
    assert model.opt.viscosity == pytest.approx(1.002e-3)
    assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST

    apply_physics(model, load_preset("windy"))
    assert model.opt.wind[0] == pytest.approx(1500.0)
    assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_EULER
    assert model.opt.gravity[2] == pytest.approx(-9810.0)


def test_preset_switch_is_reversible_on_a_live_model():
    """Mid-run preset switching depends on mjOption being plain mutable memory."""
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
    apply_physics(model, load_preset("submerged_water"))
    apply_physics(model, load_preset("dry_land"))
    assert model.opt.density == pytest.approx(1.184e-6)
