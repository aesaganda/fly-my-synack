"""Regression: fluid parameters must actually change joint kinematics.

This is a sanity check on the physics plumbing - proof that the preset reaches
MuJoCo and has a mechanical effect - and explicitly NOT biological validation.
Nothing here claims the fly behaves like a real fly underwater.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("flygym")

from body.cpg import TripodCPG  # noqa: E402
from body.sim import FlyBody  # noqa: E402
from env.loader import load_preset  # noqa: E402

STEPS = 4000


def _run(preset_name: str) -> dict:
    """Same CPG, same seed, same gait - only the environment differs.

    The brain is deliberately left out so this test isolates the physics.
    """
    body = FlyBody(load_preset(preset_name))
    cpg = TripodCPG(dt_s=body.timestep, seed=0)
    start = body._prev_pos.copy()
    angles, speeds = [], []
    for i in range(STEPS):
        phase, amplitude = cpg.step(forward=1.0, turn=0.0, per_leg_gain=np.ones(6))
        obs = body.step(phase, amplitude)
        if i % 20 == 0:
            angles.append(obs["joint_angles"])
            speeds.append(obs["speed"])
    roll, pitch = body.orientation_deg()
    body.close()
    return {
        "angles": np.asarray(angles),
        "speed": float(np.mean(speeds)),
        "displacement": float(np.linalg.norm((obs["position"] - start)[:2])),
        "roll": roll,
        "pitch": pitch,
    }


@pytest.fixture(scope="module")
def dry():
    return _run("dry_land")


@pytest.fixture(scope="module")
def water():
    return _run("submerged_water")


def test_water_changes_joint_kinematics(dry, water):
    """Identical motor commands must produce different achieved joint angles."""
    diff = np.abs(dry["angles"] - water["angles"]).mean()
    assert diff > 1e-4, f"submerging changed joint kinematics by only {diff:.2e} rad"


def test_water_slows_the_fly(dry, water):
    """~1000x the medium density has to cost something in drag.

    Measured on mean speed rather than net displacement: over a few thousand
    steps displacement is dominated by heading wander, whereas speed is the
    direct signature of drag. (Brain-driven 30k-step runs show the same thing
    more strongly: 4.7 mm/s submerged vs 6.9 mm/s on dry land.)
    """
    assert water["speed"] < dry["speed"]


def test_the_fly_stays_upright(dry, water):
    """The gait must keep the animal on its feet.

    This exists because an earlier version passed every other test while the
    fly rolled onto its back and flailed: a height-only fall check could not
    see it, since an upended fly's thorax sits well above the floor.
    """
    for name, run in (("dry_land", dry), ("submerged_water", water)):
        assert abs(run["roll"]) < 90.0, f"{name}: fly capsized (roll {run['roll']:.0f} deg)"
        assert abs(run["pitch"]) < 75.0, f"{name}: fly pitched over ({run['pitch']:.0f} deg)"


def test_dry_land_is_reproducible():
    """If this drifts, the comparison above means nothing."""
    a, b = _run("dry_land"), _run("dry_land")
    assert np.allclose(a["angles"], b["angles"])


def test_wind_displaces_the_fly():
    """MuJoCo scales wind by density; a zero-density windy preset would be a
    silent no-op, so this also guards that."""
    still = _run("dry_land")
    windy = _run("windy")
    assert not np.allclose(still["angles"], windy["angles"])
