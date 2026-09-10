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


def test_submerged_swims_rather_than_walking():
    """The submerged preset must be a different mode of locomotion, not a
    slower walk: legs rowing in synchrony, feet off the floor, no adhesion."""
    from env.loader import load_preset

    water = load_preset("submerged_water")
    land = load_preset("dry_land")
    assert water.locomotion == "swim"
    assert land.locomotion == "walk"

    body = FlyBody(water)
    assert body.swimming
    # Buoyancy is folded into effective gravity; MuJoCo's fluid model has none.
    assert abs(float(water.physics["gravity"][2])) < abs(float(land.physics["gravity"][2]))

    cpg = TripodCPG(dt_s=body.timestep, seed=0, synchronous=True)
    contacts = []
    for i in range(6000):
        phase, amplitude = cpg.step(forward=1.0, turn=0.0, per_leg_gain=np.ones(6))
        obs = body.step(phase, amplitude)
        if i > 2000:
            contacts.append(int(obs["contact_found"].sum()))
    body.close()

    # A walking fly holds ~3 feet down; a swimming one should mostly be clear.
    assert np.mean(contacts) < 1.5, f"still standing on the floor ({np.mean(contacts):.2f} feet down)"


def test_swim_thrust_needs_an_asymmetric_stroke():
    """Thrust comes from drag asymmetry, so a symmetric stroke must go nowhere.

    This is the whole propulsive mechanism: fast power stroke with the legs
    spread, slow recovery with them folded.
    """
    import body.sim as bs
    from env.loader import load_preset

    def travel(fold, power_fraction):
        old_fold, old_pf = bs.SWIM_FOLD_RAD, bs.SWIM_POWER_FRACTION
        bs.SWIM_FOLD_RAD, bs.SWIM_POWER_FRACTION = fold, power_fraction
        try:
            preset = load_preset("submerged_water")
            preset.physics["gravity"] = [0.0, 0.0, 0.0]  # isolate thrust
            body = FlyBody(preset)
            body.sim.mj_data.qpos[2] += 25.0  # suspend clear of the floor
            body.sim.mj_data.qvel[:] = 0
            cpg = TripodCPG(dt_s=body.timestep, seed=0, synchronous=True)
            start = None
            for i in range(12000):
                phase, amplitude = cpg.step(1.0, 0.0, np.ones(6))
                obs = body.step(phase, amplitude)
                if i == 2000:
                    start = obs["position"].copy()
            end = body._prev_pos.copy()
            body.close()
            return float(np.linalg.norm((end - start)[:2]))
        finally:
            bs.SWIM_FOLD_RAD, bs.SWIM_POWER_FRACTION = old_fold, old_pf

    asymmetric = travel(fold=1.8, power_fraction=0.35)
    symmetric = travel(fold=0.0, power_fraction=0.5)
    assert asymmetric > 5 * max(symmetric, 1e-4), (
        f"asymmetric stroke {asymmetric:.3f} mm vs symmetric {symmetric:.3f} mm - "
        "thrust should come from the asymmetry"
    )


def test_cold_is_sluggish_in_muscle_as_well_as_nerve():
    """Temperature must reach the muscle, not only the membrane.

    An ectotherm in the cold has slower, weaker muscle too. With only the
    neural Q10 wired up, cold merely stepped less often; it did not look
    sluggish. The position actuators stand in for muscle, so their gain scales
    with temperature - inverted, since warm muscle is faster while a warm
    membrane time constant is shorter.
    """
    from env.loader import load_preset

    gains, taus = {}, {}
    for name in ("cold", "dry_land", "hot"):
        preset = load_preset(name)
        body = FlyBody(preset)
        gains[name] = body.muscle_gain
        taus[name] = preset.scaled_taus({"tau_m_ms": 20.0})["tau_m_ms"]
        body.close()

    assert gains["cold"] < gains["dry_land"] < gains["hot"], gains
    assert taus["hot"] < taus["dry_land"] < taus["cold"], taus
    # 25 C is the reference, so the baseline must be left exactly alone.
    assert gains["dry_land"] == pytest.approx(20.0)
    lo, hi = bs_clamp()
    assert all(lo <= g <= hi for g in gains.values())


def bs_clamp():
    import body.sim as bs

    return bs.MUSCLE_GAIN_CLAMP
