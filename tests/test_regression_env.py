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


def test_humid_reduces_grip_and_costs_travel():
    """Humidity acts on GRIP, and the cost is a closed-loop effect.

    The humid preset used to differ from dry air only by a 0.6% density change
    and two small sensory gains, so it behaved identically. It now models the
    mechanical effect that matters: a damp substrate reduces tarsal adhesion.

    Note the effect needs the brain in the loop. Driving the same body from a
    fixed CPG, lower adhesion actually makes the fly slightly FASTER
    (11.7 against 11.2 mm/s over 40k steps). The slowdown comes from reduced
    grip changing the load the legs report, which changes the descending drive.
    So this test runs a full Session rather than an open-loop gait - measured
    across seeds 0-2 the two are cleanly separated, 10.64 +- 0.08 mm/s dry
    against 5.84 +- 0.20 humid.
    """
    from env.loader import load_preset
    from session import Session

    dry = load_preset("dry_land")
    humid = load_preset("humid_air")

    # Grip is lower, but the air is NOT denser - humid air is in fact slightly
    # lighter, and the preset does not pretend otherwise.
    assert (humid.adhesion_gain or 200.0) < (dry.adhesion_gain or 200.0)
    assert float(humid.physics["density"]) < float(dry.physics["density"])

    def run(name):
        sess = Session(preset=name, connectome="synthetic", subset="motor",
                       synthetic_size=3000, connectome_dir="/tmp/flysim-test", seed=0)
        summary = sess.run(12000)
        sess.body.close()
        return summary

    dry_run, humid_run = run("dry_land"), run("humid_air")
    assert humid_run["net_speed_mm_s"] < dry_run["net_speed_mm_s"], "damp ground should cost travel"
    # Slipping specifically: more of the leg motion fails to become travel.
    assert humid_run["slip_ratio"] > dry_run["slip_ratio"], (
        f"slip {humid_run['slip_ratio']:.2f} humid vs {dry_run['slip_ratio']:.2f} dry"
    )


def test_grip_peaks_at_intermediate_humidity():
    """Attachment is NOT monotonic in humidity.

    Insect tarsal pads need some moisture to form the capillary bridges that
    create grip, so bone-dry air weakens them; a condensed film at high humidity
    makes them slip, so saturated air weakens them too. Best grip is in the
    middle. This is the shape the three air presets encode, and it is the reason
    dry_land is not the fastest preset.
    """
    from env.loader import load_preset

    dry = load_preset("dry_land").adhesion_gain
    mid = load_preset("temperate").adhesion_gain
    wet = load_preset("humid_air").adhesion_gain

    assert dry < mid > wet, f"grip should peak in the middle: {dry} / {mid} / {wet}"
    # Dry air should still be the better of the two extremes here - a dried pad
    # grips worse than an optimal one but better than one on a wet film.
    assert wet < dry

    # And the air itself must NOT be doing the work: humid air is slightly
    # lighter than dry, so no preset may claim extra drag for humidity.
    densities = [float(load_preset(n).physics["density"])
                 for n in ("dry_land", "temperate", "humid_air")]
    assert densities[0] > densities[1] > densities[2], densities
    assert max(densities) / min(densities) < 1.02, "humidity must not fake a drag effect"


def body_timestep() -> float:
    """MuJoCo timestep the body runs at."""
    return 1e-4


def test_windy_gusts_actually_vary_the_wind():
    """Gusts must reach mjOption, and must swing wide enough to matter.

    A steady wind cannot blow the fly around - below ~1.5 m/s it merely biases
    the path, at 2.0 it rolls the fly onto its back for good, and at 2.5 it
    lifts it off the floor entirely. Gusts are what buffet it, and only slow
    ones: fast gusts average out and left the fly faster and straighter than
    steady wind.
    """
    from env.loader import load_preset

    windy = load_preset("windy")
    assert windy.wind_gust is not None, "windy should gust, not blow steadily"
    assert windy.wind_gust_hz < 1.0, "fast gusts average out and do not buffet"
    assert windy.wind_gust[1] > 0, "lateral gusts are what push it off heading"

    # Must cover a FULL gust period or the sample catches only part of the
    # swing: at 0.3 Hz one cycle is 3.3 s, i.e. 33k steps at the 1e-4 timestep.
    steps = int(1.2 / windy.wind_gust_hz / body_timestep())
    body = FlyBody(windy)
    winds = []
    for _ in range(steps):
        body.step(np.zeros(6), np.zeros(6))
        winds.append(body.sim.mj_model.opt.wind[0])
    body.close()

    lo, hi = min(winds), max(winds)
    assert hi - lo > 1000.0, f"gusts too weak to buffet: {lo:.0f}..{hi:.0f}"
    # And must stay under the ~2 m/s speed that capsizes the fly outright.
    assert hi < 2000.0, f"gust peak {hi:.0f} would roll the fly onto its back"


def test_hot_is_frantic_not_fast():
    """35 C sits PAST the thermal optimum, so hot must not just be a quicker
    version of temperate.

    Ectotherm locomotor performance rises to an optimum around 25-30 C and then
    declines. Here the Q10 speed-up outruns the body: the network commands a
    stride the legs cannot track. The signature is high leg speed with poor
    travel - the most active preset, not the most effective.
    """
    from session import Session

    def run(name):
        sess = Session(preset=name, connectome="synthetic", subset="motor",
                       synthetic_size=3000, connectome_dir="/tmp/flysim-test", seed=0)
        summary = sess.run(12000)
        sess.body.close()
        return summary

    hot, mid = run("hot"), run("temperate")

    # Legs working harder than at the optimum...
    assert hot["mean_speed_mm_s"] > mid["mean_speed_mm_s"], "hot should be the more active"
    # ...while more of that motion is wasted.
    assert hot["slip_ratio"] > mid["slip_ratio"], (
        f"slip {hot['slip_ratio']:.2f} hot vs {mid['slip_ratio']:.2f} temperate"
    )


def test_cold_drags_rather_than_walking():
    """10 C is cold enough that the fly stops running a tripod gait.

    Both Q10 channels bite at once - membrane time constants stretch to ~70 ms
    and muscle gain falls to ~5.7 - so the fly cannot hold itself up properly.
    The signature is postural, not just slow: a walking fly keeps about three
    feet down at any moment, a cold-stunned one sags and drags four or more.

    This is the mirror of the hot preset. Both are slow, but cold has LOW leg
    speed (the legs barely move) while hot has the highest of any preset (the
    legs move faster than they can usefully track), so net speed alone would
    make the two look alike.
    """
    from env.loader import load_preset

    cold = load_preset("cold")
    mid = load_preset("temperate")
    assert cold.temperature_c < 12.0, "15 C is not cold enough to impair a fly"

    # The Q10 must not be truncated by the guard rail - the upper tau bound is
    # not stability-critical and previously clipped this preset by 13%.
    wanted = 20.0 * (float(cold.neural["q10"]) ** ((25.0 - cold.temperature_c) / 10.0))
    assert cold.scaled_taus({"tau_m_ms": 20.0})["tau_m_ms"] == pytest.approx(wanted), (
        "tau_clamp_ms is truncating the cold preset's Q10"
    )

    def feet_down(preset):
        body = FlyBody(preset)
        cpg = TripodCPG(dt_s=body.timestep, seed=0)
        counts = []
        for i in range(6000):
            phase, amplitude = cpg.step(0.4, 0.0, np.ones(6))
            obs = body.step(phase, amplitude)
            if i > 2000:
                counts.append(int(obs["contact_found"].sum()))
        body.close()
        return float(np.mean(counts))

    assert feet_down(cold) > feet_down(mid), "a cold fly should drag more feet than it lifts"
